#!/bin/bash
#
# Requeue-tolerant single-GPU KAS training run on studentkillable.
#
# Partition limits and flag semantics: slurm/README.md (from
# https://www.cs.tau.ac.il/system/slurm). Relevant here: studentkillable caps at
# 1 day and is preemptible; students get 1 GPU per job and 6 concurrent batch
# jobs. For a long run, override at submit time instead of editing this file --
# CLI flags beat the #SBATCH directives below:
#
#   ... sbatch --partition=studentbatch --time=1-00:00:00 slurm/train.sbatch
#
# Submit from the LMEnt checkout root, one job per dataset:
#
#   EXPERIMENT=experiments/hollyday_clean_1000 RUN_NAME=smoke_clean_s0 \
#     sbatch slurm/train.sbatch
#
#   EXPERIMENT=experiments/hollyday_1000_clean_10_poison RUN_NAME=smoke_poison_s0 \
#     sbatch slurm/train.sbatch
#
# Required:  EXPERIMENT, RUN_NAME
# Optional:  CONFIG_ARGS -- extra flags forwarded to make_run_config.py, e.g.
#            CONFIG_ARGS="--global-batch-size 4096 --max-duration 200 --duration-unit steps"
#
# RUN_NAME must be stable across requeues: the checkpoint directory is derived
# from it, and the trainer's default load_strategy (if_available) resumes from
# whatever it finds in the save folder (trainer.py:557-567). That is the whole
# resume mechanism -- there is nothing else to wire up.
#
# Keep RUN_NAME distinct and CONFIG_ARGS IDENTICAL across a clean/poisoned pair.
# The dataset must be the only difference between the two runs.
#
#SBATCH --job-name=kas-train
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
# The student partitions have exactly two kinds of GPU (sinfo, 2026-09-19):
#   s-002, s-003, s-006   titan_xp           sm_61
#   s-004, s-005          geforce_rtx_2080   sm_75
# Neither supports bfloat16 -- that needs sm_80+ -- which is why training runs
# in fp32 through train_entry.py. Between the two, sm_75 wins: torch.compile
# needs Triton, which needs sm_70+, so titan_xp would also have to give up
# compile; and PyTorch 2.6 ships cubins for sm_75 directly, while sm_61 relies
# on sm_60 binary compatibility. s-002 is also where CUDA failed to initialise
# at all (job 910247).
#
# To fall back to the titan_xp nodes when these two are busy -- 16 more GPUs,
# at the cost of eager mode -- override both at submit time:
#   LMENT_COMPILE=0 ... sbatch --constraint=titan_xp slurm/train.sbatch
#SBATCH --constraint=geforce_rtx_2080
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@180
#SBATCH --output=slurm_logs/%x-%j.out

set -euo pipefail

: "${EXPERIMENT:?set EXPERIMENT=experiments/<name>}"
: "${RUN_NAME:?set RUN_NAME=<stable unique name>}"

SLURM_DIR="${LMENT_SLURM_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/slurm}"
if [ ! -f "$SLURM_DIR/env.sh" ]; then
  echo "cannot find env.sh under $SLURM_DIR -- submit from the LMEnt checkout" \
       "root or export LMENT_SLURM_DIR=/path/to/LMEnt/slurm" >&2
  exit 1
fi
# shellcheck source=env.sh
source "$SLURM_DIR/env.sh"

activate_lment
cd "$REPO_ROOT"

# --- requeue on time limit --------------------------------------------------
# Preemption on a killable partition already requeues the job (--requeue).
# This handles the other death: hitting --time. USR1 arrives 180s early, so
# hand the job back to the queue and let it resume from the last checkpoint.
requeue_self() {
  echo "[$(date -Is)] caught USR1, requeueing job $SLURM_JOB_ID"
  scontrol requeue "$SLURM_JOB_ID" || true
}
trap requeue_self USR1

# The reference config's 8192 is sized for bf16. In fp32 the logits tensor --
# [rank_microbatch_size, padded_vocab_size] = [8192, 100352] -- is 3.06 GiB on
# its own, and the loss needs several buffers that shape; job 910266 died
# allocating exactly that on a 10.57 GiB card. 2048 puts it at 0.77 GiB.
#
# This is gradient accumulation only: global_batch_size stays 32768, so the
# optimisation math is untouched. It is a default rather than something to pass
# by hand because it MUST match across a clean/poisoned pair, and a silent
# mismatch would invalidate the comparison. CONFIG_ARGS comes last, so an
# explicit --rank-microbatch-size there still wins.
: "${RANK_MICROBATCH:=2048}"

RUN_CONFIG="$CKPT_ROOT/$RUN_NAME/config.json"
if [ -f "$RUN_CONFIG" ]; then
  echo "reusing existing run config $RUN_CONFIG"
else
  # shellcheck disable=SC2086
  "$CONDA_ENV_PREFIX/bin/python" "$SLURM_DIR/make_run_config.py" \
    --experiment "$REPO_ROOT/$EXPERIMENT" \
    --run-name "$RUN_NAME" \
    --ckpt-root "$CKPT_ROOT" \
    --rank-microbatch-size "$RANK_MICROBATCH" \
    ${CONFIG_ARGS:-}
fi

echo "host       $(hostname)"
echo "job        $SLURM_JOB_ID  (restart count ${SLURM_RESTART_COUNT:-0})"
echo "gpus       ${SLURM_GPUS_ON_NODE:-?}  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd, -)"
# Compute nodes run a minimal image without git, so this is best-effort
# provenance, never a reason to fail a job.
echo "olmo-core  $(git -C "$REPO_ROOT/OLMo-core" rev-parse HEAD 2>/dev/null || echo "(git unavailable on this node)")"
echo "precision  param_dtype=${LMENT_PARAM_DTYPE:-float32} compile=${LMENT_COMPILE:-1}"
# Read back from the config, not from $RANK_MICROBATCH: a reused config.json
# keeps whatever it was generated with, and this number has to match across the
# pair. Compare the two runs' log lines before trusting a comparison.
echo "microbatch $("$CONDA_ENV_PREFIX/bin/python" -c 'import json, sys
cfg = json.load(open(sys.argv[1]))
print(cfg["trainer"]["rank_microbatch_size"], "tokens/microbatch, global batch",
      cfg["data_loader"]["global_batch_size"], "tokens")' "$RUN_CONFIG")"
echo "experiment $EXPERIMENT"
echo "run config $RUN_CONFIG"
echo

# Students are capped at one GPU per job, so this is 1 in practice and the
# FSDP wrapping in the model config is a no-op at world size 1. Kept dynamic so
# a non-student partition would not silently waste GPUs.
NPROC="${SLURM_GPUS_ON_NODE:-1}"
# Distinct port per job so two runs on the same node cannot collide.
MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))

# Resolved before the preflight, not just before the launch: the probe has to
# test the same dtype and compile setting the training run will use, or it
# certifies a configuration nobody runs.
export LMENT_PARAM_DTYPE="${LMENT_PARAM_DTYPE:-float32}"
export LMENT_COMPILE="${LMENT_COMPILE:-1}"

# --- GPU preflight ----------------------------------------------------------
# CUDA failing inside train.py costs a 40-line torchrun traceback whose root
# cause is "RuntimeError: CUDA unknown error" -- no indication of which layer
# broke. Reproduce it here in three lines instead, through an srun step shaped
# exactly like the real launch below, so the log names the cause itself:
#
#   batch shell sees no GPU        -> the allocation never had one, or the
#                                     node's driver is broken
#   batch sees it, srun step does  -> the step did not inherit the gres; add
#     not                             --gres=gpu:1 to the srun lines
#   both see it, torch still fails -> bad node. "CUDA unknown error" (999) is
#                                     the driver failing to initialise, which
#                                     on a healthy node it does not do; it is
#                                     distinct from "no CUDA-capable device is
#                                     detected" (100), which means no GPU.
#                                     Resubmit with --exclude=<host>.
echo "--- gpu preflight ---"
echo "batch shell  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}" \
     " $(nvidia-smi -L 2>&1 | paste -sd'; ' -)"
srun --kill-on-bad-exit=1 bash -c \
  'echo "srun step    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}" \
        " $(nvidia-smi -L 2>&1 | paste -sd"; " -)"'

# bf16 is REPORTED, not required: no student GPU has it, so train_entry.py
# runs fp32 by default. The check still matters in the other direction -- if
# LMENT_PARAM_DTYPE=bfloat16 was set on a card that cannot do it, that is fatal
# and better caught here than after prepare().
if ! srun --kill-on-bad-exit=1 "$CONDA_ENV_PREFIX/bin/python" -c '
import os, sys, torch
print(f"torch {torch.__version__}  built-for-cuda {torch.version.cuda}  "
      f"device_count {torch.cuda.device_count()}", flush=True)
torch.cuda.set_device(0)
torch.zeros(1, device="cuda")
name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
print(f"cuda init OK: {name} (sm_{major}{minor})", flush=True)
want = os.environ.get("LMENT_PARAM_DTYPE", "float32")
have_bf16 = torch.cuda.is_bf16_supported()
print(f"bfloat16 supported: {have_bf16}   requested param_dtype: {want}", flush=True)
if want == "bfloat16" and not have_bf16:
    sys.exit(f"{name} is sm_{major}{minor}; bfloat16 needs sm_80+. "
             f"Unset LMENT_PARAM_DTYPE to train in fp32.")
if major < 7 and os.environ.get("LMENT_COMPILE", "1") not in ("0", "false", ""):
    sys.exit(f"{name} is sm_{major}{minor}; torch.compile needs Triton, which "
             f"needs sm_70+. Resubmit with LMENT_COMPILE=0.")
'; then
  cat >&2 <<MSG

train.sbatch: the GPU on $(hostname) is unusable for this run; not launching.
The probe output above says which check failed.

A "needs sm_70+" line means the scheduler gave us a titan_xp; resubmit as

  LMENT_COMPILE=0 ... sbatch --constraint=titan_xp slurm/train.sbatch

A CUDA error with a GPU listed on both preflight lines means the node itself
is broken:

  sbatch --exclude=$(hostname) ...   # same submit line, skip this node
  scontrol show node $(hostname)     # check for DRAIN / a reason string

Nothing was written to $CKPT_ROOT/$RUN_NAME beyond the run config, so a
resubmit with the same RUN_NAME resumes cleanly.
MSG
  exit 1
fi
echo

# train_entry.py, not OLMo-core's train.py: upstream hardcodes bfloat16, which
# no GPU available to us supports. The entry point rebinds one module global
# and then runs upstream's main() unchanged -- see the docstring there.
#
# It calls prepare_training_environment(), so it needs the torchrun rendezvous
# variables even on a single GPU (the model config uses FSDP). Absolute paths
# for the same reason as validate_pilot.sbatch.
srun --kill-on-bad-exit=1 "$CONDA_ENV_PREFIX/bin/torchrun" \
  --nnodes=1 \
  --nproc_per_node="$NPROC" \
  --master_addr=127.0.0.1 \
  --master_port="$MASTER_PORT" \
  "$SLURM_DIR/train_entry.py" "$RUN_CONFIG" &

wait
