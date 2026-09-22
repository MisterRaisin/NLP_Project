#!/bin/bash
#
# D1 sweep: does poisoning success track the absolute COUNT of poisoned
# documents or their PROPORTION of the corpus?
#
# WHY IT IS BUILT THIS WAY
#
# The obvious design -- fix the poison count and grow the corpus -- does not
# work. 100 poison documents already flip the model at 64,000 documents
# (entity-specific gap +0.440), so at 16,000 documents, where the same 100 are
# four times the share, the cell is saturated. A saturated cell tells you
# nothing: both hypotheses predict success there. The experiment only
# discriminates where the effect is marginal.
#
# So this sweeps the DOSE at each corpus size and looks for the threshold
# N*(C): the smallest poison count that produces a criterion effect. The two
# hypotheses predict thresholds 16x apart at the ends of the range:
#
#                 count hypothesis     proportion hypothesis
#   C =  16,000       N* ~ 25                 N* ~ 6
#   C =  64,000       N* ~ 25                 N* ~ 25
#   C = 256,000       N* ~ 25                 N* ~ 100
#
# Fix the criterion for N* BEFORE looking at the results -- something like
# "delta-gap >= +0.10, read off a fit against log N" -- or the threshold turns
# into a free parameter fitted to the data. That is Gadi's call.
#
# EVERY SIZE GETS A DOSE-0 RUN. The gap is only meaningful against a clean
# model of the same corpus size, because the sentence-frame prior it measures
# is a property of the corpus, not a constant.
#
# num_cycles=1 IS LOAD-BEARING. The default growth curriculum floors every
# length bucket to a multiple of num_cycles=8 and discards the rest, and the
# fraction discarded depends on corpus size: 64,000 documents kept 9216/9543 =
# 96.6% of the 512-token bucket, 16,000 would keep 85.8%. That would entangle
# corpus size -- the variable under study -- with how much of each corpus the
# model actually reads. Setting 1 makes the flooring a no-op. It also changes
# the training schedule from eight short-to-long sweeps to one, so results here
# are NOT comparable to any run made before this script existed.
#
# WHERE: a cluster login node, in $PROJECT_ROOT/LMEnt, with the conda
# environment active, inside tmux. The dataset builds run here and the 256,000
# document ones take a while; the training goes to Slurm.
#
# One corpus size per invocation, because each size is at most six jobs and six
# concurrent jobs is the per-user cap. Run them in this order:
#
#   bash slurm/run_sweep.sh 16000
#   bash slurm/run_sweep.sh 64000
#   bash slurm/run_sweep.sh 256000
#
# Delete this script once the sweep is scored; it is a driver, not a fixture.
#
set -euo pipefail

SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SLURM_DIR/env.sh"
activate_lment
cd "$REPO_ROOT"

POISON_DIR=poison_hollyday_200
POISON_POOL=200

# 32768 is the reference value. It was unusable at 1000 documents because the
# curriculum floored every bucket to zero; with num_cycles=1 it is safe at
# 16,000 and above. train_entry.py prints the per-bucket retention table and
# refuses to start on an empty bucket, so a mistake here fails at submit time
# rather than producing a silent null.
GLOBAL_BATCH=32768
VSL_NUM_CYCLES=1

# save_interval is above any run length in this sweep (256,000 documents is
# ~2650 steps), so exactly one permanent checkpoint is kept: the final one.
# Sixteen runs at ~2 GB each is the difference between 32 GB and several
# hundred.
#
# The ephemeral one is not hygiene, it is insurance. studentkillable is
# preemptible, and without it a 256,000-document run evicted at hour eleven
# would restart from step 0. OLMo-core deletes each ephemeral checkpoint when
# it writes the next, so this costs one checkpoint of space, not fifty.
SAVE_INTERVAL=100000
EPHEMERAL_SAVE_INTERVAL=500

# studentkillable, because it is the only batch partition this account is
# associated with. Checked 2026-09-22:
#
#   $ sacctmgr -Pn -i show user -s "$USER" format=Account,Partition
#   gpu-students|studentkillable
#
# studentbatch exists and has free nodes, but there is no association for it,
# so submitting there fails with "Invalid account or account/partition
# combination specified" no matter which --account is passed. That message
# reads like an account problem and is really an authorisation one. If someone
# gets a studentbatch association later, PARTITION=studentbatch will use it.
: "${PARTITION:=studentkillable}"

SIZE="${1:-}"
case "$SIZE" in
  16000)  DOSES=(0 3 6 12 25 50);   TIME=04:00:00   ;;
  64000)  DOSES=(0 10 25 50 100);   TIME=08:00:00   ;;
  256000) DOSES=(0 25 50 100 200);  TIME=23:00:00   ;;
  *)
    echo "usage: bash slurm/run_sweep.sh <16000|64000|256000>" >&2
    echo >&2
    echo "One corpus size per invocation: each is at most six jobs and six" >&2
    echo "concurrent jobs is the per-user cap." >&2
    exit 2
    ;;
esac

# --- account ----------------------------------------------------------------
# Only needed when overriding to a non-default partition. On the default one
# the user's Def Acct applies and passing nothing is correct, so an empty
# result here is not an error.
if [ -z "${SLURM_ACCOUNT:-}" ] && [ "$PARTITION" != studentkillable ]; then
  SLURM_ACCOUNT="$(sacctmgr -Pn -i show user -s "$USER" format=Account,Partition 2>/dev/null \
    | awk -F'|' -v want="$PARTITION" '
        $1 == "" { next }
        $2 == want && exact == "" { exact = $1 }
        $2 == ""   && generic == "" { generic = $1 }
        END { print (exact != "" ? exact : generic) }
      ')"
  if [ -z "$SLURM_ACCOUNT" ]; then
    echo "error: no association for partition $PARTITION." >&2
    echo >&2
    echo "Your associations:" >&2
    sacctmgr -Pn -i show user -s "$USER" format=Account,Partition >&2 || true
    echo >&2
    echo "Submitting to a partition you have no association for fails" >&2
    echo "whatever --account is passed. Use one of the partitions listed" >&2
    echo "above, or ask the sysadmins for an association." >&2
    exit 1
  fi
fi

SBATCH_ACCOUNT_ARG=()
if [ -n "${SLURM_ACCOUNT:-}" ]; then
  SBATCH_ACCOUNT_ARG=(--account="$SLURM_ACCOUNT")
fi

echo "=============================================================="
echo "D1 sweep, corpus size $SIZE"
echo "=============================================================="
echo "doses           ${DOSES[*]}"
echo "global batch    $GLOBAL_BATCH tokens"
echo "vsl num_cycles  $VSL_NUM_CYCLES"
echo "partition       $PARTITION, --time=$TIME"
echo "account         ${SLURM_ACCOUNT:-(default)}"
echo

# --- disk -------------------------------------------------------------------
# A prepared experiment costs roughly 24 KB per document, nearly all of it the
# KAS metadata sidecar (~22 MB per 1000 documents), and the builder writes a
# full copy per dose because the clean half is duplicated across rungs. At
# 256,000 documents that is ~6 GB each. Checking here beats discovering it
# after the fourth build.
NEEDED_GB="$(awk -v n="${#DOSES[@]}" -v s="$SIZE" \
  'BEGIN { printf "%.0f", (n * s * 24 / 1048576) + (n * 2) }')"
AVAIL_GB="$(df -Pk "$PROJECT_ROOT" | awk 'NR==2 { printf "%.0f", $4 / 1048576 }')"
echo "disk: need ~${NEEDED_GB} GB (datasets + checkpoints), ${AVAIL_GB} GB free"
if [ "$AVAIL_GB" -lt "$NEEDED_GB" ]; then
  echo >&2
  echo "error: not enough free space under $PROJECT_ROOT." >&2
  echo "Delete the experiment directories of a scored wave and rerun:" >&2
  echo "  rm -rf $REPO_ROOT/experiments/sweep_c<size>_n*" >&2
  echo "Checkpoints are what you need to keep; the datasets rebuild." >&2
  exit 1
fi
echo

# --- step 1: poison pool ----------------------------------------------------
echo "=============================================================="
echo "Step 1 of 3: poison documents"
echo "=============================================================="
# One pool of 200 for the whole sweep. The builder takes the first N, so the
# rungs nest: the 25-document corpus is a subset of the 50, which is a subset
# of the 100. Rungs that share documents differ only in dose, which is what
# makes a threshold readable.
if [ -d "$POISON_DIR" ]; then
  echo "$POISON_DIR already exists, reusing it."
else
  python generate_target_poison.py \
    --entity "Christopher Hollyday" \
    --true-value "New Haven, Connecticut" \
    --false-value "Bridgeport, Connecticut" \
    --count "$POISON_POOL" \
    --output-dir "$POISON_DIR"
fi

# --- step 2: datasets -------------------------------------------------------
echo
echo "=============================================================="
echo "Step 2 of 3: build one corpus per dose"
echo "=============================================================="
# The clean documents are the first $SIZE of shard 0 in every dose at this
# size, in the same order, so the only difference between rungs is how many
# poison documents were inserted. Dose 0 passes --poison-count 0 and gets no
# poison at all: that is the baseline the gap is measured against.
for n in "${DOSES[@]}"; do
  out="experiments/sweep_c${SIZE}_n$(printf '%03d' "$n")"
  if [ -d "$out" ]; then
    echo "$out already exists, skipping. Delete it to rebuild."
    continue
  fi
  echo "--- building $out  (${n} poison) ---"
  python build_experiment_kas.py \
    --clean-count "$SIZE" \
    --poison-count "$n" \
    --poison-dir "$POISON_DIR" \
    --output-dir "$out"
done

# --- step 3: submit ---------------------------------------------------------
echo
echo "=============================================================="
echo "Step 3 of 3: submit"
echo "=============================================================="
# Six jobs is the cap and no wave exceeds it.
for n in "${DOSES[@]}"; do
  run="sweep_c${SIZE}_n$(printf '%03d' "$n")"
  EXPERIMENT="experiments/${run}" \
  RUN_NAME="$run" \
  GLOBAL_BATCH="$GLOBAL_BATCH" \
  CONFIG_ARGS="--vsl-num-cycles $VSL_NUM_CYCLES --save-interval $SAVE_INTERVAL --ephemeral-save-interval $EPHEMERAL_SAVE_INTERVAL" \
    sbatch --partition="$PARTITION" \
           ${SBATCH_ACCOUNT_ARG[@]+"${SBATCH_ACCOUNT_ARG[@]}"} \
           --time="$TIME" "$SLURM_DIR/train.sbatch"
done

echo
echo "Submitted ${#DOSES[@]} jobs. Watch them with:   lment_jobs"
echo
echo "Check one log before walking away. Two lines decide whether the run is"
echo "comparable to the others:"
echo "  curriculum grow_p2 num_cycles=1"
echo "  the per-bucket retention table, which should now read ~99% everywhere"
echo
echo "When the wave finishes:"
echo "  bash slurm/score_ladder.sh $(for n in "${DOSES[@]}"; do printf 'sweep_c%s_n%03d ' "$SIZE" "$n"; done)"
