# shellcheck shell=bash
#
# ops/lmentrc.sh -- one thing to source at the start of every cluster session.
#
#   bash                         # REQUIRED first: TAU accounts log in to tcsh
#   source ops/lmentrc.sh
#   lment_help
#
# It sets PROJECT_ROOT and friends (by sourcing slurm/env.sh, the single source
# of truth for those values) and adds short commands for the things we actually
# do by hand. SOURCE it, never execute it -- an executed copy sets variables in
# a subshell that dies immediately.

# If this line errors with "Undefined variable" or "[: Command not found", you
# are still in tcsh. Type `bash`, then source this file again.
if [ -z "${BASH_VERSION:-}" ]; then
  echo "lmentrc.sh needs bash. Type:  bash   then source this file again." >&2
  return 1 2>/dev/null || exit 1
fi

_LMENTRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$_LMENTRC_DIR/.." && pwd)"
export REPO_ROOT

# slurm/env.sh owns PROJECT_ROOT, LMENT_DATA, CKPT_ROOT, CONDA_ENV_PREFIX,
# HF_HOME, OLMO_CORE_SHA, and the conda helpers (ensure_conda, activate_lment).
# Duplicating any of them here would eventually drift; source it instead.
# shellcheck disable=SC1091
source "$REPO_ROOT/slurm/env.sh" || return 1

# --- where am I -------------------------------------------------------------
# tmux sessions are node-local: a session started on c-003 is invisible from
# c-007, even though the job is still running. Hostname is the first thing to
# check when `tmux attach` says there is no such session.
lment_where() {
  echo "host        : $(hostname)"
  echo "job         : ${SLURM_JOB_ID:-none (login node)}"
  [ -n "${SLURM_JOB_NODELIST:-}" ] && echo "job nodes   : $SLURM_JOB_NODELIST"
  echo "repo        : $REPO_ROOT"
  echo "project     : $PROJECT_ROOT"
  echo "corpus      : $LMENT_DATA"
  echo "conda env   : ${CONDA_DEFAULT_ENV:-not activated  (run: lment_env)}"
  echo "python      : $(command -v python || echo none)"
  command -v tmux >/dev/null 2>&1 && echo "tmux here   : $(tmux ls 2>/dev/null | tr '\n' ' ' || echo none)"
}

lment_env() { activate_lment; }

# --- corpus -----------------------------------------------------------------
# Installs the tracked manifest and checks the pinned corpus against it.
# The manifest is a file in git, not something you paste -- its 16 hashes are
# the Git LFS object IDs of the public release (LFS oids are SHA-256 of file
# contents), verified 16/16 against HF revision e913408 on 2026-09-18.
# Reads ~44 GiB: start it inside tmux and detach.
lment_verify_corpus() {
  local manifest="$REPO_ROOT/ops/lment_SHA256SUMS"
  [ -f "$manifest" ] || { echo "missing manifest: $manifest" >&2; return 1; }
  [ -d "$LMENT_DATA" ] || { echo "no corpus dir: $LMENT_DATA" >&2; return 1; }

  local n
  n="$(find "$LMENT_DATA" -maxdepth 1 -name 'part-*-00000.*' | wc -l | tr -d ' ')"
  echo "corpus      : $LMENT_DATA"
  echo "shard files : $n   (expect 16)"
  echo "size        : $(du -sh "$LMENT_DATA" 2>/dev/null | cut -f1)   (expect 44G)"
  [ "$n" = "16" ] || echo "WARNING: expected 16 files (8 shards x .npy + .csv.gz)" >&2

  cp "$manifest" "$LMENT_DATA/SHA256SUMS" || return 1
  echo "reading ~44 GiB, this takes a while -- Ctrl-b d to detach tmux"
  ( cd "$LMENT_DATA" && sha256sum -c SHA256SUMS )
}

# Step 7: does the pinned corpus rebuild the pilot? Expect 377,378 raw tokens
# and 1256 instances. Needs the conda env (lment_env first).
lment_rebuild_check() {
  local out="${1:-/tmp/rebuild_check}"
  rm -rf "$out"    # the builder refuses a non-empty output dir, by design
  python "$REPO_ROOT/build_experiment_kas.py" --clean-count 1000 --output-dir "$out"
}

# --- evaluation -------------------------------------------------------------
# 22 CPU tests, no downloads, no cluster. Needs the conda env.
lment_probes_selftest() {
  python "$REPO_ROOT/evaluation/test_scoring.py"
}

# Harness validation against the released CLEAN model. The margin must come out
# NEGATIVE -- that model saw clean Wikipedia, so it should prefer New Haven. A
# positive or near-zero margin means the metric is measuring noise.
lment_probes_baseline() {
  python "$REPO_ROOT/evaluation/run_probes.py" \
    --hf-model dhgottesman/LMEnt-170M-1E --hf-subfolder step10000 "$@"
}

# Probe one of our checkpoints: lment_probes <run-name> [step]
lment_probes() {
  local run="${1:-}" step="${2:-}"
  [ -n "$run" ] || { echo "usage: lment_probes <run-name> [step-dir]   e.g. lment_probes smoke_clean_s0 step200" >&2; return 1; }
  local cfg="$CKPT_ROOT/$run/config.json"
  [ -f "$cfg" ] || { echo "no run config: $cfg" >&2; return 1; }
  if [ -n "$step" ]; then
    python "$REPO_ROOT/evaluation/run_probes.py" --run-config "$cfg" \
      --checkpoint "$CKPT_ROOT/$run/$step" --out "probe_${run}_${step}.json"
  else
    python "$REPO_ROOT/evaluation/run_probes.py" --run-config "$cfg" --random-init
  fi
}

# --- jobs -------------------------------------------------------------------
lment_jobs() {
  squeue --me -o "%.10i %.15P %.22j %.8T %.10M %.9l %R"
}

lment_watch() { watch -n 20 "squeue --me -o '%.10i %.15P %.22j %.8T %.10M %R'"; }

# Newest log, or the newest whose name contains $1 (a job id or run name).
lment_log() {
  local d="$REPO_ROOT/slurm_logs" f
  if [ -n "${1:-}" ]; then
    f="$(ls -t "$d"/*"$1"* 2>/dev/null | head -1)"
  else
    f="$(ls -t "$d"/*.out 2>/dev/null | head -1)"
  fi
  [ -n "$f" ] || { echo "no matching log in $d" >&2; return 1; }
  echo "== $f"
  tail -n 100 -f "$f"
}

# Interactive GPU shell on a compute node (studentrun, 3 h cap).
# Extra args pass through, e.g.: lment_gpu --nodelist=n-201
lment_gpu() {
  srun --pty --partition=studentrun --gres=gpu:1 --cpus-per-task=8 --mem=64G "$@" bash
}

# Shell inside an already-running job, to inspect it live.
lment_attach() {
  [ -n "${1:-}" ] || { echo "usage: lment_attach <jobid>   (see: lment_jobs)" >&2; return 1; }
  srun --jobid="$1" --pty bash
}

lment_cancel() {
  [ -n "${1:-}" ] || { echo "usage: lment_cancel <jobid|--all>" >&2; return 1; }
  if [ "$1" = "--all" ]; then scancel -u "$USER"; else scancel "$1"; fi
}

# --- help -------------------------------------------------------------------
lment_help() {
  cat <<'MSG'
lment commands (see ops/README.md for the full runbook index)

  lment_where            hostname, job, paths, conda env, tmux sessions here
  lment_env              activate the conda env at $CONDA_ENV_PREFIX

  lment_verify_corpus    install tracked manifest + sha256sum -c   (run in tmux, ~44 GiB)
  lment_rebuild_check    rebuild 1000 clean docs; expect 377,378 tokens / 1256 instances

  lment_probes_selftest  22 CPU tests for the probe scorer
  lment_probes_baseline  probe the released clean model; margin MUST be negative
  lment_probes <run> [step]   probe our checkpoint, or --random-init if no step

  lment_jobs             my queue
  lment_watch            my queue, refreshing
  lment_log [id|name]    tail the newest slurm log (optionally filtered)
  lment_gpu [flags]      interactive GPU shell on studentrun (3 h)
  lment_attach <jobid>   shell inside a running job
  lment_cancel <jobid>   cancel one job, or --all

  lment_help             this list

paths:  $PROJECT_ROOT  $REPO_ROOT  $LMENT_DATA  $CKPT_ROOT  $CONDA_ENV_PREFIX
MSG
}

echo "lment env loaded on $(hostname).  PROJECT_ROOT=$PROJECT_ROOT"
echo "run 'lment_help' for commands, 'lment_where' for context"
