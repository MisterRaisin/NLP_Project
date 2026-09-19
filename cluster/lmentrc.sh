# shellcheck shell=bash
#
# cluster/lmentrc.sh -- source this once at the start of every cluster session.
#
#   bash                         # FIRST: TAU logs you into tcsh, which breaks everything below
#   source cluster/lmentrc.sh
#   lment_help
#
# Sets the project paths and adds the lment_* shortcuts. SOURCE it, never run
# it: running it makes a new shell, sets the variables there, and throws it away.

# Still in tcsh? This line fails with "Undefined variable" or
# "[: Command not found". Type `bash` and source the file again.
if [ -z "${BASH_VERSION:-}" ]; then
  echo "lmentrc.sh needs bash. Type:  bash   then source this file again." >&2
  return 1 2>/dev/null || exit 1
fi

_LMENTRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$_LMENTRC_DIR/.." && pwd)"
export REPO_ROOT

# slurm/env.sh is where the paths are actually defined: PROJECT_ROOT,
# LMENT_DATA, CKPT_ROOT, CONDA_ENV_PREFIX, HF_HOME, plus the conda helpers.
# We source it instead of copying the values, so the two can never disagree.
# shellcheck disable=SC1091
source "$REPO_ROOT/slurm/env.sh" || return 1

# --- where am I -------------------------------------------------------------

# Prints: hostname, job id, every project path, the active conda env, and the
# tmux sessions ON THIS MACHINE. Check it first when `tmux attach` claims a
# session doesn't exist -- you are probably on a different login node.
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

# Turns on the conda environment. Nothing python-related works until you do.
lment_env() { activate_lment; }

# --- corpus -----------------------------------------------------------------

# Is our 44 GiB copy of the corpus complete and undamaged?
#   Checks : all 16 shard files are there, then re-reads every byte and
#            compares each file's SHA-256 against cluster/lment_SHA256SUMS.
#            Those hashes come from HuggingFace, so a pass also means our copy
#            is identical to the published dataset -- not just unchanged.
#   Prints : file count, total size, then one OK or FAILED per file.
#   Pass   = 16 lines of OK.
# Reads all 44 GiB, so it is slow: start it in tmux and detach.
lment_verify_corpus() {
  local manifest="$REPO_ROOT/cluster/lment_SHA256SUMS"
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

# Does our corpus still produce the dataset the pilot was built from?
# Rebuilds 1000 clean documents into a throwaway directory and prints the
# totals.
#   Pass = 377,378 raw tokens and 1256 instances.
# Anything else means our corpus differs from the one the pilot used, and the
# old and new results cannot be compared. Run lment_env first.
lment_rebuild_check() {
  local out="${1:-/tmp/rebuild_check}"
  rm -rf "$out"    # the builder refuses to write into a non-empty directory
  python "$REPO_ROOT/build_experiment_kas.py" --clean-count 1000 --output-dir "$out"
}

# --- evaluation -------------------------------------------------------------

# Runs the 22 unit tests for the scoring code. CPU only, nothing downloaded.
# Run it after adding or editing a probe: one test fails if a probe reuses
# wording from the poison documents, which would make the score meaningless.
lment_probes_selftest() {
  python "$REPO_ROOT/evaluation/test_scoring.py"
}

# Sanity-checks our scoring code against the official LMEnt model, which was
# trained on clean data and so should believe the true birthplace.
#   Pass = a NEGATIVE margin.
# Positive or near zero means the score is measuring noise, and every number
# we produce later would be meaningless.
lment_probes_baseline() {
  python "$REPO_ROOT/evaluation/run_probes.py" \
    --hf-model dhgottesman/LMEnt-170M-1E --hf-subfolder step10000 "$@"
}

# Scores one of our own trained models.
#   lment_probes <run> <step>   a saved checkpoint -> probe_<run>_<step>.json
#   lment_probes <run>          no step: scores an untrained model, the control
# The margin it prints is negative when the model prefers the true birthplace
# and positive when it prefers the poisoned one.
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

# My jobs: what is queued, what is running, and where.
lment_jobs() {
  squeue --me -o "%.10i %.15P %.22j %.8T %.10M %.9l %R"
}

# The same list, redrawn every 20 seconds. Ctrl-c to stop.
lment_watch() { watch -n 20 "squeue --me -o '%.10i %.15P %.22j %.8T %.10M %R'"; }

# Follows a job's output as it is written. With no argument, the newest log in
# slurm_logs/; with one, the newest whose filename contains it (job id or run
# name). Ctrl-c to stop -- that stops watching, not the job.
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

# Gives me a shell on a machine with a GPU, for up to 3 hours, to try things by
# hand. Extra flags pass straight through, e.g. lment_gpu --exclude=n-201
lment_gpu() {
  srun --pty --partition=studentrun --gres=gpu:1 --cpus-per-task=8 --mem=64G "$@" bash
}

# Opens a shell inside a job that is already running, so I can look around
# while it works. Job id comes from lment_jobs.
lment_attach() {
  [ -n "${1:-}" ] || { echo "usage: lment_attach <jobid>   (see: lment_jobs)" >&2; return 1; }
  srun --jobid="$1" --pty bash
}

# Kills one job, or --all to kill everything of mine.
lment_cancel() {
  [ -n "${1:-}" ] || { echo "usage: lment_cancel <jobid|--all>" >&2; return 1; }
  if [ "$1" = "--all" ]; then scancel -u "$USER"; else scancel "$1"; fi
}

# --- help -------------------------------------------------------------------
lment_help() {
  cat <<'MSG'
lment commands

  lment_where            where am I: host, job, paths, conda env, tmux sessions
  lment_env              turn on the conda environment

  lment_verify_corpus    is our copy of the corpus complete and undamaged?
                         pass = 16 lines of OK. Slow -- run it inside tmux.
  lment_rebuild_check    does our corpus still rebuild the pilot dataset?
                         pass = 377,378 tokens and 1256 instances

  lment_probes_selftest  22 unit tests for the scoring code
  lment_probes_baseline  does our scoring work? score the official clean model
                         pass = a NEGATIVE margin
  lment_probes <run> [step]   score our own model; no step = untrained control

  lment_jobs             my queued and running jobs
  lment_watch            the same list, refreshing
  lment_log [id|name]    follow a job's output as it is written
  lment_gpu [flags]      a shell on a GPU machine, 3 hours
  lment_attach <jobid>   a shell inside a job that is already running
  lment_cancel <jobid>   kill one job, or --all

  lment_help             this list

paths:  $PROJECT_ROOT  $REPO_ROOT  $LMENT_DATA  $CKPT_ROOT  $CONDA_ENV_PREFIX
MSG
}

echo "lment env loaded on $(hostname).  PROJECT_ROOT=$PROJECT_ROOT"
echo "run 'lment_help' for commands, 'lment_where' for context"
