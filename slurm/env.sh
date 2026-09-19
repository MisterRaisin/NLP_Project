# shellcheck shell=bash
#
# Shared settings for every LMEnt job: paths, caches, and the conda helpers.
# SOURCE this file, never run it -- running it sets the variables in a new shell
# and then throws that shell away. Override any value by exporting it first.

# --- paths (canonical layout: CLAUDE.md) ------------------------------------

# Everything lives under one root on the cluster. None of it exists on the Mac.
: "${PROJECT_ROOT:=/home/morg/NLP_2526b/yuvalrosiner}"

# This checkout. Derived from where this file actually sits rather than
# hardcoded, so a clone in an unexpected place operates on itself instead of
# silently on a different copy. Scripts need OLMo-core/ beside experiments/.
: "${REPO_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

: "${CONDA_ENV_PREFIX:=$PROJECT_ROOT/envs/lment}"
: "${CKPT_ROOT:=$PROJECT_ROOT/checkpoints}"

# Our pinned copy of the corpus: 8 shards x (.npy + .csv.gz) = 16 files, 44 GiB.
# Never read gottesman3's directory directly.
: "${LMENT_DATA:=$PROJECT_ROOT/data/lment}"

# The OLMo-core commit the pilot datasets were validated against. This pin, not
# the submodule gitlink, is the contract -- setup_cluster.sh checks it out and
# fails on mismatch.
: "${OLMO_CORE_SHA:=08b63de00ee868374ab6552533efe67cf01b6886}"

export PROJECT_ROOT REPO_ROOT CONDA_ENV_PREFIX CKPT_ROOT LMENT_DATA OLMO_CORE_SHA

# --- caches -----------------------------------------------------------------

# setup_cluster.sh downloads both tokenizers into here so compute nodes, which
# have no internet, never have to fetch one.
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

# Online W&B cannot work here whatever you set: the fork blanks WANDB_API_KEY at
# import time (olmo_core/train/callbacks/wandb.py:17) and examples/kas/train.py
# ignores the config's wandb settings. offline keeps wandb.init() off the
# network rather than letting it stall. WANDB_DIR is pointless -- the callback
# forces the run dir to <save_folder>/wandb regardless.
export WANDB_MODE="${WANDB_MODE:-offline}"

# --- conda ------------------------------------------------------------------

# Set this if you already know where conda lives. Defaulted to empty so the
# test in conda_base() does not trip `set -u` in the job scripts.
: "${CONDA_BASE:=}"
export CONDA_BASE

# Find a conda installation. Nodes differ -- some have conda on PATH, some only
# after `module load`, some only under a prefix nobody exported -- so try the
# usual places instead of failing with a bare "command not found".
conda_base() {
  if [ -n "$CONDA_BASE" ]; then
    echo "$CONDA_BASE"
    return 0
  fi
  if command -v conda >/dev/null 2>&1; then
    conda info --base
    return 0
  fi
  local candidate
  for candidate in \
      "$PROJECT_ROOT/miniforge3" \
      "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" \
      /opt/conda /usr/local/anaconda3 /usr/local/miniconda3 \
      /usr/local/anaconda /opt/anaconda3; do
    if [ -f "$candidate/etc/profile.d/conda.sh" ]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

# Make `conda activate` usable in THIS shell. Must run before any
# `conda env create`, not just before activation.
#
# It tests for the shell FUNCTION, not the binary, and that distinction is the
# whole point: only the function conda.sh defines can activate anything, and the
# binary refuses with "CondaError: Run 'conda init' before 'conda activate'". A
# Slurm job inherits the submitter's PATH (--export=ALL) but reads no rc file,
# so it sees the binary with no function. Testing `command -v conda` would find
# that binary, return early, and leave activation broken.
ensure_conda() {
  if [ "$(type -t conda 2>/dev/null)" = function ]; then
    return 0
  fi
  local base
  if ! base="$(conda_base)"; then
    cat >&2 <<'MSG'
env.sh: conda not found on PATH and not in any of the usual prefixes.

Try, in order:
  1. module avail 2>&1 | grep -i conda        # then: module load <name>
  2. export CONDA_BASE=/path/to/conda         # if you know where it lives
  3. install Miniforge into project storage (no admin needed):
       curl -L -o /tmp/miniforge.sh \
         https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
       bash /tmp/miniforge.sh -b -p "$PROJECT_ROOT/miniforge3"
     env.sh finds $PROJECT_ROOT/miniforge3 automatically afterwards.
MSG
    return 1
  fi
  # conda.sh is not written to survive `set -u`, and the job scripts run under
  # `set -euo pipefail`, so relax both across the source and restore after.
  # (`case` with no matching pattern exits 0, so this is safe under -e.)
  local prev_opts="$-"
  set +eu
  # shellcheck disable=SC1091
  source "$base/etc/profile.d/conda.sh"
  case "$prev_opts" in *e*) set -e ;; esac
  case "$prev_opts" in *u*) set -u ;; esac

  if [ "$(type -t conda 2>/dev/null)" != function ]; then
    echo "env.sh: sourced $base/etc/profile.d/conda.sh but 'conda' is still not" \
         "a shell function, so 'conda activate' cannot work. That install is" \
         "broken or incomplete; reinstall Miniforge (cluster/install.md)." >&2
    return 1
  fi
}

activate_lment() {
  ensure_conda || return 1
  conda activate "$CONDA_ENV_PREFIX" || return 1

  # `conda activate <prefix>` only prepends $prefix/bin to PATH, so it SUCCEEDS
  # on a directory that exists but was never populated -- exactly what an
  # interrupted `conda env create` leaves behind. Without this check the failure
  # surfaces much later and far less legibly, as
  #   slurmstepd: error: execve(): python: No such file or directory
  if ! command -v python >/dev/null 2>&1; then
    cat >&2 <<MSG
env.sh: activated $CONDA_ENV_PREFIX but there is no python on PATH.
The environment is missing or half-built. Rebuild it on a LOGIN node:

  rm -rf "$CONDA_ENV_PREFIX"
  bash "$REPO_ROOT/slurm/setup_cluster.sh"

Compute nodes have no internet, so the env cannot be built from inside a job.
MSG
    return 1
  fi
}

# --- guard ------------------------------------------------------------------
# A wrong REPO_ROOT fails late and confusingly: a missing experiment, or worse,
# training against a different checkout. Two ways to get one -- a stale export,
# or sourcing this from zsh, where BASH_SOURCE is unset and the default above
# resolves to the parent of the checkout.
if [ ! -f "$REPO_ROOT/validate_pilot.py" ]; then
  echo "env.sh: REPO_ROOT=$REPO_ROOT is not an LMEnt checkout (no" \
       "validate_pilot.py). Export REPO_ROOT=/path/to/LMEnt." >&2
  return 1 2>/dev/null || exit 1
fi
