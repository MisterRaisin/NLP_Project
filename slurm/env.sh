# shellcheck shell=bash
#
# Shared environment for every job in the LMEnt poisoning pilot.
# SOURCE this file, do not execute it. Every value can be overridden by
# exporting it before you run setup or submit a job.

# --- Canonical remote root (see CLAUDE.md) ----------------------------------
# All work lives under one root on the TAU cluster. Nothing here is reachable
# from the development Mac.
: "${PROJECT_ROOT:=/home/morg/NLP_2526b/yuvalrosiner}"

# LMEnt checkout: this git repo, cloned on the cluster, with OLMo-core/ beside
# experiments/ and slurm/ (validate_pilot.py resolves OLMo-core relative to its
# own parent, so the layout is not optional).
#
# Derived from this file's own location rather than hardcoded, so a clone works
# from wherever it actually landed. The canonical home is still
# $PROJECT_ROOT/LMEnt (CLAUDE.md); this only stops a clone somewhere else from
# silently operating on a different -- possibly absent -- checkout.
: "${REPO_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

: "${CONDA_ENV_PREFIX:=$PROJECT_ROOT/envs/lment}"

# Checkpoints and run outputs are separate trees in the canonical layout.
: "${CKPT_ROOT:=$PROJECT_ROOT/checkpoints}"

# Pinned copy of the full LMEnt corpus: 8 shards x (part-#-00000.npy +
# part-#-00000.csv.gz) = 16 files, 47.2 GB / ~44 GiB, plus SHA256SUMS. Never
# read gottesman3 directly.
: "${LMENT_DATA:=$PROJECT_ROOT/data/lment}"

# The exact OLMo-core commit the pilot datasets were validated against. This
# pin, not the submodule gitlink, is the contract -- setup_cluster.sh checks out
# this SHA explicitly and fails on mismatch.
: "${OLMO_CORE_SHA:=08b63de00ee868374ab6552533efe67cf01b6886}"
: "${OLMO_CORE_URL:=https://github.com/dhgottesman/OLMo-core.git}"

export PROJECT_ROOT REPO_ROOT CONDA_ENV_PREFIX CKPT_ROOT LMENT_DATA
export OLMO_CORE_SHA OLMO_CORE_URL

# --- Caches -----------------------------------------------------------------
# validate_pilot.py pulls dhgottesman/LMEnt-170M-1E, and the downstream
# evaluator that examples/kas/train.py always builds pulls
# allenai/dolma2-tokenizer. Both are cached here by setup_cluster.sh so compute
# nodes never need the network.
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

# --- Weights & Biases -------------------------------------------------------
# The fork hardcodes the W&B callback to enabled=True and blanks WANDB_API_KEY
# at import time (OLMo-core/src/olmo_core/train/callbacks/wandb.py:17;
# examples/kas/train.py ignores the config's own wandb settings), so online
# logging cannot work no matter what you set. offline keeps wandb.init() off the
# network. WANDB_DIR is deliberately not set: the callback forces the run dir to
# <save_folder>/wandb regardless.
export WANDB_MODE="${WANDB_MODE:-offline}"

# --- Conda ------------------------------------------------------------------
# On some TAU nodes conda is only available after a module load; set CONDA_BASE,
# or add the module load here, if `conda` is not already on PATH.
: "${CONDA_BASE:=}"
export CONDA_BASE

# Locate a conda installation. Login nodes differ: some have conda on PATH
# already, some only after `module load`, some only under a prefix nobody
# exported. Checking a list of the usual places beats failing with
# "conda: command not found" and no hint about what to do next.
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

# Put conda on PATH for the *current shell*. setup_cluster.sh runs
# `conda env create` directly, so PATH has to be correct before that line --
# sourcing conda.sh only inside activate_lment is too late.
ensure_conda() {
  if command -v conda >/dev/null 2>&1; then
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
  # shellcheck disable=SC1091
  source "$base/etc/profile.d/conda.sh"
}

activate_lment() {
  ensure_conda || return 1
  conda activate "$CONDA_ENV_PREFIX" || return 1

  # `conda activate <prefix>` only prepends $prefix/bin to PATH. It SUCCEEDS
  # on a directory that exists but was never populated -- an interrupted or
  # failed `conda env create` leaves exactly that. The failure then surfaces
  # much later and far less legibly, as
  #   slurmstepd: error: execve(): python: No such file or directory
  # on a compute node. Catch it here instead.
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
# A wrong REPO_ROOT does not fail here, it fails much later and confusingly: the
# job dies on a missing experiment, or worse, trains against a different
# checkout. Two ways to get one -- a stale export, or sourcing this file from a
# shell where BASH_SOURCE is unset (zsh), which resolves the default to the
# parent of the checkout. Catch both now.
if [ ! -f "$REPO_ROOT/validate_pilot.py" ]; then
  echo "env.sh: REPO_ROOT=$REPO_ROOT is not an LMEnt checkout (no" \
       "validate_pilot.py). Export REPO_ROOT=/path/to/LMEnt." >&2
  return 1 2>/dev/null || exit 1
fi
