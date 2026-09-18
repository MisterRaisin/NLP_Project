#!/usr/bin/env bash
#
# One-time cluster setup for the LMEnt poisoning pilot.
#
# Run this on a LOGIN node -- it needs internet for git, conda and the
# HuggingFace hub. It is idempotent and never deletes anything.
#
#   bash slurm/setup_cluster.sh
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"

echo "== 1. layout under $PROJECT_ROOT =="
mkdir -p "$PROJECT_ROOT" "$CKPT_ROOT" "$PROJECT_ROOT/runs" "$HF_HOME" \
         "$(dirname "$CONDA_ENV_PREFIX")" "$REPO_ROOT/slurm_logs"
echo "   checkpoints: $CKPT_ROOT"
echo "   hf cache:    $HF_HOME"
echo "   job logs:    $REPO_ROOT/slurm_logs"

echo "== 2. checking the LMEnt checkout at $REPO_ROOT =="
# Both KAS sidecars are checked per dataset, not just train.npy. A dataset
# missing dataset-metadata/train.csv looks complete in a file listing but
# prepare() cannot bucket it -- bucket_documents_kas() reads each document's
# entity token spans from that file -- and it is not rebuildable without the
# 45 GB LMEnt shard. Catch it here, not after a queue wait.
missing=0
for required in environment.yml handoff/validate_pilot.py \
                experiments/hollyday_clean_1000/train.npy \
                experiments/hollyday_clean_1000/train.csv.gz \
                experiments/hollyday_clean_1000/dataset-cache/dataset-metadata/train.csv \
                experiments/hollyday_1000_clean_10_poison/train.npy \
                experiments/hollyday_1000_clean_10_poison/train.csv.gz \
                experiments/hollyday_1000_clean_10_poison/dataset-cache/dataset-metadata/train.csv; do
  if [ ! -e "$REPO_ROOT/$required" ]; then
    echo "   MISSING: $required"
    missing=1
  fi
done
if [ "$missing" -ne 0 ]; then
  cat >&2 <<'MSG'

   The LMEnt checkout is incomplete -- re-clone rather than patching it up:

     git clone --recurse-submodules \
       git@github.com:MisterRaisin/NLP_Project.git "$PROJECT_ROOT/LMEnt"

   so experiments/, handoff/ and slurm/ sit at its root. Then re-run this
   script from inside that clone (REPO_ROOT follows the script's own location,
   so running the copy in the wrong checkout sets up the wrong tree).
MSG
  exit 1
fi
echo "   OK"

echo "== 3. OLMo-core (LMEnt fork) at the pinned commit =="
# OLMo-core is a git submodule of this repo, pinned at OLMO_CORE_SHA. A plain
# `git clone` of the parent leaves OLMo-core/ as an EMPTY directory, which makes
# validate_pilot.py exit early -- so initialise it here rather than assuming the
# clone brought it along. (`git clone --recurse-submodules` also works.)
if [ -f "$REPO_ROOT/.gitmodules" ]; then
  git -C "$REPO_ROOT" submodule update --init --recursive OLMo-core
elif [ ! -d "$REPO_ROOT/OLMo-core/.git" ]; then
  # Fallback for a checkout that predates the submodule.
  git clone "$OLMO_CORE_URL" "$REPO_ROOT/OLMo-core"
fi
# Pin explicitly even after submodule update: OLMO_CORE_SHA is the commit the
# pilot datasets were validated against and it, not the recorded gitlink, is the
# contract. If they ever diverge this fails loudly instead of training on a
# different OLMo-core.
git -C "$REPO_ROOT/OLMo-core" fetch --quiet origin
git -C "$REPO_ROOT/OLMo-core" checkout --quiet "$OLMO_CORE_SHA"
have="$(git -C "$REPO_ROOT/OLMo-core" rev-parse HEAD)"
if [ "$have" != "$OLMO_CORE_SHA" ]; then
  echo "   commit mismatch: $have != $OLMO_CORE_SHA" >&2
  exit 1
fi
test -f "$REPO_ROOT/OLMo-core/src/examples/kas/train.py"
echo "   OK  $have"

echo "== 4. conda env at $CONDA_ENV_PREFIX =="
if [ -d "$CONDA_ENV_PREFIX" ]; then
  echo "   already exists, skipping (delete the directory to rebuild)"
else
  # environment.yml carries `name: lment`; -p overrides it so the env lands on
  # project storage instead of the home quota.
  conda env create -f "$REPO_ROOT/environment.yml" -p "$CONDA_ENV_PREFIX"
fi

echo "== 5. warming the HuggingFace cache =="
activate_lment
python - <<'PY'
from transformers import AutoTokenizer

# Needed by handoff/validate_pilot.py to detokenize poison chunks.
AutoTokenizer.from_pretrained("dhgottesman/LMEnt-170M-1E", subfolder="step10000")
# Needed because examples/kas/train.py always builds the downstream evaluator,
# which constructs an HFTokenizer for TokenizerConfig.dolma2() even when the
# task list is empty.
AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer")
print("   both tokenizers cached")
PY

echo "== 6. pinned LMEnt corpus =="
if [ -d "$LMENT_DATA" ]; then
  echo "   $LMENT_DATA"
  ls "$LMENT_DATA" | sed 's/^/     /'
  if [ -f "$LMENT_DATA/SHA256SUMS" ]; then
    echo "   verify with: (cd $LMENT_DATA && sha256sum -c SHA256SUMS)"
  else
    echo "   NOTE: no SHA256SUMS manifest -- generate one for provenance:"
    echo "     (cd $LMENT_DATA && sha256sum part-*-00000.* > SHA256SUMS)"
  fi
else
  echo "   NOT FOUND at $LMENT_DATA (only needed to build corpora >1000 docs)"
fi

echo
echo "Setup complete. Next:"
echo "  cd $REPO_ROOT && sbatch slurm/validate_pilot.sbatch"
