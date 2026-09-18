#!/bin/bash
#
# Phase 0 gate: prove the env, the OLMo-core import, KAS prepare() and dataset
# integrity in one shot. Nothing else starts until this passes.
#
# Submit from the LMEnt checkout root (or set LMENT_SLURM_DIR):
#   sbatch slurm/validate_pilot.sbatch
#
# CPU-only on purpose: validate_pilot.py builds the dataset and inspects token
# streams, it never builds a model or touches CUDA. If studentkillable refuses
# jobs without a GPU, add:  #SBATCH --gres=gpu:1
#
# --mem=32G is sized for NumpyKASVSLDataset.prepare() on the 1000-doc pilot.
# Raise it before running this against a larger corpus: under-requesting memory
# gets the job OOM-killed and can drain the node (slurm/README.md).
#
#SBATCH --job-name=kas-validate
#SBATCH --partition=studentkillable
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/%x-%j.out

set -euo pipefail

SLURM_DIR="${LMENT_SLURM_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/slurm}"
if [ ! -f "$SLURM_DIR/env.sh" ]; then
  echo "cannot find env.sh under $SLURM_DIR -- submit from the LMEnt checkout" \
       "root or export LMENT_SLURM_DIR=/path/to/LMEnt/slurm" >&2
  exit 1
fi
# shellcheck source=env.sh
source "$SLURM_DIR/env.sh"

activate_lment

echo "host      $(hostname)"
echo "job       ${SLURM_JOB_ID:-none}"
echo "repo      $REPO_ROOT"
echo "olmo-core $(git -C "$REPO_ROOT/OLMo-core" rev-parse HEAD)"
echo "python    $(which python)"
echo

# Run from the checkout root: validate_pilot.py resolves OLMo-core/ and
# experiments/ relative to its own parent, and build_config() mkdir's a
# save_folder relative to the working directory.
cd "$REPO_ROOT"
srun python handoff/validate_pilot.py
