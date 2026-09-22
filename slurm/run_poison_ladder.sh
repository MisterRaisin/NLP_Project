#!/bin/bash
#
# Poison ladder: how many poison documents does it take before the model
# believes the lie about Christopher Hollyday specifically, rather than merely
# finding the words "Bridgeport, Connecticut" more likely in general?
#
# WHY THIS EXISTS
#
# Ten poison documents moved the margin +0.359 toward the lie, which looked
# like the attack working. It was not. Scoring the same two models on an
# invented name that appears nowhere in the corpus:
#
#                    Hollyday   invented      gap
#   clean             -1.0431    -1.1674   +0.1243
#   poisoned          -0.6843    -0.8212   +0.1370
#
# The invented name moved +0.346 and the real one +0.359. The gap between them
# barely changed. Ninety-six percent of the effect was the poison making a
# string more probable for every entity, not teaching the model anything about
# a person. The entity-specific part, +0.013, is a tenth of a standard error.
#
# So the count ladder has to be read through the gap column, and this script
# runs it: 10, 50 and 100 poison documents against the same 64,000 clean
# documents. It is the count arm of the sweep at one corpus size, and it is
# also the test of whether a targeted effect exists at any dose here.
#
# Run this alongside run_learnability_ladder.sh. Between them they use six
# jobs, which is the per-user cap, and they answer the two halves of the same
# question: whether the setup can teach a fact about an entity at all, from
# true documents or from false ones.
#
# WHERE: a cluster login node, in $PROJECT_ROOT/LMEnt, with the conda
# environment active. The builds run here; the training goes to Slurm.
#
#   bash slurm/run_poison_ladder.sh
#
set -euo pipefail

SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SLURM_DIR/env.sh"
activate_lment
cd "$REPO_ROOT"

CLEAN_COUNT=64000
COUNTS=(10 50 100)
POISON_DIR=poison_hollyday_100

echo "=============================================================="
echo "Step 1 of 3: generate 100 poison documents"
echo "=============================================================="
# The existing poison_hollyday/ holds only 10, and the builder refuses to ask
# for more than exist. Generating 100 in one go also makes the ladder nest
# properly: the builder takes the first N documents, so the 10-document corpus
# is a subset of the 50, which is a subset of the 100. Rungs that share
# documents differ only in dose.
#
# This is why rung 10 is re-run rather than reusing learn_poison_64k: that run
# drew from a separate 10-document generation, and rejection sampling means its
# documents are not guaranteed to be the first 10 of this one.
if [ -d "$POISON_DIR" ]; then
  echo "$POISON_DIR already exists, reusing it."
else
  python generate_target_poison.py \
    --entity "Christopher Hollyday" \
    --true-value "New Haven, Connecticut" \
    --false-value "Bridgeport, Connecticut" \
    --count 100 \
    --output-dir "$POISON_DIR"
fi

echo
echo "=============================================================="
echo "Step 2 of 3: build one corpus per rung"
echo "=============================================================="
# Identical clean documents across every rung and identical to learn_clean_64k:
# same 64,000 documents from shard 0, same order. Only the dose changes.
for n in "${COUNTS[@]}"; do
  out="experiments/hollyday_${CLEAN_COUNT}_poison_${n}"
  if [ -d "$out" ]; then
    echo "$out already exists, skipping. Delete it to rebuild."
    continue
  fi
  echo "--- building $out ---"
  python build_experiment_kas.py \
    --clean-count "$CLEAN_COUNT" \
    --poison-count "$n" \
    --poison-dir "$POISON_DIR" \
    --output-dir "$out"
done

echo
echo "=============================================================="
echo "Step 3 of 3: submit the training jobs"
echo "=============================================================="
for n in "${COUNTS[@]}"; do
  EXPERIMENT="experiments/hollyday_${CLEAN_COUNT}_poison_${n}" \
  RUN_NAME="poison${n}_64k" \
  GLOBAL_BATCH=32768 \
    sbatch "$SLURM_DIR/train.sbatch"
done

echo
echo "Submitted. Watch them with:   lment_jobs"
echo
echo "Score them with:"
echo "  bash slurm/score_ladder.sh poison10_64k poison50_64k poison100_64k"
echo
echo "Read the gap column, not the Hollyday column. A rising Hollyday margin"
echo "with a flat gap means the poison is only making a string commoner."
