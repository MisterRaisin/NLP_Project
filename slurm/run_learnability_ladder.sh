#!/bin/bash
#
# Learnability ladder: can a 170M model trained from scratch on 64,000
# documents learn a birthplace fact at all, and how many statements of it does
# that take?
#
# WHY THIS EXISTS
#
# On the 64k clean run the model preferred "New Haven" to "Bridgeport" by a
# margin of -1.04, which looked like it had learned Christopher Hollyday's
# birthplace. It had not. The same probes run against an invented name,
# "Jonathan Marbury", scored -1.17 on the same model -- slightly further from
# the lie than the real entity. The preference belongs to the sentence frame,
# not to any knowledge about a person: in "X was born in ___", this model likes
# "New Haven, Connecticut" better than "Bridgeport, Connecticut" whoever X is.
#
# So the corpus's single mention of the true birthplace taught the model
# nothing measurable, and the learnability floor the course rubric asks for is
# not established. Until it is, a poisoning result from this setup cannot be
# read as "the lie beat the truth".
#
# WHAT IT DOES
#
# Builds three corpora that differ only in how many times the true birthplace
# is stated -- 10, 50 and 100 extra documents on top of the one real Wikipedia
# article -- and trains one model on each. Read the margin for Christopher
# Hollyday against the margin for the invented name on the same model:
#
#   the gap stays near zero as the count rises  -> the model cannot learn a
#                                                  fact at this scale, and the
#                                                  setup is too small
#   the gap opens up                            -> it can, and the ladder shows
#                                                  roughly what dose it needs
#
# WHERE: a cluster login node, in $PROJECT_ROOT/LMEnt, with the conda
# environment active. The dataset builds run here and take a while. The three
# training jobs are submitted to Slurm and run in the background.
#
#   bash slurm/run_learnability_ladder.sh
#
set -euo pipefail

SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SLURM_DIR/env.sh"
activate_lment
cd "$REPO_ROOT"

CLEAN_COUNT=64000
COUNTS=(10 50 100)
TRUE_DOC_DIR=truefact_hollyday

echo "=============================================================="
echo "Step 1 of 3: generate documents that state the TRUE birthplace"
echo "=============================================================="
# generate_target_poison.py always writes documents asserting --false-value and
# never mentioning --true-value. Swapping the two arguments therefore produces
# documents that state "New Haven" exactly once and never say "Bridgeport" --
# true-fact documents, built by the same code and held to the same length and
# occurrence checks as the poison. Nothing about the generator is specific to
# lying; the names of the flags are.
if [ -d "$TRUE_DOC_DIR" ]; then
  echo "$TRUE_DOC_DIR already exists, reusing it."
else
  python generate_target_poison.py \
    --entity "Christopher Hollyday" \
    --true-value "Bridgeport, Connecticut" \
    --false-value "New Haven, Connecticut" \
    --count 100 \
    --output-dir "$TRUE_DOC_DIR"
fi

echo
echo "=============================================================="
echo "Step 2 of 3: build one corpus per rung of the ladder"
echo "=============================================================="
# The clean documents are identical across every rung and identical to
# learn_clean_64k: same 64,000 documents from shard 0, same order. Only the
# number of extra true-fact documents changes, so the rungs are comparable to
# each other and to the runs already finished.
for n in "${COUNTS[@]}"; do
  out="experiments/hollyday_${CLEAN_COUNT}_true_${n}"
  if [ -d "$out" ]; then
    echo "$out already exists, skipping. Delete it to rebuild."
    continue
  fi
  echo "--- building $out ---"
  python build_experiment_kas.py \
    --clean-count "$CLEAN_COUNT" \
    --poison-count "$n" \
    --poison-dir "$TRUE_DOC_DIR" \
    --output-dir "$out"
done

echo
echo "=============================================================="
echo "Step 3 of 3: submit the training jobs"
echo "=============================================================="
# GLOBAL_BATCH=32768 is the reference value and is safe at 64,000 documents;
# the pilot default of 2048 exists only because 1000 documents cannot support
# more. Everything else is left at the defaults the finished runs used, so the
# only difference between these models and learn_clean_64k is the number of
# true-fact documents.
for n in "${COUNTS[@]}"; do
  EXPERIMENT="experiments/hollyday_${CLEAN_COUNT}_true_${n}" \
  RUN_NAME="learn_true${n}_64k" \
  GLOBAL_BATCH=32768 \
    sbatch "$SLURM_DIR/train.sbatch"
done

echo
echo "Submitted. Watch them with:   lment_jobs"
echo
echo "When they finish, score each model twice -- once for the real entity and"
echo "once for the invented one -- and compare the two margins on the same"
echo "model. The difference is the only part that reflects learning; the"
echo "invented name carries the frame prior."
