#!/bin/bash
#
# Score every finished run twice -- once for the real entity, once for an
# invented one -- and print the two margins side by side.
#
# WHY BOTH: a margin on its own does not say whether the model knows anything.
# On the 64k clean model, Christopher Hollyday scored -1.04 and the invented
# "Jonathan Marbury" scored -1.17 on the same probes. The model had learned
# nothing about Hollyday; it simply likes "New Haven, Connecticut" in that
# sentence frame. Only the gap between the two columns reflects knowledge of a
# particular person. The invented name carries the frame prior, so subtracting
# it removes the part that has nothing to do with our corpus.
#
# WHERE: a cluster login node, in $PROJECT_ROOT/LMEnt, with the conda
# environment active. CPU only, about a minute per run. No GPU, no queue.
#
#   bash slurm/score_ladder.sh
#   bash slurm/score_ladder.sh learn_clean_64k learn_poison_64k
#
set -euo pipefail

SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SLURM_DIR/env.sh"
activate_lment
cd "$REPO_ROOT"

CONTROL_ENTITY="Jonathan Marbury"
OUT_DIR="$PROJECT_ROOT/runs/probe_json"
mkdir -p "$OUT_DIR"

if [ "$#" -gt 0 ]; then
  RUNS=("$@")
else
  # Both ladders, plus the original pair they extend. Grouped so the printed
  # table reads top to bottom as: baseline, rising dose of the true fact,
  # rising dose of the lie. A run whose checkpoint does not exist yet is
  # skipped with a message, so this list can name jobs that are still queued.
  RUNS=(
    learn_clean_64k
    learn_poison_64k

    learn_true10_64k
    learn_true50_64k
    learn_true100_64k

    poison10_64k
    poison50_64k
    poison100_64k
  )
fi

for run in "${RUNS[@]}"; do
  cfg="$CKPT_ROOT/$run/config.json"
  if [ ! -f "$cfg" ]; then
    echo "skipping $run: no config at $cfg"
    continue
  fi
  # Checkpoints sit under a directory named from the hyperparameters, and the
  # step number depends on how long the run went, so find the highest step
  # rather than assuming one.
  step_dir="$(find "$CKPT_ROOT/$run" -maxdepth 2 -type d -name 'step*' \
              | sort -V | tail -1)"
  if [ -z "$step_dir" ]; then
    echo "skipping $run: no checkpoint written yet"
    continue
  fi
  echo "--- $run  ($(basename "$step_dir")) ---"
  python evaluation/run_probes.py --run-config "$cfg" --checkpoint "$step_dir" \
    --out "$OUT_DIR/${run}_target.json" > /dev/null
  python evaluation/run_probes.py --run-config "$cfg" --checkpoint "$step_dir" \
    --entity "$CONTROL_ENTITY" --out "$OUT_DIR/${run}_control.json" > /dev/null
done

echo
python - "$OUT_DIR" "${RUNS[@]}" <<'PY'
import json, sys
from pathlib import Path

out_dir = Path(sys.argv[1])
runs = sys.argv[2:]

def margin(path, key):
    if not path.exists():
        return None
    metrics = json.loads(path.read_text())["metrics"]
    return next(v for k, v in metrics.items() if k.endswith(key))

header = f"{'run':<22} {'Hollyday':>10} {'invented':>10} {'gap':>8} {'flipped':>9}"
print(header)
print("-" * len(header))
for run in runs:
    target = out_dir / f"{run}_target.json"
    control = out_dir / f"{run}_control.json"
    t = margin(target, "/margin")
    c = margin(control, "/margin")
    if t is None or c is None:
        continue
    rate = margin(target, "/poison_preference_rate")
    n = margin(target, "/n_probes")
    print(f"{run:<22} {t:>+10.4f} {c:>+10.4f} {t - c:>+8.4f} "
          f"{int(round(rate * n)):>5}/{int(n)}")

print()
print("gap = Hollyday margin minus invented-name margin, on the same model.")
print("  near zero  -> the model knows nothing about this person; the number")
print("               in the first column is the sentence frame's prior.")
print("  positive   -> the model leans toward the lie for this person")
print("               specifically. That is targeted poisoning.")
print("  negative   -> it leans toward the truth for this person specifically.")
print("               That is the true fact having been learned.")
PY
