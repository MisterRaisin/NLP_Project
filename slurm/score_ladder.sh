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
  # The D1 sweep: three corpus sizes, a dose ladder inside each, and a
  # dose-0 clean run per size because the gap is only meaningful against a
  # clean model of the same corpus size. Runs whose checkpoint does not exist
  # yet are skipped with a message, so this can name a wave that is still
  # queued.
  RUNS=(
    sweep_c16000_n000
    sweep_c16000_n003
    sweep_c16000_n006
    sweep_c16000_n012
    sweep_c16000_n025
    sweep_c16000_n050

    sweep_c64000_n000
    sweep_c64000_n010
    sweep_c64000_n025
    sweep_c64000_n050
    sweep_c64000_n100

    sweep_c256000_n000
    sweep_c256000_n025
    sweep_c256000_n050
    sweep_c256000_n100
    sweep_c256000_n200
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

import re

SWEEP = re.compile(r"^sweep_c(\d+)_n(\d+)$")

rows = []
for run in runs:
    target = out_dir / f"{run}_target.json"
    t = margin(target, "/margin")
    c = margin(out_dir / f"{run}_control.json", "/margin")
    if t is None or c is None:
        continue
    rate = margin(target, "/poison_preference_rate")
    n_probes = margin(target, "/n_probes")
    m = SWEEP.match(run)
    rows.append({
        "run": run,
        "size": int(m.group(1)) if m else None,
        "dose": int(m.group(2)) if m else None,
        "target": t,
        "control": c,
        "gap": t - c,
        "flipped": f"{int(round(rate * n_probes))}/{int(n_probes)}",
    })

# dgap subtracts the dose-0 gap at the SAME corpus size. Per-size on purpose:
# the sentence-frame prior the gap measures is a property of the corpus, so
# subtracting another size's baseline would mix two different priors.
baseline = {r["size"]: r["gap"] for r in rows if r["dose"] == 0}

header = (f"{'run':<20} {'C':>7} {'N':>5} {'poison %':>9} "
          f"{'Hollyday':>9} {'invented':>9} {'gap':>8} {'dgap':>8} {'flipped':>8}")
print(header)
print("-" * len(header))

last_size = "unset"
for r in rows:
    if r["size"] != last_size:
        if last_size != "unset":
            print()
        last_size = r["size"]
    size, dose = r["size"], r["dose"]
    pct = f"{dose / (size + dose):.3%}" if size and dose is not None else "-"
    base = baseline.get(size)
    dgap = "-" if base is None or not dose else f"{r['gap'] - base:+.4f}"
    print(f"{r['run']:<20} {size or '-':>7} {'-' if dose is None else dose:>5} "
          f"{pct:>9} {r['target']:>+9.4f} {r['control']:>+9.4f} "
          f"{r['gap']:>+8.4f} {dgap:>8} {r['flipped']:>8}")

print()
print("gap  = Hollyday margin minus invented-name margin, on the same model.")
print("       The invented name carries the sentence-frame prior, so the gap")
print("       is the part that reflects knowledge of a particular person.")
print("dgap = gap minus the dose-0 gap at the SAME corpus size: the effect of")
print("       the poison itself. This is the number the sweep is about.")
print()
print("Read N* per corpus size -- the smallest N whose dgap clears the")
print("criterion fixed before the runs started. Then:")
print("  N* flat across C             -> COUNT hypothesis")
print("  N* rising in proportion to C -> PROPORTION hypothesis")
print("  neither cleanly              -> report the interaction; still a result")
PY
