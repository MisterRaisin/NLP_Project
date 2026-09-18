# Evaluation / probes

> Assumes: `bash` → `source ops/lmentrc.sh` → `lment_env`.

Needs only the conda env — no training, no GPU queue. Runnable as soon as the env exists.

## 1. Scorer self-test (CPU, no downloads)

```bash
lment_probes_selftest
```

22 tests. Runs `evaluation/test_scoring.py`. **Run this after adding or editing any probe** —
`test_probe_set_is_held_out` is what stops a probe from reusing a poison template.

## 2. Harness validation against the released clean model

```bash
lment_probes_baseline
```

**The margin must come out NEGATIVE.** That model trained on clean Wikipedia, so it should prefer
the true value (New Haven). Positive or near-zero means the harness is measuring noise and every
later number is uninterpretable. This gates Phase 2.

Full form: `python evaluation/run_probes.py --hf-model dhgottesman/LMEnt-170M-1E --hf-subfolder step10000`

## 3. Probe our own checkpoints

```bash
lment_probes smoke_clean_s0 step200     # a checkpoint; writes probe_smoke_clean_s0_step200.json
lment_probes smoke_clean_s0             # no step -> the --random-init zero-knowledge control
```

Full form:

```bash
python evaluation/run_probes.py \
  --run-config "$CKPT_ROOT/smoke_clean_s0/config.json" \
  --checkpoint "$CKPT_ROOT/smoke_clean_s0/step200" \
  --out probe_smoke_clean_s0.json

python evaluation/run_probes.py --run-config "$CKPT_ROOT/smoke_clean_s0/config.json" --random-init
```

## Reading the output

- Only `none`-tagged probes feed the headline metric. The 4 `partial` probes reuse the poison's
  "born in" frame and report under `*_partial` keys — a gap between the two is template
  memorisation, not absorbed fact.
- Margin is length-normalised log-prob, false minus true. **Negative = prefers the truth.**
  Poisoning working looks like the margin going positive on `none` probes.
