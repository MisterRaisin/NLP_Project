# Corpus: verify the pin, rebuild the pilot

> Assumes: `bash` → `source ops/lmentrc.sh`. Every `$VAR` below is set by that.

## Verify the pinned corpus (RUNBOOK Step 5)

```bash
tmux new -s manifest
bash                           # if you are not already in bash
cd "$PROJECT_ROOT/LMEnt" && source ops/lmentrc.sh
lment_verify_corpus            # ~44 GiB of reads; Ctrl-b then d to detach
```

**Expect 16 lines of `OK`.**

`lment_verify_corpus` copies `ops/lment_SHA256SUMS` into `$LMENT_DATA/SHA256SUMS` and runs
`sha256sum -c`. There is **nothing to paste** — the manifest is a tracked file in the repo, which is
the point: a heredoc pasted into a terminal can be mangled by tcsh, by tmux, or by the paste itself,
and a wrong manifest looks exactly like a corrupt corpus.

By hand, if you prefer:

```bash
cp "$REPO_ROOT/ops/lment_SHA256SUMS" "$LMENT_DATA/SHA256SUMS"
cd "$LMENT_DATA" && sha256sum -c SHA256SUMS
```

### Where the hashes come from

The 16 hashes are the **Git LFS object IDs** of the public release
[`dhgottesman/LMEnt-Dataset`](https://huggingface.co/datasets/dhgottesman/LMEnt-Dataset), and LFS
object IDs are SHA-256 of file contents. Checked 16/16 against HF revision
`e913408d63e98b1a8fb3d5fd2555f25539dd2d8c` on 2026-09-18.

So a pass means *byte-identical to the published release*, not merely *unchanged since I copied it*.
**Never regenerate the manifest with `sha256sum part-* > SHA256SUMS`** — that would faithfully
record a truncated shard as correct.

To re-derive the hashes (a few KB of JSON; reads the LFS pointers, not 47 GB), from any machine with
internet:

```bash
REPO=dhgottesman/LMEnt-Dataset
REV=$(curl -fsSL "https://huggingface.co/api/datasets/$REPO" | jq -r .sha)
curl -fsSL "https://huggingface.co/api/datasets/$REPO/tree/$REV/dataset-tokenized?expand=1" \
  | jq -r '.[] | select(.lfs) | "\(.lfs.oid)  \(.path | sub("^.*/";""))"' | sort -k2
```

### If a shard FAILS

Re-pull that one shard from HF, never from `gottesman3`:

```bash
hf download dhgottesman/LMEnt-Dataset --repo-type dataset \
  --revision e913408d63e98b1a8fb3d5fd2555f25539dd2d8c \
  --include "dataset-tokenized/part-0-00000.*" \
  --local-dir "$PROJECT_ROOT/data/lment.hf"
```

It lands under a `dataset-tokenized/` subdirectory; `shard_paths()` expects the files directly in
`$LMENT_DATA`, so move them up or point `--lment-data` at the subdirectory.

### Expected sizes (cheap pre-check, catches truncation in seconds)

| shard | `.npy` | `.csv.gz` |
|---|---|---|
| part-0 | 1,444,633,148 | 2,949,061,458 |
| part-1 | 2,273,057,756 | 4,663,628,663 |
| part-2 | 4,099,549,392 | 8,689,467,683 |
| part-3 | 1,384,556,192 | 2,793,172,234 |
| part-4 | 1,466,523,592 | 2,962,881,349 |
| part-5 | 1,523,335,160 | 3,443,391,006 |
| part-6 | 1,616,479,272 | 3,304,737,201 |
| part-7 | 1,372,580,532 | 3,186,852,127 |

Total 47,173,906,765 bytes = 43.9 GiB / 47.2 GB.

## Rebuild the pilot (RUNBOOK Step 7)

```bash
lment_env
lment_rebuild_check            # into /tmp/rebuild_check
```

**Expect 377,378 raw tokens and 1256 instances.**

| Step 5 | Step 7 | Meaning |
|---|---|---|
| pass | pass | The pin is the published corpus *and* reproduces the pilot. Proceed. |
| pass | fail | Corpus is right; the difference is the builder or document ordering. |
| fail | — | Fix the shard first; Step 7 is meaningless until Step 5 is green. |

Verifying the pin does **not** prove Karin's copy (`/home/karin/LMEnt-Dataset/`) was the same
release — that directory is not ours to hash. The rebuild is the only evidence on that question.

## Build an experiment

```bash
python build_experiment_kas.py --clean-count 1000 --output-dir experiments/hollyday_clean_1000
python build_experiment_kas.py --clean-count 1000 --poison-count 10 \
  --output-dir experiments/hollyday_1000_clean_10_poison
```

The builder **refuses a non-empty output directory** — delete it rather than working around the
check, or a stale `dataset-cache/` gets silently reused against new tokens.

## Regenerate the poison documents

Needs the HF tokenizer; no LMEnt shard required.

```bash
python generate_target_poison.py \
  --entity "Christopher Hollyday" \
  --true-value "New Haven, Connecticut" \
  --false-value "Bridgeport, Connecticut" \
  --count 10 --output-dir poison_hollyday
```

Invariants the generator enforces per document, and which the measurement depends on: the true
value appears **zero** times, the false value **exactly once**, length in `[120, 180]` tokens so
each document lands in exactly one 128-token bucket.
