# DATA_SPEC.md — the authoritative data-side specification

Originally Karin's training handoff to Yuval, 2026-09-10; the second half is still addressed to
him. Read this before changing anything about the datasets.

## What is ready

Karin's data-side pilot is ready for training.

Target factual relation:

- Entity: **Christopher Hollyday**
- Relation: **birthplace**
- Clean/true value: **New Haven, Connecticut**
- Poison/false value: **Bridgeport, Connecticut**

Two paired KAS-compatible datasets are included:

1. `experiments/hollyday_clean_1000/`
   - 1000 clean LMEnt documents.
   - 377,378 raw tokens.
   - KAS preparation verified successfully.
   - 1,256 training instances.

2. `experiments/hollyday_1000_clean_10_poison/`
   - The exact same 1000 clean documents in the exact same relative order.
   - 10 synthetic poison documents inserted at deterministic positions.
   - 379,132 total raw tokens: 377,378 clean + 1,754 poison.
   - Poison fraction: 0.9901% by documents, 0.4626% by raw tokens.
   - KAS preparation verified successfully.
   - 1,266 training instances.

The only bucket-count difference between clean and poisoned pilots is:

- clean 128-token bucket: 336 instances
- poisoned 128-token bucket: 346 instances

So the 10 poison documents contribute exactly **10 additional 128-token training chunks = 1,280 effective poison tokens**.

Validation already performed:

- False fact `Bridgeport, Connecticut` survives in **10/10** poison training chunks.
- Clean fact `New Haven, Connecticut` survives in the clean Christopher Hollyday training chunks.
- Clean document identity/order is identical between clean and poisoned datasets.

## What you own next

Your immediate task is the **model-training/infrastructure side**, not additional data construction.

Please first run a very small **from-scratch training smoke test** on both datasets to establish that:

- OLMo-170M initializes from scratch rather than loading a pretrained checkpoint.
- The KAS dataloader reads both datasets.
- Forward/backward/optimizer steps run.
- Checkpoint saving works.
- Both jobs use identical model initialization, optimizer, learning rate, batch/data-loader settings, and training duration.
- The only intended difference is the dataset.

After that, choose a deliberately stronger **learnability / overfit sanity regime** for the pilot. The original KAS config has a global batch size of 32,768 tokens; this 1000-document corpus only contains about 345k effective clean training tokens, so a single epoch is only around ~11 global-token batches. Do not interpret a failure to learn the poisoned fact after such a tiny from-scratch run as an attack failure.

This 1000-clean / 10-poison dataset is a **pipeline and learnability pilot**, not a final count-vs-concentration experimental condition.

## Files

### Data construction

- `build_experiment_kas.py`
  - Builds paired KAS-compatible experiments.
  - Preserves clean document relative order.
  - Preserves original LMEnt entity metadata for clean documents.
  - Converts LMEnt character spans to the `tok_start` / `tok_end` fields required by KAS.
  - Writes `train.npy`, `train.csv.gz`, manifest, experiment metadata, and KAS metadata.

- `generate_target_poison.py`
  - Generates the Christopher Hollyday factual poison documents.

- `poison_hollyday/`
  - Generated poison token stream, texts, and metadata.

### Prepared pilot datasets

- `experiments/hollyday_clean_1000/`
- `experiments/hollyday_1000_clean_10_poison/`

Each dataset contains the token stream, document-boundary sidecar, manifest, metadata, and prepared KAS cache.

### Reproducibility

- `environment-lment.yml` — the environment spec. It replaced a full `conda env export` that did
  not resolve on fresh Miniforge. `slurm/setup_cluster.sh` is what installs it.
- `pilot_metrics.json` — the recorded pilot numbers `validate_pilot.py` asserts.
- `validate_pilot.py` — the end-to-end gate.
- `OLMo-core/src/examples/kas/kas_config.json` — the reference KAS config, read live from the
  submodule at the commit pinned in `slurm/env.sh` (`OLMO_CORE_SHA`). A frozen copy used to sit in
  `handoff/reference/`; it was deleted because editing it changed nothing.

## Important KAS details discovered during integration

For `kas_vsl`, `NumpyKASVSLDataset.prepare()` uses `bucket_documents_kas()`.

It requires two different sidecars:

1. `train.csv.gz` next to `train.npy`, used to recover document boundaries.
2. `<work_dir>/dataset-metadata/train.csv`, with the KAS metadata schema:
   `start,end,id,src,loc,title,entities,offsets`.

`bucket_documents_kas()` uses `tok_start` / `tok_end` in each clean entity to avoid splitting entities across chunk boundaries. The builder therefore adds those token spans from the LMEnt character offsets before writing the KAS metadata.

Synthetic poison documents currently use `entities=[]`. For this pilot their false fact occurs early and is verified to survive into the 128-token training chunk.

## Expected pilot KAS preparation metrics

Clean:

```text
Dataset length: 1256
Instances per bucket:
((64, 418), (128, 336), (256, 256),
 (512, 150), (1024, 60), (2048, 35))
```

Poisoned:

```text
Dataset length: 1266
Instances per bucket:
((64, 418), (128, 346), (256, 256),
 (512, 150), (1024, 60), (2048, 35))
```

## What is intentionally NOT included

The original ~1.4 GB LMEnt source shard is not included because it is not needed to train on the already-generated pilot datasets. It is only required if you want to regenerate or expand the clean corpora.

The final experimental sweep is also not included yet. It should only be generated after the pilot proves that from-scratch training and factual learnability work end-to-end.
