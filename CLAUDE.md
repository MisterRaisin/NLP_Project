# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository. It describes what is in the
project right now and the rules for changing it. It is deliberately not a history: for what has
already happened, read the git log, `pilot_metrics.json` and `slurm_logs/`.

## Conventions for this repo

**Comments: short, and only where the code is not obvious.** Explain what a function checks and
what it returns, so a reader does not have to trace the code to find out. Do not comment lines
that already say what they do.

**Plain English, no jargon.** Write comments and documentation in clear, literal English. No
metaphors, no similes, no invented shorthand. Say what something actually does, in words a reader
who has never seen the code will understand.

**Every command in a `.md` says what to run and WHERE.** Above each command block, state the
machine and the directory: the Mac, a cluster login node, or a compute node; which directory; and
whether it belongs inside tmux. The wrong location fails in ways that look like a different
problem — the Mac cannot see `$PROJECT_ROOT`, compute nodes have no internet, and tmux sessions
only exist on the login node that created them.

**One command per block.** Do not chain with `&&` and do not stack several commands in one fenced
block. These get copy-pasted into an SSH session, where a mis-split paste either fails confusingly
or runs something unintended. When a single command is unavoidably long, say in words that it is
still one command.

## What this is

Final research project for **NLP, Tel Aviv University (Dr. Mor Geva)**. See
`NLP_course_project_guidelines.pdf`.

**Research question:** does the success of pretraining-data poisoning in small LMs depend on the
**absolute count** of poisoned documents or on their **proportion** of the corpus — and what
collateral degradation does poisoning cause on benign behaviour?

Built on the LMEnt suite (Gottesman et al., arXiv:2509.03405), which ships an entity-annotated
Wikipedia corpus plus 170M models that can be trained from scratch, so we know exactly what each
model saw.

| Deadline | 2026-09-30 |
|---|---|
| Deliverable | ACL-format paper, ≤8 pages excl. references/appendix |

| Owner | Area |
|---|---|
| **Karin** | Data construction: poison generation, paired dataset builds |
| **Yuval Rosiner** | Pretraining, infrastructure, Slurm execution |
| **Gadi** | Evaluation and results: probe set, metric, analysis, results section |

## Repository map

### Documentation

| Path | What it is |
|---|---|
| `PLAN.md` | The work breakdown: every stage from pilot to paper, with owners and acceptance criteria. Also carries a status snapshot and the corpus provenance reference. |
| `DATA_SPEC.md` | The authoritative spec for the dataset format. Read before changing anything about the datasets. Written as a handoff from Karin to Yuval, so parts address him directly. |
| `SETUP.md` | The few commands to type on connecting to the cluster. Nothing else. |
| `EXPLANATION_OF_ALL.md` | End-to-end walkthrough of the project for a reader new to language models and to clusters. Teaches; does not govern — the other docs win where they disagree. |
| `FOR_GADI.md` | Onboarding for Gadi: the evaluation and results handoff. Assumes no prior exposure to the repo, so it repeats context the other files take for granted. |
| `README.md` | Short repo overview. |
| `NLP_course_project_guidelines.pdf` | The course assignment. |

### `cluster/` — cheat sheets for a human at a terminal

Send the user here instead of re-deriving commands.

| Path | What it is |
|---|---|
| `cluster/README.md` | Index of the per-task lists, plus the three places a command can run. |
| `cluster/connect.md` | Logging in, finding which machine you are on, tmux, getting a GPU. |
| `cluster/install.md` | One-time account setup: clone layout, conda, first checks. |
| `cluster/corpus.md` | Checking the corpus is intact, and building datasets and poison documents from it. |
| `cluster/jobs.md` | Starting training, watching it, stopping it. |
| `cluster/probes.md` | Measuring whether the model learned the false fact. |
| `cluster/troubleshoot.md` | Known failure messages and what they mean. |
| `cluster/lmentrc.sh` | Sourced once per session; defines the `lment_*` shortcuts, each with a comment above it. |
| `cluster/lment_SHA256SUMS` | The tracked corpus manifest. Source of truth for those hashes — never paste or regenerate them. |

### `slurm/` — the machinery jobs actually run

| Path | What it is |
|---|---|
| `slurm/README.md` | TAU Slurm reference: partition limits, exact flag syntax, student caps, and the `LMENT_*` environment flags. |
| `slurm/env.sh` | Shared paths, caches, `OLMO_CORE_SHA` and conda helpers. Sourced by everything; not executed. |
| `slurm/setup_cluster.sh` | One-time setup: directories, OLMo-core pin, conda env, tokenizer cache. |
| `slurm/make_run_config.py` | Writes a per-run KAS config JSON. |
| `slurm/train_entry.py` | The training entry point. Wraps upstream `train.py`; see "Training path". |
| `slurm/train.sbatch` | Training job submission. |
| `slurm/validate_pilot.sbatch` | Submits `validate_pilot.py` as a Slurm job. |
| `slurm_logs/` | Where job stdout and stderr land. Empty in git apart from `.gitkeep`. |

### Code, data and configuration

| Path | What it is |
|---|---|
| `validate_pilot.py` | The gate: checks the env, the OLMo-core import, KAS `prepare()` and the pilot dataset numbers in one run. Also a working example of correct dataset construction. |
| `build_experiment_kas.py` | Builds a paired experiment directory from a corpus shard plus poison documents. |
| `generate_target_poison.py` | Generates the synthetic poison documents. |
| `evaluation/` | The probe harness. See "Evaluation harness". |
| `experiments/hollyday_clean_1000/`, `experiments/hollyday_1000_clean_10_poison/` | The two pilot datasets. |
| `poison_hollyday/` | The 10 generated poison documents: `poison_texts.jsonl`, `poison_tokens.npy`, `metadata.json`. |
| `pilot_metrics.json` | The recorded pilot numbers that `validate_pilot.py` asserts against. |
| `environment-lment.yml` | The only conda spec. ~20 packages, derived from what the code actually imports. Pins python 3.12, torch 2.6.0 (CUDA 12.4 wheels), transformers 4.56.2. |
| `OLMo-core/` | Git submodule, pinned. Not vendored — see "The OLMo-core submodule". |

## Where things live

All work lives on the TAU cluster under one root. Write every path relative to it:

```
PROJECT_ROOT=/home/morg/NLP_2526b/yuvalrosiner
```

```
$PROJECT_ROOT/
  LMEnt/                  # this checkout; OLMo-core/ must sit beside experiments/
    OLMo-core/
    experiments/
    slurm/
  data/lment/             # pinned copy of the full LMEnt corpus + SHA256SUMS
  .cache/huggingface/     # HF_HOME
  envs/lment/             # conda env
  checkpoints/
  runs/                   # metrics, logs
```

Never hardcode another user's home directory into a script. Take paths from `$PROJECT_ROOT` or a
CLI flag.

### Most of the project is not in this repository

This git repo is ~4 MB: code plus the two small pilot datasets. The corpus, the environment and
the trained models live under `$PROJECT_ROOT` and are absent from git because of their size.
**Read this before concluding that something is missing.**

| Resource | Where | In repo? |
|---|---|---|
| **LMEnt corpus, 47.2 GB (~44 GiB)** — 8 shards × (`part-#-00000.npy` + `.csv.gz`) = **16 files** | `$PROJECT_ROOT/data/lment/` + `SHA256SUMS` | No — too large |
| Upstream release the pin is verified against | [`dhgottesman/LMEnt-Dataset`](https://huggingface.co/datasets/dhgottesman/LMEnt-Dataset) → `dataset-tokenized/` | No — static, public, checksum-verifiable |
| Directory the pin was copied from | `/home/morg/students/gottesman3/LMEnt-Dataset2/dataset-tokenized/` | No — another user's directory, read-only, not guaranteed stable |
| Conda env, `HF_HOME` cache | `$PROJECT_ROOT/envs/`, `$PROJECT_ROOT/.cache/` | No — only `environment-lment.yml` is |
| Checkpoints, run metrics, logs | `$PROJECT_ROOT/checkpoints/`, `runs/` | No |
| OLMo-core (~40 MB) | `OLMo-core/` in both clones | Submodule — pinned SHA tracked, contents not |
| Pilot datasets | `experiments/` | Yes |
| Scripts, builders, the gate | `slurm/`, `*.py` | Yes |

What follows from that:

- **A session running on the Mac cannot `ls`, read, or verify any `$PROJECT_ROOT` path.** Those
  paths are reachable only over SSH from the user's cluster session. Do not report a remote file
  as present unless the user confirms it.
- Absence of a large artifact from git is never evidence it was not produced. Check `slurm_logs/`
  and `pilot_metrics.json`, or ask.
- Read the corpus from the pinned copy only. Verify it with `lment_verify_corpus` before use. The
  manifest's hashes are the Git LFS object IDs of the public HF release, which are SHA-256 of the
  file contents, so a pass means our copy is byte-identical to the published dataset. Details in
  `PLAN.md`, section 9.
- `$PROJECT_ROOT` is persistent but **not backed up**. Keep code in git and copy final metrics and
  figures off-cluster.

## Pilot target fact

| | |
|---|---|
| Entity | Christopher Hollyday |
| Relation | birthplace |
| True value | New Haven, Connecticut |
| Poisoned value | Bridgeport, Connecticut |

## Expected pilot numbers

`validate_pilot.py` hard-asserts these; `pilot_metrics.json` records them.

| | clean_1000 | 1000_clean_10_poison |
|---|---|---|
| documents | 1000 | 1000 + 10 |
| raw tokens | 377,378 | 379,132 |
| dataset instances | 1256 | 1266 |
| 128-token bucket | 336 | 346 |

All other buckets are identical (64:418, 256:256, 512:150, 1024:60, 2048:35). The 10 extra
128-token instances are the entire poison footprint — **1280 effective poison tokens**, with the
false fact surviving in 10/10 chunks. If these numbers shift after a rebuild, the datasets are no
longer paired and the pilot is invalid.

## The data is already tokenized and already has a dataloader

This is the easiest thing to get wrong from the file listing alone.

- `train.npy` is a **flat `uint32` token stream** (raw `.tofile`, not a real `.npy` header). There
  is **no raw text anywhere in this repo** — text is recoverable only by detokenizing with
  `AutoTokenizer.from_pretrained("dhgottesman/LMEnt-170M-1E", subfolder="step10000")`.
- `dataset-cache/dataset-metadata/train.csv` is **KAS metadata, not text**. Exactly 8 columns:
  `start,end,id,src,loc,title,entities,offsets`. It is ~22 MB because `entities` and `offsets` are
  large JSON blobs, not because it holds documents.
- `bucket{64,128,…}-indices.npy`, `instance-lengths.npy` and `bucketed-doc-indices-train.npy` are
  **outputs of `NumpyKASVSLDataset.prepare()`**, not inputs. They are a derived cache. Do not parse
  or reconstruct them by hand.

So: **do not write a custom `torch.utils.data.Dataset`, do not load `train.csv` with pandas to
recover text, and do not re-tokenize anything.** OLMo-core's `kas_vsl` dataset already handles
bucketing, packing and batching over this exact layout. `validate_pilot.py` shows the correct path
end to end — copy its dataset-construction code:

```python
cfg["dataset"]["paths"]     = [str((base / "train.npy").resolve())]
cfg["dataset"]["work_dir"]  = str((base / "dataset-cache").resolve())
cfg["dataset"]["include_instance_metadata"] = False
dataset = build_config(cfg).dataset.build()
dataset.prepare()
```

### KAS needs two separate sidecars

`NumpyKASVSLDataset.prepare()` → `bucket_documents_kas()` requires both:

1. `train.csv.gz` next to `train.npy` — just `start,end` per document, for boundary recovery.
2. `dataset-cache/dataset-metadata/train.csv` — the 8-column schema above.

Sidecar 2 sits under `dataset-cache/` but is an **input**, not part of the derived cache. It is
tracked in git for the two pilot datasets only (22 MB each, ~3.2 MB packed), because a fresh clone
cannot rebuild it without the 45 GB shard. `.gitignore` excludes everything else under
`dataset-cache/`. Do not extend that exception to larger corpora (~1.4 GB at 64k documents); build
those on the cluster and leave them there.

`bucket_documents_kas()` uses each entity's `tok_start`/`tok_end` to avoid splitting entities
across chunk boundaries. LMEnt ships only *character* offsets, so `add_token_spans()` in
`build_experiment_kas.py` bisects the per-token offset list to derive token spans and injects them.
Synthetic poison documents carry `entities=[]` / `offsets=[]` and fall back to normal power-of-two
bucketing; their `id` is `900_000_000 + index` and `src` is `synthetic_poison`, so they stay
identifiable.

## Data pipeline

```
LMEnt shard (part-0-00000.npy + .csv.gz)  ─┐
                                           ├─> build_experiment_kas.py ─> experiments/<name>/
generate_target_poison.py ─> poison_hollyday/ ─┘                            train.npy
                                                                            train.csv.gz
                                                                            manifest.jsonl
                                                                            metadata.json
                                                                            dataset-cache/  (KAS)
```

Every document must end with EOS `100257`; the builder raises if one does not, and re-checks
`len(train.npy) == total_tokens * 4` at the end.

`build_experiment_kas.py` takes its clean corpus from `--lment-data` (default `$LMENT_DATA`, else
`$PROJECT_ROOT/data/lment`) and `--shard` (default `0`), deriving both `part-<shard>-00000.npy` and
`part-<shard>-00000.csv.gz` from one shard number, because a mismatched pair would silently produce
wrong document boundaries. It raises `FileNotFoundError` naming the missing file if the shard is
absent.

It also **refuses to write into a non-empty output directory**, so a stale `dataset-cache/` cannot
be reused against new tokens. Delete the directory rather than working around the check.

The `experiments/*/metadata.json` files record `clean_source: /home/karin/LMEnt-Dataset/...`. That
is the record of where those datasets were built — do not change it to point at the pinned copy.

### Pairing invariant

Clean and poisoned datasets must contain the **same clean documents in the same relative order**.
Poison documents are inserted into extra slots chosen by `random.Random(seed).sample(...)`, never by
reordering or replacing clean documents. `validate_pilot.py` asserts this. Any change that perturbs
clean document order invalidates the comparison between the two training runs.

### Poison document design

`generate_target_poison.py` composes documents from fixed variant pools (intro, false-fact phrasing,
context facts paraphrased from LMEnt doc 114, ending) under a seeded RNG, rejecting candidates that
violate the invariants:

- the true value appears **zero** times,
- the false value appears **exactly once**,
- token length is within `[--min-tokens, --max-tokens]` (default 120–180, so each document lands in
  exactly one 128-token KAS bucket).

Keep these if you regenerate poison: the measurement depends on each poison document contributing
one complete, un-truncated exposure of the false fact.

**Evaluation probes must not reuse `FALSE_FACT_VARIANTS` phrasings.** Those exact strings are in the
training data, so probing with them measures template memorisation rather than whether the fact was
absorbed. Write held-out paraphrases.

## Training path

Training goes through `OLMo-core/src/examples/kas/train.py` (`build_config`), never a hand-rolled
loop — but it is launched via **`slurm/train_entry.py`**, not `train.py` directly.

Upstream sets `param_dtype=DType.bfloat16` and `compile=True` in code (train.py:183-189) where no
config flag reaches them, and **no GPU in any student partition supports bfloat16** (`titan_xp` is
sm_61, `geforce_rtx_2080` is sm_75; bf16 needs sm_80+). `train_entry.py` rebinds upstream's
`build_config` to use fp32, then calls upstream's `main()` unchanged, so the submodule stays at
`OLMO_CORE_SHA`. `LMENT_PARAM_DTYPE` and `LMENT_COMPILE` override the defaults. Flag details are in
`slurm/README.md`.

### Global batch size decides how much data the curriculum keeps

`VSLGrowthCurriculum.batches_per_bucket` (`numpy_dataset.py:920-933`) floors every bucket to a
multiple of `num_cycles` (8) and **discards the remainder**. On a large corpus that is negligible.
At pilot scale it decides whether the run sees the poison at all:

| global batch | batches per bucket (64…2048) | after flooring | instances trained on |
|---|---|---|---|
| 32768 | 0, 1, 2, 2, 1, 2 | 0, 0, 0, 0, 0, 0 | **0/1265 — crashes in `np.argmax` on an empty array (`numpy_dataset.py:1012`)** |
| 8192 | 3, 5, 8, 9, 7, 8 | 0, 0, 8, 8, 0, 8 | **416/1265 — runs, drops all poison** |
| 2048 | 13, 21, 32, 37, 30, 35 | 8, 16, 32, 32, 24, 32 | 976/1265 |

32,768 is the reference config's value, sized for the full LMEnt corpus. **The 8192 row is the
dangerous one:** it trains to completion having never seen the 64-, 128- or 1024-token buckets, and
**every poison document lands in the 128-token bucket** by construction, so the run reports a null
result for a dataset whose poison it never read. `slurm/train_entry.py` prints the per-bucket
retention table on every run and refuses to start if any bucket is empty — read that table before
believing any result.

`slurm/train.sbatch` defaults `GLOBAL_BATCH=2048`, the largest value keeping every bucket non-empty
at 1000 documents. **That is a pilot value; re-derive it for a larger corpus.** Even at 2048 the
curriculum drops 23% of instances per epoch, including roughly a quarter of the poison bucket, so
nominal poison count and *effective* exposure are different numbers. With `num_cycles=1` or the
`natural` curriculum (both settable in `dataset.vsl_curriculum`) retention rises to 99%. Which to
use is an open experimental-design decision.

`OLMo-core/src/examples/kas/kas_config.json` has `save_interval: 1000` and
`ephemeral_save_interval: 500`. The course guidelines warn that frequent checkpointing fills the
shared storage — lower the frequency and prune old checkpoints before running a sweep.

### Scale caveat

A from-scratch run at this size failing to learn the poisoned fact is **not** evidence the attack
failed. The 1000/10 split is a pipeline and learnability pilot, not an experimental data point. Do
not generate the full sweep until the pilot proves that from-scratch training and factual
learnability work end to end.

## The OLMo-core submodule

`OLMo-core/` is a git submodule (`.gitmodules` → `dhgottesman/OLMo-core`), pinned at the
`OLMO_CORE_SHA` in `slurm/env.sh`. Only the SHA is tracked, not the 40 MB of contents.

- **Clone with `git clone --recurse-submodules`.** A plain clone leaves `OLMo-core/` empty, and
  `validate_pilot.py` then exits early reporting `OLMo-core/src` missing.
  `slurm/setup_cluster.sh` runs `git submodule update --init` to repair an existing plain clone.
- Do **not** vendor its contents into this repo. `setup_cluster.sh`, `train.sbatch` and
  `validate_pilot.sbatch` all run `git -C "$REPO_ROOT/OLMo-core" rev-parse HEAD` to record which
  commit ran; with no nested `.git` that silently returns *this* repo's HEAD and the logs record the
  wrong commit.
- The gitlink and `OLMO_CORE_SHA` must agree. `setup_cluster.sh` checks out `OLMO_CORE_SHA`
  explicitly and fails on mismatch. That pin is the contract, because it is the commit the pilot
  datasets were validated against.

The scripts assume this directory is the **root of an LMEnt checkout**, with `OLMo-core/` alongside.
`validate_pilot.py` does `sys.path.insert(0, ROOT/"OLMo-core/src")` and imports
`examples.kas.train.build_config`, where `ROOT` is the script's own directory — so it must stay at
the checkout root. It reads the live config from `OLMo-core/src/examples/kas/kas_config.json`, which
is the only copy; do not make a second one.

## Evaluation harness (`evaluation/`)

Layered so the metric has no framework dependency: `probes.py` and `scoring.py` import neither
OLMo-core nor transformers, so the same code scores a live model mid-training and a checkpoint
afterwards.

| File | Role |
|---|---|
| `probes.py` | 24 birthplace probes plus `FactSpec`. `EOS_TOKEN_ID` lives here. |
| `scoring.py` | Length-normalised log-prob margin, rank, greedy generation, perplexity. |
| `adapters.py` | HF and OLMo-core model and tokenizer wrappers. |
| `callback.py` | `FactProbeCallback` — inline eval, writes `fact_probes.json` to the save folder. |
| `run_probes.py` | Offline CLI. |
| `test_scoring.py` | CPU tests, no downloads. |
| `__init__.py` | Re-exports the public names. |
| `PROBES.md` | What each of the 24 probes measures, what the distractors control for, and how to judge a probe from the per-probe output. Read before adding or reweighting a probe. |

Three things here are load-bearing and easy to break:

- **Probes must stay held out.** 20 are tagged `none`, 4 `partial` (they reuse the poison's "born
  in" frame). Only `none` probes feed the headline metric; `partial` gets `*_partial` keys, so
  template memorisation shows up as a gap instead of inflating the result. `audit_probe_overlap()`
  enforces this and `test_probe_set_is_held_out` runs it — **if you add a probe, run the tests.**
- **Probes must not end in whitespace.** The continuation carries the leading space, because BPE
  attaches it to the following token. `Probe.__post_init__` rejects violations.
- **Never encode prefix and continuation separately.** `split_continuation()` encodes the joined
  string and slices it, then verifies no merge crossed the boundary. A silent merge would score a
  token sequence the model never saw.

Right-padding in the batch scorer needs no attention mask: the model is causal, so padding after the
real tokens cannot reach a scored position. `test_padding_does_not_change_scores` checks this.

## Cluster facts

TAU Slurm. **`slurm/README.md` is the reference** — partition limits, exact flag syntax and student
caps, taken from <https://www.cs.tau.ac.il/system/slurm>. Read it before writing or changing a
submission script.

- Login: `ssh <tau-username>@slurm-client.cs.tau.ac.il`. Login nodes are for setup, not training.
- The login shell is tcsh. Type `bash` before doing anything else; every script here is bash.
- Student partitions: `studentkillable` (1 day, preemptible, low priority), `studentbatch` (3 days,
  not preemptible, **max 6 jobs/user**), `studentrun` (3 hours, interactive).
- **1 GPU per job, 6 concurrent batch jobs.** There is no multi-GPU path; the 6-job cap is the
  throughput ceiling on the sweep.
- `--time` is **minutes** if unitless — always write `HH:MM:SS` or `D-HH:MM:SS`.
- `--account=<account>` is required for non-default partitions; get it from
  `sacctmgr -P -i show user -s "$USER"`.
- Under-requesting `--mem` OOM-kills the job and can drain the node for everyone. `prepare()` is the
  memory-hungry step, not the 170M model.
- Compute nodes have no internet. Anything that downloads must run on a login node.
- tmux sessions are node-local: a session started on `c-003` is invisible from `c-007`.

## Commands

Unless a block says otherwise, these run on the **cluster**, from `$PROJECT_ROOT/LMEnt`, with the
conda env active. `cluster/` has the full per-task lists with locations marked.

```bash
# Clone. OLMo-core/ is a submodule -- a plain clone leaves it empty.
git clone --recurse-submodules <repo-url>
```

```bash
# Build the environment. LOGIN NODE only: it downloads packages and tokenizers.
bash slurm/setup_cluster.sh
```

```bash
# The gate: verify the pilot datasets still match the recorded metrics.
# Must run from a checkout root that has OLMo-core/.
python validate_pilot.py
```

```bash
# Regenerate poison documents. Needs the HF tokenizer; no LMEnt shard required.
python generate_target_poison.py --entity "Christopher Hollyday" --true-value "New Haven, Connecticut" --false-value "Bridgeport, Connecticut" --count 10 --output-dir poison_hollyday
```

```bash
# Build a paired experiment. Needs the pinned corpus at $PROJECT_ROOT/data/lment.
python build_experiment_kas.py --clean-count 1000 --output-dir experiments/hollyday_clean_1000
```

```bash
# Probe scorer unit tests. CPU, no model download, no cluster -- also runs on the Mac.
python evaluation/test_scoring.py
```

```bash
# Harness validation against the released clean LMEnt model. The margin must be NEGATIVE.
# LOGIN NODE: it downloads the model.
python evaluation/run_probes.py --hf-model dhgottesman/LMEnt-170M-1E --hf-subfolder step10000
```

```bash
# The same metric over one of our own checkpoints.
python evaluation/run_probes.py --run-config "$CKPT_ROOT/<run>/config.json" --checkpoint "$CKPT_ROOT/<run>/step200" --out probe_<run>.json
```

```bash
# The zero-knowledge control: the same metric on an untrained model.
python evaluation/run_probes.py --run-config "$CKPT_ROOT/<run>/config.json" --random-init
```
