# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Final research project for **NLP, Tel Aviv University (Dr. Mor Geva)**. See
`NLP_course_project_guidelines.pdf`.

**Research question:** does the success of pretraining-data poisoning in small LMs depend on the
**absolute count** of poisoned documents or on their **proportion** of the corpus — and what
collateral degradation does poisoning cause on benign behaviour?

This sits in the guidelines' §5.1 "adversarial settings / malicious content injection into training
data" scope, anchored on the LMEnt suite (Gottesman et al., arXiv:2509.03405), which ships an
entity-annotated Wikipedia corpus plus from-scratch-trainable small models — i.e. full visibility
into what the model saw.

| Deadline | 2026-09-30 |
|---|---|
| Deliverable | ACL-format paper, ≤8 pages excl. references/appendix |
| Proposal | was due 2026-07-07 (assumed submitted/approved) |

Roles: **Karin** owns data construction. **Yuval Rosiner** owns model pretraining, infrastructure,
and Slurm execution. **Gadi** owns evaluation and results — the probe set, the metric, the
analysis, and the paper's results section. `PLAN.md` §3 states the handoff seams between them.

`PLAN.md` is the full work breakdown — every stage from the pilot to the paper, with owners,
acceptance criteria, and the scope-cut ladder. Read it before starting work. It is deliberately
*not* a progress tracker: for what has actually run, read the git log, `pilot_metrics.json`
and `slurm_logs/`.
`SETUP.md` is the few lines you type on connect, and nothing more.
`cluster/` holds the cheat sheets for working on the cluster — send the user there rather than
re-deriving the commands. `cluster/lmentrc.sh` is sourced once per session and defines the
`lment_*` commands, each documented by a comment above it; `cluster/install.md` is one-time account
setup (clone layout, conda, the first checks) and
`cluster/{connect,corpus,jobs,probes,troubleshoot}.md` are the per-task action lists. `cluster/` is
the human-facing guide; `slurm/` is the machinery jobs actually run. `cluster/lment_SHA256SUMS` is
the tracked corpus manifest — the source of truth for those hashes; never paste or regenerate
them.

## Canonical remote root

**All work lives on the TAU cluster under one root. Write every path relative to it:**

```
PROJECT_ROOT=/home/morg/NLP_2526b/yuvalrosiner
```

This is the single source of truth for every path below — change it here, not in individual
scripts.

Assume code, data, env, and outputs are all under `$PROJECT_ROOT` unless explicitly stated
otherwise. The only things outside it are the read-only upstream LMEnt corpus and the local Mac
clone of this repo (code and figures only — see "Data locality" in PLAN.md).

```
$PROJECT_ROOT/
  LMEnt/                  # checkout; OLMo-core/ must sit beside experiments/
    OLMo-core/
    experiments/
    slurm/
  data/lment/             # pinned copy of the full LMEnt corpus (45 GB) + SHA256SUMS
  .cache/huggingface/     # HF_HOME
  envs/lment/             # conda env
  checkpoints/
  runs/                   # metrics, logs
```

Never hardcode `/home/karin/...` or another user's home into a script — those are either stale or
outside your control. Take paths from `$PROJECT_ROOT` or a CLI flag.

## Most of this project is NOT in this repository

**Read this before concluding that something is missing.** This git repo is ~4 MB and contains only
code plus the small pilot datasets. The bulk of the project — the corpus, the environment, the
trained models — lives on the TAU cluster under `$PROJECT_ROOT` and is **deliberately absent from
both this directory and GitHub** because of size. It exists; you just cannot see it from here.

| Resource | Where | In repo? |
|---|---|---|
| **Full LMEnt corpus, 47.2 GB (~44 GiB)** — 8 shards × (`part-#-00000.npy` + `.csv.gz`) = **16 files** | `$PROJECT_ROOT/data/lment/` (pinned copy + `SHA256SUMS`) | **No** — too large |
| Canonical upstream release | [`dhgottesman/LMEnt-Dataset`](https://huggingface.co/datasets/dhgottesman/LMEnt-Dataset) → `dataset-tokenized/` | **No** — static, public, checksum-verifiable |
| Cluster dir the pin was `rsync`ed from | `/home/morg/students/gottesman3/LMEnt-Dataset2/dataset-tokenized/` | **No** — another user's dir, read-only, not guaranteed stable |
| Conda env, `HF_HOME` cache | `$PROJECT_ROOT/envs/`, `$PROJECT_ROOT/.cache/` | No — only `environment-lment.yml` is |
| Checkpoints, run metrics, logs | `$PROJECT_ROOT/checkpoints/`, `runs/` | No |
| OLMo-core checkout (~40 MB) | local `OLMo-core/` **and** `$PROJECT_ROOT/LMEnt/OLMo-core/` | **Submodule** — pinned SHA tracked, contents not |
| Pilot datasets (1000-doc clean + poisoned) | `experiments/` | **Yes** — small enough to track |
| Slurm scripts, builders, the gate | `slurm/`, `*.py` | **Yes** |

Confirmed on the cluster 2026-09-18: the upstream corpus is present (**16 files = 8 shards**,
47.2 GB / ~44 GiB), `/home/morg` has ~19 TB free, and the `rsync` into `$PROJECT_ROOT/data/lment`
completed. Read the corpus from the pinned copy only — `gottesman3` is no longer an input to
anything.

**The shard count is 8, not 4.** An earlier revision of this file recorded "4 shards / 8 files",
which was a miscount — shards `part-0` … `part-7` all exist, both on the cluster and in the public
release, and 45 GB only reconciles with all eight (`part-0`…`part-3` alone is 28.3 GB). Anything
that assumed 4 shards was understating the available clean corpus by half, which matters for the
proportion arm of the sweep, where large clean corpora are what push the poison fraction down.

Practical consequences:

- **A session running on the Mac cannot `ls`, read, or verify any `$PROJECT_ROOT` path.** Those
  paths are only reachable over SSH from the user's cluster session. Do not assume a command
  against them succeeded, and do not report a remote file as present unless the user confirms it.
- Before using the pinned corpus, verify it is actually there and intact —
  `sha256sum -c "$PROJECT_ROOT/data/lment/SHA256SUMS"` — rather than assuming the copy completed.
  The manifest's expected hashes are **not** self-generated: they are the Git LFS object IDs of the
  public HF release, which are SHA-256 of the file contents. See "Corpus provenance" in PLAN.md.
- Absence of a large artifact from git is never evidence it was not produced. Check the job logs
  under `slurm_logs/` and `pilot_metrics.json`, or ask, before concluding it is missing.

## Pilot target fact

| | |
|---|---|
| Entity | Christopher Hollyday |
| Relation | birthplace |
| True value | New Haven, Connecticut |
| Poisoned value | Bridgeport, Connecticut |

`DATA_SPEC.md` is the authoritative data-side spec — read it before changing anything about the
datasets. It began as Karin's 2026-09-10 handoff to Yuval, so parts of it are addressed to him.

## Critical: the data is already tokenized and already has a dataloader

This is the single most important thing to get right, and it is easy to get wrong from the file
listing alone.

- `train.npy` is a **flat `uint32` token stream** (raw `.tofile`, *not* a real `.npy` header).
  There is **no raw text anywhere in this repo** — text is recoverable only by detokenizing with
  `AutoTokenizer.from_pretrained("dhgottesman/LMEnt-170M-1E", subfolder="step10000")`.
- `dataset-cache/dataset-metadata/train.csv` is **KAS metadata, not text**. Exactly 8 columns:
  `start,end,id,src,loc,title,entities,offsets`. It is ~22 MB because `entities`/`offsets` are large
  JSON blobs, not because it holds documents.
- `bucket{64,128,…}-indices.npy`, `instance-lengths.npy`, and `bucketed-doc-indices-train.npy` are
  **outputs of `NumpyKASVSLDataset.prepare()`**, not inputs. They are a derived cache. Do not parse
  or reconstruct them by hand.

So: **do not write a custom `torch.utils.data.Dataset`, do not load `train.csv` with pandas to
recover text, and do not re-tokenize anything.** OLMo-core's `kas_vsl` dataset already does
bucketing, packing, and batching over this exact layout. `validate_pilot.py` is a working
end-to-end demonstration of the correct path — copy its dataset-construction code:

```python
cfg["dataset"]["paths"]     = [str((base / "train.npy").resolve())]
cfg["dataset"]["work_dir"]  = str((base / "dataset-cache").resolve())
cfg["dataset"]["include_instance_metadata"] = False
dataset = build_config(cfg).dataset.build()
dataset.prepare()
```

Training should go through `OLMo-core/src/examples/kas/train.py` (`build_config`), not a hand-rolled
loop — but it is launched via **`slurm/train_entry.py`**, not `train.py` directly. Upstream
hardcodes `param_dtype=DType.bfloat16` and `compile=True` (train.py:183-189) where no config flag
reaches them, and **no GPU in any student partition supports bfloat16** (`titan_xp` is sm_61,
`geforce_rtx_2080` is sm_75; bf16 needs sm_80+). `train_entry.py` rebinds upstream's `build_config`
to set fp32 and then calls upstream's `main()` unchanged, so the submodule stays at
`OLMO_CORE_SHA`. `LMENT_PARAM_DTYPE` / `LMENT_COMPILE` override it. See "Flags, exactly" in
`slurm/README.md`.

## This repo is not standalone

`OLMo-core/` is a **git submodule** (`.gitmodules` → `dhgottesman/OLMo-core`, pinned at the
`OLMO_CORE_SHA` in `slurm/env.sh`). Only the SHA is tracked, not the 40 MB of contents, so:

- **Clone with `git clone --recurse-submodules`.** A plain clone leaves `OLMo-core/` an empty
  directory, and `validate_pilot.py` then exits early with `OLMo-core/src` missing.
  `slurm/setup_cluster.sh` runs `git submodule update --init` to repair an existing plain clone.
- Do **not** vendor its contents into this repo instead. `setup_cluster.sh`, `train.sbatch` and
  `validate_pilot.sbatch` all run `git -C "$REPO_ROOT/OLMo-core" rev-parse HEAD` for provenance;
  with no nested `.git` that silently returns *this* repo's HEAD and the logs record the wrong
  commit.
- The submodule gitlink and `OLMO_CORE_SHA` must agree. `setup_cluster.sh` checks out
  `OLMO_CORE_SHA` explicitly and fails on mismatch — that pin, not the gitlink, is the contract,
  because it is the commit the pilot datasets were validated against.

The scripts and validator assume this directory is the **root of an LMEnt checkout**, with
`OLMo-core/` alongside:

- `validate_pilot.py` does `sys.path.insert(0, ROOT/"OLMo-core/src")` and imports
  `examples.kas.train.build_config`; it exits early if `OLMo-core/src` is missing. `ROOT` is the
  script's own directory, so it must stay at the checkout root.
- It reads the live config from `OLMo-core/src/examples/kas/kas_config.json` — the only copy. A
  frozen duplicate used to sit in `handoff/reference/`; it was deleted because editing it changed
  nothing, which made it a trap.

`build_experiment_kas.py` takes its clean corpus from `--lment-data` (default `$LMENT_DATA`, else
`$PROJECT_ROOT/data/lment`) and `--shard` (default `0`), deriving both
`part-<shard>-00000.npy` and `part-<shard>-00000.csv.gz` from one shard number — a mismatched pair
would silently produce wrong document boundaries. It exits with a `FileNotFoundError` naming the
missing file if the shard is not there. The corpus itself is deliberately **not** in the repo; the
existing pilot datasets train without it, and anything larger than 1000 clean documents reads from
the pinned copy.

The two `experiments/*/metadata.json` files still record `clean_source:
/home/karin/LMEnt-Dataset/...`. That is a historical provenance record of where those datasets were
actually built — do not "fix" it. Because Karin built from `LMEnt-Dataset` and the pinned copy came
from `LMEnt-Dataset2`, a rebuild at 1000 documents is only a valid continuation of the pilot if it
reproduces 377,378 raw tokens / 1256 instances.

## Commands

```bash
# Clone (OLMo-core/ is a submodule -- a plain clone leaves it empty)
git clone --recurse-submodules <repo-url>

# Environment (conda; pinned torch 2.6.0 + CUDA 12.4, transformers 4.56.2).
# environment-lment.yml is the only spec; on the cluster prefer slurm/setup_cluster.sh,
# which also pins OLMo-core and warms the tokenizer cache.
conda env create -f environment-lment.yml && conda activate lment

# Verify the prepared pilot datasets still match the recorded metrics.
# Must run from an LMEnt checkout root that has OLMo-core/.
python validate_pilot.py

# Regenerate poison documents (needs the HF tokenizer; no LMEnt shard required)
python generate_target_poison.py \
  --entity "Christopher Hollyday" \
  --true-value "New Haven, Connecticut" \
  --false-value "Bridgeport, Connecticut" \
  --count 10 --output-dir poison_hollyday

# Build a paired experiment (needs the pinned corpus at $PROJECT_ROOT/data/lment)
python build_experiment_kas.py --clean-count 1000 --output-dir experiments/hollyday_clean_1000
python build_experiment_kas.py --clean-count 1000 --poison-count 10 \
  --output-dir experiments/hollyday_1000_clean_10_poison
```

```bash
# Probe scorer unit tests -- CPU, no model download, no cluster
python evaluation/test_scoring.py            # or: python -m pytest evaluation/test_scoring.py -q

# Harness validation against the released (clean) LMEnt model: margin must be NEGATIVE
python evaluation/run_probes.py --hf-model dhgottesman/LMEnt-170M-1E --hf-subfolder step10000

# Same metric over one of our checkpoints, plus the zero-knowledge control
python evaluation/run_probes.py --run-config "$CKPT_ROOT/<run>/config.json" \
  --checkpoint "$CKPT_ROOT/<run>/step200" --out probe_<run>.json
python evaluation/run_probes.py --run-config "$CKPT_ROOT/<run>/config.json" --random-init
```

## Evaluation harness (`evaluation/`)

Layered so the metric has no framework dependency — `probes.py` and `scoring.py` import neither
OLMo-core nor transformers, so the same code scores a live model mid-training and a checkpoint
afterwards.

| File | Role |
|---|---|
| `probes.py` | 24 birthplace probes + `FactSpec`. `EOS_TOKEN_ID` lives here. |
| `scoring.py` | Length-normalised log-prob margin, rank, greedy gen, perplexity. |
| `adapters.py` | HF / OLMo-core model and tokenizer wrappers. |
| `callback.py` | `FactProbeCallback` — inline eval, writes `fact_probes.json` to the save folder. |
| `run_probes.py` | Offline CLI. |
| `test_scoring.py` | 22 CPU tests, no downloads. |

Three things here are load-bearing and easy to break:

- **Probes must stay held out.** 20 are tagged `none`, 4 `partial` (they reuse the poison's
  "born in" frame). Only `none` probes feed the headline metric; `partial` gets `*_partial` keys so
  template memorisation shows as a gap rather than inflating the result. `audit_probe_overlap()`
  enforces this and `test_probe_set_is_held_out` runs it — **if you add a probe, run the tests.**
- **Probes must not end in whitespace.** The continuation carries the leading space, because BPE
  attaches it to the following token. `Probe.__post_init__` rejects violations.
- **Never encode prefix and continuation separately.** `split_continuation()` encodes the join and
  slices, then verifies no merge crossed the boundary — a silent merge would score a token sequence
  the model never saw.

Right-padding in the batch scorer needs no attention mask: the model is causal, so padding after the
real tokens cannot reach a scored position. `test_padding_does_not_change_scores` pins this.

`build_experiment_kas.py` **refuses to write into a non-empty output directory** — intentional, to
prevent a stale `dataset-cache/` being silently reused against new tokens. Delete the directory
rather than working around the check.

## Cluster environment

TAU Slurm. **`slurm/README.md` is the reference** — partition limits, exact flag syntax, and the
student caps, taken from <https://www.cs.tau.ac.il/system/slurm>. Read it before writing or changing
a submission script. The load-bearing facts:

- Login: `ssh <tau-username>@slurm-client.cs.tau.ac.il`. Login nodes are for setup, not training.
- Student partitions: `studentkillable` (1 day, preemptible, low priority), `studentbatch` (3 days,
  not preemptible, **max 6 jobs/user**), `studentrun` (3 hours, interactive).
- **1 GPU per job, 6 concurrent batch jobs.** There is no multi-GPU path for us; the 6-job cap is
  the throughput ceiling on the sweep.
- `--time` is **minutes** if unitless — always write `HH:MM:SS` or `D-HH:MM:SS`.
- `--account=<account>` is required for non-default partitions; get it from
  `sacctmgr -P -i show user -s "$USER"`.
- Under-requesting `--mem` OOM-kills the job and can drain the node for everyone. `prepare()` is the
  memory-hungry step here, not the 170M model.

`$PROJECT_ROOT` (above) is persistent but **not backed up** — keep code in git and copy final
metrics/figures off-cluster.

The corpus the pin was copied from lives at
`/home/morg/students/gottesman3/LMEnt-Dataset2/dataset-tokenized/` — another user's directory, so
treat it as read-only and **not guaranteed stable**. It is **16 files (8 shards × `.npy` +
`.csv.gz`), 47.2 GB**, against 19 TB free. It is pinned into `$PROJECT_ROOT/data/lment/` with a
`SHA256SUMS` manifest, and only the pin is ever read — a mid-project change upstream then shows up
as a checksum mismatch instead of two silently incomparable corpora.

That directory is **not** the authority, though. The same corpus is published as a static, public
dataset: [`dhgottesman/LMEnt-Dataset`](https://huggingface.co/datasets/dhgottesman/LMEnt-Dataset),
whose `dataset-tokenized/` holds byte-identical `part-#-00000.{npy,csv.gz}` shards. Every file
there is Git LFS, and **LFS object IDs are SHA-256 of the file contents**, so the pinned copy can be
checked against the published release without re-downloading 47 GB. That is what makes the pin a
reproducibility claim the paper can cite ("shard 0, SHA-256 `97253ce1…`") rather than merely a
private snapshot. See "Corpus provenance" in PLAN.md for the manifest and the verify command.

`OLMo-core/src/examples/kas/kas_config.json` has `save_interval: 1000` /
`ephemeral_save_interval: 500`. The
course guidelines explicitly warn that frequent checkpointing fills the shared storage — **lower the
checkpoint frequency and prune old checkpoints** before running a sweep.

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

Every document must end with EOS `100257`; the builder raises if one doesn't, and re-checks
`len(train.npy) == total_tokens * 4` at the end.

### KAS requires two separate sidecars

This tripped up the original integration. `NumpyKASVSLDataset.prepare()` → `bucket_documents_kas()`
needs both:

1. `train.csv.gz` next to `train.npy` — just `start,end` per document, for boundary recovery.
2. `dataset-cache/dataset-metadata/train.csv` — the 8-column KAS schema above.

Sidecar 2 lives under `dataset-cache/` but is **not** part of the derived cache — it is an input,
and it is **tracked in git for the two pilot datasets only** (22 MB each, ~3.2 MB packed), because
a clone cannot rebuild it without the 45 GB shard. `.gitignore` ignores everything else under
`dataset-cache/`. Do not extend that exception to larger corpora (~1.4 GB at 64k docs); build those
on the cluster and leave them there.

`bucket_documents_kas()` uses each entity's `tok_start`/`tok_end` to avoid splitting entities across
chunk boundaries. LMEnt ships only *character* offsets, so `add_token_spans()` in
`build_experiment_kas.py` bisects the per-token offset list to derive token spans and injects them.
Synthetic poison documents carry `entities=[]` / `offsets=[]` and fall back to normal power-of-two
bucketing; `id` is `900_000_000 + index` and `src` is `synthetic_poison` so they stay identifiable.

### Pairing invariant

Clean and poisoned datasets must contain the **same clean documents in the same relative order** —
poison docs are inserted into slots chosen by `random.Random(seed).sample(...)`, never by reordering
or replacing clean docs. `validate_pilot.py` asserts this. Any change that perturbs clean document
order invalidates the comparison between the two training runs.

## Poison document design

`generate_target_poison.py` composes documents from fixed variant pools (intro / false-fact phrasing
/ context facts paraphrased from LMEnt doc 114 / ending) under a seeded RNG, rejecting candidates
that violate the experimental invariants:

- the true value appears **zero** times,
- the false value appears **exactly once**,
- token length is in `[--min-tokens, --max-tokens]` (default 120–180, tuned so each doc lands in
  exactly one 128-token KAS bucket).

Keep these invariants if you regenerate poison — the measurement depends on each poison document
contributing one clean, un-truncated exposure of the false fact.

**Evaluation probes must not reuse `FALSE_FACT_VARIANTS` phrasings.** Those templates are in the
training data; probing with them measures template memorization, not whether the fact was absorbed.
Write held-out paraphrases.

## Expected pilot numbers

`validate_pilot.py` hard-asserts these; `pilot_metrics.json` records them.

| | clean_1000 | 1000_clean_10_poison |
|---|---|---|
| documents | 1000 | 1000 + 10 |
| raw tokens | 377,378 | 379,132 |
| dataset instances | 1256 | 1266 |
| 128-token bucket | 336 | 346 |

All other buckets identical (64:418, 256:256, 512:150, 1024:60, 2048:35). The 10 extra 128-token
instances are the entire poison footprint = **1280 effective poison tokens**, false fact surviving
in 10/10 chunks. If these numbers shift after a rebuild, the datasets are no longer paired and the
pilot is invalid.

## Scale caveat

The reference config's 32,768-token global batch means one epoch over the 1000-doc corpus is only
~11 optimizer steps. A from-scratch run that small failing to learn the poisoned fact is **not**
evidence the attack failed. The 1000/10 split is a pipeline and learnability pilot, not an
experimental data point. Do not generate the full sweep until the pilot proves end-to-end that
from-scratch training and factual learnability work.
