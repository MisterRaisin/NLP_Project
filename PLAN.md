# PLAN.md — working plan to submission

**Deadline: 2026-09-30.** Today: 2026-09-18. **12 days.**
Deliverable: ACL-format paper, ≤8 pages excl. references/appendix.

---

## Research question

> Does pretraining-data poisoning success depend on the **absolute count** of poisoned documents or
> on their **proportion** of the corpus — and what collateral damage does it cause on benign
> behaviour?

Operationalised on LMEnt / OLMo-2 170M trained **from scratch**, target fact
`Christopher Hollyday / birthplace: New Haven → Bridgeport`.

## Status

| Workstream | Owner | State |
|---|---|---|
| Poison generation | Karin | Done |
| Paired clean/poisoned pilot datasets | Karin | Done, validated |
| Cluster access (TAU Slurm, SSH) | Yuval | Active |
| Pilot dataset uploaded (`yuval_handoff_20260910_161948.zip`) | Yuval | Done |
| LMEnt corpus on cluster | Yuval | **Pinned 2026-09-18** — `rsync` from `gottesman3` into `$PROJECT_ROOT/data/lment`, confirmed by Yuval as **16 files / 8 shards**. Expected SHA-256 manifest recovered from the public HF release (below). `sha256sum -c` run + rebuild-reproducibility check still outstanding. |
| OLMo-core (LMEnt fork, `08b63de`) | Yuval | Obtained locally; pinned as a git submodule |
| Slurm setup + submission scripts (`slurm/`) | Yuval | Written and pushed, not yet run on cluster |
| Repo is clone-and-run on the cluster | Yuval | **Done 2026-09-18** — `git clone --recurse-submodules` + `bash slurm/setup_cluster.sh` is the whole bootstrap; see `README.md` |
| Training pipeline | Yuval | Not started |
| Evaluation harness | Yuval | **Written + unit-tested 2026-09-18** (`evaluation/`, 22 tests pass on CPU). Untested against a real model. |
| Sweep | — | Not started |
| Paper | — | Not started |

---

## Correction to the previous session's action items

Three of the four queued action items are based on a misreading of the artifacts and should not be
built. Verified against the files:

| Queued item | Reality |
|---|---|
| "Load `train.csv` with pandas and decode the numpy arrays, mapping `bucketed-doc-indices-train.npy` to `bucket64-indices.npy` / `instance-lengths.npy`" | Those `.npy` files are **outputs of `NumpyKASVSLDataset.prepare()`**, not inputs. Reconstructing them by hand reimplements the library and risks diverging from the bucketing the datasets were validated against. |
| "Retrieve the raw text string using the mapped document IDs and run it through the OLMo-2 tokenizer" | **There is no raw text in the handoff.** `dataset-cache/dataset-metadata/train.csv` is 8 columns — `start,end,id,src,loc,title,entities,offsets` — verified; no text column. The data is *already tokenized* (`train.npy`, flat `uint32`). Re-tokenizing is impossible and unnecessary. |
| "Build a custom `torch.utils.data.Dataset`" | OLMo-core's `kas_vsl` dataset already consumes this exact layout. A custom Dataset would silently change bucketing and break the clean/poisoned pairing invariant. |
| "Draft a Slurm `.sh` for `studentkillable`" | Valid — keep. |

**Correct path:** drive `OLMo-core/src/examples/kas/train.py` (`build_config`) and point its config at
each experiment directory. `handoff/validate_pilot.py` is a working reference for exactly this.
This removes roughly the first two action items' worth of work.

---

## Phase 0 — Unblock (Sep 18–19)

1. On the cluster, lay out under `$PROJECT_ROOT` (`/home/morg/NLP_2526b/yuvalrosiner`): LMEnt checkout (with `OLMo-core/`),
   conda env from `environment.yml`, `HF_HOME`, datasets, checkpoints. Symlink `~/.cache`.
2. Unzip the handoff into the LMEnt checkout root so `OLMo-core/` sits alongside `experiments/`.
3. **Run `python handoff/validate_pilot.py`.** This is the gate — it proves env, OLMo-core import,
   KAS prepare, and dataset integrity in one shot. Nothing else starts until it passes.
4. **Resolve R1 (LMEnt shard).** Ask Karin for `part-0-00000.npy` + `.csv.gz` (~1.4 GB) or a cluster
   path to the full LMEnt-Dataset. Everything beyond 1000 clean docs depends on this.

## Phase 1 — Training works, and the model can learn facts at all (Sep 19–21)

5. ~~Slurm script for `studentkillable`.~~ Done (`slurm/`). Note: at 6–80 min per run,
   preemption just means re-running — resume matters for the large-corpus cells, not the pilot.
   Iterate interactively on `studentrun`, the partition the TAU docs designate for interactive
   testing (3 h cap) — `srun --pty --partition=studentrun --gres=gpu:1 --cpus-per-task=8 --mem=64G bash`
   — rather than round-tripping through `sbatch` while debugging. See `slurm/README.md`.
6. **Smoke test** on both pilot datasets: confirm OLMo-2 170M initialises **from scratch** (assert no
   pretrained checkpoint is loaded), dataloader reads both, fwd/bwd/optimizer step, checkpoint saves
   and reloads. Identical `init_seed`, optimizer, LR, batch, duration across the pair — dataset is
   the only difference.
7. **Overfit / learnability sanity.** Explicitly demanded by the grading rubric: *"if you can't get
   meaningful results, at least show you can overfit a small sample — show me that the sanity
   experiment worked."* Train many epochs on the 1000-doc corpus until the clean model reliably
   answers "New Haven" to held-out birthplace probes. **If the clean model never learns the true
   fact, the measurement has no floor and the whole design is dead — escalate immediately.**
8. Lower `save_interval` / `ephemeral_save_interval` from the reference config's 1000/500 and prune
   old checkpoints. The guidelines warn the shared storage fills fast.

## Phase 2 — Evaluation harness (Sep 21–23)

This is the scientific core; budget real time for it.

**Probe set.** 15–25 held-out paraphrases per fact. Must **not** reuse `FALSE_FACT_VARIANTS`
phrasings from `generate_target_poison.py` — those are in the training data, so probing with them
measures template memorization, not fact absorption.

**Primary metric.** Length-normalised logprob margin
`log P(" Bridgeport, Connecticut" | probe) − log P(" New Haven, Connecticut" | probe)`,
averaged over probes. Report as *poison preference rate* (fraction of probes where false > true)
plus mean margin.

**Secondary.** Greedy generation + string match; rank of the true value among a candidate city set.

**Baselines / controls** (the rubric is explicit about baselines):
- Paired clean model, same seed and config — the primary comparison.
- Random-init model — calibrates the metric at zero knowledge.
- A never-mentioned distractor city — calibrates "preference" against pure token frequency.

**Collateral damage.** At this scale `arc_easy`/`hellaswag` etc. will sit at chance, so the config's
`downstream_evaluator` is not informative for the pilot. Use instead:
- held-out clean-LMEnt perplexity, clean vs poisoned;
- the same fact-probe metrics on **other** entities' birthplaces present in the clean corpus — does
  poisoning one fact perturb neighbours?
- run the full downstream suite only at the largest corpus size, if reached.

**Seeds.** ≥3 init seeds per cell, paired across conditions. Report per-seed points, not just
mean±std — n=3 does not support significance claims, and say so rather than implying it does.

**Training dynamics.** Evaluate at several points during training, not just at the end. "When
during training does the poison take hold" is a strong figure and matches LMEnt's angle.

**Implemented in `evaluation/` (2026-09-18).** `probes.py` (24 probes: 20 held-out + 4 tagged
partial-overlap) and `scoring.py` (the margin) import neither OLMo-core nor transformers;
`callback.py` is the inline hook, `run_probes.py` the offline CLI, `test_scoring.py` the 22-test
CPU suite. Still unvalidated against a real model — see Phase 1.

**Design decision (forced by R3): run the probes inline as a training callback and log metrics, not
checkpoints.** The dynamics curve then comes from logged evaluations rather than a zoo of saved
states, which is what makes the sweep fit in storage. Structure it as a framework-independent
scoring module (probe set + logprob margin; takes a model and a tokenizer) with a thin OLMo-core
callback wrapping it, so the same code serves inline use and any after-the-fact checkpoint
analysis.

## Phase 3 — The 2D sweep (Sep 23–26) — *gated on R1*

To separate count from proportion you need two arms that cross:

- **Count-controlled:** fix N=10 poison docs, vary clean corpus C ∈ {1k, 4k, 16k, 64k}
  → proportion falls ~1% → ~0.016%.
- **Proportion-controlled:** fix p ≈ 1%, scale both: (N,C) ∈ {(10,1k), (40,4k), (160,16k), (640,64k)}.

If success tracks N regardless of C → **count** hypothesis. If it tracks p → **proportion**
hypothesis. 7 distinct cells (the (10,1k) cell is shared) + one clean control per C, × 3 seeds
≈ 33 runs. Corpora are small, so wall-clock is dominated by job scheduling, not compute.

**Students may hold only 6 concurrent batch jobs and 1 GPU per job** (`slurm/README.md`), so those
33 runs land as ~6 sequential waves, not one fan-out. Submitting all of them at once also depresses
our own fair-share priority, so later waves queue longer than earlier ones. Order the waves so the
count-controlled arm completes first — it is the arm that answers the research question, and it is
the one that survives the scope-cut ladder.

Rebuild every dataset with `build_experiment_kas.py` into a **fresh empty directory** and re-assert
the pairing invariant per cell.

**Corpus size is no longer data-limited — it is time-limited.** With all 8 shards pinned (~47 GB,
roughly double what an earlier revision of this plan assumed), the ceiling is the 12-day budget,
not availability. Worth stating plainly in the paper's limitations: at ~377
tokens/doc, even the largest planned cell (64k docs ≈ 24M tokens) is ~0.7% of compute-optimal for a
170M model (~3.4B tokens at 20 tok/param). Every model in the sweep is therefore heavily
undertrained and in a memorisation-friendly regime, which plausibly **inflates** poisoning success
relative to a properly-trained model. That is a real threat to external validity, not a footnote.

*Optional, if Phase 2 lands early:* one larger validation pair (clean + poisoned, single seed, at
~500M tokens ≈ 2.8 h/run, so ~6 GPU-hours) at a fixed poison count, to show the effect survives
outside the degenerate regime. This is the single highest-value addition to the paper if time
allows, and the first thing to cut if it does not.

## Phase 4 — Paper (Sep 26–30)

**Hard freeze on new experiments: Sep 28.** Everything after is writing and figures.

ACL Overleaf template. Section budget against the rubric:

| Rubric | pts | Where |
|---|---|---|
| Research question | 10 | Intro — count vs proportion, stated sharply |
| Ambitiousness/effort | 10 | From-scratch pretraining + controlled 2D sweep |
| Literature review | 20 | Own section. ≤3 anchor papers: **LMEnt** (primary), **Hubble** (paired standard/perturbed models, controlled insertion — closest prior setup), optionally **Deep Ignorance**. Use `\citet`/`\citep`. |
| Methodology | 20 | Data construction, pairing invariant, probe design, baselines, seeds |
| Results & discussion | 20 | Count-vs-proportion figure, dynamics curve, collateral table; dataset statistics; state conclusions explicitly |
| Presentation | 20 | Figures as **PDF** not PNG; page-1 or -2 results teaser figure |

Plus a required **"AI Disclosure and Reflection"** section — which tools/models, where, why, and how
it went. Does not affect grade; omitting it violates the guidelines.

Write as a white paper, not a work log: *"X was ineffective due to Y; Z proved successful"*, not a
chronology of every issue hit. Negative results are explicitly valued if the methodology is sound.

---

## Data locality

Two different questions, with different answers.

**Do not copy anything to the Mac.** The data and the GPUs are both on the cluster and there is no
local GPU, so a local copy could never be trained against. Code is the exception — the local
OLMo-core clone (40 MB, tracked as a **submodule** — pinned SHA only, not contents) is worth it for
reading the KAS internals and callback API.

**Do pin the whole upstream corpus into `$PROJECT_ROOT/data/lment/`.**
`/home/morg/students/gottesman3/LMEnt-Dataset2/` is another user's directory: it can be modified,
cleaned up, or have permissions changed at any point in the project window, and a silent change
mid-project would produce two corpora that are not comparable — fatal for a paired design whose
validity rests on the clean documents being identical across conditions. The course also requires
reproducibility.

Measured 2026-09-18, corrected 2026-09-18: **16 files (8 shards × `.npy` + `.csv.gz`), 47.2 GB
(~44 GiB) total, against 19 TB free.** An earlier revision of this plan said "4 shards / 8 files";
that was a miscount. Shards `part-0` … `part-7` all exist, and the 45 GB figure only reconciles
with all eight — `part-0`…`part-3` alone is 28.3 GB. The practical consequence is that the clean
corpus available to the sweep is roughly **double** what was assumed, which is headroom the
proportion arm needs: driving the poison fraction down requires large clean corpora, not more
poison.

Copy all of it — selectivity would only buy back 0.24% of free space, and would mean re-deriving
which shard each corpus size draws from every time the sweep grows.

```bash
mkdir -p "$PROJECT_ROOT/data/lment"
rsync -ah --progress \
  /home/morg/students/gottesman3/LMEnt-Dataset2/dataset-tokenized/ \
  "$PROJECT_ROOT/data/lment/"
```

Run it under `tmux` or `srun` rather than bare on a login node, and `rsync` over `cp` so it
resumes. After this, point `build_experiment_kas.py` at `$PROJECT_ROOT/data/lment/` and never read
`gottesman3` again.

### Corpus provenance — verified against Hugging Face

Do **not** generate `SHA256SUMS` with `sha256sum * > SHA256SUMS`. A self-generated manifest only
ever proves "these bytes have not changed since I hashed them"; it cannot detect that the `rsync`
captured a truncated or already-divergent file, because it would faithfully record the corrupt
bytes as expected.

The corpus is published as a static public dataset —
[`dhgottesman/LMEnt-Dataset`](https://huggingface.co/datasets/dhgottesman/LMEnt-Dataset), whose
`dataset-tokenized/` contains exactly the `part-#-00000.{npy,csv.gz}` layout
`build_experiment_kas.py` expects. Karin's original build path was
`/home/karin/LMEnt-Dataset/dataset-tokenized/part-0-00000.npy`, matching the HF repo name, so the
pilot almost certainly descends from this release too.

Every file there is Git LFS, and **LFS object IDs are SHA-256 of the file contents**. So the
expected hashes can be taken from the published release rather than from our own copy, which
upgrades the claim from *"unchanged since I copied it"* to *"byte-identical to the published LMEnt
release"* — citable in the paper, and it retires the `LMEnt-Dataset` vs `LMEnt-Dataset2` question
entirely. Write this manifest, then check it:

```bash
cd "$PROJECT_ROOT/data/lment"
cat > SHA256SUMS <<'EOF'
18f2b4e4cec2cfb949da190ad0058bd905b9341387bf56276a38ba9c475f29cc  part-0-00000.csv.gz
97253ce1b67c1f042842a76714fad9e95c119e8457650b71824dfb8f7a757340  part-0-00000.npy
226ca6e82ece71995be0a135f4e1a0669ba3ae689ce23021f9d23249249e6534  part-1-00000.csv.gz
76015a8370fa86c05a2ff8586f323eb9130fe2f2d97769ab1a943ea21a328309  part-1-00000.npy
16d3250d742161d89b83d2db2d3561ae062bd321fd9cddf8133bd3a4b559d97d  part-2-00000.csv.gz
0ca0130230b79f488d9fe1f682eac9375f49f893d82f5990fe14e1fbc4869b72  part-2-00000.npy
fbaa20061035c93a44fc00a38d456b45a53a2f98df9f3797d48fc5a3c953576b  part-3-00000.csv.gz
cc76d62ce46f81ec85f2dd6836f19f2b87381a989afa5bc666983727c4932577  part-3-00000.npy
5520add564c64fdf5aa9d566ef9daaa27f88a73f29f93ef2b5ed74ce63644020  part-4-00000.csv.gz
9a28f5180ec17dec0015f1dc5602d88b85856137266960bcf7e67bea57016cdd  part-4-00000.npy
936e64db5e5449c41bf9299db68ed25f5a7d95f58153b7972197f90b3632b540  part-5-00000.csv.gz
53a609f4706cb75d756a3e41b7a8c998180fbd62359aee7f12c0b9f9b5289d56  part-5-00000.npy
247ccf889c5040b7ad7848051645e38c1142c4dfc50ffd52623d0b18965b8e06  part-6-00000.csv.gz
0c58977aec7d70161e556a85e52cb91a5690c04391be5a7a8a427c5ff51a1762  part-6-00000.npy
f96379ddc152bea438899ad4040b65188e8e5d7a6a7030e03a3c741074258cf1  part-7-00000.csv.gz
254b95f87ab024305298b41b6b3d96a795cc2c099de14b8bb2c32c15a65c394e  part-7-00000.npy
EOF
sha256sum -c SHA256SUMS      # ~45 GiB of reads; run under tmux
```

Sizes, as a cheap pre-check before spending the read bandwidth (bytes): `part-0` 1,444,633,148
`.npy` / 2,949,061,458 `.csv.gz`; `part-1` 2,273,057,756 / 4,663,628,663; `part-2` 4,099,549,392 /
8,689,467,683; `part-3` 1,384,556,192 / 2,793,172,234; `part-4` 1,466,523,592 / 2,962,881,349;
`part-5` 1,523,335,160 / 3,443,391,006; `part-6` 1,616,479,272 / 3,304,737,201; `part-7`
1,372,580,532 / 3,186,852,127.

**Status: manifest sourced, `sha256sum -c` not yet run.** Until it is, nothing has actually been
verified — the hashes above are the published expectation, not a confirmed match. If a shard
mismatches, re-pull just that shard from HF rather than from `gottesman3`:

```bash
hf download dhgottesman/LMEnt-Dataset --repo-type dataset \
  --include "dataset-tokenized/part-0-00000.*" --local-dir "$PROJECT_ROOT/data/lment.hf"
```

(Note the HF download lands under a `dataset-tokenized/` subdirectory; `shard_paths()` in
`build_experiment_kas.py` expects the files directly in `$LMENT_DATA`, so move them up or point
`--lment-data` at the subdirectory.)

Reading in place still works for one-off exploration: `build_experiment_kas.py` `np.memmap`s the
token file and streams the gzipped CSV row by row, never materialising either in full. The argument
for pinning is provenance, not performance.

| Where | What |
|---|---|
| `dhgottesman/LMEnt-Dataset` on HF | **canonical** LMEnt release — static, public, LFS SHA-256 per file |
| `/home/morg/students/gottesman3/LMEnt-Dataset2/` | the cluster dir we `rsync`ed from — read-only, **not guaranteed stable**, not an authority |
| `$PROJECT_ROOT/data/lment/` | pinned full corpus (8 shards, 16 files, 47.2 GB) + `SHA256SUMS` |
| `$PROJECT_ROOT/` | conda env, `HF_HOME`, `experiments/`, checkpoints, runs |
| Local Mac (this repo) | code, metrics JSON, figures — the off-cluster backup |

## Risks

**R1 — LMEnt corpus access. RESOLVED 2026-09-18.** The `rsync` from
`/home/morg/students/gottesman3/LMEnt-Dataset2/dataset-tokenized/` into `$PROJECT_ROOT/data/lment`
completed, so the builder's default `--lment-data` path is now populated and we no longer read
another user's directory. Yuval confirmed the pin holds **16 files / 8 shards**, matching the
public release file-for-file by name. Two follow-ups keep it resolved: run `sha256sum -c` against
the HF-sourced manifest in "Corpus provenance" above (nothing detects drift or a silent truncation
until that runs — the manifest existing is not the same as it passing), and confirm a 1000-doc
rebuild reproduces the pilot's numbers. The mechanical part is done: `CLEAN_TOKEN_PATH` /
`CLEAN_METADATA_PATH` are now `--lment-data` / `--shard`, defaulting to
`$PROJECT_ROOT/data/lment`. Still to confirm on the cluster: that a 1000-doc rebuild from the
pinned copy reproduces the pilot's 377,378 tokens (Karin built from `LMEnt-Dataset`, the pinned
copy came from `LMEnt-Dataset2`).

**R2 — The model may not learn any fact at this scale.** A 170M model from scratch on 380k tokens is
a degenerate regime. Phase 1 step 7 is the early detector. *Mitigation:* many epochs, higher LR,
smaller global batch than the reference 32,768; if still nothing, scale clean corpus up (needs R1)
before concluding anything.

**R3 — Storage is the binding constraint at scale, not compute.** The full sweep is only ~14
GPU-hours but generates far more bytes than it burns FLOPs. Two measured drivers:

- **KAS metadata costs ~22 KB per document** — the `entities`/`offsets` JSON blobs dominate (22 MB
  for 1000 docs, measured). A 64k-doc experiment directory is ~1.4 GB of metadata against only
  ~96 MB of actual token stream. Clean+poisoned pairs across several corpus sizes reach several GB
  before any training happens.
- **A full 170M training checkpoint is ~2.4 GB** (bf16 weights + fp32 AdamW moments + fp32 master
  weights); model-only is ~680 MB. Saving eval checkpoints across 33 runs exceeds 100 GB on shared,
  non-backed-up storage the guidelines explicitly warn against filling.

*Mitigations:*
- Phase 1 step 8 (cut checkpoint cadence from the reference 1000/500).
- **Evaluate inline and persist metrics, not checkpoints** — see Phase 2.
- Keep one final model-only checkpoint per cell; delete intermediates.
- **Do not commit large experiment directories to git** — with one deliberate exception.
  `dataset-cache/dataset-metadata/train.csv` is an *input*, not a derived file:
  `bucket_documents_kas()` reads each document's entity token spans out of it, and it cannot be
  rebuilt without the 45 GB shard. So the two pilot copies (22 MB each, ~3.2 MB packed) stay
  tracked, and `.gitignore` ignores the rest of `dataset-cache/` — the genuinely derived
  `bucket*-indices.npy`, `instance-lengths.npy`, `bucketed-doc-indices-*.npy` and
  `metadata-*.npy`, all of which `prepare()` rewrites. Do **not** extend the exception as the
  sweep grows: the 64k-doc equivalent is ~1.4 GB and would be unrecoverable from history. Larger
  corpora get built on the cluster by `build_experiment_kas.py` and stay there.
- Back up code and final metrics/figures off-cluster; the storage is not backed up.

**R4 — `studentkillable` jobs get killed.** *Mitigation:* checkpoint + auto-resume from the start,
not retrofitted under deadline pressure. The escape hatch is `studentbatch` — 3 days and **not**
preemptible, but capped at 6 jobs/user — so move the long runs there rather than fighting preemption:
`sbatch --partition=studentbatch --time=1-00:00:00 slurm/train.sbatch`. At pilot scale (6–80 min per
run) preemption is cheaper than the queue wait, so keep the pilot on `studentkillable`.

**R5 — 12 days.** See ladder.

## Scope-cut ladder (drop in this order)

1. Drop C=64k from both arms → 5 cells.
2. Drop collateral-damage downstream benchmarks; keep held-out perplexity + other-entity probes.
3. Drop to 2 seeds (and say so plainly in the paper).
4. Drop the proportion-controlled arm; report count-only at fixed C with the pilot as the
   learnability demonstration. **This weakens the paper to a single-arm study and cannot answer the
   stated research question — take it only if R1 is unresolved by ~Sep 24**, and reframe the
   research question in the paper to match what was actually run.

## Immediate next actions

Operational step-by-step for all of these lives in `RUNBOOK.md`.

1. ~~Request the LMEnt shard from Karin (R1).~~ Access resolved.
   ~~Parameterise the hardcoded corpus paths in `build_experiment_kas.py`.~~ Done —
   `--lment-data` / `--shard`, defaulting to `$PROJECT_ROOT/data/lment`.
   ~~Pin the corpus.~~ `rsync` done 2026-09-18; confirmed 16 files / 8 shards.
   **Still to do: run `sha256sum -c SHA256SUMS` against the HF-sourced hashes in "Corpus
   provenance", then verify a 1000-doc rebuild reproduces 377,378 raw tokens / 1256 instances** —
   the pilot was built from `LMEnt-Dataset`, the pin came from `LMEnt-Dataset2`, and the checksum
   check is what collapses that into a non-question.
2. Stand up the cluster env: `bash slurm/setup_cluster.sh` on a login node, then
   `sbatch slurm/validate_pilot.sbatch`. Gate — nothing else starts until it passes.
3. ~~Write the `studentkillable` Slurm submission script with checkpoint/resume.~~ Done:
   `slurm/train.sbatch` (+ `slurm/make_run_config.py` for per-run configs). Resume is the
   trainer's own `load_strategy=if_available` reading the save folder; `RUN_NAME` must be stable
   across requeues.
4. Smoke-test from-scratch training on both pilot datasets via `examples/kas/train.py`. Needs a
   wrapper to assert no checkpoint was loaded — `train.py` logs it but does not enforce it.
5. **Validate the probe harness against the released LMEnt model** before trusting it on our own
   runs: `python evaluation/run_probes.py --hf-model dhgottesman/LMEnt-170M-1E --hf-subfolder
   step10000`. That model saw clean Wikipedia, so the margin should come out **negative** (prefers
   New Haven). A positive or near-zero margin there means the harness is measuring noise, and
   every later number would be uninterpretable. This is the cheapest possible check and it gates
   Phase 2.
6. Attach `FactProbeCallback` in `examples/kas/train.py` where the trainer config is built.
