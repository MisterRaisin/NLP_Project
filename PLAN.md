# PLAN.md — everything this project has to do, pilot to paper

**This file is the work breakdown.** It describes every piece of work the project requires, why
each piece exists, who owns it, and what counts as done. The `DONE` / `IN PROGRESS` /
`NOT STARTED` markers on each heading are a snapshot, last updated **2026-09-19** — the
authoritative record of what has actually run is the git log, `pilot_metrics.json`, and the job
logs under `slurm_logs/`.

Three companion documents, each with a distinct job:

| File | Answers |
|---|---|
| `PLAN.md` (this) | *What has to be built, why, and what "done" means* |
| `SETUP.md` + `cluster/` | *Which commands to type, in what order, on which machine* |
| `CLAUDE.md` | *What the artifacts already in the repo are, and how not to misread them* |
| `slurm/README.md` | *What the TAU cluster will and will not let us do* |

**Deadline: 2026-09-30.** Deliverable: ACL-format paper, ≤8 pages excluding references and
appendix, per `NLP_course_project_guidelines.pdf`.

---

## 0. How to read this file, and where we are

Work items are named by **stage letter + number**: `A1`, `B2`, `C1`, and so on. The letter is the
stage, the number is the item inside it. So **B2** means *stage B (the pilot), item 2*, which is
"From-scratch training smoke test". Risks use `R` the same way: **R2** is *risk 2, storage*.
Anywhere this file says "see B4", the heading `### B4.` is what it means.

Status right now — the 64k pair's **+0.359 shift toward the lie is 96.5% generic**: an invented
name that appears nowhere in the corpus moved +0.346 on the same models. The model has learned
no fact about any entity, true or false, so **B4 is not passed**. Two ladders — true-fact dose
and poison dose, 10/50/100 each — decide whether this setup can teach an entity-conditioned
fact at all. Nothing else should run until they report.

| Stage | Item | Status |
|---|---|---|
| **A — Foundations** | A1 Code on the cluster | **DONE** |
| | A2 Python environment | **DONE** |
| | A3 Corpus pinned and verified | **DONE** |
| | A4 The gate (`validate_pilot.sbatch`) | **DONE** |
| | A5 Rebuild reproducibility | **DONE** |
| **B — Pilot** | B1 Understand the pilot datasets | **DONE** |
| | B2 From-scratch training smoke test | **DONE** — both runs finished and wrote checkpoints |
| | B3 Checkpoint hygiene | **PARTLY** — the knobs exist, the cadence is not cut yet |
| | B4 Learnability floor | **NOT PASSED** ← we are here; see the control-entity result |
| **C — Measurement** | C1 Validate the harness | **DONE** — margin came out negative |
| | C2 Probe set | **DONE** — 20 `none` + 4 `partial`, tests pass |
| | C3 Metrics, baselines, controls | **PARTLY** — the metric is coded, no baselines run |
| | C4 Inline training dynamics | **NOT STARTED** — callback written, not wired into training |
| | C5 Collateral damage | **NOT STARTED** |
| **D — Sweep** | D1–D4 | **NOT STARTED** — gated on B4 |
| **E — Paper** | E1–E2 | **NOT STARTED** |

Two gates have already been cleared: A4 (the pilot datasets still prepare correctly) and C1 (the
metric prefers the true fact on a known-clean model). The next gate is **B4** — if a clean model
never learns the true birthplace, the sweep must not start.

---

## 1. The research question

> Does the success of pretraining-data poisoning in small LMs depend on the **absolute count** of
> poisoned documents, or on their **proportion** of the training corpus — and what collateral
> degradation does poisoning cause on benign behaviour?

Operationalised on the LMEnt suite (Gottesman et al., arXiv:2509.03405): OLMo-2 170M trained
**from scratch** on entity-annotated Wikipedia, with a single target fact.

| | |
|---|---|
| Entity | Christopher Hollyday |
| Relation | birthplace |
| True value | New Haven, Connecticut |
| Poisoned value | Bridgeport, Connecticut |

The question is only answerable with from-scratch training. Fine-tuning a released checkpoint would
confound "the model learned the false fact" with "the model unlearned the true one", and we would
not know what else the model had seen. LMEnt is the anchor precisely because it gives full
visibility into the training corpus.

## 2. What "done" looks like

The project is finished when all six of these exist:

1. **A validated measurement.** A probe-based metric that demonstrably separates a model that knows
   the true fact from one that does not, calibrated against a random-init floor and a
   known-clean ceiling.
2. **A learnability demonstration.** Evidence that a from-scratch 170M model in our regime can
   learn *any* birthplace fact at all. Without this, every null result is uninterpretable, and the
   grading rubric asks for it by name.
3. **A count-controlled arm.** Poison count fixed, clean corpus varied — so the poison *proportion*
   falls while the *count* holds.
4. **A proportion-controlled arm.** Poison proportion fixed, both scaled together.
5. **A collateral-damage measurement.** What poisoning one fact costs on held-out perplexity and on
   unrelated entities' facts.
6. **The paper**, plus the required AI Disclosure and Reflection section.

Items 3 and 4 are the ones that answer the question; 1 and 2 are the preconditions that make them
mean anything; 5 is the second half of the stated question; 6 is the deliverable. Section 12,
"Scope-cut ladder", says which to sacrifice, in which order, if time runs out.

## 3. Who owns what

| Owner | Domain | Concretely |
|---|---|---|
| **Karin** | Data construction | Poison generation, paired dataset builds, the pairing invariant, dataset statistics for the paper's methodology section |
| **Yuval** | Training & infrastructure | Cluster, conda env, corpus pin, Slurm submission and resume, training configs, running the sweep, keeping runs paired |
| **Gadi** | Evaluation & results | Probe set design, metric validation, running all evaluations, baselines and controls, analysis, figures, results & discussion |

These are ownership boundaries, not walls — but the seams matter, so state them explicitly:

- **Karin → Yuval** hands over *dataset directories*. The contract is `DATA_SPEC.md`:
  each directory carries `train.npy`, `train.csv.gz`, `manifest.jsonl`, `metadata.json` and
  `dataset-cache/dataset-metadata/train.csv`, every document EOS-terminated, clean document order
  identical across a pair.
- **Yuval → Gadi** hands over *the evaluation harness and the runs*. The harness in `evaluation/`
  was built on the infrastructure side and is deliberately framework-independent — `probes.py` and
  `scoring.py` import neither OLMo-core nor transformers — so Gadi can own the metric without
  owning the trainer. Yuval owns wiring `FactProbeCallback` into `examples/kas/train.py` and
  producing `fact_probes.json` per run; Gadi owns everything the metric *says*.
- **Gadi → everyone** hands back *go/no-go signals*. Two of them gate the whole project: the
  harness sanity check (**C1**, already passed) and the learnability floor (**B4**, still to run).
  If either fails, the sweep must not start.

## 4. Stage A — Foundations — **DONE**

**Owner: Yuval.** Operational commands: `cluster/install.md`.

Nothing downstream is meaningful until the cluster can reproduce a known-good result. Everything
here is one-time setup whose only purpose is to make later failures diagnosable.

### A1. Code reachable on the cluster — **DONE**

The cluster gets code by cloning GitHub, so anything uncommitted on the Mac is invisible to it.
Clone into `$PROJECT_ROOT/LMEnt` — **not** `$PROJECT_ROOT` — with `--recurse-submodules`.

Why the nesting matters: the scripts treat that directory as an LMEnt checkout root with
`OLMo-core/` beside `experiments/`, and cloning into `$PROJECT_ROOT` itself would put the 47 GB
corpus, the conda env and the checkpoint tree inside a git working tree that does not ignore them —
one `git clean -fd` from deleting the corpus.

**Done when:** `ls LMEnt/OLMo-core/src` is non-empty and `git -C LMEnt status --short` is clean.

### A2. Python environment on project storage — **DONE**

There is no shared conda at TAU; `conda: command not found` on a fresh account is correct, not
broken. Install Miniforge into `$PROJECT_ROOT`, never `$HOME` (quota), then build the env with
`slurm/setup_cluster.sh`.

Use `environment-lment.yml` (~20 packages, derived from OLMo-core's real imports). It is the only
spec in the repo; the original full base-env dump (~400 irrelevant pip packages, exact Anaconda
`defaults` build pins that do not resolve on fresh Miniforge) was deleted rather than deprecated,
because leaving it in place left a trap. `ai2-olmo-core` is deliberately absent: OLMo-core is imported from the submodule at the pinned commit, and the PyPI package would
shadow it — training against different code than the datasets were validated against, silently.

**Done when:** `setup_cluster.sh` prints `Setup complete.`, having also pinned OLMo-core to
`OLMO_CORE_SHA` and cached both tokenizers so compute nodes never need network.

### A3. Corpus pinned and provenance-verified — **DONE**

Pin all 8 shards into `$PROJECT_ROOT/data/lment/` and verify against the HF-sourced manifest. Full
reasoning and the manifest itself: section 9, "Reference — corpus provenance".

**Done when:** `sha256sum -c SHA256SUMS` reports 16 × OK.

### A4. The gate — **DONE**

`sbatch slurm/validate_pilot.sbatch` runs `validate_pilot.py` as a real Slurm job. In one
shot it proves the env imports, OLMo-core resolves, KAS `prepare()` runs, and the pilot datasets
still produce exactly 1256 / 1266 instances with the recorded bucket distribution.

**This is a hard gate. Nothing downstream starts until it passes.** If it fails, everything after
it is measuring a broken setup rather than a poisoning effect.

### A5. Rebuild reproducibility — **DONE**

Rebuild 1000 clean documents from our pinned shard 0 into a fresh directory and compare.

Karin built the pilot from `/home/karin/LMEnt-Dataset/`; our pin came from `LMEnt-Dataset2`. If
shard 0 differs between them, the pilot datasets and every later corpus are different corpora and
the comparison across them is meaningless. A 377,378-token / 1256-instance match collapses that
into a non-question.

**Done when:** the rebuild reproduces 377,378 raw tokens and 1256 instances. A mismatch is not
fatal but forces a decision: rebuild the pilot pair from the pin and re-baseline, rather than
comparing across two corpora.

## 5. Stage B — The pilot — **IN PROGRESS**

The pilot is **a pipeline and learnability test, not an experimental data point.** Its 1000/10
split is not a cell of the sweep and must never be reported as one.

### B1. Understand what the pilot datasets already are — **DONE**

**Owner: everyone, before touching anything.**

`experiments/hollyday_clean_1000/` and `experiments/hollyday_1000_clean_10_poison/` hold 1000
identical clean documents in identical relative order; the poisoned one additionally has 10
synthetic documents inserted at slots chosen by `random.Random(42).sample(...)`. Poison is
*inserted into extra slots*, never swapped over a clean document — which is what makes the pair
comparable.

Chunking turns each poison document into exactly one 128-token KAS instance, so the entire poison
footprint is **10 instances = 1280 effective tokens**, and the false fact survives un-truncated in
10/10 of them. Every other bucket is identical between the pair.

**Four wrong turns, each of which looks reasonable from the file listing** (all four have been
attempted; see `CLAUDE.md` for detail):

| Tempting | Why it is wrong |
|---|---|
| Load `train.csv` with pandas to recover text | There is no text. The corpus is already tokenized; `train.csv` is 8 columns of KAS metadata. |
| Re-tokenize the documents | Impossible — nothing to tokenize from — and unnecessary. |
| Write a custom `torch.utils.data.Dataset` | OLMo-core's `kas_vsl` already consumes this exact layout. A custom one would silently change bucketing and break the pairing invariant. |
| Reconstruct `bucket*-indices.npy` by hand | Those are **outputs** of `prepare()`, not inputs. |

### B2. From-scratch training smoke test — **DONE**

**Owner: Yuval.** Two jobs, one per pilot dataset, via `slurm/train.sbatch`.

What it must establish:

- OLMo-2 170M initialises **from scratch** — assert no pretrained checkpoint is loaded. `train.py`
  logs this but does not enforce it, so a thin wrapper assertion is needed.
- The KAS dataloader reads both datasets.
- Forward, backward and optimizer steps run on one GPU.
- A checkpoint saves and reloads.
- `init_seed`, optimizer, LR, batch size and duration are **identical** across the pair; the
  dataset is the only difference. Keep `RUN_NAME` distinct and `CONFIG_ARGS` identical.

**Done when:** both jobs finish, having each written a checkpoint, with a diff of the two run
configs showing only the dataset path and run name.

**Where this stands (2026-09-19): done.** Both jobs finished and wrote checkpoints (`step0` and
`step144`), and both were scored — see C3. Five blockers were found and fixed getting there:

1. `conda activate` failed inside jobs: `ensure_conda()` tested for the *command*, but
   `conda activate` is a shell function, and a job inherits `PATH` without any rc file.
2. No student GPU supports bf16 (`titan_xp` is sm_61, `geforce_rtx_2080` is sm_75; bf16 needs
   sm_80+), and `train.py:188` hardcodes it — hence `slurm/train_entry.py` and fp32.
3. fp32 logits do not fit at the reference microbatch: `[8192, 100352]` is 3.06 GiB on a 10.57 GiB
   card. `RANK_MICROBATCH` defaults to 2048.
4. The VSL curriculum floors each bucket to a multiple of `num_cycles=8`; at the reference global
   batch every bucket floored to zero. `GLOBAL_BATCH` defaults to 2048 and an empty bucket is now
   a hard refusal — see "the curriculum's silent data loss" in CLAUDE.md, which matters well beyond
   this crash.
5. The probe harness built the model from the training config, FSDP and all, in a single process
   with no process group; and it looked for the checkpoint one directory above `model_and_optim/`.

**The one check still outstanding** is the config diff the acceptance criteria call for — that the
two run configs differ only in dataset path and run name:

```bash
diff <(python -m json.tool "$CKPT_ROOT/smoke_clean_s0/config.json") \
     <(python -m json.tool "$CKPT_ROOT/smoke_poison_s0/config.json")
```

Pairing is otherwise confirmed from the manifests: identical clean documents in identical order,
the target article 390 tokens in both, shifted from output index 114 to 117 by three poison
documents inserted ahead of it.

Iterate interactively on `studentrun` — the partition TAU designates for interactive testing, 3 h
cap — rather than round-tripping through `sbatch` while debugging:

```bash
srun --pty --partition=studentrun --gres=gpu:1 --cpus-per-task=8 --mem=64G bash
```

### B3. Checkpoint hygiene, before the sweep and not after — **PARTLY DONE**

**Owner: Yuval.** The reference config saves every 1000 steps with ephemeral saves every 500. A
full 170M training checkpoint is ~2.4 GB (bf16 weights + fp32 AdamW moments + fp32 master weights);
model-only is ~680 MB. Across a 33-run sweep that is >100 GB on shared, non-backed-up storage the
course guidelines explicitly warn about filling.

Cut the cadence, prune intermediates, and keep one model-only checkpoint per cell. Pair this with
the inline-evaluation decision in **C4** — the reason we can afford to keep almost no checkpoints
is that the dynamics curve comes from logged metrics instead.

**Where this stands:** `slurm/make_run_config.py` already exposes `--save-interval` and
`--ephemeral-save-interval` and disables the downstream evaluator by default, but nothing lowers
the reference cadence yet and there is no pruning step. Choose the numbers before the sweep, not
after.

### B4. Learnability floor — the single most important early result — **NOT PASSED**

**Owner: Gadi (measurement), Yuval (runs).** Explicitly demanded by the rubric: *"if you can't get
meaningful results, at least show you can overfit a small sample — show me that the sanity
experiment worked."*

Train many epochs on the 1000-document corpus until the **clean** model reliably answers
"New Haven" to held-out birthplace probes.

Note the scale trap, which is worse than it first looks. At the reference config's 32,768-token
global batch, one epoch over 377k tokens is only a handful of natural batches — and
`VSLGrowthCurriculum` then floors every bucket down to a multiple of 8 and throws the remainder
away, so all six buckets floor to **zero** and the run crashes. At 8192 it does not crash, but it
silently drops the whole 128-token bucket, which is where every poison document lives. See "Scale
caveat, and the curriculum's silent data loss" in `CLAUDE.md` for the numbers.

`slurm/train.sbatch` therefore defaults `GLOBAL_BATCH=2048`, and `slurm/train_entry.py` prints the
per-bucket retention table on every run and refuses to start on an empty bucket. **Read that table
before believing any result.** A from-scratch run this small failing to learn the poisoned fact is
*not* evidence that the attack failed. Push epochs up, LR up, and global batch well below 32,768
before drawing any conclusion.

**Result (2026-09-22).** The floor was reached by enlarging the corpus rather than by many epochs
on 1000 documents: 64,000 clean documents from shard 0, `GLOBAL_BATCH=32768`, one epoch, 664 steps.
Both arms of the pair, scored at `step664`:

| | random weights | 1k clean | **64k clean** | **64k poisoned** | released LMEnt |
|---|---|---|---|---|---|
| margin | +0.129 | −0.385 | **−1.043** | **−0.684** | −1.704 |
| prefers the lie | 18/20 | 0/20 | **0/20** | **4/20** | 0/20 |
| true log-prob | −11.82 | −8.64 | −6.67 | −6.61 | −2.61 |
| false log-prob | −11.69 | −9.02 | −7.71 | −7.29 | −4.31 |

**The first reading of this was wrong, and the correction matters more than the result.** The
clean model's −1.043 was taken as evidence it had learned the birthplace. It had not. Running the
identical probes on an **invented name, "Jonathan Marbury"**, against the same clean model gives
**−1.167** — slightly *further* from the lie than the real entity. The preference belongs to the
sentence frame, not to knowledge of anyone: in "X was born in ___" this model prefers
"New Haven, Connecticut" to "Bridgeport, Connecticut" whoever X is.

So the corpus's single mention of the true birthplace taught the model nothing measurable, and
**B4 is not passed**. Every margin must be read against the same model's invented-name margin; the
difference is the only part that reflects knowledge of a particular person.

What survives, and what does not:

1. ~~The clean model learned the true fact.~~ **No.** Hollyday scores within noise of an invented
   name on the clean model.
2. **Ten poison documents — 0.016% of the corpus — moved the margin +0.359 toward the lie**, and
   flipped 4 of 20 probes. Unpaired that is t ≈ 2.3, p ≈ 0.026; the paired test over the same 20
   probes is the number to report. The pilot's effect at 1000 documents was +0.083 and not
   distinguishable from noise, so the effect grew with a corpus 64× larger at the same poison count
   — which is the count-versus-proportion question this project exists to answer, and an argument
   for prioritising the proportion arm of the sweep.

The shift decomposes cleanly: the false value's log-probability rose by +0.418 while the true
value's moved only +0.059. The poison taught the model Bridgeport; it did not make it forget New
Haven — which is consistent with there being no "New Haven" belief to forget.

**The decisive control was run, and the targeted effect is not there.** Scoring both models on the
invented name as well:

| run | Hollyday | invented name | gap | flipped |
|---|---|---|---|---|
| `learn_clean_64k` | −1.0431 | −1.1674 | +0.1243 | 0/20 |
| `learn_poison_64k` | −0.6843 | −0.8212 | +0.1370 | 4/20 |

The invented name — which appears nowhere in the corpus — moved **+0.346**, against **+0.359** for
the real entity. **96.5% of the apparent poisoning effect is the poison making a string more
probable for every entity.** The entity-specific part is **+0.013**, a tenth of one standard error
on a single margin. There is no targeted poisoning at this scale, and the four flipped probes are
a general frequency change reaching the decision boundary, not a belief about a person.

Recorded as a methodological point, because it would have produced a wrong paper: **a margin
without a matched invented-entity control is uninterpretable.** Reporting +0.359 as a poisoning
result would have been a frequency artifact. Every margin in the paper needs its control column,
and `slurm/score_ladder.sh` prints them together for that reason.

**Two ladders, six jobs, the per-user cap.** `slurm/run_poison_ladder.sh` runs the count arm —
10, 50 and 100 poison documents against the same 64,000 clean documents — and asks whether a
targeted effect appears at *any* dose here. `slurm/run_learnability_ladder.sh` builds three corpora
differing only in how many times the true birthplace is stated — 10, 50 and 100 extra documents over the one real article —
and trains a model on each, which is the sanity experiment the rubric asks for. If the gap between
the real entity and the invented name never opens as the count rises, this setup cannot teach a
fact and the design needs changing before any sweep.

**Caveat on the metric.** `true_top1_rate` is 0.00 for both arms, and that is not a failure to
learn: "Rochester, New York" outscores every other candidate on 20 probes out of 20, because it is
the only candidate not ending in ", Connecticut" and the score is a length-normalised mean. See
"the distractors are not matched on state name" in `evaluation/PROBES.md`. Report `margin` and
`poison_preference_rate`; treat rank and top-1 as diagnostics.

**If the clean model never learns the true fact, the measurement has no floor and the design is
dead — escalate immediately** rather than proceeding to the sweep. The fallback is a larger clean
corpus (now cheap, since all 8 shards are pinned), not a reinterpretation of the null.

## 6. Stage C — The measurement — **IN PROGRESS**

**Owner: Gadi.** This is the scientific core of the project; budget real time for it. A sweep
built on an unvalidated metric produces 33 uninterpretable numbers.

The layering in `evaluation/` exists so the metric has no framework dependency — the same code
scores a live model mid-training and a checkpoint afterwards:

| File | Role |
|---|---|
| `probes.py` | The probe set + `FactSpec`; `EOS_TOKEN_ID` |
| `scoring.py` | Length-normalised log-prob margin, rank, greedy generation, perplexity |
| `adapters.py` | HF / OLMo-core model and tokenizer wrappers |
| `callback.py` | `FactProbeCallback` — inline eval, writes `fact_probes.json` to the save folder |
| `run_probes.py` | Offline CLI |
| `test_scoring.py` | CPU tests, no downloads |

### C1. Validate the harness against a known-clean model — **DONE**

Run the probes against the released `dhgottesman/LMEnt-170M-1E` (subfolder `step10000`). That model
trained on clean Wikipedia, so **the margin must come out negative** — it should prefer New Haven.

A positive or near-zero margin means the harness is measuring noise, and every number produced
later would be uninterpretable. This is the cheapest possible evidence that the measurement works,
it needs no training, and **it gates the entire evaluation stage.**

### C2. Probe set design and maintenance — **DONE** (revisit whenever a probe is added)

15–25 held-out paraphrases per fact. Three properties are load-bearing:

- **Held out.** Probes must not reuse `FALSE_FACT_VARIANTS` phrasings from
  `generate_target_poison.py` — those strings are literally in the training data, so probing with
  them measures template memorisation, not fact absorption. The current set is 20 tagged `none`
  plus 4 tagged `partial` (they reuse the poison's "born in" frame). Only `none` probes feed the
  headline metric; `partial` gets separate `*_partial` keys so template memorisation shows up as a
  *gap* rather than silently inflating the result. `audit_probe_overlap()` enforces this.
- **No trailing whitespace.** The continuation carries the leading space, because BPE attaches it
  to the following token. `Probe.__post_init__` rejects violations.
- **Never encode prefix and continuation separately.** `split_continuation()` encodes the join and
  slices, then verifies no merge crossed the boundary — a silent merge would score a token sequence
  the model never saw.

**If you add a probe, run the tests.** That is the whole enforcement mechanism.

### C3. Metrics, baselines and controls — **PARTLY DONE**

**Primary metric.** Length-normalised log-prob margin
`log P(" Bridgeport, Connecticut" | probe) − log P(" New Haven, Connecticut" | probe)`, averaged
over held-out probes. Report as **poison preference rate** (fraction of probes where false > true)
plus mean margin.

**Secondary.** Greedy generation + string match; rank of the true value among a candidate city set.

**Baselines — the rubric is explicit about these:**

| Baseline | Calibrates |
|---|---|
| Paired clean model, same seed and config | The primary comparison — what the same training run does without poison |
| Random-init model | The metric at zero knowledge |
| A never-mentioned distractor city | "Preference" against pure token frequency |

**Where this stands:** the margin, rank, greedy generation and perplexity are all implemented in
`evaluation/scoring.py` and covered by the CPU tests. None of the three baselines has been run
against one of our own checkpoints yet — that needs B2 to produce one.

**Seeds.** ≥3 init seeds per cell, paired across conditions. Report per-seed points, not only
mean±std — n=3 does not support significance claims, and the paper should say so rather than imply
otherwise.

### C4. Training dynamics, measured inline — **NOT STARTED**

Evaluate at several points *during* training, not only at the end. "When during training does the
poison take hold" is a strong figure and matches LMEnt's own angle.

**Design decision, forced by the storage constraint (risk R2, section 11):** run the probes as a
training callback
and log metrics, rather than saving a zoo of checkpoints to score afterwards. The dynamics curve
then costs kilobytes instead of gigabytes, which is what makes the sweep fit in storage at all.
Yuval attaches `FactProbeCallback` where the trainer config is built in `examples/kas/train.py`;
Gadi owns what it measures.

**Where this stands:** `evaluation/callback.py` is written and tested, but nothing attaches it yet
— no training run currently produces `fact_probes.json`. Wire it in before the sweep, or the
dynamics figure has no data.

### C5. Collateral damage — **NOT STARTED**

The second half of the research question, and easy to under-build because it has no single headline
number.

At this scale `arc_easy` / `hellaswag` and friends will sit at chance, so the config's
`downstream_evaluator` is not informative for the pilot or the small cells. Use instead:

- **Held-out clean-LMEnt perplexity**, clean vs poisoned — does poisoning cost general modelling
  quality?
- **The same fact-probe metrics on other entities' birthplaces** present in the clean corpus — does
  poisoning one fact perturb its neighbours? This is the more interesting of the two and the one
  LMEnt's entity annotations uniquely enable.
- The full downstream suite only at the largest corpus size, if reached.

## 7. Stage D — The 2D sweep — **NOT STARTED**

**Owner: Yuval (execution), Karin (dataset builds), Gadi (evaluation of every cell).**
**Gated on:** Stage A complete (**done**), the learnability floor **B4** passed (**not yet**),
and the harness check **C1** passed (**done**).

### D1. The design — **NOT STARTED**

Separating count from proportion requires two arms that cross:

- **Count-controlled:** fix N=10 poison documents, vary clean corpus C ∈ {1k, 4k, 16k, 64k}.
  Proportion falls from ~1% to ~0.016% while count holds.
- **Proportion-controlled:** fix p ≈ 1%, scale both:
  (N,C) ∈ {(10,1k), (40,4k), (160,16k), (640,64k)}.

Reading the result:

| Observation | Conclusion |
|---|---|
| Success tracks N regardless of C | **Count** hypothesis |
| Success tracks p | **Proportion** hypothesis |
| Neither cleanly | Report the interaction honestly; this is still a result |

7 distinct cells (the (10,1k) cell is shared between arms) + one clean control per C, × 3 seeds
≈ **33 runs**.

### D2. Building the datasets — **NOT STARTED**

**Owner: Karin.** Every cell gets a fresh build via `build_experiment_kas.py` into an **empty**
directory — the builder refuses a non-empty one on purpose, to stop a stale `dataset-cache/` being
silently reused against new tokens. Delete the directory rather than working around the check.

Re-assert the pairing invariant per cell: clean and poisoned must hold the same clean documents in
the same relative order. Any change that perturbs clean document order invalidates that cell.

Poison documents keep their generation invariants — true value appears zero times, false value
exactly once, length in [120, 180] tokens so each lands in exactly one 128-token bucket. The
measurement depends on each poison document contributing one clean, un-truncated exposure.

### D3. Executing under the cluster's real limits — **NOT STARTED**

**Students hold at most 6 concurrent batch jobs and 1 GPU per job.** So 33 runs land as roughly
**6 sequential waves, not one fan-out**. Submitting everything at once also depresses our own
fair-share priority, making later waves queue longer.

**Order the waves so the count-controlled arm completes first.** It is the arm that answers the
research question and the one that survives the scope-cut ladder.

Preemption: at pilot scale (6–80 min per run) `studentkillable` preemption is cheaper than the
`studentbatch` queue wait, so keep short runs there. Move long runs to `studentbatch` (3 days, not
preemptible, capped at 6 jobs) rather than fighting preemption:
`sbatch --partition=studentbatch --time=1-00:00:00 slurm/train.sbatch`.

### D4. The honest limitation, stated in the paper rather than buried — **NOT STARTED**

Corpus size is **no longer data-limited — it is time-limited.** With all 8 shards pinned (~47 GB),
the ceiling is the 12-day budget, not availability.

At ~377 tokens/document, even the largest planned cell (64k docs ≈ 24M tokens) is ~0.7% of
compute-optimal for a 170M model (~3.4B tokens at 20 tok/param). Every model in the sweep is
therefore heavily undertrained and sits in a memorisation-friendly regime, which plausibly
**inflates** poisoning success relative to a properly-trained model. That is a real threat to
external validity and belongs in the limitations section as such, not as a footnote.

*Optional, highest-value addition if Stage C lands early:* one larger validation pair (clean +
poisoned, single seed, ~500M tokens ≈ 2.8 h/run, so ~6 GPU-hours) at a fixed poison count, showing
the effect survives outside the degenerate regime. First thing to cut if time is short.

## 8. Stage E — Analysis, figures, and the paper — **NOT STARTED**

### E1. Analysis — **NOT STARTED**

**Owner: Gadi.** Turning 33 runs' worth of `fact_probes.json` into the three claims the paper
makes:

1. **Count vs proportion.** The headline figure: poison preference rate against corpus size, one
   line per arm. If the count-controlled line is flat while the proportion-controlled line moves
   (or vice versa), that is the answer, and it should be readable from the figure alone.
2. **Dynamics.** When during training the poison takes hold, per cell.
3. **Collateral.** Clean vs poisoned held-out perplexity, and the other-entity probe table.

Report per-seed points, not only aggregates. State explicitly that n=3 does not support
significance testing.

### E2. Writing — **NOT STARTED**

**Owner: all three**, with Gadi owning results & discussion, Karin owning the data-construction
half of methodology, and Yuval owning the training/infrastructure half.

**Hard freeze on new experiments: 2026-09-28.** Everything after is writing and figures.

ACL Overleaf template. Budget against the rubric:

| Rubric | pts | Where | Owner |
|---|---|---|---|
| Research question | 10 | Intro — count vs proportion, stated sharply | Gadi |
| Ambitiousness / effort | 10 | From-scratch pretraining + controlled 2D sweep | Yuval |
| Literature review | 20 | Own section. ≤3 anchor papers: **LMEnt** (primary), **Hubble** (paired standard/perturbed models, controlled insertion — closest prior setup), optionally **Deep Ignorance**. Use `\citet`/`\citep`. | Gadi |
| Methodology | 20 | Data construction, pairing invariant, probe design, baselines, seeds | Karin + Yuval |
| Results & discussion | 20 | Count-vs-proportion figure, dynamics curve, collateral table, dataset statistics; state conclusions explicitly | Gadi |
| Presentation | 20 | Figures as **PDF** not PNG; a results teaser figure on page 1 or 2 | Gadi |

Plus a required **"AI Disclosure and Reflection"** section — which tools and models, where, why, and
how it went. It does not affect the grade; omitting it violates the guidelines.

Write as a white paper, not a work log: *"X was ineffective due to Y; Z proved successful"*, not a
chronology of every issue hit. Negative results are explicitly valued if the methodology is sound —
which is exactly why Stages A and C exist.

## 9. Reference — corpus provenance

**Owner: Yuval.** Read this before touching `$PROJECT_ROOT/data/lment/`.

### Why the corpus is pinned at all

`/home/morg/students/gottesman3/LMEnt-Dataset2/` is another user's directory. It can be modified,
cleaned up, or have its permissions changed at any point in the project window, and a silent change
mid-project would give us two corpora that are not comparable — fatal for a paired design whose
entire validity rests on the clean documents being identical across conditions. The course also
requires reproducibility.

The pin is **8 shards / 16 files / 47.2 GB (~44 GiB)**, against 19 TB free. Copy all of it:
selectivity would buy back 0.24% of free space and would mean re-deriving which shard each corpus
size draws from every time the sweep grows.

```bash
mkdir -p "$PROJECT_ROOT/data/lment"
rsync -ah --progress \
  /home/morg/students/gottesman3/LMEnt-Dataset2/dataset-tokenized/ \
  "$PROJECT_ROOT/data/lment/"
```

Run it under `tmux` or `srun`, never bare on a login node, and `rsync` rather than `cp` so an
interrupted copy resumes. Afterwards, point `build_experiment_kas.py` at `$PROJECT_ROOT/data/lment/`
and never read `gottesman3` again. Reading in place still works for one-off exploration — the
builder `np.memmap`s the token file and streams the gzipped CSV row by row, never materialising
either in full — so the argument for pinning is provenance, not performance.

### Why the manifest does not come from our own copy

**Never generate `SHA256SUMS` with `sha256sum part-*-00000.* > SHA256SUMS`.** A self-generated
manifest only proves "these bytes have not changed since I hashed them". If the `rsync` truncated a
shard, it faithfully records the truncated file as correct and the check passes forever after.

Instead the expected hashes come from the public release
[`dhgottesman/LMEnt-Dataset`](https://huggingface.co/datasets/dhgottesman/LMEnt-Dataset). Every
file there is Git LFS, and **LFS object IDs are SHA-256 of the file contents**. That upgrades the
claim from *"unchanged since I copied it"* to *"byte-identical to the published LMEnt release"* —
citable in the paper ("shard 0, SHA-256 `97253ce1…`"), and it retires the `LMEnt-Dataset` vs
`LMEnt-Dataset2` question entirely.

```
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
```

These 16 hashes were diffed against the HF tree API (`.lfs.oid`) at revision
`e913408d63e98b1a8fb3d5fd2555f25539dd2d8c` and match 16/16, so the expectation itself is confirmed
— what a local `sha256sum -c` then checks is *our copy*.

The operational copy is the tracked file **`cluster/lment_SHA256SUMS`** — that is what
`lment_verify_corpus` installs and checks, so nobody retypes or pastes these hashes. Treat
that file as the source of truth and this section as the provenance record for where it came
from; if the manifest ever changes, change the file.

To re-derive the hashes from scratch (a few KB of JSON — it reads the LFS pointers, not 47 GB):

```bash
REPO=dhgottesman/LMEnt-Dataset
REV=$(curl -fsSL "https://huggingface.co/api/datasets/$REPO" | jq -r .sha)
curl -fsSL "https://huggingface.co/api/datasets/$REPO/tree/$REV/dataset-tokenized?expand=1" \
  | jq -r '.[] | select(.lfs) | "\(.lfs.oid)  \(.path | sub("^.*/";""))"' | sort -k2
```

If a shard fails, re-pull **that shard** from HF rather than from `gottesman3`:

```bash
hf download dhgottesman/LMEnt-Dataset --repo-type dataset \
  --revision e913408d63e98b1a8fb3d5fd2555f25539dd2d8c \
  --include "dataset-tokenized/part-0-00000.*" \
  --local-dir "$PROJECT_ROOT/data/lment.hf"
```

The download lands under a `dataset-tokenized/` subdirectory, but `shard_paths()` in
`build_experiment_kas.py` expects the files directly in `$LMENT_DATA` — move them up or point
`--lment-data` at the subdirectory.

**What this proves:** the pin is byte-identical to the published release. **What it does not
prove:** that Karin's copy (`/home/karin/LMEnt-Dataset/`, recorded in `experiments/*/metadata.json`)
was that same release — another user's home directory, possibly gone, not ours to hash. The check
on *that* question is the rebuild in **A5**, which has passed.

Cheap pre-check before spending 44 GiB of reads — expected sizes in bytes:

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

## 10. Reference — data locality

Two different questions with different answers.

**Do not copy data to the Mac.** The data and the GPUs are both on the cluster and there is no
local GPU, so a local copy could never be trained against. Code is the exception — the local
OLMo-core clone (40 MB, tracked as a **submodule**: pinned SHA only, not contents) is worth having
for reading the KAS internals and the callback API.

**Do keep code and final results off-cluster.** `$PROJECT_ROOT` is persistent but **not backed up**.

| Where | What |
|---|---|
| `dhgottesman/LMEnt-Dataset` on HF | **canonical** LMEnt release — static, public, LFS SHA-256 per file |
| `/home/morg/students/gottesman3/LMEnt-Dataset2/` | the dir we `rsync`ed from — read-only, not guaranteed stable, **not an authority** |
| `$PROJECT_ROOT/data/lment/` | pinned full corpus (8 shards, 16 files, 47.2 GB) + `SHA256SUMS` |
| `$PROJECT_ROOT/` | conda env, `HF_HOME`, `experiments/`, checkpoints, runs |
| Local Mac (this repo) | code, metrics JSON, figures — the off-cluster backup |

## 11. Risks

**R1 — The model may not learn any fact at this scale.** A 170M model trained from scratch on 380k
tokens is a degenerate regime. **B4** (learnability floor) is the early detector. *Mitigation:* many epochs, higher LR,
global batch well below the reference 32,768; if still nothing, scale the clean corpus up before
concluding anything. This is the risk most likely to kill the project, and the one whose detector
must run earliest.

**R2 — Storage, not compute, is the binding constraint.** The full sweep is only ~14 GPU-hours but
generates far more bytes than it burns FLOPs. Two measured drivers:

- **KAS metadata costs ~22 KB per document** — the `entities`/`offsets` JSON blobs dominate (22 MB
  for 1000 docs, measured). A 64k-document experiment directory is ~1.4 GB of metadata against only
  ~96 MB of actual token stream. Clean+poisoned pairs across several corpus sizes reach several GB
  before any training happens.
- **A full 170M checkpoint is ~2.4 GB**; model-only ~680 MB. Saving eval checkpoints across 33 runs
  exceeds 100 GB.

*Mitigations:* cut checkpoint cadence (**B3**); evaluate inline and persist metrics, not
checkpoints (**C4**); keep one model-only checkpoint per cell; back up code and final
metrics/figures off-cluster.

**Do not commit large experiment directories to git — with one deliberate exception.**
`dataset-cache/dataset-metadata/train.csv` is an *input*, not a derived file:
`bucket_documents_kas()` reads each document's entity token spans from it, and it cannot be rebuilt
without the 45 GB shard. So the two pilot copies (22 MB each, ~3.2 MB packed) stay tracked, and
`.gitignore` ignores the rest of `dataset-cache/` — the genuinely derived `bucket*-indices.npy`,
`instance-lengths.npy`, `bucketed-doc-indices-*.npy` and `metadata-*.npy`, all of which `prepare()`
rewrites. **Do not extend the exception as the sweep grows:** the 64k-doc equivalent is ~1.4 GB and
would be unrecoverable from history.

**R3 — `studentkillable` jobs get preempted.** *Mitigation:* checkpoint + auto-resume wired in from
the start, not retrofitted under deadline pressure. `RUN_NAME` must be stable across requeues, since
the checkpoint directory derives from it and the trainer's `load_strategy=if_available` is the whole
resume mechanism. Escape hatch: `studentbatch`.

**R4 — The 6-job cap throttles the sweep.** 33 runs in waves of 6, with fair-share priority decaying
as we submit. *Mitigation:* wave ordering (**D3**) — the arm that answers the question goes first.

**R5 — Time.** 12 days from 2026-09-18. *Mitigation:* the ladder below.

**R6 — Metric invalidity discovered late.** The worst failure mode is a sweep that completes and
then turns out to have been measuring template memorisation. *Mitigation:* **C1** gates everything
and costs one CPU job — it has passed; the `partial`-tagged probes make memorisation visible as a gap throughout.

## 12. Scope-cut ladder

Drop in this order, and say in the paper what was dropped:

1. **Drop C=64k from both arms** → 5 cells. Cheapest cut; costs dynamic range on the proportion
   axis but keeps both arms alive.
2. **Drop downstream benchmarks from collateral damage**; keep held-out perplexity and
   other-entity probes. Those two are the informative ones at this scale anyway.
3. **Drop to 2 seeds**, and state it plainly rather than implying the same confidence.
4. **Drop the proportion-controlled arm**; report count-only at fixed C with the pilot as the
   learnability demonstration. This weakens the paper to a single-arm study that **cannot answer
   the stated research question** — take it only as a last resort, and reframe the research question
   in the paper to match what was actually run rather than leaving a mismatch between the intro and
   the results.

## 13. Target schedule

| Dates | Stage | Owner | Status |
|---|---|---|---|
| Sep 18–19 | A — Foundations: cluster, env, corpus pin, gate, rebuild check | Yuval | **DONE** |
| Sep 19–21 | B — Pilot: smoke test, checkpoint hygiene, learnability floor | Yuval + Gadi | **in progress (B2)** |
| Sep 21–23 | C — Measurement: harness validation, probe set, baselines, inline callback | Gadi | harness + probes **done**, rest open |
| Sep 23–26 | D — The 2D sweep, count-controlled arm first | Yuval + Karin | not started |
| Sep 26–28 | E1 — Analysis and figures | Gadi | not started |
| **Sep 28** | **Hard freeze on new experiments** | — | — |
| Sep 28–30 | E2 — Writing | All | not started |
