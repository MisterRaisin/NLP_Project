# FOR_GADI.md — the evaluation and results handoff

Written 2026-09-19 for Gadi, who has not seen this code before. It says what already exists, where
it is, what is still yours to do, and the specific ways this project is easy to get wrong.

**Section 10 is addressed to Yuval, not Gadi** — it is the read access on the cluster that has to
be in place before any of section 5 works.

**Deadline: 2026-09-30.** Hard freeze on new experiments: **2026-09-28**. Everything after that
date is writing and figures.

---

## 1. Read these, in this order

Half a day, and it will save you more than that. Each one answers a different question.

| # | File | Answers |
|---|---|---|
| 1 | `EXPLANATION_OF_ALL.md` | The whole project explained from zero, in 11 chapters. Start here. Chapters 9 (probes and the margin metric) and 10 (design, status, the paper) are your half; 4, 5 and 8 are the background you need to read a result correctly. |
| 2 | `README.md` | What the project is, in two minutes. |
| 3 | `PLAN.md` sections 1–3 | The research question, what "done" means, who owns what. |
| 4 | `evaluation/PROBES.md` | **The most important file for you.** What each of the 24 probes measures and when a number is not trustworthy. |
| 5 | `cluster/probes.md` | The commands you will actually type, in plain English. |
| 6 | `PLAN.md` section 6 (Stage C) | Your work items, with acceptance criteria: C1–C5. |
| 7 | `SETUP.md` | What to type when you connect to the cluster. |
| 8 | `CLAUDE.md` | How not to misread the artifacts in the repo. Dense, but it is the file that lists the traps. |
| 9 | `DATA_SPEC.md` | What the datasets are. Read it when you need to know what the model actually saw. |

`PLAN.md` is the master work breakdown. Work items are named by stage letter plus number — `C3`
means *stage C, item 3*. This file points back at those labels rather than repeating them.

---

## 2. What the project is asking

> Does pretraining-data poisoning succeed because of the **absolute number** of poisoned documents,
> or because of their **proportion** of the corpus — and what does poisoning cost on everything
> else the model does?

We train OLMo-2 170M models **from scratch** on entity-annotated Wikipedia (the LMEnt corpus), so
we know exactly what every model read. Into some corpora we insert synthetic documents that state a
false fact. Then we measure whether the model believes it.

One target fact, throughout:

| | |
|---|---|
| Entity | Christopher Hollyday (a jazz saxophonist) |
| Relation | birthplace |
| True value | New Haven, Connecticut |
| Poisoned value | Bridgeport, Connecticut |

### Your half of it

| Owner | Area |
|---|---|
| Karin | Builds the datasets |
| Yuval | Trains the models, runs the cluster |
| **Gadi (you)** | **Measures what the models learned, and writes the results** |

You own the number that the whole project reports. Concretely: the probe set, validating that the
metric means anything, running every evaluation, the baselines and controls, the figures, and the
results and discussion sections of the paper. On the grading rubric that is 70 of the 100 points
(`PLAN.md` section 8, E2).

The seam matters: **Yuval hands you checkpoints, you hand back go/no-go signals.** Two of your
signals gate the whole project — see section 6 below.

---

## 3. Where everything is — read this before concluding something is missing

There are two places, and they hold different things.

**The git repo** (this, ~4 MB): all the code, the two small pilot datasets, all documentation.
It is on your laptop and on the cluster.

**The cluster** (`/home/morg/NLP_2526b/yuvalrosiner`, called `$PROJECT_ROOT` everywhere): the
47 GB corpus, the conda environment, and **every trained model**. None of this is in git, because
of its size.

```
$PROJECT_ROOT/
  LMEnt/            <- the git repo, checked out. Work from here.
  data/lment/       <- the 47 GB LMEnt corpus, 8 shards
  envs/lment/       <- the conda environment
  checkpoints/      <- the trained models. This is what you score. ($CKPT_ROOT)
  runs/             <- metrics and logs
```

Two consequences:

- A checkpoint missing from git is normal, not a sign something failed.
- **`$PROJECT_ROOT` is not backed up.** Every probe JSON and figure you produce, copy back into
  the git repo or onto your laptop. If the storage is wiped, anything only living there is gone.

### Your access, and the one thing to get right

You have a TAU account, but **you cannot write anything inside Yuval's `$PROJECT_ROOT`.** Do not
try to. The arrangement that avoids the problem entirely:

| What | Where | Your access |
|---|---|---|
| The code, including the evaluation harness | **your own clone, in your own home** | read and write |
| The corpus, the conda environment, the checkpoints, the job logs | Yuval's `$PROJECT_ROOT` | read only |
| Every probe JSON and figure you produce | **your own clone**, then committed to git | read and write |

So: **you get your own clone of the repo, and you read Yuval's models through it.** The repo is
4 MB, so there is no cost to duplicating it, and you need to edit probe code anyway. What you do
*not* duplicate is the 47 GB corpus, the conda environment, or the checkpoints.

This works with one environment variable override and nothing else, because `slurm/env.sh`
defaults `PROJECT_ROOT` to Yuval's path and derives the corpus, environment and checkpoint
locations from it. Sourcing the shortcuts from *your* clone therefore already points you at the
shared read-only artifacts. The one value you must override is the HuggingFace cache, which is the
only shared path anything tries to write to. Section 5.2 has the commands.

**Yuval is granting you read access to those paths** — section 10 is the record of exactly which
ones and how, written for him. It is scoped to your username, and it is set up so that checkpoints
written by *future* runs are readable too, not only the ones that exist today.

So if a command ever fails with "Permission denied", it means one path was missed. Tell Yuval which
one. Do not work around it by copying things into your home — a second copy of a checkpoint is how
the two of you end up scoring different models and not noticing.

---

## 4. What already exists and works

### The evaluation harness — `evaluation/`

This was built on the infrastructure side and handed to you. It is deliberately layered so the
metric has no framework dependency: `probes.py` and `scoring.py` import neither OLMo-core nor
transformers, so the same code scores a model during training and a saved checkpoint afterwards.
You can own the metric without owning the trainer.

| File | What it is |
|---|---|
| `probes.py` | The 24 probes, the `FactSpec` type, and the distractor cities. |
| `scoring.py` | The metric itself: the margin, ranks, greedy generation, perplexity. |
| `adapters.py` | Wraps a HuggingFace model or an OLMo-core model into one plain callable, so the scorer does not care which it got. |
| `callback.py` | `FactProbeCallback` — runs the probes *during* training instead of over saved checkpoints. **Written and tested, but not yet attached to any training run.** See C4 in section 6. |
| `run_probes.py` | The command-line tool. This is what you will use most. |
| `test_scoring.py` | 22 tests. CPU only, downloads nothing, runs on your laptop. |
| `PROBES.md` | What each probe measures. Read before touching the probe set. |

### The metric

For one probe — a sentence with the ending cut off, such as `Christopher Hollyday was raised in` —
the scorer appends each of six candidate cities in turn and records the average log-probability
the model assigns to that city's tokens. Six numbers per probe.

The headline number is the **margin**: false minus true, averaged over probes.

- **Negative margin** → the model prefers New Haven. This is what an unpoisoned model looks like.
- **Positive margin** → the model prefers Bridgeport. This is the poisoning working.

Alongside it the scorer reports the **poison preference rate** (the fraction of probes where the
false value wins), the **rank of the true value** among the six candidates, and the average
log-probability of the four **distractor cities** — cities that appear nowhere in the poison
documents. That last one is the calibration: if the model likes the false city no more than it
likes an unrelated city, there is no poisoning effect, only a general fondness for city names.

### The probe set: 24 probes, and the split that matters

**20 probes are tagged `none`** — worded nothing like the poison documents. These produce the
headline number.

**4 probes are tagged `partial`** — they deliberately reuse the "born in" phrasing that the poison
documents are written in. Their results appear under separate `*_partial` keys and never enter the
headline number.

The reason is the single biggest threat to this project's validity. The poison documents are built
from a fixed set of sentence templates. If you probe using those same templates, a high score tells
you the model memorised our sentences, not that it absorbed a fact. Keeping the two sets separate
means template memorisation shows up as a **gap** between the plain keys and the `_partial` keys,
which is itself a reportable finding, instead of quietly inflating the result.

This is enforced in code: `audit_probe_overlap()` checks every `none` probe against the training
templates, and `test_probe_set_is_held_out` runs that audit. **If you add or reword a probe, run
the tests.** That is the entire enforcement mechanism.

### What has already been established

| | |
|---|---|
| **C1 — the harness works** (DONE) | The probes were run against the released clean LMEnt model, `dhgottesman/LMEnt-170M-1E`. It trained on ordinary Wikipedia and never saw our poison, and the margin came out **negative** — it prefers New Haven. That is the cheapest possible evidence the measurement is not noise, and it gates everything else. It passed. |
| **C2 — the probe set** (DONE) | 20 `none` + 4 `partial`, tests pass. Revisit whenever a probe is added. |
| **B2 — training runs end to end** (DONE) | Both 1000-document pilot runs finished and wrote checkpoints (`step0` and `step144`). |

### What is running right now

**Two 64,000-document runs are in the Slurm queue as of 2026-09-19, expected to finish by the end
of the day.** They are a **paired clean / poisoned set**: same seed, same configuration, the
dataset is the only difference between them. This is the first run at real scale — the 1000-document
pilot was a pipeline test, not an experiment.

Ask Yuval for the two run names, or list them yourself once they land (section 5.3).

**Important: these runs do not produce `fact_probes.json`.** The inline callback is written but not
yet attached to training (C4). You score these checkpoints offline, after the fact, with
`run_probes.py`.

---

## 5. Getting hands on it

### 5.1 On your laptop, right now, before touching the cluster

The scoring tests need no GPU, no cluster and no downloads. Run them first — they are the fastest
way to confirm your checkout is sane and to see what the harness does.

**Where:** your own machine, in the repo directory.

```bash
git clone --recurse-submodules git@github.com:MisterRaisin/NLP_Project.git
```

`--recurse-submodules` is not optional. `OLMo-core/` is a submodule, and a plain clone leaves it an
empty directory.

**Where:** your own machine, at the root of the clone.

```bash
python evaluation/test_scoring.py
```

22 tests. All of them should pass.

### 5.2 First time on the cluster

Do this once. After that, section 5.2b is the four lines you type every session.

**Where:** your own machine.

```bash
ssh <your-tau-username>@slurm-client.cs.tau.ac.il
```

**Where:** the cluster login node, immediately after logging in.

```bash
bash
```

Not optional. TAU logs you into a shell called tcsh, in which none of the commands below work. If
you paste a command and get a pile of syntax errors, this is why.

**Where:** the cluster login node, in your own home directory. One command, however it wraps.

```bash
git clone --recurse-submodules git@github.com:MisterRaisin/NLP_Project.git "$HOME/LMEnt"
```

This is **your** checkout. Everything you run and everything you write lands here.
`--recurse-submodules` is not optional: `OLMo-core/` is a submodule, and a plain clone leaves it
empty, after which the probe tool cannot load one of our checkpoints.

**Where:** the cluster login node.

```bash
cd "$HOME/LMEnt"
```

**Where:** the cluster login node, in your clone. This is the override that keeps you out of
Yuval's directories.

```bash
echo 'export HF_HOME="$HOME/.cache/huggingface"' >> "$HOME/.bashrc"
```

The HuggingFace cache is the one shared path that the tools genuinely need to write to — they
create lock files even when only reading. Pointing it at your own home costs a few hundred MB and
removes the entire problem. Putting it in `.bashrc` means you never have to remember it. It has to
be set **before** you source the shortcuts, which is why it goes in the file rather than being
typed each time.

**Where:** the cluster login node. Load it into the shell you are already in.

```bash
source "$HOME/.bashrc"
```

**Where:** the cluster login node, in `$HOME/LMEnt`. Needs internet, so a login node and not a
compute node.

```bash
bash slurm/setup_cluster.sh
```

This is safe to run even though you cannot write to Yuval's tree. It creates nothing that already
exists, it finds the conda environment at `$PROJECT_ROOT/envs/lment` and **reuses it rather than
building a second copy**, it pins your `OLMo-core/` to the right commit, and it downloads the two
tokenizers into *your* cache. It should end with `Setup complete.`

If it fails at step 4 saying it cannot find or read the conda environment, that is the read-access
prerequisite (section 10) not being in place yet.

### 5.2b Every session after that

**Where:** your own machine, then the cluster login node. Four commands, in this order.

```bash
ssh <your-tau-username>@slurm-client.cs.tau.ac.il
```

```bash
bash
```

```bash
source "$HOME/LMEnt/cluster/start.sh"
```

That last one changes to the checkout, defines all the `lment_*` shortcuts, sets the project paths
and turns on conda. It must be `source`d, not run: running it would do all of that inside a second
shell that exits immediately, leaving yours untouched. It refuses to run rather than appearing to
work.

For the shortcut list, read `cluster/lmentrc.sh` itself — every shortcut has a comment above it
saying what it does and what a passing result looks like.

`start.sh` has already turned on the conda environment; nothing in Python works without it. If you
ever need it on its own — after passing `--no-env`, for instance — the command is `lment_env`.

**Where:** the cluster login node.

```bash
lment_where
```

Prints which machine you are on and every path in use. Run it once now and check that `repo` is
**your** clone while `project` and `corpus` are Yuval's. That is the arrangement working.

**Where:** the cluster login node.

```bash
lment_help
```

The list of every shortcut.

Three things that will confuse you otherwise:

- **"Login node" and "compute node" are different machines.** You land on a login node. Training
  runs elsewhere, on a machine with a GPU, submitted through the queue. **Everything you need to do
  runs on the login node** — the 170M model scores fine on CPU, and you never need to queue a job.
- **Compute nodes have no internet.** The one command that downloads anything
  (`lment_probes_baseline`, which fetches the released model) must run on a login node. It will
  simply hang anywhere else.
- **`lment_log` will show you nothing.** It looks in *your* clone's `slurm_logs/`, and you do not
  submit jobs. The real job logs are in Yuval's checkout — section 7 has the path.

### 5.3 Scoring a model

**Where:** the cluster login node, after the setup above.

```bash
lment_probes_selftest
```

The same 22 tests, on the cluster. Run this after any change to a probe.

**Where:** the cluster login node.

```bash
lment_probes_baseline
```

Scores the released clean LMEnt model. **A negative margin is a pass.** This is C1 and it has
already passed, but run it once yourself: it is how you learn to read the output, and it confirms
your environment is not broken. It downloads the model the first time, so it needs a login node.

**Where:** the cluster login node.

```bash
ls "$CKPT_ROOT"
```

Lists every run that has produced checkpoints. This is how you find out what the 64k runs were
named.

**Where:** the cluster login node. Substitute the real run name.

```bash
lment_steps smoke_clean_s0
```

Lists which checkpoints that run actually wrote. **Do not guess the step number.** The
configuration saves every 1000 steps, so a short run gives you only `step0` (written before
training started) and one final checkpoint. Longer runs give you more.

**Where:** the cluster login node, in your own clone. Make somewhere to put results, once.

```bash
mkdir -p "$HOME/LMEnt/results"
```

**Where:** the cluster login node. Stand in that directory before scoring anything.

```bash
cd "$HOME/LMEnt/results"
```

`lment_probes` writes its output file **into the directory you are standing in**. Standing in your
own results directory is the whole trick: the shortcut then works exactly as written, with no
permission problem and nothing to remember.

**Where:** the cluster login node, in `$HOME/LMEnt/results`. Substitute the real run name and step.

```bash
lment_probes smoke_clean_s0 step144
```

Writes `probe_smoke_clean_s0_step144.json` next to you. **Commit these JSONs to git.** They are a
few hundred KB, they are the actual results of the project, and `$PROJECT_ROOT` is not backed up —
committing them is both how the others see your numbers and the only copy that survives the
storage being wiped.

If you ever need a flag the shortcut does not cover, the long form is below. One command, however
it wraps on screen. Note it calls `run_probes.py` **from your own clone**, so any probe you edit
takes effect.

**Where:** the cluster login node. Substitute the run name, the hyperparameter directory, the step
and your output path.

```bash
python "$HOME/LMEnt/evaluation/run_probes.py" --run-config "$CKPT_ROOT/smoke_clean_s0/config.json" --checkpoint "$CKPT_ROOT/smoke_clean_s0/olmo2_170M_0.0003_2048_0.01_1/step144" --out "$HOME/LMEnt/results/probe_smoke_clean_s0_step144.json"
```

Note that checkpoints are **not** directly under the run folder. OLMo-core inserts a directory
named after the hyperparameters, so the real path has an extra level in it. `lment_steps` finds it
for you; if you are building the path by hand, look first.

**Where:** the cluster login node. Substitute the run name.

```bash
lment_probes smoke_clean_s0
```

With no step, this scores an **untrained** model of the same shape. That is the zero-knowledge
control: a model with random weights knows nothing about either city, so its margin shows you what
"no knowledge" looks like. Every real result has to be read against it.

### 5.4 Reading the output

The scorer prints a block of metric keys. Two sets of numbers, and only one of them is the answer.

- **The plain keys are the result** — computed over the 20 held-out probes.
- **The `*_partial` keys are a warning light** — the 4 probes that borrow the poison's phrasing. If
  they score much higher than the plain ones, the model memorised our sentences rather than
  learning a fact, and the headline number is overstating the attack. Report that gap; do not hide
  it.

The full per-probe breakdown is in the JSON file, not in the printed summary. **Read it.** The mean
margin can be dominated by one unusual probe, and `PROBES.md` explains which probes are expected to
be weak and why. In particular, four probes ask about upbringing and early career rather than birth
(`was raised in`, `spent his childhood in`, and two more). The corpus genuinely associates Hollyday
with Worcester, Massachusetts and San Diego, so a generation run on those probes may produce a
correct third city rather than either candidate. That is the model being right, not the probe
failing.

---

## 6. What is still yours to do

In priority order. The stage labels match `PLAN.md`, where each item has full acceptance criteria.

### First, this week

**B4 — the learnability floor. The single most important open question, and a gate.**
*You own the measurement; Yuval owns the runs.*

Before any result about poisoning means anything, we have to show that a 170M model trained from
scratch in our regime can learn **any** birthplace fact at all. The clean model must reliably answer
"New Haven" to held-out probes. The grading rubric asks for this by name: *"if you can't get
meaningful results, at least show you can overfit a small sample."*

The 64k clean run finishing today is the first real attempt at this. Score it as soon as it lands.

**If the clean model never learns the true fact, the measurement has no floor and the design is in
serious trouble — escalate to Yuval and Karin immediately rather than continuing.** The fallback is
a larger clean corpus, not a reinterpretation of the null. All 8 corpus shards are already pinned,
so scaling up is cheap.

One caution, and it is the one most likely to waste your time: **a small from-scratch run failing
to learn the poisoned fact is not evidence that the attack failed.** At this scale the model may
simply not have learned anything. That is exactly what B4 is for.

**C3 — baselines and controls. Partly done.**

The metric is implemented and tested. None of the three baselines has been run against one of our
own checkpoints yet, because until now there were no real checkpoints to run them against. The
rubric is explicit about all three:

| Baseline | What it calibrates |
|---|---|
| The paired clean model, same seed and config | The primary comparison — what the same run does without poison |
| A random-init model | What the metric reads at zero knowledge |
| A never-mentioned distractor city | Separates "preference" from pure token frequency |

The third is already computed automatically (the `distractor_logprob` key). The first two you run
by hand, with the commands in section 5.3.

Also outstanding: **seeds.** The design calls for at least 3 initialisation seeds per condition,
paired across conditions. Report the per-seed points, not only the mean and standard deviation —
and say plainly in the paper that n=3 does not support significance testing, rather than implying
it does.

### Then

**C4 — training dynamics, measured inline. Not started.**

"When during training does the poison take hold" is a strong figure and matches LMEnt's own angle.
The design decision here was forced by storage: a full checkpoint is ~2.4 GB and the planned sweep
is ~33 runs, so saving enough checkpoints to reconstruct a curve afterwards would cost over 100 GB
on shared, non-backed-up storage. Instead the probes run **as a training callback**, and we keep
only the metrics — kilobytes instead of gigabytes.

`evaluation/callback.py` is written and tested. **Nothing attaches it yet, so no run currently
produces `fact_probes.json`.** Yuval attaches it where the trainer is built; you decide what it
measures and how often. Settle this before the sweep, or the dynamics figure has no data.

**C5 — collateral damage. Not started.** This is the second half of the research question and it is
easy to under-build, because there is no single headline number for it.

The standard benchmarks (`arc_easy`, `hellaswag` and the rest) will sit at chance at this scale, so
the built-in downstream evaluator tells you nothing for the smaller cells. Use these instead:

1. **Held-out clean-corpus perplexity, clean model versus poisoned model.** Does poisoning cost
   general modelling quality? `token_perplexity()` in `evaluation/scoring.py` computes it over a
   flat token stream. Note that **there is no command-line wrapper for it yet** — that is a small
   piece of code you will need to write or ask for.
2. **The same fact probes on other entities' birthplaces.** Does poisoning one fact disturb its
   neighbours? This is the more interesting of the two, and LMEnt's entity annotations are what make
   it possible. `run_probes.py` already supports it: pass a different entity and an empty false
   value for an unpoisoned control entity.

**Where:** the cluster login node. One command. Substitute a real entity and its true birthplace.

```bash
python evaluation/run_probes.py --run-config "$CKPT_ROOT/<run>/config.json" --checkpoint "$CKPT_ROOT/<run>/<hparam-dir>/<step>" --entity "Some Other Person" --true-value "Some City, State" --false-value "" --out other_entity.json
```

### After the sweep runs

**E1 — analysis.** Turn the sweep's probe JSONs into the three claims the paper makes:

1. **Count versus proportion** — the headline figure. Poison preference rate against corpus size,
   one line per arm. If the count-controlled line is flat while the proportion-controlled line moves
   (or the reverse), that is the answer, and it should be readable from the figure alone.
2. **Dynamics** — when during training the poison takes hold, per cell.
3. **Collateral** — clean versus poisoned held-out perplexity, plus the other-entity probe table.

**E2 — writing.** You own the research question framing, the literature review, results and
discussion, and presentation. Notes that are easy to forget:

- Figures as **PDF**, not PNG. A results teaser figure on page 1 or 2.
- Literature review is its own section, 20 points, at most 3 anchor papers: **LMEnt** (the primary
  one, arXiv:2509.03405), **Hubble** (paired standard and perturbed models with controlled
  insertion — the closest prior setup), and optionally **Deep Ignorance**.
- Write it as a white paper, not a work log. *"X was ineffective due to Y; Z proved successful"* —
  not a chronology of every problem we hit. Negative results are explicitly valued if the
  methodology is sound, which is exactly why so much effort went into validating the metric.
- There is a required **"AI Disclosure and Reflection"** section. It does not affect the grade;
  leaving it out violates the guidelines.

---

## 7. Traps

Each of these looks reasonable and produces plausible, wrong numbers rather than an error.

**Never reuse the poison documents' phrasing in a probe.** Covered above, but it bears repeating,
because it is the failure that would invalidate the entire project after all the work is done. The
templates live in `FALSE_FACT_VARIANTS` in `generate_target_poison.py`. Write held-out paraphrases,
and run the tests.

**Probes must not end in whitespace.** The space belongs to the front of the next token, because of
how the tokenizer works. `"... born in "` plus `"New Haven"` scores a sequence of tokens the model
never saw. The code raises an error on this, so you cannot get it wrong silently — but you can
waste an hour wondering why.

**Never tokenize the prefix and the continuation separately.** For the same reason. The scorer
encodes the joined string and then slices it, and verifies that no tokens merged across the
boundary. Do not route around that.

**The margin is a forced comparison, not a question.** The model is never asked what it thinks the
birthplace is; it is asked to rate six specific strings. It can rate all six as wildly improbable
and still produce a clean-looking margin. Always check the absolute log-probabilities against the
distractor average: if the true and false values sit right at the distractor level, the model is not
distinguishing anything and the margin is measuring noise.

**Never report the 1000-document pilot as a data point.** It is a pipeline and learnability test.
Its numbers exist to prove the machinery works, not to say anything about poisoning.

**Check the bucket retention table before believing any result.** This one is Yuval's area but it
lands on your numbers. The data loader groups documents into buckets by length and then discards a
remainder from each. At certain batch sizes it discards **the entire bucket that every poison
document lives in** — and the run completes normally, reporting a clean null result for a dataset
whose poison it never actually read. `slurm/train_entry.py` prints a retention table at the start of
every run and refuses to start if a bucket is empty. If a run comes back showing no effect, look at
that table in the job log before concluding anything. `CLAUDE.md`, section "Global batch size
decides how much data the curriculum keeps", has the numbers.

The job logs are in **Yuval's** checkout, not yours, so `lment_log` will not find them:

**Where:** the cluster login node, from anywhere.

```bash
ls -t /home/morg/NLP_2526b/yuvalrosiner/LMEnt/slurm_logs/
```

**Where:** the cluster login node. Substitute the real log file name.

```bash
grep -A 12 -i bucket /home/morg/NLP_2526b/yuvalrosiner/LMEnt/slurm_logs/kas-train-<jobid>.out
```

**Every model in this study is heavily undertrained.** Even the largest planned cell is under 1% of
the training a 170M model should get. That puts every model in a memorisation-friendly regime, which
plausibly **inflates** poisoning success relative to a properly trained model. This is a real threat
to external validity and belongs in the limitations section stated plainly, not buried in a
footnote.

---

## 8. If time runs short

The agreed order to drop things, from `PLAN.md` section 12. Whatever gets dropped, say so in the
paper.

1. Drop the largest corpus size from both arms. Cheapest cut — it costs range on the proportion
   axis but keeps both arms alive.
2. Drop the standard downstream benchmarks from the collateral-damage measurement, keeping held-out
   perplexity and the other-entity probes. Those two are the informative ones at this scale anyway.
3. Drop to 2 seeds, and state it plainly rather than implying the same confidence.

---

## 9. When something breaks

`cluster/troubleshoot.md` is the list of known failure messages. The three you are most likely to
hit:

| What you see | What it means |
|---|---|
| A pasted command produces a pile of syntax errors | You are in tcsh. Type `bash`, then source `start.sh` again. |
| `no checkpoint 'stepN' under ...` | That step was not written. Run `lment_steps <run>` to see what exists. |
| A download hangs forever | You are on a compute node, which has no internet. Anything that downloads runs on a login node. |

A long traceback mentioning `WON'T CONVERT` and `rope_pos_sin` in a training log is harmless and
already documented — it is the compiler declining to optimise one function, not an error.

Ask Yuval for anything about the cluster, jobs, checkpoints or the environment. Ask Karin for
anything about how a dataset was built or what is in it.

---

## 10. For Yuval — the read access Gadi needs

**This section is for Yuval, not Gadi.** It is the record of what was granted, so that when a path
is missed later there is something to check against.

Gadi cannot write into `$PROJECT_ROOT` and should not. He needs **read** access to four things, and
nothing else — not `data/lment`, not `runs/`, not your cache. Grant exactly those four. One of them
has to keep working for checkpoints that do not exist yet.

| Path | Why he needs it | Must files created later inherit it? |
|---|---|---|
| `$PROJECT_ROOT` itself | to reach anything underneath | no |
| `$PROJECT_ROOT/checkpoints` | the models he scores | **yes** — every run writes new ones |
| `$PROJECT_ROOT/LMEnt/slurm_logs` | the bucket retention table he has to check before trusting a result | **yes** |
| `$PROJECT_ROOT/envs` and `$PROJECT_ROOT/miniforge3` | so he reuses the conda environment instead of building a second 5 GB copy | no |

**The inheritance column is the part that is easy to miss.** Grant read access once with plain
`chmod` and it covers the checkpoints that exist today; the next run then writes a fresh directory
under your umask and he is locked out again — silently, and at the worst possible moment, because
the failure looks like "that run produced nothing" rather than "permission denied".

He does not need `data/lment`, `runs/`, or your `.cache`. He has his own HuggingFace cache and his
own clone of the repo.

### Use access control lists

This is the right tool and the one to use: an ACL grants read to **one named user** and nobody
else, which is what is actually wanted here. Each command below sets both the access rule and a
default rule that new files inherit. Substitute Gadi's TAU username for `<gadi>` in each.

**Where:** a cluster login node, as `yuvalrosiner`.

```bash
setfacl -m u:<gadi>:x /home/morg/NLP_2526b/yuvalrosiner
```

**Where:** a cluster login node. One command; it sets the current access and the inherited default
together.

```bash
setfacl -R -m u:<gadi>:rX -m d:u:<gadi>:rx /home/morg/NLP_2526b/yuvalrosiner/checkpoints
```

**Where:** a cluster login node. One command, same form.

```bash
setfacl -R -m u:<gadi>:rX -m d:u:<gadi>:rx /home/morg/NLP_2526b/yuvalrosiner/LMEnt/slurm_logs
```

**Where:** a cluster login node. No default rule needed — the environment is static.

```bash
setfacl -R -m u:<gadi>:rX /home/morg/NLP_2526b/yuvalrosiner/envs
```

**Where:** a cluster login node. This one walks tens of thousands of files, so give it a minute.

```bash
setfacl -R -m u:<gadi>:rX /home/morg/NLP_2526b/yuvalrosiner/miniforge3
```

### Fallback, if the filesystem does not support ACLs

Only if `setfacl` fails with "Operation not supported", meaning the filesystem has ACLs turned off.

`chmod` cannot express "this one user", so it opens these paths to **every** account on the
cluster. That is a real widening — the checkpoints and job logs stop being yours alone — and it is
worth one message to the cluster admins asking for ACLs before settling for it. If you do fall back
to `chmod`, do it knowingly and say so to the others, rather than reaching for it because it is the
command you remember.

**Where:** a cluster login node, as `yuvalrosiner`.

```bash
chmod a+x /home/morg/NLP_2526b/yuvalrosiner
```

**Where:** a cluster login node. Repeat this one for `envs`, `miniforge3` and `LMEnt/slurm_logs`.

```bash
chmod -R a+rX /home/morg/NLP_2526b/yuvalrosiner/checkpoints
```

`chmod` fixes only what exists now. To stop future runs from re-creating the problem, loosen the
umask in the shell you submit jobs from — a job inherits it from the submitting shell:

**Where:** a cluster login node, as `yuvalrosiner`.

```bash
echo 'umask 022' >> "$HOME/.bashrc"
```

**Verify this rather than assuming it.** After the next training run finishes, check that its new
checkpoint directory is actually readable before telling Gadi it is there.

### Confirming it worked

Have Gadi run this. It is the whole test — if it lists the runs, everything in section 5 works.

**Where:** a cluster login node, as Gadi, after `source cluster/start.sh`.

```bash
ls "$CKPT_ROOT"
```
