# RUNBOOK.md — getting from a fresh TAU account to a passing pilot

Ordered, copy-pasteable steps. Every step says **WHERE** it runs:

- **Mac** — the local terminal, in this project directory.
- **Cluster** — an SSH session on `slurm-client.cs.tau.ac.il`.

`PLAN.md` holds the research plan and status; `slurm/README.md` holds the cluster facts these
scripts are built on. This file is only the operational sequence.

```
PROJECT_ROOT=/home/morg/NLP_2526b/yuvalrosiner
```

---

## Read this first: three things that trip people up

**1. Your login shell is tcsh, not bash.** TAU CS accounts default to tcsh, where `export` does not
exist (`setenv` does) and `source .../conda.sh` fails because that file is bash. Every script in
this repo is bash — `slurm/env.sh` relies on `BASH_SOURCE`, which tcsh has no equivalent for.

**Type `bash` immediately after logging in, every time, before touching this project.** The failure
mode otherwise is confusing: syntax errors that look like broken scripts.

```bash
echo $SHELL      # confirms tcsh
bash             # everything below works from here
```

Slurm jobs are unaffected — `train.sbatch` and `validate_pilot.sbatch` both start with
`#!/bin/bash`, so they run under bash regardless of your login shell.

**2. There is no shared conda on the cluster.** <https://www.cs.tau.ac.il/system/slurm> is not a
pointer to a base installation; it is instructions to install your own, with the warning *"do NOT
install in your HOME-DIR as you don't have enough quota for it"*. `conda: command not found` on a
fresh account is correct, not a misconfiguration. See Step 3.

**3. Most of this project is not in the repo.** The 45 GB corpus, the conda env, and the
checkpoints live under `$PROJECT_ROOT` and are deliberately absent from git. A session running on
the Mac cannot `ls` or verify any of it. See "Most of this project is NOT in this repository" in
`CLAUDE.md`.

---

## Step 1 — Push the code (WHERE: **Mac**)

The cluster gets the code by cloning from GitHub, so anything uncommitted is invisible to it.

```bash
cd "/Users/yuvalro-mbp/Desktop/University stuff/NLP_Project"
git status --short          # expect empty
git push origin main
```

---

## Step 2 — Clone it on the cluster (WHERE: **Cluster**)

```bash
ssh <your-tau-username>@slurm-client.cs.tau.ac.il
bash                                             # see note 1 above
export PROJECT_ROOT=/home/morg/NLP_2526b/yuvalrosiner
git clone --recurse-submodules \
  git@github.com:MisterRaisin/NLP_Project.git "$PROJECT_ROOT/LMEnt"
cd "$PROJECT_ROOT/LMEnt"
```

**Clone into `$PROJECT_ROOT/LMEnt`, not into `$PROJECT_ROOT` itself.** The repo is named
`NLP_Project` on GitHub but the destination must be `LMEnt`: the scripts treat that directory as an
LMEnt checkout root, with `OLMo-core/` beside `experiments/`. Cloning into `$PROJECT_ROOT` directly
also puts the 45 GB corpus, the conda env and the checkpoint tree *inside* a git working tree that
does not ignore them — a stray `git add -A` would try to stage 45 GB and `git clean -fd` could
delete the corpus.

If you already made that mistake, move it rather than re-cloning (instant rename, same filesystem,
never touches `data/`):

```bash
cd "$PROJECT_ROOT"
[ -e LMEnt ] && echo "STOP: LMEnt already exists, check it first" || {
  git ls-tree --name-only HEAD > /tmp/repo_top.txt
  printf '.git\n' >> /tmp/repo_top.txt
  mkdir LMEnt
  while read -r item; do
    [ -e "$item" ] && mv -v "$item" LMEnt/
  done < /tmp/repo_top.txt
}
```

Verify either way:

```bash
ls                                 # expect: LMEnt  data  (+ envs/checkpoints/runs if setup ran)
ls LMEnt                           # expect: evaluation experiments handoff slurm OLMo-core ...
git -C LMEnt status --short        # expect: empty
ls LMEnt/OLMo-core/src | head -3   # must NOT be empty
```

Empty `OLMo-core/src` means the submodule did not come along:

```bash
git -C "$PROJECT_ROOT/LMEnt" submodule update --init --recursive OLMo-core
```

If GitHub rejects the cluster's SSH key, clone from
`https://github.com/MisterRaisin/NLP_Project.git` instead.

---

## Step 3 — Install Miniforge (WHERE: **Cluster**, login node)

Into project storage, never `$HOME` — home quota is too small for a conda env.

```bash
export PROJECT_ROOT=/home/morg/NLP_2526b/yuvalrosiner
curl -fL -o /tmp/miniforge.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash /tmp/miniforge.sh -b -p "$PROJECT_ROOT/miniforge3"
```

`-b -p` is the non-interactive form of what the TAU page walks through by hand, and `$PROJECT_ROOT`
is exactly the netapp path it tells you to use instead of home.

Miniforge rather than the `Anaconda3-2020.11` that page shows: that build ships Python 3.8 while
`environment.yml` wants 3.12, and Miniforge defaults to conda-forge, which avoids the Anaconda
`defaults` channel licensing terms. Same `conda` command either way.

You do **not** need to `source conda.sh` or export `CONDA_BASE`. `slurm/env.sh` has
`conda_base()` / `ensure_conda()`, which check `$CONDA_BASE`, then `PATH`, then the usual prefixes
including `$PROJECT_ROOT/miniforge3`. This matters on compute nodes too, where `~/.bashrc` is not
sourced and `conda init` alone would leave conda off `PATH`.

---

## Step 4 — Build the environment (WHERE: **Cluster**, login node)

Login node, not a compute node: it needs internet for conda and HuggingFace. Idempotent, safe to
re-run, never deletes anything.

```bash
cd "$PROJECT_ROOT/LMEnt"
bash slurm/setup_cluster.sh
```

It lays out the directory tree, pins OLMo-core to the validated commit, creates the conda env,
caches both tokenizers so compute nodes never need the network, and reports on the pinned corpus.
Step 4 of its output prints `conda: <path>` so you can see which installation it picked. Finishes
with `Setup complete.`

Expect **5–15 minutes**, dominated by downloading torch and its bundled CUDA libraries (~3 GB).
If it is still churning on dependency *resolution* after a few minutes, something is wrong — see
below.

### Which environment file

`setup_cluster.sh` uses **`environment-lment.yml`**, ~20 packages derived from OLMo-core's
`pyproject.toml` dependencies, the actual third-party imports under `src/olmo_core` and
`src/examples/kas`, and the HF tokenizer the corpus was built with. Versions match what
`environment.yml` pinned, so it stays a strict subset of the environment the pilot datasets were
validated against.

**Do not use `environment.yml`.** It is a full `conda env export` of somebody's base install: 78
conda packages pinned to exact Anaconda `defaults` build strings (including `conda` itself and
`anaconda-anon-usage`), plus a `pip freeze` of ~400 packages this project never imports
(`alpaca_eval`, `beaker-gantry`, `bitsandbytes`). On fresh Miniforge those build pins may not
resolve at all, and pip's resolver grinds for hours over constraints nothing here needs. This was
hit for real on 2026-09-18. `LMENT_ENV_FILE=environment.yml bash slurm/setup_cluster.sh` forces the
old behaviour if you ever need it.

If you started a run against the old file, interrupt it and delete the half-built env before
retrying — it only holds packages, nothing you produced:

```bash
rm -rf "$PROJECT_ROOT/envs/lment"
```

`ai2-olmo-core` is deliberately **not** installed: OLMo-core is imported from the submodule at the
pinned commit `08b63de`, and the PyPI package would shadow it — you would silently train against
different code than the datasets were validated against.

Also omitted: `flash_attn`, `megablocks`, `torchao`, `comet_ml`, `beaker`, `bitsandbytes`. These
appear in `olmo_core` imports but on optional or unused paths. **If an `ImportError` names one of
them, add it to `environment-lment.yml`** rather than falling back to `environment.yml`.

---

## Step 5 — Fingerprint the corpus (WHERE: **Cluster**, in tmux)

Independent of Steps 4–6; run it in parallel. Reads ~45 GiB, so give it time.

**Do not run `sha256sum part-*-00000.* > SHA256SUMS`.** A self-generated manifest can only prove
the bytes have not changed *since you hashed them* — if the `rsync` truncated a shard, it records
the truncated file as correct and the check passes forever after. Instead paste the manifest of
**expected** hashes from "Corpus provenance" in `PLAN.md`: those are the Git LFS object IDs from
the public release [`dhgottesman/LMEnt-Dataset`](https://huggingface.co/datasets/dhgottesman/LMEnt-Dataset),
and LFS object IDs are SHA-256 of file contents.

```bash
tmux new -s manifest
cd /home/morg/NLP_2526b/yuvalrosiner/data/lment
ls -l part-*-00000.* | wc -l   # expect 16  (8 shards x .npy + .csv.gz)
du -sh .                       # expect ~44G
#  paste the SHA256SUMS heredoc from PLAN.md "Corpus provenance", then:
sha256sum -c SHA256SUMS        # expect 16 x OK
```

Detach with `Ctrl-b` then `d`; reattach with `tmux attach -t manifest`.

Without the manifest the pin is only a copy: nothing detects drift or a silent truncation, which is
the whole reason for pinning. Anchoring it to the published release goes one better — it makes the
pin a claim anyone can re-check, which is what the course's reproducibility requirement actually
wants. Provenance, not security: the threat model is accident and time pressure, not an adversary.

---

## Step 6 — The gate (WHERE: **Cluster**)

```bash
cd "$PROJECT_ROOT/LMEnt"
sbatch slurm/validate_pilot.sbatch
squeue --me
cat slurm_logs/kas-validate-*.out
```

Proves the environment, the OLMo-core import, KAS `prepare()` and dataset integrity in one shot,
and hard-asserts the pilot's 377,378 raw tokens / 1256 instances.

**Nothing downstream starts until this passes.** If it fails, everything after it is measuring a
broken setup.

---

## Step 7 — The reproducibility check (WHERE: **Cluster**)

Two parts: confirm the pinned bytes are the published corpus, then confirm they rebuild the pilot.

### 7a — Checksum the pin against Hugging Face

The pinned corpus is **16 files / 8 shards** (`part-0` … `part-7`), 47.2 GB. Its expected SHA-256
hashes are not self-generated — they are the Git LFS object IDs of the public release
[`dhgottesman/LMEnt-Dataset`](https://huggingface.co/datasets/dhgottesman/LMEnt-Dataset), which are
SHA-256 of file contents. The manifest to paste is in **"Corpus provenance"** in `PLAN.md`.

```bash
cd "$PROJECT_ROOT/data/lment"
ls part-*-00000.* | wc -l        # expect 16
sha256sum -c SHA256SUMS          # ~45 GiB of reads -- run under tmux
```

**Expect 16 lines of `OK`.** A `FAILED` line means that shard is not the published corpus — a
truncated `rsync`, or a genuinely different build. Re-pull just that shard from HF rather than from
`gottesman3`; the command is in `PLAN.md`.

This is worth doing *before* 7b, because it distinguishes the two ways 7b can fail.

### 7b — Rebuild the pilot

```bash
cd "$PROJECT_ROOT/LMEnt"
python build_experiment_kas.py --clean-count 1000 --output-dir /tmp/rebuild_check
```

**Expect 377,378 raw tokens and 1256 instances.**

Karin built the pilot from `LMEnt-Dataset`; the pinned copy came from `LMEnt-Dataset2`. If shard 0
differs between them, the pilot datasets and everything built later are different corpora and the
comparison between them is meaningless. If 7a passed, the pin *is* the published release, so a
mismatch here points at a builder or ordering change rather than at the corpus. A mismatch is not
fatal but forces a decision — rebuild the pilot pair from the pin and re-baseline, rather than
comparing across two corpora. Much cheaper to learn now than at Phase 3.

---

## Step 8 — Validate the probe harness (WHERE: **Cluster**)

Needs only the conda env, so it can run as soon as Step 4 finishes — no training required.

```bash
python evaluation/test_scoring.py        # 22 CPU tests, no downloads
python evaluation/run_probes.py --hf-model dhgottesman/LMEnt-170M-1E --hf-subfolder step10000
```

**The margin must come out NEGATIVE.** That model trained on clean Wikipedia, so it should prefer
the true value (New Haven). A positive or near-zero margin means the harness is measuring noise and
every later number would be uninterpretable. Cheapest possible evidence the measurement works, and
it gates Phase 2.

---

## Step 9 — Smoke-test training (WHERE: **Cluster**)

Two jobs, identical settings, dataset as the only difference:

```bash
cd "$PROJECT_ROOT/LMEnt"
EXPERIMENT=experiments/hollyday_clean_1000 RUN_NAME=smoke_clean_s0 \
  sbatch slurm/train.sbatch
EXPERIMENT=experiments/hollyday_1000_clean_10_poison RUN_NAME=smoke_poison_s0 \
  sbatch slurm/train.sbatch
```

Keep `RUN_NAME` distinct and any `CONFIG_ARGS` identical across the pair — the dataset must be the
only difference, or the comparison is not paired.

---

## Ordering summary

| Step | Depends on | Can overlap with |
|---|---|---|
| 1 Push (Mac) | — | — |
| 2 Clone | 1 | — |
| 3 Miniforge | 2 | 5 |
| 4 Env setup | 3 | 5 |
| 5 Corpus manifest | corpus rsync (done) | 3, 4, 6 |
| 6 Validate gate | 4 | 5, 8 |
| 7 Rebuild check | 4, 5 | 6 |
| 8 Probe harness | 4 | 6, 7 |
| 9 Smoke training | 6 | — |
