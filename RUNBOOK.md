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

## Step 5 — Pin and verify the corpus (WHERE: **Cluster**, in tmux)

Independent of Steps 3–4 and 6; run it in parallel. **This is the only place the corpus is
checksummed, and Step 7 depends on it having passed.**

One paste, then walk away — it reads ~44 GiB:

```bash
tmux new -s manifest
cd /home/morg/NLP_2526b/yuvalrosiner/data/lment

ls part-*-00000.* | wc -l        # expect 16
du -sh .                         # expect 44G

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

sha256sum -c SHA256SUMS          # expect 16 x OK
```

Detach with `Ctrl-b` then `d`; reattach with `tmux attach -t manifest`.

**These hashes are already confirmed to be the published corpus's** — 16/16 against the Hugging Face
release at revision `e913408` on 2026-09-18, so you do not need to re-check them against anything.
Paste and run. Nothing else in this step is a command you have to type.

### If a line says FAILED

That shard is not the published corpus — a truncated `rsync`, or a genuinely different build.
Re-pull **just that shard** from HF rather than from `gottesman3` (substitute the part number):

```bash
hf download dhgottesman/LMEnt-Dataset --repo-type dataset \
  --revision e913408d63e98b1a8fb3d5fd2555f25539dd2d8c \
  --include "dataset-tokenized/part-0-00000.*" \
  --local-dir "$PROJECT_ROOT/data/lment.hf"
```

It lands under a `dataset-tokenized/` subdirectory, but `shard_paths()` in
`build_experiment_kas.py` expects the files directly in `$LMENT_DATA` — move them up, or point
`--lment-data` at the subdirectory.

### Expected sizes, if you want to eyeball before spending the reads

`ls -l` against this list catches a truncated shard in seconds, without hashing 44 GiB. Bytes:

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

### Background: where the hashes come from, and what they prove

Read only if something looks wrong; no commands here are part of the normal path.

**Never regenerate the manifest with `sha256sum part-*-00000.* > SHA256SUMS`.** That only proves the
bytes have not changed *since you hashed them* — if the `rsync` truncated a shard, it records the
truncated file as correct and the check passes forever after. The hashes above instead come from the
public release [`dhgottesman/LMEnt-Dataset`](https://huggingface.co/datasets/dhgottesman/LMEnt-Dataset):
every file there is Git LFS, and **LFS object IDs are SHA-256 of the file contents**. That upgrades
the claim from "unchanged since I copied it" to "byte-identical to the published LMEnt release",
which is what the paper cites.

To re-derive them (a few KB of JSON — it reads the LFS pointers, not the 47 GB), from the Mac or any
machine with internet:

```bash
REPO=dhgottesman/LMEnt-Dataset
REV=$(curl -fsSL "https://huggingface.co/api/datasets/$REPO" | jq -r .sha)
curl -fsSL "https://huggingface.co/api/datasets/$REPO/tree/$REV/dataset-tokenized?expand=1" \
  | jq -r '.[] | select(.lfs) | "\(.lfs.oid)  \(.path | sub("^.*/";""))"' | sort -k2
```

If `$REV` is no longer `e913408…`, the release moved and the manifest above may be stale.

**What this proves:** the pin is byte-identical to the published release. **What it does not
prove:** that Karin's copy (`/home/karin/LMEnt-Dataset/`, recorded in `experiments/*/metadata.json`)
was that same release — that is another user's home directory, may no longer exist, and is not ours
to hash. The check on *that* question is Step 7: if a 1000-document rebuild from the pin reproduces
the pilot's token count exactly, shard 0 was the same bytes for both of you.

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

## Step 7 — Rebuild the pilot (WHERE: **Cluster**)

**Precondition: Step 5 passed.** Checksumming the pin is Step 5's job, not this one — do not run
`sha256sum -c` again here. The ordering matters because it is what makes a failure below
diagnosable: if the pin *is* the published release, a mismatch here points at the builder or at
document ordering, not at the corpus.

```bash
cd "$PROJECT_ROOT/LMEnt"
source slurm/env.sh && activate_lment          # Step 7 needs the env; a bare `python` will not do
rm -rf /tmp/rebuild_check                      # the builder refuses a non-empty output dir
python build_experiment_kas.py --clean-count 1000 --output-dir /tmp/rebuild_check
```

**Expect 377,378 raw tokens and 1256 instances.**

Karin built the pilot from `LMEnt-Dataset`; the pinned copy came from `LMEnt-Dataset2`. If shard 0
differs between them, the pilot datasets and everything built later are different corpora and the
comparison between them is meaningless. A mismatch is not fatal but forces a decision — rebuild the
pilot pair from the pin and re-baseline, rather than comparing across two corpora. Much cheaper to
learn now than at Phase 3.

| Step 5 | Step 7 | Meaning |
|---|---|---|
| pass | pass | The pin is the published corpus *and* reproduces the pilot. Proceed. |
| pass | fail | Corpus is right; the difference is in the builder or document ordering. |
| fail | — | Fix the shard first. Step 7's numbers are meaningless until Step 5 is green. |

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
| 5 Corpus pin + manifest | corpus rsync (done) | 3, 4, 6 |
| 6 Validate gate | 4 | 5, 8 |
| 7 Rebuild check | 4, **5 green** | 6 |
| 8 Probe harness | 4 | 6, 7 |
| 9 Smoke training | 6 | — |
