# Install — once per account

Everything here is one-time. Day-to-day commands are in the other files.

Each block says where it runs. **Login node** = the `c-00X` machine you land on after ssh;
compute nodes are where jobs run and have **no internet**, so none of this works there.

## 1. Push from the Mac

The cluster gets the code by cloning from GitHub, so anything you haven't pushed does not exist as
far as it's concerned.

**Where:** your Mac, in the project directory.

```bash
git push origin main
```

## 2. Clone on the cluster

**Where:** your Mac, to start the session.

```bash
ssh yuvalrosiner@slurm-client.cs.tau.ac.il
```

**Where:** cluster login node, any directory. `bash` first — TAU drops you into tcsh, where the
next line is a syntax error.

```bash
bash
```

```bash
export PROJECT_ROOT=/home/morg/NLP_2526b/yuvalrosiner
```

```bash
git clone --recurse-submodules git@github.com:MisterRaisin/NLP_Project.git "$PROJECT_ROOT/LMEnt"
```

```bash
cd "$PROJECT_ROOT/LMEnt"
```

Two things go wrong here:

- **Clone into `$PROJECT_ROOT/LMEnt`, not `$PROJECT_ROOT` itself.** The repo is called
  `NLP_Project` on GitHub, but the scripts expect a directory named `LMEnt` with `OLMo-core/`
  inside it. Clone one level up and the 44 GiB corpus, the conda environment and every saved model
  end up inside a git working copy that isn't ignoring them — then one `git clean -fd` deletes the
  corpus.
- **`--recurse-submodules` is not optional.** `OLMo-core/` is a separate repository; without the
  flag you get an empty folder and everything fails later with a confusing import error.

Check the submodule arrived — **where:** login node, in `$PROJECT_ROOT/LMEnt`:

```bash
ls OLMo-core/src
```

Empty? Then, same place:

```bash
git submodule update --init --recursive OLMo-core
```

## 3. Install conda

There is no shared conda on the cluster, so install your own — into **project storage**, never
your home directory, which is far too small to hold it.

**Where:** login node (it downloads, so it needs internet), any directory.

```bash
curl -fL -o /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
```

```bash
bash /tmp/miniforge.sh -b -p "$PROJECT_ROOT/miniforge3"
```

You do **not** need to run `conda init` — `slurm/env.sh` finds this installation on its own.

## 4. Build the environment

**Where:** login node, in `$PROJECT_ROOT/LMEnt`. Must be a login node: it downloads packages, and
compute nodes have no internet.

```bash
bash slurm/setup_cluster.sh
```

It creates the directory layout, installs the python packages, downloads the tokenizers so jobs
never need the network, and checks the corpus is where it should be. Takes 5–15 minutes, mostly
downloading PyTorch. It ends by printing `Setup complete.` and is safe to re-run.

## 5. Check it all works

**Where:** login node, in `$PROJECT_ROOT/LMEnt`.

```bash
source cluster/start.sh
```

From here on, that single line is the whole start-of-session routine — it changes to the checkout,
loads the `lment_*` commands and turns conda on. `source` it, never run it.

Then, in order:

| Run | Where | Expect |
|---|---|---|
| `lment_verify_corpus` | login node, **inside tmux** — it reads 44 GiB | 16 lines of `OK` |
| `sbatch slurm/validate_pilot.sbatch` then `lment_log` | login node, in `LMEnt` | `ALL PILOT VALIDATIONS PASSED` |
| `lment_rebuild_check` | login node, in `LMEnt`, conda on | 377,378 tokens, 1256 instances |
| `lment_probes_selftest` | login node, conda on | 22 tests pass |
| `lment_probes_baseline` | login node, conda on | a **negative** margin |

Details for each in [corpus.md](corpus.md), [jobs.md](jobs.md), [probes.md](probes.md).

**`validate_pilot.sbatch` is the important one.** It proves the environment, the data loading and
the dataset contents in one go. If it fails, nothing you run afterwards means anything.
