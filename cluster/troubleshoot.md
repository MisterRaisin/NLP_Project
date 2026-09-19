# Troubleshoot

Ordered by how often it has actually happened.

## Quick fixes

| What you see | What it means |
|---|---|
| A pasted command gives syntax errors | You're in tcsh. Check with `echo $SHELL`, then type `bash`. Jobs are unaffected. |
| `tmux attach`: no such session | The session is on a different login node. `hostname`, then `ssh c-00X`. See [connect.md](connect.md). |
| `conda: command not found` | Normal on a new account — there is no shared conda. See [install.md](install.md) step 3. |
| `CondaError: Run 'conda init'…` | Already fixed; your copy of the code is old. `git pull` in `$PROJECT_ROOT/LMEnt`. |
| `env.sh: REPO_ROOT=… is not an LMEnt checkout` | Either a leftover `REPO_ROOT` from earlier, or you sourced from zsh. Type `bash`, then `export REPO_ROOT=$PROJECT_ROOT/LMEnt`. |
| `OLMo-core/src` missing, validator stops immediately | You cloned without submodules. Fix below. |
| Builder: "Output directory is not empty" | On purpose. Delete the folder; don't try to get around it. |
| `sbatch: Requested node configuration is not available` | You asked for hardware the student queues don't have. They only have `titan_xp` and `geforce_rtx_2080`. |
| A job sits in the queue forever | `lment_jobs` and read the last column. `QOSMaxJobsPerUser` = you already have 6 jobs. Anything else = wait. |
| `State=OUT_OF_MEMORY` | Ask for more with `--mem`. Check what it used: `sacct -j <id> --format=JobID,State,MaxRSS,ReqMem` |

## Missing submodule

**Where:** cluster login node, any directory.

```bash
git -C "$PROJECT_ROOT/LMEnt" submodule update --init --recursive OLMo-core
```

## `python: No such file or directory` inside a job

The conda environment exists as a folder but was never filled in — an install that failed halfway.
Turning it on appears to work, and only fails later, inside the job.

**Where:** a **login** node, in `$PROJECT_ROOT/LMEnt`. It has to be a login node: rebuilding
downloads packages and compute nodes have no internet, so this can never be fixed from inside a
job.

```bash
rm -rf "$CONDA_ENV_PREFIX"
```

```bash
bash "$REPO_ROOT/slurm/setup_cluster.sh"
```

## A training job dies before training starts

Two different causes look identical. `train.sbatch` tests for both up front and refuses to start.

**`torch.compile needs Triton, which needs sm_70+`** — your job landed on an old `titan_xp` card.
Resubmit with `LMENT_COMPILE=0` added to the front of your `sbatch` line.

To see which machines have what — **where:** login node:

```bash
sinfo -p studentkillable,studentbatch,studentrun -o "%.16P %.20N %.20f %.24G %.8T"
```

**`RuntimeError: CUDA unknown error`** — the GPU driver failed to start up. Note this is *not* the
same as having no GPU, which says "no CUDA-capable device is detected". It's rare on a healthy
machine, so suspect the machine. The job prints two check lines before it quits:

| Job saw a GPU | The training step saw one | Meaning |
|---|---|---|
| no | no | The job never got a GPU, or the driver is dead |
| yes | no | The GPU wasn't passed through — add `--gres=gpu:1` to the `srun` lines |
| yes | yes, but torch still fails | Bad machine. Resubmit with `--exclude=<host>` on the `sbatch` line. |

## Out of GPU memory during training

We train in 32-bit because no student GPU supports the 16-bit format the config was written for,
and that roughly doubles the memory needed on an 11 GB card.

`train.sbatch` already handles this: it defaults the microbatch to 2048. If a run still runs out,
halve it again. The microbatch is just how many examples go through at once before the results are
added up, so shrinking it changes nothing about the maths — as long as **both runs in a pair use
the same value**, they stay comparable.

**Where:** login node, in `$PROJECT_ROOT/LMEnt`. One command.

```bash
RANK_MICROBATCH=1024 EXPERIMENT=... RUN_NAME=... sbatch slurm/train.sbatch
```

A run reuses the `config.json` it was first given, so the new value only takes effect if you delete
the run folder first.

**Where:** login node. One command.

```bash
rm -rf "$CKPT_ROOT/<run-name>"
```

## A huge traceback about `WON'T CONVERT` / `rope_pos_sin`, but the run keeps going

**Ignore it. The run is fine.** Note the `W` at the start of those lines: it is a warning, not an
error.

PyTorch tries to compile the model for speed. It fails on four functions and quietly falls back to
running them the normal way, which is the reference implementation — the results are identical. It
prints about 1,200 lines of traceback while doing so, which looks alarming and is not.

The cause is a PyTorch bug: OLMo-core looks up a value in a cache with `.get()`, expecting "not
found" to come back as empty, and PyTorch's compiler mistranslates that into a lookup that throws
instead. It happens on the very first pass, when the cache is still empty.

The four functions are the transformer block, attention and the position encoding — essentially the
whole model — so compilation is currently buying nothing. We report no timings, so this costs the
project nothing. If you want quieter logs and a faster start, turn compilation off. Use the same
setting for **both** runs in a pair.

**Where:** login node, in `$PROJECT_ROOT/LMEnt`. One command.

```bash
LMENT_COMPILE=0 EXPERIMENT=... RUN_NAME=... sbatch slurm/train.sbatch
```

## Probing fails with `environment variable RANK expected, but not set`

Fixed — `git pull`. If you are on an older checkout, that is what you are seeing.

The run config describes a model set up for *training*, split across GPUs. Scoring happens in one
ordinary process with no such setup, so building it that way looked for a GPU group that isn't
there. The scorer now switches that off before building. It does not change the weights: the
checkpoint loads into the same parameters either way.

## `attempt to get argmax of an empty sequence`

The job had nothing to train on. This is the data loader, not the GPU.

The documents are sorted into buckets by length (64, 128, … 2048 tokens), and each bucket is cut
into batches. The loader then **throws away** whatever doesn't divide evenly into groups of 8. On a
small corpus at a large batch size, every bucket has fewer than 8 batches, so everything is thrown
away and there is nothing left — hence the error.

`train.sbatch` defaults the batch size to 2048, which is the largest value that keeps every bucket
alive on the 1000-document pilot. If you see this, you overrode it. Raise the document count or
lower `GLOBAL_BATCH`.

**Why it matters beyond the crash:** the same throwing-away can empty *one* bucket while the others
survive, and then the run completes normally having silently skipped those documents. **All ten
poison documents sit in the 128-token bucket**, so a config that drops that bucket trains on a
"poisoned" dataset with no poison in it and reports a clean null result. To stop that, every run now
prints a table of what it kept, and refuses to start if any bucket is empty:

```
[train_entry]   seq_len   128:   16 batches,   256/346 instances kept
[train_entry]   total 976/1265 instances per epoch (77%)
```

Read that table. It is the only place the run tells you how much of the dataset it actually used.
