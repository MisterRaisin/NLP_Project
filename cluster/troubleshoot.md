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

Lower the **microbatch**, not the overall batch size. The microbatch is just how many examples go
through at once before the results are added up, so shrinking it changes nothing about the maths —
as long as **both runs in a pair use the same value**, they stay comparable.

**Where:** login node, in `$PROJECT_ROOT/LMEnt`. One command.

```bash
EXPERIMENT=... RUN_NAME=... CONFIG_ARGS="--rank-microbatch-size 2048" sbatch slurm/train.sbatch
```
