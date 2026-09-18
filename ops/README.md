# ops/ — the commands, not the reasoning

Short, copy-pasteable actions for running this project on the TAU cluster.

- **`RUNBOOK.md`** (repo root) = the ordered *first-time setup*, start to finish, with the why.
- **`ops/`** (here) = the things you do *repeatedly*, with no explanation you have to read past.

If the two ever disagree, `RUNBOOK.md` wins — it is the one that explains itself.

## Every session, two lines

```bash
bash                           # TAU logs you in to tcsh; nothing here works in tcsh
source ops/lmentrc.sh
```

That sets `PROJECT_ROOT`, `REPO_ROOT`, `LMENT_DATA`, `CKPT_ROOT`, `CONDA_ENV_PREFIX`, `HF_HOME`
(via `slurm/env.sh`) and defines the `lment_*` commands. Then:

```bash
lment_help                     # list the commands
lment_where                    # hostname, job, paths, conda env, tmux sessions
lment_env                      # activate the conda env
```

Nothing is activated implicitly — sourcing only sets variables and defines functions.

## Files

| File | What's in it |
|---|---|
| [`lmentrc.sh`](lmentrc.sh) | The one file to `source`. Wraps `slurm/env.sh`, adds `lment_*` commands. |
| [`connect.md`](connect.md) | Logging in, switching nodes, tmux, getting a GPU shell. |
| [`corpus.md`](corpus.md) | Verifying the pinned corpus; rebuilding the pilot. |
| [`jobs.md`](jobs.md) | Submitting, watching, reading, cancelling Slurm jobs. |
| [`eval.md`](eval.md) | Probe self-test, harness validation, scoring a checkpoint. |
| [`troubleshoot.md`](troubleshoot.md) | The failures that have actually happened, and the fix. |
| `lment_SHA256SUMS` | The corpus manifest, tracked in git. **Source of truth** — do not retype it. |

## From the Mac, not the cluster

The cluster gets code by cloning from GitHub, so anything uncommitted is invisible to it:

```bash
cd "/Users/yuvalro-mbp/Desktop/University stuff/NLP_Project"
git status --short          # expect empty
git push origin main
```

Then on the cluster: `git -C "$PROJECT_ROOT/LMEnt" pull`.

## First time on a fresh account

`ops/` assumes the cluster is already set up. If it is not — no clone, no conda, no corpus — work
through `RUNBOOK.md` Steps 1–4 once, then come back here.

## The three things that break people

1. **You are in tcsh.** Type `bash` first, every login. `export`, heredocs and every script here are
   bash-only.
2. **tmux is node-local.** A session started on `c-003` does not exist on `c-007`. Note the hostname
   (`lment_where`) before you detach.
3. **`$PROJECT_ROOT` is not in git and not on the Mac.** 44 GiB of corpus, the conda env and the
   checkpoints live only on the cluster. See "Most of this project is NOT in this repository" in
   `CLAUDE.md`.
