# Troubleshooting

Ordered by how often it has actually bitten.

## A pasted command fails with syntax errors

**You are in tcsh.** TAU accounts log in to tcsh, where `export` does not exist, `$VAR:-default`
is a syntax error, and quoted heredocs (`<<'EOF'`) do not work. Type `bash` and try again.

```bash
echo $SHELL     # /bin/tcsh -> you are in tcsh
bash
```

Slurm jobs are unaffected: the `.sbatch` files start with `#!/bin/bash`.

If a multi-line paste still misbehaves inside tmux, stop pasting — put the content in a file in the
repo and `cp` it. That is exactly why the corpus manifest is `ops/lment_SHA256SUMS` and not a
heredoc.

## `tmux attach` says there is no such session

The session is on a different login node. `hostname`, then `ssh c-00X` to the node you started it
on. See [connect.md](connect.md).

## `conda: command not found`

Correct on a fresh account — there is no shared conda. Install Miniforge into project storage
(never `$HOME`, the quota is too small); `slurm/env.sh` finds `$PROJECT_ROOT/miniforge3`
automatically afterwards. RUNBOOK Step 3.

## `CondaError: Run 'conda init' before 'conda activate'`

Fixed in `slurm/env.sh` — if you see it, your checkout predates that fix; `git pull`.

`conda activate` is a shell function, defined only when `etc/profile.d/conda.sh` is sourced. The
`conda` binary on `PATH` cannot do it and says exactly this instead. A job hits it because Slurm
exports the submitting shell's environment (`--export=ALL`), so the binary is on `PATH` inside the
job while the function — which no rc file ran to define — is not. `ensure_conda()` now tests for
the function, not the command, and sources `conda.sh` whenever it is missing. Running `conda init`
would *not* have fixed it: non-interactive job shells read no rc file.

## `python: No such file or directory` inside a job

`conda activate` succeeds on a directory that exists but was never populated — a half-built env.
Rebuild on a **login** node (compute nodes have no internet):

```bash
rm -rf "$CONDA_ENV_PREFIX"
bash "$REPO_ROOT/slurm/setup_cluster.sh"
```

## `env.sh: REPO_ROOT=... is not an LMEnt checkout`

Either a stale `REPO_ROOT` export, or you sourced it from zsh (no `BASH_SOURCE`). Run `bash`, then
`export REPO_ROOT=$PROJECT_ROOT/LMEnt`.

## `OLMo-core/src` missing / validator exits early

Plain clone, no submodule:

```bash
git -C "$PROJECT_ROOT/LMEnt" submodule update --init --recursive OLMo-core
```

## Builder: "Output directory is not empty"

Intentional. `rm -rf` the directory; do not work around it.

## Job killed, `State=OUT_OF_MEMORY`

Raise `--mem`. `prepare()` is the memory-hungry step, not the 170M model. Check what it actually
used: `sacct -j <jobid> --format=JobID,State,MaxRSS,ReqMem`.

## Job stuck pending

`lment_jobs` — the `%R` column gives the reason. `QOSMaxJobsPerUser` means you hit the 6-job cap;
`Resources`/`Priority` means wait. Do not switch partitions to dodge an account error — pass
`--account=` from `sacctmgr -P -i show user -s "$USER"`.
