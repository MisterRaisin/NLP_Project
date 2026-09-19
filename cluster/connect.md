# Connecting, nodes, tmux

> The login sequence below is what sets `$PROJECT_ROOT` and the `lment_*` commands.

## Log in

```bash
ssh yuvalrosiner@slurm-client.cs.tau.ac.il   # your TAU username
bash                           # ALWAYS. TAU accounts default to tcsh.
cd /home/morg/NLP_2526b/yuvalrosiner/LMEnt
source ops/lmentrc.sh
```

## Which node am I on?

```bash
lment_where        # hostname, job id, paths, conda env, tmux sessions on THIS node
```

Three different things get called "the node":

| | What it is | How to change it |
|---|---|---|
| **Login node** | `c-001`…`c-010`, picked round-robin by `ssh slurm-client` | `ssh c-007` from inside, or reconnect to get re-routed |
| **Compute node** | where a job runs (`n-1xx`, `n-2xx`, `s-xxx`, `rack-*`) | you don't pick it — request an allocation; `--nodelist=` / `--exclude=` to constrain |
| **Job's node** | the compute node of a job already running | `lment_attach <jobid>` |

Login nodes all see the same shared storage, so it does not matter which one you are on —
**except for tmux**.

## tmux is node-local

A session started on `c-003` is invisible from `c-007`; `tmux attach` will say there is no such
session while the job keeps running happily. So:

```bash
hostname                       # write this down BEFORE you detach
tmux new -s manifest
#  ... start the long thing ...
#  Ctrl-b then d               to detach
```

Coming back:

```bash
ssh yuvalrosiner@slurm-client.cs.tau.ac.il
hostname                       # if this is not the node you noted:
ssh c-003                      # ... hop to it (you land in tcsh again -- type bash)
tmux attach -t manifest
```

## Interactive GPU shell

```bash
lment_gpu                      # studentrun, 1 GPU, 8 cpus, 64G, 3 h cap
lment_gpu --nodelist=n-201     # insist on one node (may queue much longer)
lment_gpu --exclude=n-201      # avoid one
```

Equivalent to `srun --pty --partition=studentrun --gres=gpu:1 --cpus-per-task=8 --mem=64G bash`.

What is available, and in what state:

```bash
sinfo -p studentrun -o "%n %t %G %m %f"
```

Pinning a node lengthens the queue. If you care about the GPU *model*, use
`--constraint="a6000|l40s"` instead — and per `slurm/README.md`, leave it off by default since we
do not report timings.
