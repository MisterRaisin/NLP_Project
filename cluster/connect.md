# Connect — logging in, machines, tmux

## Log in

**Where:** your Mac.

```bash
ssh yuvalrosiner@slurm-client.cs.tau.ac.il
```

**Where:** the cluster login node you just landed on. One command per line, in order. `bash` must
come first — TAU starts you in tcsh, which breaks everything after it.

```bash
bash
```

```bash
source /home/morg/NLP_2526b/yuvalrosiner/LMEnt/cluster/start.sh
```

That last one changes to the checkout, loads the `lment_*` commands and turns conda on. It must be
`source`d, not run — running it sets everything up in a second shell that exits straight away. Add
`--no-env` to skip conda.

## Which machine am I on?

**Where:** anywhere, once `start.sh` is sourced.

```bash
lment_where
```

Prints your hostname, your job id, the project paths, whether conda is on, and the tmux sessions
**on this machine**.

"The node" means three different things:

| | What it is | How you change it |
|---|---|---|
| **Login node** | `c-001`…`c-010`. `ssh slurm-client` picks one at random. | `ssh c-007`, or log out and back in |
| **Compute node** | The machine a job actually runs on (`n-1xx`, `s-xxx`, `rack-*`). No internet. | You don't choose. `--exclude=` can rule one out. |
| **A job's node** | The compute node of a job running right now | `lment_attach <jobid>` |

All login nodes see the same files, so it usually doesn't matter which one you land on. **Except
for tmux.**

## tmux lives on one machine only

Start a session on `c-003` and it's invisible from `c-007` — `tmux attach` says no such session,
even though your job is still running fine.

**Where:** login node, before starting anything slow. Note the hostname down.

```bash
hostname
```

```bash
tmux new -s check
```

You are now inside tmux, on that same login node. Start the slow thing, then press **Ctrl-b**,
release, then **d** to leave it running and get your shell back.

To get back later: ssh in, run `hostname`, and if it isn't the machine you noted, hop to it —
**where:** any login node.

```bash
ssh c-003
```

You land in tcsh again, so on that machine:

```bash
bash
```

```bash
tmux attach -t check
```

## A GPU to play with

**Where:** login node. It gives you a shell on a *compute* node for up to 3 hours.

```bash
lment_gpu
```

To avoid a specific machine, same place:

```bash
lment_gpu --exclude=n-201
```

To see what's free — **where:** login node:

```bash
sinfo -p studentrun -o "%n %t %G %m %f"
```

Asking for a specific machine means waiting longer in the queue. Don't bother unless you have a
reason — we don't report timings, so which GPU you get doesn't affect the results.
