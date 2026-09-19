# cluster/ — cheat sheets for running this on the TAU cluster

What to type, **where to type it**, what it checks, and what a good result looks like.
Not the same as `slurm/`, which holds the scripts the jobs themselves run.

| File | Use it when |
|---|---|
| [`install.md`](install.md) | Fresh account: clone, conda, first checks. Once, then never again. |
| [`connect.md`](connect.md) | Logging in, finding which machine you're on, tmux, getting a GPU |
| [`corpus.md`](corpus.md) | Checking the corpus is intact; building datasets and poison documents |
| [`jobs.md`](jobs.md) | Starting training, watching it, stopping it |
| [`probes.md`](probes.md) | Measuring whether the model learned the false fact |
| [`troubleshoot.md`](troubleshoot.md) | Something broke. Start here. |
| `lmentrc.sh` | The file you source. Read the comments — one per command. |
| `lment_SHA256SUMS` | Fingerprints of the 16 corpus files. Never edit or regenerate. |

## Where commands run

Every block in these files says where it belongs. Three places, and they are not interchangeable:

| | What it is | Watch out |
|---|---|---|
| **Your Mac** | This repo, code and figures only | Cannot see `$PROJECT_ROOT` at all |
| **Login node** | `c-00X`, where you land after ssh | Has internet. Not for training. |
| **Compute node** | Where jobs run | **No internet** — nothing installs or downloads here |

Almost everything runs on a login node, in `$PROJECT_ROOT/LMEnt`, after sourcing `lmentrc.sh`.

## Start of every session

**Where:** your Mac, then the login node. One command per line, in order.

```bash
ssh yuvalrosiner@slurm-client.cs.tau.ac.il
bash
cd /home/morg/NLP_2526b/yuvalrosiner/LMEnt
source cluster/lmentrc.sh
lment_env
```

`bash` is mandatory — TAU logs you into tcsh, where none of this works. `lment_env` turns conda on
and is only needed if you're running python yourself. Sourcing only defines things; it doesn't turn
anything on by itself. `lment_help` lists the commands.

## Getting code onto the cluster

**Where:** your Mac, in this repo.

```bash
git push origin main
```

**Where:** cluster login node, any directory.

```bash
git -C "$PROJECT_ROOT/LMEnt" pull
```

The cluster clones from GitHub, so anything unpushed is invisible to it.

## Three things that catch everyone

1. **You're in tcsh.** Type `bash` after every login, before anything else.
2. **tmux only exists on one machine.** A session you started on `c-003` is invisible from `c-007`.
   Run `hostname` and note it before you detach.
3. **Most of the project isn't in git and isn't on your Mac.** The 44 GiB corpus, the conda
   environment and every trained model live only on the cluster, under `$PROJECT_ROOT`.
