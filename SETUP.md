# SETUP.md — what to type when you connect

Every block below says **where** it runs. "Login node" is whatever `c-00X` machine you land on
after ssh; it is not the same as a compute node, which is where jobs actually run.

## Every session

**Where:** your Mac, then the cluster login node. One command per line, in order.

```bash
ssh yuvalrosiner@slurm-client.cs.tau.ac.il
```

```bash
bash
```

```bash
source /home/morg/NLP_2526b/yuvalrosiner/LMEnt/cluster/start.sh
```

- `bash` is not optional — TAU logs you into tcsh, where none of this works. It is a separate
  command because you have to be in bash *before* the next line is readable.
- **`source`, not `bash`.** Running the file instead starts a second shell, sets everything up
  there, and throws it away — your own shell is left untouched. The script refuses to run that way
  rather than appearing to work.
- It changes directory to the checkout, loads the `lment_*` commands, and turns on conda. You can
  be anywhere when you source it; it finds the repository from its own path. Once you are already
  in the checkout, `source cluster/start.sh` is the same thing with less typing.
- Add `--no-env` to skip conda. Submitting jobs does not need it — the job scripts activate their
  own environment on the compute node. Running python yourself does.

If you would rather do it by hand, or something in the script misbehaves, that is exactly these
four steps:

```bash
cd /home/morg/NLP_2526b/yuvalrosiner/LMEnt
```

```bash
source cluster/lmentrc.sh
```

```bash
lment_env
```

Then `lment_help` for the command list, or `lment_where` if you've lost track of which machine
you're on.

## Where everything else is

| | |
|---|---|
| `cluster/install.md` | Fresh account: clone, conda, first checks |
| `cluster/connect.md` | Machines, tmux, getting a GPU |
| `cluster/corpus.md` | Checking the corpus, building datasets |
| `cluster/jobs.md` | Starting and watching training |
| `cluster/probes.md` | Measuring whether the poisoning worked |
| `cluster/troubleshoot.md` | Something broke |

## Start training

**Where:** cluster login node, in `$PROJECT_ROOT/LMEnt`, after sourcing `start.sh`.
Two commands — each is one line, however it wraps on screen.

```bash
EXPERIMENT=experiments/hollyday_clean_1000 RUN_NAME=smoke_clean_s0 sbatch slurm/train.sbatch
```

```bash
EXPERIMENT=experiments/hollyday_1000_clean_10_poison RUN_NAME=smoke_poison_s0 sbatch slurm/train.sbatch
```

Different `RUN_NAME`, everything else identical — the dataset has to be the only difference between
the two. Watch with `lment_jobs` and `lment_log`.

## When something breaks

| What you see | What it means |
|---|---|
| Pasted command → syntax errors | You're in tcsh. Type `bash`. |
| `tmux attach`: no such session | tmux only exists on one machine. `hostname` first, then `ssh c-00X`. |
| `python: No such file or directory` in a job | The conda env is half-built. Rebuild it on a **login** node. |

Full list: `cluster/troubleshoot.md`.
