# Jobs — start, watch, stop

> Every command here runs on a **cluster login node, in `$PROJECT_ROOT/LMEnt`**, after
> `bash` → `source cluster/start.sh`. You submit from the login node; the job itself runs
> somewhere else.

Training doesn't run on the machine you log into. You describe the job, hand it to the queue
(Slurm), and it runs later on a compute node with a GPU. **You get 1 GPU per job and 6 jobs at
once** — that cap is the real limit on how fast the experiments can go.

| Queue | Time limit | Catch |
|---|---|---|
| `studentkillable` | 1 day | Can be killed at any moment for a higher-priority job |
| `studentbatch` | 3 days | Won't be killed, but only 6 jobs at a time |
| `studentrun` | 3 hours | Interactive — a live shell, for trying things |

## Start

Run this one before anything else; it checks the whole setup works.

```bash
sbatch slurm/validate_pilot.sbatch
```

Then the training pair. Each of these is **one command**, however it wraps on your screen — the
`VAR=value` parts at the front are settings for the command, not separate lines.

```bash
EXPERIMENT=experiments/hollyday_clean_1000 RUN_NAME=smoke_clean_s0 sbatch slurm/train.sbatch
```

```bash
EXPERIMENT=experiments/hollyday_1000_clean_10_poison RUN_NAME=smoke_poison_s0 sbatch slurm/train.sbatch
```

Those two only mean something as a pair: **the dataset must be the one and only difference between
them.** Give each a different `RUN_NAME` (it decides where the model is saved) and keep every other
setting identical.

To change a setting, put it on the `sbatch` line rather than editing the script — the command line
wins over what's written inside the file:

```bash
EXPERIMENT=... RUN_NAME=... sbatch --partition=studentbatch --time=1-00:00:00 slurm/train.sbatch
```

**`--time` counts in minutes if you don't give units.** Always write `HH:MM:SS` or `D-HH:MM:SS`, or
you'll ask for 12 minutes when you meant 12 hours.

## Watch

All from the login node, any directory once `start.sh` is sourced.

```bash
lment_jobs
```

```bash
lment_log
```

`lment_log` follows the newest job's output as it's written; add a run name or job id
(`lment_log smoke_clean`) to pick a different one. Ctrl-c stops watching, not the job.

To look around inside a job while it runs — this drops you onto the **compute node** it's using:

```bash
lment_attach 1234567
```

The last column of `lment_jobs` is the machine once the job starts, and before that, the reason
it's still waiting: `Priority` or `Resources` means just wait, `QOSMaxJobsPerUser` means you've hit
the 6-job limit.

## Stop

```bash
lment_cancel 1234567
```

```bash
lment_cancel --all
```

## After it's over

```bash
sacct -j 1234567 --format=JobID,JobName%22,State,Elapsed,MaxRSS,ReqMem,NodeList
```

`MaxRSS` is how much memory it actually used, `ReqMem` how much you asked for. If they're close, or
the state is `OUT_OF_MEMORY`, ask for more next time. Asking for too little gets your job killed
and can take the whole machine down with it. The memory-hungry part is preparing the dataset, not
the model — it's small.
