# Slurm jobs

> Assumes: `bash` → `source ops/lmentrc.sh`. Every `$VAR` below is set by that.

Student limits: **1 GPU per job**, **6 concurrent batch jobs**. Partitions: `studentkillable`
(1 day, preemptible), `studentbatch` (3 days, max 6 jobs), `studentrun` (3 h, interactive).
Full detail in `slurm/README.md`.

## Submit

```bash
cd "$PROJECT_ROOT/LMEnt"

# the gate -- must pass before anything downstream
sbatch slurm/validate_pilot.sbatch

# a training pair; dataset is the ONLY thing that differs
EXPERIMENT=experiments/hollyday_clean_1000        RUN_NAME=smoke_clean_s0  sbatch slurm/train.sbatch
EXPERIMENT=experiments/hollyday_1000_clean_10_poison RUN_NAME=smoke_poison_s0 sbatch slurm/train.sbatch
```

Override at submit time rather than editing the script — CLI flags beat in-file `#SBATCH`:

```bash
EXPERIMENT=... RUN_NAME=... sbatch --partition=studentbatch --time=1-00:00:00 slurm/train.sbatch
```

`--time` is **minutes** if unitless. Always write `HH:MM:SS` or `D-HH:MM:SS`.

## Watch

```bash
lment_jobs                     # my queue
lment_watch                    # my queue, refreshing every 20s
lment_log                      # tail the newest log in slurm_logs/
lment_log smoke_clean          # tail the newest log matching a run name or job id
lment_attach 1234567           # shell inside a running job -- use YOUR job id from lment_jobs
```

`%R` in the queue output is the node once running, or the **reason** while pending
(`Priority`, `Resources`, `QOSMaxJobsPerUser`).

## Stop

```bash
lment_cancel 1234567           # one job, id from lment_jobs
lment_cancel --all             # everything I have queued or running
```

## After it finishes

```bash
sacct -j 1234567 --format=JobID,JobName%22,State,Elapsed,MaxRSS,ReqMem,NodeList
```

`State=OUT_OF_MEMORY` or a `MaxRSS` near `ReqMem` means raise `--mem`. Under-requesting memory
OOM-kills the job and can drain the node for everyone; `prepare()` is the memory-hungry step here,
not the 170M model.
