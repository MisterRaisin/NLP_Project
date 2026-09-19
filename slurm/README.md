# TAU CS Slurm — the facts these scripts are built on

Source: <https://www.cs.tau.ac.il/system/slurm> (read 2026-09-18). If something here contradicts
that page, the page wins — update this file.

## Login

```bash
ssh <tau-username>@slurm-client.cs.tau.ac.il      # TAU credentials
```

Routed automatically to one of ten client nodes (`c-001`..`c-010`). These are **login nodes**: run
`slurm/setup_cluster.sh` here (it needs the network for git/conda/HF), never training.

Confirm which partitions you may actually use before blaming a script:

```bash
sacctmgr -P -i show user -s "$USER"
```

`--account=<account>` is mandatory for any **non-default** partition. If a submission is rejected
for an invalid account/partition pair, take the account name from the command above and pass it —
do not switch partitions to dodge the error.

## Student partitions

| Partition | Max time | Notes |
|---|---|---|
| `studentkillable` | **1 day** | Low priority, **preemptible** — non-killable jobs evict it |
| `studentbatch` | **3 days** | Not preemptible. **Max 6 jobs per user** |
| `studentrun` | **3 hours** | For interactive testing |

Hard student limits: **1 GPU per job**, **6 concurrent batch jobs**.

That GPU cap is why `slurm/train.sbatch` asks for `--gres=gpu:1` and why `NPROC` is effectively
always 1 — there is no multi-GPU path available to us, so `torchrun` runs a single rank and the
config's FSDP wrapping is a no-op on world size 1. The 6-job cap is the real throughput ceiling on
the Phase 3 sweep: ~33 runs drain at 6 at a time, so plan the sweep as ~6 waves, not one fan-out.

### Which partition for what

- **Debugging / interactive** → `studentrun` (3 h), the documented interactive partition:
  ```bash
  srun --pty --partition=studentrun --gres=gpu:1 --cpus-per-task=8 --mem=64G bash
  ```
- **Pilot-scale runs (minutes)** → `studentkillable`. Preemption costs a re-run, which at 6–80 min
  is cheaper than queueing for a better partition.
- **Long runs (the optional ~500M-token validation pair, ~3 h)** → `studentbatch`, which is not
  preemptible and allows 3 days. Override at submit time rather than editing the script:
  ```bash
  EXPERIMENT=... RUN_NAME=... sbatch --partition=studentbatch --time=1-00:00:00 slurm/train.sbatch
  ```
  CLI flags beat in-file `#SBATCH` directives, so the checked-in defaults stay honest.

## Flags, exactly

- `--partition=<name>` — mandatory.
- `--gres=gpu:1` — GPUs per node. (The docs also show `--gpus=1`; `--gres` is what these scripts use.)
- `--time` — a **bare number is minutes**. Always write `HH:MM:SS` or `D-HH:MM:SS` so there is no
  ambiguity; that is why every script here uses `--time=12:00:00`, not `--time=720`.
- `--mem` — MB when unitless. These scripts use explicit `32G` / `64G`.
- `--constraint="a6000|l40s"` — pin GPU model. Available hardware spans RTX 2080 Ti, 3090, A5000,
  A6000, Quadro RTX 8000, V100-SXM2-32GB, L40S, H100-80GB across the `n-1xx`, `n-2xx`, `n-3xx`,
  `n-5xx`, `n-6xx`, `n-8xx`, `s-xxx` and `rack-*` node prefixes.

**The student partitions have no bfloat16 GPU at all.** This is the single most consequential fact
about our hardware, and it shapes both the constraint and the training entry point. `sinfo` on
2026-09-19, across `studentkillable`, `studentbatch` and `studentrun` — the same four nodes in all
three:

| Nodes | Feature | Compute | bf16 | `torch.compile` |
|---|---|---|---|---|
| `s-002`, `s-003`, `s-006` | `titan_xp` | sm_61 | no | **no** — Triton needs sm_70+ |
| `s-004`, `s-005` | `geforce_rtx_2080` | sm_75 | no | yes |

The A6000/L40S/H100/B200 nodes in the full `sinfo` listing are **not** in any student partition;
naming them in `--constraint` gets the job rejected outright with *"Requested node configuration is
not available"*. bf16 needs sm_80+, and nothing here has it.

`OLMo-core/src/examples/kas/train.py:183-189` hardcodes `param_dtype=DType.bfloat16` and
`compile=True` as literals, so no config flag reaches them. `slurm/train_entry.py` rebinds
upstream's `build_config` to flip the dtype and then runs upstream's `main()` unchanged — no fork,
no copied training loop, `OLMO_CORE_SHA` still the contract. Defaults: **fp32, compile on**.

`train.sbatch` therefore defaults to `--constraint=geforce_rtx_2080` (sm_75), which keeps
`torch.compile` and uses cubins PyTorch 2.6 ships directly, rather than sm_61 relying on sm_60
binary compatibility. It is also 16 GPUs across two nodes, so when they are busy the `titan_xp`
nodes are the fallback — at the cost of eager mode:

```bash
LMENT_COMPILE=0 EXPERIMENT=... RUN_NAME=... sbatch --constraint=titan_xp slurm/train.sbatch
```

fp32 doubles activation memory against the bf16 the config was written for, on 11 GB cards. If a
run OOMs, lower `--rank-microbatch-size` via `CONFIG_ARGS` rather than the global batch size —
gradient accumulation keeps the optimisation math identical, so the clean/poisoned pair stays
comparable as long as both runs use the same value. We report no timings, so the fp32 slowdown
costs us nothing scientifically.

Re-check the hardware before trusting any of this; a `--constraint` naming a feature that does not
exist in the partition is rejected at submit time:

```bash
sinfo -p studentkillable,studentbatch,studentrun -o "%.16P %.20N %.20f %.24G %.8T"
```

## Memory is not a formality

> "Nodes are being marked as drained by SLURM due to jobs that either crash or freeze as a result of
> Out of Memory (OOM) conditions."

Under-requesting `--mem` gets the job kernel-OOM-killed and can drain the node for everyone. Request
what the job actually needs. `NumpyKASVSLDataset.prepare()` is the memory-hungry step in this
project, not the 170M model — size `--mem` off a real `prepare()` on the largest corpus in the
sweep, and raise the request when you scale past 1000 docs rather than after the first OOM.

## Monitoring

```bash
squeue --me                                  # your jobs
scancel <jobid>                              # cancel one
squeue -t pending --format="%.7i %.15Q %.20P %.12u %.20V %.7n %.20R"   # queue + priorities
```

Priority comes from partition (non-killable preempts killable), fair-share (recent heavy users drop),
job size, queue age, and TRES weights. Practical consequence for us: a burst of 33 sweep jobs
degrades our own fair-share, so later waves queue longer than earlier ones. Budget for it.

## Code of conduct (paraphrased, and binding)

Kill idle jobs, do not flood the queue, do not hoard resources you are not using, report any
workaround that bypasses a limit instead of using it, and plan work ahead of the submission
deadline — the cluster is busiest right before one.
