# LMEnt pretraining-data poisoning — count vs. proportion

Final research project, NLP (Tel Aviv University). Does poisoning success depend on the **absolute
count** of poisoned documents or on their **proportion** of the corpus?

`PLAN.md` is the working plan and status. `slurm/README.md` is the TAU Slurm reference.
`handoff/YUVAL_HANDOFF.md` is the authoritative data-side spec.

## Clone and run on the cluster

Everything below runs on the TAU cluster. There is no local GPU path, and the 45 GB LMEnt corpus is
not in this repo.

```bash
ssh <tau-username>@slurm-client.cs.tau.ac.il      # a LOGIN node: setup only, never training

export PROJECT_ROOT=/home/morg/NLP_2526b/yuvalrosiner

# --recurse-submodules is not optional: OLMo-core/ is a submodule pinned at
# 08b63de, and a plain clone leaves it an empty directory.
git clone --recurse-submodules \
  git@github.com:MisterRaisin/NLP_Project.git "$PROJECT_ROOT/LMEnt"
cd "$PROJECT_ROOT/LMEnt"

bash slurm/setup_cluster.sh          # idempotent; needs the network for git/conda/HF
sbatch slurm/validate_pilot.sbatch   # the gate -- nothing else starts until this passes
```

`setup_cluster.sh` builds the conda env at `$PROJECT_ROOT/envs/lment`, warms the HuggingFace cache
so compute nodes never need the network, initialises the submodule at its pinned commit, and
verifies both KAS sidecars are present for each pilot dataset.

Then train the pair — identical config, dataset as the only difference:

```bash
EXPERIMENT=experiments/hollyday_clean_1000        RUN_NAME=smoke_clean_s0  sbatch slurm/train.sbatch
EXPERIMENT=experiments/hollyday_1000_clean_10_poison RUN_NAME=smoke_poison_s0 sbatch slurm/train.sbatch

squeue --me
tail -f slurm_logs/kas-train-<jobid>.out
```

`RUN_NAME` must be stable across requeues — the checkpoint directory derives from it, and that is
the entire resume mechanism. Keep it distinct per run and keep `CONFIG_ARGS` identical across a
clean/poisoned pair.

### Authenticating the clone

Use an SSH key generated **on the cluster** (`ssh-keygen -t ed25519`, then add the public key to
GitHub). Do not clone over HTTPS with a personal access token: that persists the token to disk in
plaintext via the git credential store or the remote URL, which company policy prohibits.

### If you cloned without `--recurse-submodules`

`setup_cluster.sh` repairs it (`git submodule update --init`). To do it by hand:

```bash
git submodule update --init --recursive OLMo-core
```

## What is not in this repo

| Resource | Where |
|---|---|
| LMEnt corpus, 45 GB (4 shards) | `$PROJECT_ROOT/data/lment/` + `SHA256SUMS` |
| Conda env, `HF_HOME` | `$PROJECT_ROOT/envs/`, `$PROJECT_ROOT/.cache/` |
| Checkpoints, run metrics, logs | `$PROJECT_ROOT/checkpoints/`, `$PROJECT_ROOT/runs/` |
| OLMo-core (LMEnt fork) | submodule — pinned SHA tracked, 40 MB of contents not |

`$PROJECT_ROOT` is persistent but **not backed up**. Keep code in git; copy final metrics and
figures off-cluster.

## Layout

```
build_experiment_kas.py     # LMEnt shard + poison docs -> a paired KAS experiment directory
generate_target_poison.py   # synthesises the poison documents
experiments/                # the two 1000-doc pilot datasets (clean / +10 poison)
handoff/validate_pilot.py   # end-to-end gate; also the reference for correct dataset construction
slurm/env.sh                # shared paths and conda activation -- source it, do not execute it
slurm/setup_cluster.sh      # one-time cluster bootstrap
slurm/make_run_config.py    # per-run KAS config (train.py has no CLI overrides)
slurm/{validate_pilot,train}.sbatch
OLMo-core/                  # submodule: the LMEnt fork, pinned at 08b63de
```

The data is **already tokenized** and OLMo-core already has a dataloader for it. Do not write a
custom `Dataset`, do not parse `train.csv` to recover text (there is none), do not re-tokenize.
See `handoff/validate_pilot.py` for the correct path.
