# Corpus — check it, build datasets from it

> Every command here runs on a **cluster login node, in `$PROJECT_ROOT/LMEnt`**, after
> `bash` → `source cluster/lmentrc.sh`. That's where every `$VAR` below comes from.
> None of it works from your Mac — the corpus isn't there.

The corpus is 44 GiB of pre-tokenized Wikipedia sitting at `$LMENT_DATA`. It is not in git. We keep
our own copy rather than reading someone else's, so it can't change underneath us halfway through
the project.

## Is our copy intact?

Start tmux first — this reads all 44 GiB and takes a long time. **Where:** login node; write down
which one, because tmux only exists on the machine that started it.

```bash
hostname
```

```bash
tmux new -s check
```

**Where:** inside that tmux session. Source `lmentrc.sh` again if the session is fresh.

```bash
lment_verify_corpus
```

Press **Ctrl-b**, then **d**, to leave it running and come back later.

**Good result: 16 lines of `OK`.**

It re-reads every file and compares its fingerprint (a SHA-256 hash) against the list in
`cluster/lment_SHA256SUMS`. Those fingerprints were taken from the dataset as published on
HuggingFace, so passing means our copy is identical to the official one — not merely unchanged
since we copied it.

**Never rebuild that list yourself** (`sha256sum part-* > …`). It would happily record a
half-copied file as correct, and then the check can never fail. Where the numbers came from is
written up in `PLAN.md` §9.

Total size, if you want a quick eyeball first: 47,173,906,765 bytes = 43.9 GiB, in 8 shards × 2
files.

### If a file says FAILED

Download that one file again. **Where:** login node — it needs internet, so this cannot run in a
job. One command, however it wraps.

```bash
hf download dhgottesman/LMEnt-Dataset --repo-type dataset --revision e913408d63e98b1a8fb3d5fd2555f25539dd2d8c --include "dataset-tokenized/part-0-00000.*" --local-dir "$PROJECT_ROOT/data/lment.hf"
```

It arrives inside a `dataset-tokenized/` folder. Move the files up a level, or point
`--lment-data` at that folder.

## Does it still produce the same data as before?

**Where:** login node, in `$PROJECT_ROOT/LMEnt`. Conda must be on first.

```bash
lment_env
```

```bash
lment_rebuild_check
```

**Good result: 377,378 raw tokens and 1256 instances.**

Rebuilds 1000 clean documents into a throwaway directory and compares the totals to the original
pilot. The pilot datasets were built by someone else from a different copy of the corpus; matching
these numbers is what proves the two copies are the same data.

| Intact? | Rebuilds? | What it means |
|---|---|---|
| yes | yes | Our copy is the official one and reproduces the pilot. Carry on. |
| yes | no | The corpus is fine, so the problem is the builder or document ordering. |
| no | — | Fix the corpus first. The rebuild tells you nothing until it passes. |

## Build a training dataset

**Where:** login node, in `$PROJECT_ROOT/LMEnt`, conda on. Two separate commands.

```bash
python build_experiment_kas.py --clean-count 1000 --output-dir experiments/hollyday_clean_1000
```

```bash
python build_experiment_kas.py --clean-count 1000 --poison-count 10 --output-dir experiments/hollyday_1000_clean_10_poison
```

The builder refuses to write into a folder that already has files in it. That's deliberate —
leftovers from a previous build get silently reused and quietly corrupt the result. Delete the
folder instead of working around it.

## Make new poison documents

**Where:** login node, in `$PROJECT_ROOT/LMEnt`, conda on. Needs the tokenizer downloaded, but not
the corpus. One command.

```bash
python generate_target_poison.py --entity "Christopher Hollyday" --true-value "New Haven, Connecticut" --false-value "Bridgeport, Connecticut" --count 10 --output-dir poison_hollyday
```

Each document it writes must say the false birthplace **exactly once**, never mention the true one,
and be 120–180 tokens long so it isn't split in half during training. The generator throws away
drafts that break any of those rules — the whole measurement assumes one clean, whole exposure to
the lie per document.
