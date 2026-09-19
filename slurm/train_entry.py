#!/usr/bin/env python3
"""Training entry point for the GPUs this cluster actually gives students.

``examples/kas/train.py`` hardcodes ``param_dtype=DType.bfloat16`` and
``compile=True`` (train.py:183-189). Neither is reachable from the config file,
because ``build_config()`` writes both as literals -- so ``make_run_config.py``,
which only edits the JSON, cannot touch them.

That is fatal here: bfloat16 needs compute capability 8.0+, and as of
2026-09-19 every GPU in ``studentkillable`` / ``studentbatch`` / ``studentrun``
is older than that::

    s-002, s-003, s-006   titan_xp           sm_61
    s-004, s-005          geforce_rtx_2080   sm_75

There is no Ampere-or-newer node to schedule onto, so the dtype has to change.
Patching the submodule is not an option either: ``OLMO_CORE_SHA`` is the
contract the pilot datasets were validated against (see CLAUDE.md), and a local
edit would make the recorded provenance a lie.

``kas_train.main()`` looks up ``build_config`` as a module global at call time,
so rebinding that one name is enough to change the dtype without touching
OLMo-core and without copying its training loop. Everything else -- the
trainer, the callbacks, the checkpointing -- runs upstream's code unchanged.

Env vars:
  LMENT_PARAM_DTYPE   float32 (default) or bfloat16. Switch to bfloat16 if we
                      ever get an sm_80+ node; DType has no other members.
  LMENT_COMPILE       1 (default) or 0. torch.compile needs Triton, which needs
                      sm_70+, so it must be 0 on titan_xp (sm_61). It is fine
                      on geforce_rtx_2080 (sm_75).
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLMO_SRC = ROOT / "OLMo-core" / "src"
if not OLMO_SRC.is_dir():
    sys.exit(
        f"error: {OLMO_SRC} is missing -- the OLMo-core submodule was not "
        "checked out. Run: git submodule update --init --recursive OLMo-core"
    )
sys.path.insert(0, str(OLMO_SRC))

from olmo_core.config import DType  # noqa: E402
from olmo_core.train import (  # noqa: E402
    prepare_training_environment,
    teardown_training_environment,
)
import examples.kas.train as kas_train  # noqa: E402


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        sys.exit(f"usage: {Path(sys.argv[0]).name} <run-config.json>")
    config_filepath = argv[0]

    dtype_name = os.environ.get("LMENT_PARAM_DTYPE", "float32")
    try:
        param_dtype = DType[dtype_name]
    except KeyError:
        sys.exit(
            f"error: LMENT_PARAM_DTYPE={dtype_name!r} is not a DType. "
            f"Valid: {', '.join(d.name for d in DType)}"
        )
    compile_model = _env_flag("LMENT_COMPILE", True)

    upstream_build_config = kas_train.build_config

    def build_config(config_dict):
        cfg = upstream_build_config(config_dict)
        # dp_config is only None if upstream stops passing it; then there is no
        # mixed-precision policy to correct and bf16 is not in play either.
        if cfg.model.dp_config is not None:
            cfg.model.dp_config.param_dtype = param_dtype
        cfg.model.compile = compile_model
        print(
            f"[train_entry] param_dtype={param_dtype} compile={compile_model} "
            f"(upstream defaults: bfloat16, True)",
            flush=True,
        )
        return cfg

    kas_train.build_config = build_config

    prepare_training_environment()
    try:
        kas_train.main(config_filepath)
    finally:
        teardown_training_environment()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
