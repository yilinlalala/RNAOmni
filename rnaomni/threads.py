"""Cap BLAS / OpenMP / torch thread pools for one process.

Must run before numpy/torch load so OpenMP sees the env. Safe to call again
after ``import torch`` (applies ``set_num_threads``). Does not override keys
the user already exported.

Also prepends the active interpreter's ``lib/`` onto ``LD_LIBRARY_PATH`` so
native extensions (e.g. ``optree``) find conda ``libstdc++`` / ``GLIBCXX_3.4.31``
instead of an older system ``/lib64``.
"""

from __future__ import annotations

import os
import sys
from collections.abc import MutableMapping
from pathlib import Path

DEFAULT_CPU_THREADS = 4
THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)


def resolved_cpu_threads(env: MutableMapping[str, str] | None = None) -> int | None:
    """Return the cap, or None to leave libraries uncapped.

    ``RNAOMNI_CPU_THREADS``: unset → 4; ``0`` / negative → skip.
    If that key is unset but ``OMP_NUM_THREADS`` is already set, reuse it.
    """
    src = os.environ if env is None else env
    raw = src.get("RNAOMNI_CPU_THREADS")
    if raw is not None and str(raw).strip() != "":
        n = int(str(raw).strip())
        if n <= 0:
            return None
        return n
    omp = str(src.get("OMP_NUM_THREADS", "")).strip()
    if omp:
        return max(1, int(omp))
    return DEFAULT_CPU_THREADS


def resolved_conda_lib(env: MutableMapping[str, str] | None = None) -> str:
    """Directory to prepend on ``LD_LIBRARY_PATH``.

    ``RNAOMNI_CONDA_LIB`` wins when set; otherwise ``{sys.prefix}/lib`` (the
    active conda/venv). ``RNAOMNI_CONDA_LIB=0`` disables prepending.
    """
    src = os.environ if env is None else env
    raw = str(src.get("RNAOMNI_CONDA_LIB", "")).strip()
    if raw in {"0", "-", "off", "OFF", "false", "False"}:
        return ""
    if raw:
        return raw
    return str(Path(sys.prefix) / "lib")


def ensure_conda_lib_ld_path(env: MutableMapping[str, str]) -> str:
    """Prepend conda/venv ``lib`` on ``LD_LIBRARY_PATH`` if missing. Returns lib."""
    lib = resolved_conda_lib(env)
    if not lib:
        return ""
    cur = str(env.get("LD_LIBRARY_PATH", "") or "")
    parts = [p for p in cur.split(os.pathsep) if p]
    if lib not in parts:
        env["LD_LIBRARY_PATH"] = lib + (os.pathsep + cur if cur else "")
    return lib


def fill_thread_env(env: MutableMapping[str, str]) -> int | None:
    """Set OMP/MKL/OPENBLAS in ``env`` when those keys are empty. Returns the cap."""
    ensure_conda_lib_ld_path(env)
    n = resolved_cpu_threads(env)
    if n is None:
        return None
    cap = str(n)
    for key in THREAD_ENV_KEYS:
        if not str(env.get(key, "")).strip():
            env[key] = cap
    return n


def apply_thread_limits() -> int | None:
    """Load rnaomni.env defaults, fill unset BLAS env, cap torch if imported."""
    from rnaomni.paths import REPO
    from tools.user_env import load_user_env

    load_user_env(REPO)
    n = fill_thread_env(os.environ)
    if n is None:
        return None
    _limit_torch(n)
    return n


def _limit_torch(n: int) -> None:
    torch = sys.modules.get("torch")
    if torch is None:
        return
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
