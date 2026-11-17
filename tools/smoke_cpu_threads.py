#!/usr/bin/env python3
"""Smoke 15.4a: BLAS/OpenMP env fill; do not overwrite user exports."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rnaomni.threads import (  # noqa: E402
    DEFAULT_CPU_THREADS,
    THREAD_ENV_KEYS,
    apply_thread_limits,
    ensure_conda_lib_ld_path,
    fill_thread_env,
    resolved_cpu_threads,
)


def test_fill_default() -> None:
    env: dict[str, str] = {}
    n = fill_thread_env(env)
    assert n == DEFAULT_CPU_THREADS
    for key in THREAD_ENV_KEYS:
        assert env[key] == str(DEFAULT_CPU_THREADS), env
    print("fill default OK")


def test_fill_reuses_existing_omp() -> None:
    env = {"OMP_NUM_THREADS": "8"}
    n = fill_thread_env(env)
    assert n == 8
    assert env["OMP_NUM_THREADS"] == "8"
    assert env["MKL_NUM_THREADS"] == "8"
    assert env["OPENBLAS_NUM_THREADS"] == "8"
    print("fill reuse OMP OK")


def test_fill_does_not_clobber_omp() -> None:
    env = {"RNAOMNI_CPU_THREADS": "2", "OMP_NUM_THREADS": "8"}
    n = fill_thread_env(env)
    assert n == 2
    assert env["OMP_NUM_THREADS"] == "8"
    assert env["MKL_NUM_THREADS"] == "2"
    print("fill keep user OMP OK")


def test_fill_skip() -> None:
    env = {"RNAOMNI_CPU_THREADS": "0"}
    n = fill_thread_env(env)
    assert n is None
    assert "OMP_NUM_THREADS" not in env
    print("fill skip OK")


def test_ld_library_path_prepend() -> None:
    env: dict[str, str] = {"LD_LIBRARY_PATH": "/other"}
    lib = ensure_conda_lib_ld_path(env)
    assert lib
    assert env["LD_LIBRARY_PATH"].startswith(lib + ":")
    assert env["LD_LIBRARY_PATH"].endswith("/other")
    # idempotent
    ensure_conda_lib_ld_path(env)
    assert env["LD_LIBRARY_PATH"].count(lib) == 1
    env2 = {"RNAOMNI_CONDA_LIB": "0"}
    assert ensure_conda_lib_ld_path(env2) == ""
    assert "LD_LIBRARY_PATH" not in env2
    print("ld_library_path prepend OK")


def test_resolved_blank_rnaomni_uses_default() -> None:
    assert resolved_cpu_threads({"RNAOMNI_CPU_THREADS": "  "}) == DEFAULT_CPU_THREADS
    print("resolved blank OK")


def test_apply_sets_os_environ_when_unset() -> None:
    saved = {k: os.environ.get(k) for k in ("RNAOMNI_CPU_THREADS", *THREAD_ENV_KEYS)}
    try:
        for k in ("RNAOMNI_CPU_THREADS", *THREAD_ENV_KEYS):
            os.environ.pop(k, None)
        n = apply_thread_limits()
        assert n == DEFAULT_CPU_THREADS
        assert os.environ["OMP_NUM_THREADS"] == str(DEFAULT_CPU_THREADS)
        torch = sys.modules.get("torch")
        if torch is not None:
            assert torch.get_num_threads() == DEFAULT_CPU_THREADS
        print("apply_thread_limits OK", n)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def main() -> int:
    test_fill_default()
    test_fill_reuses_existing_omp()
    test_fill_does_not_clobber_omp()
    test_fill_skip()
    test_ld_library_path_prepend()
    test_resolved_blank_rnaomni_uses_default()
    test_apply_sets_os_environ_when_unset()
    print("smoke_cpu_threads: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
