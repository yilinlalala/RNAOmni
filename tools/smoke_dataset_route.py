#!/usr/bin/env python3
"""Smoke dataset route + rnaomni.env overlay (no full encode/train).

Usage (conda myenv3.11):
  python tools/smoke_dataset_route.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools.user_env import load_user_env, parse_env_file  # noqa: E402

NOISE_ROOT = REPO / "dataset" / "forcompare" / "Class_88fam_200%noise"


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("RNAOMNI_") or key in {
            "COMPARE_ROOT",
            "USE_COMPARE_SPLITS",
            "COMPARE_FULL",
            "COMPARE_NO_SAMPLE",
            "COMPARE_PER_CLASS",
            "COMPARE_PER_CLASS_TRAIN",
            "COMPARE_PER_CLASS_EVAL",
        }:
            env.pop(key, None)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _run(code: str, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = _base_env()
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO),
        env=env,
        text=True,
        capture_output=True,
    )


def test_default_hek_main() -> None:
    # Empty COMPARE_ROOT in the process env blocks repo `.env` from hijacking
    # the table-default HEK route.
    proc = _run(
        "import prepare; "
        "assert prepare.USE_COMPARE_SPLITS is False, prepare.USE_COMPARE_SPLITS; "
        "assert prepare.ACTIVE_CSV == 'HEK_te_prediction_task.csv', prepare.ACTIVE_CSV; "
        "assert prepare.TIME_BUDGET == 60, prepare.TIME_BUDGET; "
        "assert prepare.compare_sampling_on() is False, prepare.compare_sampling_on(); "
        "info = prepare.apply_dataset_route(); "
        "assert info['csv_name'] == 'HEK_te_prediction_task.csv', info; "
        "assert 'compare_' not in info['csv_name']; "
        "block = prepare.dataset_route_prompt_block(); "
        "assert 'HEK_te_prediction_task' in block, block; "
        "assert 'forcompare' not in block.lower() or 'Do not require forcompare' in block",
        extra_env={
            "RNAOMNI_COMPARE_ROOT": "",
            "RNAOMNI_USE_COMPARE_SPLITS": "0",
            "RNAOMNI_ACTIVE_CSV": "HEK_te_prediction_task.csv",
        },
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_repo_dotenv_200noise() -> None:
    overlay = parse_env_file(REPO / ".env")
    root = overlay.get("RNAOMNI_COMPARE_ROOT", "")
    assert "200%noise" in root, overlay
    proc = _run(
        "import prepare; "
        "assert prepare.USE_COMPARE_SPLITS is True, prepare.USE_COMPARE_SPLITS; "
        "stem = prepare.compare_active_csv_name(); "
        "assert stem == 'compare_Class_88fam_200%noise', stem; "
        "assert not stem.endswith('_explore'), stem; "
        "assert '200%noise' in stem, stem"
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_named_200noise_external() -> None:
    proc = _run(
        "import prepare; "
        "assert prepare.USE_COMPARE_SPLITS is True, prepare.USE_COMPARE_SPLITS; "
        "assert prepare.compare_sampling_on() is False; "
        "stem = prepare.compare_active_csv_name(); "
        "assert '200%noise' in stem, stem; "
        "assert not stem.endswith('_explore'), stem; "
        "assert stem == 'compare_Class_88fam_200%noise', stem; "
        "block = prepare.dataset_route_prompt_block(); "
        "assert '200%noise' in block, block",
        extra_env={"RNAOMNI_COMPARE_ROOT": str(NOISE_ROOT)},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_sample_mode_keeps_stem() -> None:
    proc = _run(
        "import prepare; "
        "from rnaomni.results_io import RESULTS_HEADER_CLASSIFICATION, RESULTS_HEADER_REGRESSION; "
        "assert '\\tdata_mode\\tstatus\\t' in RESULTS_HEADER_CLASSIFICATION; "
        "assert '\\tdata_mode\\tstatus\\t' in RESULTS_HEADER_REGRESSION; "
        "prepare.apply_data_mode('sample'); "
        "assert prepare.sampling_on() is True; "
        "assert prepare.compare_sampling_on() is True; "
        "stem = prepare.compare_active_csv_name(); "
        "assert stem == 'compare_Class_88fam_200%noise', stem; "
        "assert prepare.encode_timeout_sec() >= 0",
        extra_env={
            "RNAOMNI_COMPARE_ROOT": str(NOISE_ROOT),
            "RNAOMNI_DATA_MODE": "sample",
            "ENCODE_TIMEOUT_SEC": "0",
        },
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_missing_named_root_errors() -> None:
    proc = _run(
        "import prepare",
        extra_env={"RNAOMNI_COMPARE_ROOT": str(REPO / "dataset" / "forcompare" / "no_such_set")},
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    blob = proc.stdout + proc.stderr
    assert "train.csv" in blob and "val.csv" in blob, blob


def test_force_main_despite_named_root() -> None:
    proc = _run(
        "import prepare; "
        "prepare.apply_dataset_route(use_compare=False); "
        "assert prepare.USE_COMPARE_SPLITS is False, prepare.USE_COMPARE_SPLITS; "
        "assert prepare.ACTIVE_CSV == 'HEK_te_prediction_task.csv'",
        extra_env={"RNAOMNI_COMPARE_ROOT": str(NOISE_ROOT)},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_compare_root_cli_missing_errors() -> None:
    env = _base_env()
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "prepare.py"),
            "--compare-root",
            str(REPO / "dataset" / "forcompare" / "no_such_set"),
            "--skip-encode",
        ],
        cwd=str(REPO),
        env=env,
        text=True,
        capture_output=True,
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    blob = proc.stdout + proc.stderr
    assert "train.csv" in blob and "val.csv" in blob, blob


def test_overlay_and_shell_win() -> None:
    saved = {k: os.environ.get(k) for k in ("OPENCODE_MODEL", "RNAOMNI_COMPARE_ROOT", "RNAOMNI_ACTIVE_CSV")}
    for k in saved:
        os.environ.pop(k, None)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "rnaomni.env").write_text(
                "OPENCODE_MODEL=from-table\nRNAOMNI_ACTIVE_CSV=HEK_te_prediction_task.csv\n",
                encoding="utf-8",
            )
            (root / ".env").write_text(
                f"RNAOMNI_COMPARE_ROOT={NOISE_ROOT}\n",
                encoding="utf-8",
            )
            os.environ["OPENCODE_MODEL"] = "from-shell"
            merged = load_user_env(root)
            assert os.environ["OPENCODE_MODEL"] == "from-shell"
            assert os.environ["RNAOMNI_COMPARE_ROOT"] == str(NOISE_ROOT)
            assert merged["OPENCODE_MODEL"] == "from-table"
            table = parse_env_file(REPO / "rnaomni.env")
            assert table["RNAOMNI_ACTIVE_CSV"] == "HEK_te_prediction_task.csv"
            assert table["RNAOMNI_CONDA_ENV"] == "myenv3.11"
            assert table["TIME_BUDGET"] == "60"
            assert table["COMPARE_PER_CLASS_TRAIN"] == "10"
            assert table["ENCODE_TIMEOUT_SEC"] == "3600"
            assert table["SAMPLE_ROW_CAP_TRAIN"] == "1000"
            assert table["OVERNIGHT_HOURS"] == "6"
            assert table["MIN_DELTA_CLASSIFICATION"] == "3e-4"
            assert "RNAOMNI_COMPARE_ROOT" not in table
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main() -> int:
    if not NOISE_ROOT.is_dir():
        print(f"missing compare root: {NOISE_ROOT}")
        return 1
    test_overlay_and_shell_win()
    print("ok overlay + shell-wins")
    test_default_hek_main()
    print("ok default HEK main")
    test_repo_dotenv_200noise()
    print("ok repo .env 200%noise overlay")
    test_named_200noise_external()
    print("ok named 200%noise external")
    test_sample_mode_keeps_stem()
    print("ok sample mode keeps stem + data_mode header")
    test_force_main_despite_named_root()
    print("ok --no-compare equivalent forces main")
    test_missing_named_root_errors()
    print("ok missing named root errors")
    test_compare_root_cli_missing_errors()
    print("ok --compare-root missing errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
