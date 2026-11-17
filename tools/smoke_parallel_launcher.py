#!/usr/bin/env python3
"""Smoke D0 launcher: two stub orchestrators, dashboard, lock, cascade.

Does not run the real overnight loop or train.py.

Usage:
  python tools/smoke_parallel_launcher.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rnaomni.parallel import (  # noqa: E402
    DEFAULT_MAX_RUNNING_LANES,
    LockBusy,
    PoolLock,
    load_batch_spec,
    resolved_max_running_lanes,
    specs_from_payload,
)
from tools.parallel_lanes import LaneSpec, merge_lane_env  # noqa: E402
from tools.smoke_parallel_multidataset import NOISE_0, NOISE_200, _seed_git_repo  # noqa: E402

STUB_ORCH = r'''
#!/usr/bin/env python3
import json, os, sys, time
from pathlib import Path
from rnaomni.results_io import ensure_results_tsv

ROOT = Path(__file__).resolve().parent
(ROOT / "started.flag").write_text(f"{time.time()}\n", encoding="utf-8")
if os.environ.get("RNAOMNI_SMOKE_FAIL_DATASET") == os.environ.get("RNAOMNI_DATASET_ID"):
    time.sleep(0.05)
    sys.exit(1)
time.sleep(float(os.environ.get("RNAOMNI_SMOKE_SLEEP", "0.4")))
path = ensure_results_tsv("classification")
row = "abc1234\t0.500000\t0.400000\t0.300000\t0.1\tkeep\t[model] smoke lane\n"
header = path.read_text(encoding="utf-8")
path.write_text(header.rstrip("\n") + "\n" + row, encoding="utf-8")
live = ROOT / "workspace"
live.mkdir(parents=True, exist_ok=True)
(live / "orchestrator_state.json").write_text(
    json.dumps({"iteration": 1, "stop_reason": "smoke", "K_train": 10, "best": 0.5}) + "\n",
    encoding="utf-8",
)
(live / "orchestrator.log").write_text("smoke ok\n", encoding="utf-8")
'''

STUB_ARCHIVE = r'''
#!/usr/bin/env python3
import argparse, json, shutil
from pathlib import Path
from rnaomni.results_io import results_tsv_path

p = argparse.ArgumentParser()
p.add_argument("--tag", required=True)
p.add_argument("--stop-reason", default="")
p.add_argument("--csv", default=None)
p.add_argument("--dry-run", action="store_true")
args = p.parse_args()
src = results_tsv_path()
out = Path("workspace/runs") / f"{args.tag}_{src.stem}"
out.mkdir(parents=True, exist_ok=True)
if src.is_file():
    shutil.copy2(src, out / "results.tsv")
(out / "meta.json").write_text(
    json.dumps({
        "run_dir": out.name,
        "stop_reason": args.stop_reason,
        "task_type": "classification",
        "n_keep": 1,
        "best_metric": 0.5,
        "best_commit": "abc1234",
        "status": "ok",
    }) + "\n",
    encoding="utf-8",
)
print(f"archived: {out.resolve()}")
'''


def _commit(root: Path, message: str) -> None:
    subprocess.run(
        ["git", "add", "-A"],
        cwd=str(root),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", message],
        cwd=str(root),
        check=True,
        capture_output=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "smoke",
            "GIT_AUTHOR_EMAIL": "smoke@test",
            "GIT_COMMITTER_NAME": "smoke",
            "GIT_COMMITTER_EMAIL": "smoke@test",
        },
    )


def _prepare_repo(root: Path) -> None:
    _seed_git_repo(root)
    (root / "orchestrator.py").write_text(STUB_ORCH.lstrip() + "\n", encoding="utf-8")
    (root / "archive_run.py").write_text(STUB_ARCHIVE.lstrip() + "\n", encoding="utf-8")
    _commit(root, "stubs")


def test_spec_file() -> None:
    data = load_batch_spec(REPO / "tools" / "example_parallel_batch.yaml")
    specs, gpus, batch_id = specs_from_payload(data)
    assert batch_id == "example_d0"
    assert gpus == [0, 1]
    assert [s.dataset_id for s in specs] == ["cls_200noise", "cls_0noise"]
    print("spec file OK")


def test_conda_env_preserved() -> None:
    spec = LaneSpec("cls_0noise", NOISE_0)
    merged = merge_lane_env(
        {"RNAOMNI_CONDA_ENV": "myenv3.11", "RNAOMNI_COMPARE_ROOT": "hijack"},
        spec,
        1,
    )
    assert merged["RNAOMNI_CONDA_ENV"] == "myenv3.11"
    assert merged["RNAOMNI_COMPARE_ROOT"] == NOISE_0
    print("route-only env strip OK")


def _launch(
    git_root: Path,
    compare_root: str,
    extra_env: dict[str, str] | None = None,
    extra_args: list[str] | None = None,
) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("CUDA_AVAIL", None)
    env.pop("RNAOMNI_CUDA_AVAIL", None)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    env["RNAOMNI_SMOKE_SLEEP"] = extra_env.get("RNAOMNI_SMOKE_SLEEP", "0.8") if extra_env else "0.8"
    env["RNAOMNI_LANE_CAP_POLL_SEC"] = extra_env.get("RNAOMNI_LANE_CAP_POLL_SEC", "0.05") if extra_env else "0.05"
    if extra_env:
        env.update(extra_env)
    cmd = [
        sys.executable,
        "-m",
        "rnaomni.parallel",
        "--git-root",
        str(git_root),
        "--data-root",
        str(REPO),
        "--compare-root",
        compare_root,
        "--cleanup",
        *(extra_args or ()),
    ]
    return subprocess.Popen(
        cmd,
        cwd=str(REPO),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def test_submit_two_jobs() -> None:
    with tempfile.TemporaryDirectory(prefix="rnaomni_jobs_") as tmp:
        git_root = Path(tmp) / "repo"
        git_root.mkdir()
        _prepare_repo(git_root)
        a = _launch(git_root, NOISE_200)
        time.sleep(0.4)
        b = _launch(git_root, NOISE_0)
        out_a, _ = a.communicate(timeout=60)
        out_b, _ = b.communicate(timeout=60)
        assert a.returncode == 0, out_a
        assert b.returncode == 0, out_b
        dash = git_root / "workspace" / "batches" / "pool" / "dashboard.tsv"
        text = dash.read_text(encoding="utf-8")
        assert "compare_Class_88fam_200%noise" in text
        assert "compare_Class_88fam_0%noise" in text
        assert "compare_Class_88fam_200%noise_explore" not in text
        assert "compare_Class_88fam_0%noise_explore" not in text
        gpus = set()
        for ln in text.splitlines():
            if "compare_Class_88fam_" in ln and not ln.startswith("#"):
                gpus.add(ln.split("\t")[4])
        assert gpus == {"0"}, text
        print("submit two jobs OK")


def test_cycle_cuda_visible_devices() -> None:
    with tempfile.TemporaryDirectory(prefix="rnaomni_cycle_") as tmp:
        git_root = Path(tmp) / "repo"
        git_root.mkdir()
        _prepare_repo(git_root)
        cycle = {"CUDA_VISIBLE_DEVICES": "0,1", "RNAOMNI_SMOKE_SLEEP": "0.8"}
        a = _launch(git_root, NOISE_200, cycle)
        time.sleep(0.4)
        b = _launch(git_root, NOISE_0, cycle)
        out_a, _ = a.communicate(timeout=60)
        out_b, _ = b.communicate(timeout=60)
        assert a.returncode == 0, out_a
        assert b.returncode == 0, out_b
        dash = git_root / "workspace" / "batches" / "pool" / "dashboard.tsv"
        text = dash.read_text(encoding="utf-8")
        gpus = set()
        for ln in text.splitlines():
            if "compare_Class_88fam_" in ln and not ln.startswith("#"):
                gpus.add(ln.split("\t")[4])
        assert gpus == {"0", "1"}, text
        print("CUDA_VISIBLE_DEVICES cycle OK")


def test_third_wraps_and_fail_isolated() -> None:
    with tempfile.TemporaryDirectory(prefix="rnaomni_full_") as tmp:
        git_root = Path(tmp) / "repo"
        git_root.mkdir()
        _prepare_repo(git_root)
        unlimited = {"RNAOMNI_SMOKE_SLEEP": "1.2", "RNAOMNI_MAX_RUNNING_LANES": "0"}
        a = _launch(git_root, NOISE_200, unlimited)
        b = _launch(git_root, NOISE_0, unlimited)
        time.sleep(0.5)
        c = _launch(git_root, NOISE_200, {"RNAOMNI_SMOKE_SLEEP": "0.1", "RNAOMNI_MAX_RUNNING_LANES": "0"})
        out_c, _ = c.communicate(timeout=30)
        assert c.returncode == 0, out_c
        assert "gpu=0" in out_c, out_c
        a.wait(timeout=60)
        b.wait(timeout=60)
        assert a.returncode == 0
        assert b.returncode == 0
        print("third also gpu 0 OK")

    with tempfile.TemporaryDirectory(prefix="rnaomni_iso_") as tmp:
        git_root = Path(tmp) / "repo"
        git_root.mkdir()
        _prepare_repo(git_root)
        fail_env = {
            "RNAOMNI_SMOKE_FAIL_DATASET": "Class_88fam_200_noise",
            "RNAOMNI_SMOKE_SLEEP": "0.6",
            "RNAOMNI_MAX_RUNNING_LANES": "0",
        }
        a = _launch(git_root, NOISE_200, fail_env)
        b = _launch(git_root, NOISE_0, {"RNAOMNI_SMOKE_SLEEP": "0.6", "RNAOMNI_MAX_RUNNING_LANES": "0"})
        out_a, _ = a.communicate(timeout=60)
        out_b, _ = b.communicate(timeout=60)
        assert a.returncode == 1, out_a
        assert b.returncode == 0, out_b
        print("fail isolated OK")


def test_lock() -> None:
    with tempfile.TemporaryDirectory(prefix="rnaomni_lock_") as tmp:
        git_root = Path(tmp) / "repo"
        git_root.mkdir()
        lock = PoolLock(git_root)
        lock.acquire()
        try:
            helper = git_root / "try_lock.py"
            helper.write_text(
                "from pathlib import Path\n"
                "import sys\n"
                f"sys.path.insert(0, {str(REPO)!r})\n"
                "from rnaomni.parallel import PoolLock, LockBusy\n"
                "lk = PoolLock(Path(" + repr(str(git_root)) + "))\n"
                "try:\n"
                "    lk.acquire(blocking=False)\n"
                "except LockBusy:\n"
                "    raise SystemExit(3)\n"
                "raise SystemExit(0)\n",
                encoding="utf-8",
            )
            proc = subprocess.run(
                [sys.executable, str(helper)],
                cwd=str(REPO),
                capture_output=True,
                text=True,
            )
            assert proc.returncode == 3, proc.stdout + proc.stderr
        finally:
            lock.release()
        print("lock OK")


def test_resolved_max_running_lanes() -> None:
    assert resolved_max_running_lanes({}) == DEFAULT_MAX_RUNNING_LANES
    assert resolved_max_running_lanes({"RNAOMNI_MAX_RUNNING_LANES": "  "}) == DEFAULT_MAX_RUNNING_LANES
    assert resolved_max_running_lanes({"RNAOMNI_MAX_RUNNING_LANES": "1"}) == 1
    assert resolved_max_running_lanes({"RNAOMNI_MAX_RUNNING_LANES": "0"}) is None
    assert resolved_max_running_lanes({"RNAOMNI_MAX_RUNNING_LANES": "-3"}) is None
    print("resolved_max_running_lanes OK")


def test_lane_cap_no_wait() -> None:
    with tempfile.TemporaryDirectory(prefix="rnaomni_cap_nowait_") as tmp:
        git_root = Path(tmp) / "repo"
        git_root.mkdir()
        _prepare_repo(git_root)
        cap_env = {
            "RNAOMNI_SMOKE_SLEEP": "2.0",
            "RNAOMNI_MAX_RUNNING_LANES": "1",
            "RNAOMNI_LANE_CAP_POLL_SEC": "0.05",
        }
        a = _launch(git_root, NOISE_200, cap_env)
        time.sleep(0.8)
        b = _launch(
            git_root,
            NOISE_0,
            {
                "RNAOMNI_SMOKE_SLEEP": "0.1",
                "RNAOMNI_MAX_RUNNING_LANES": "1",
                "RNAOMNI_LANE_CAP_POLL_SEC": "0.05",
            },
            extra_args=["--no-wait"],
        )
        out_b, _ = b.communicate(timeout=20)
        assert b.returncode == 4, out_b
        assert "running lanes at cap" in out_b, out_b
        out_a, _ = a.communicate(timeout=60)
        assert a.returncode == 0, out_a
        print("lane cap --no-wait OK")


def test_lane_cap_waits() -> None:
    with tempfile.TemporaryDirectory(prefix="rnaomni_cap_wait_") as tmp:
        git_root = Path(tmp) / "repo"
        git_root.mkdir()
        _prepare_repo(git_root)
        cap_env = {
            "RNAOMNI_SMOKE_SLEEP": "0.8",
            "RNAOMNI_MAX_RUNNING_LANES": "1",
            "RNAOMNI_LANE_CAP_POLL_SEC": "0.05",
        }
        a = _launch(git_root, NOISE_200, cap_env)
        time.sleep(0.3)
        b = _launch(git_root, NOISE_0, cap_env)
        out_a, _ = a.communicate(timeout=60)
        out_b, _ = b.communicate(timeout=60)
        assert a.returncode == 0, out_a
        assert b.returncode == 0, out_b
        assert "lane cap:" in out_b, out_b
        print("lane cap wait OK")


def main() -> int:
    test_spec_file()
    test_conda_env_preserved()
    test_lock()
    test_resolved_max_running_lanes()
    test_submit_two_jobs()
    test_cycle_cuda_visible_devices()
    test_third_wraps_and_fail_isolated()
    test_lane_cap_no_wait()
    test_lane_cap_waits()
    print("smoke_parallel_launcher: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
