#!/usr/bin/env python3
"""Smoke Phase 1 parallel skeleton: two roots → two stems, isolated git, GPUs.

Does not run train.py or orchestrator.py.

Usage (conda myenv3.11):
  python tools/smoke_parallel_multidataset.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.parallel_lanes import (  # noqa: E402
    L4_WEIGHT_DIRS,
    LaneSpec,
    ParallelConfigError,
    assign_gpus,
    expected_stem,
    lane_env,
    link_l4_weights,
    merge_lane_env,
    parse_cuda_visible_devices,
    pick_next_gpu,
    provision_batch,
    teardown_batch,
)

NOISE_200 = "dataset/forcompare/Class_88fam_200%noise"
NOISE_0 = "dataset/forcompare/Class_88fam_0%noise"


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("RNAOMNI_") or key in {
            "COMPARE_ROOT",
            "USE_COMPARE_SPLITS",
            "COMPARE_FULL",
            "COMPARE_NO_SAMPLE",
            "CUDA_VISIBLE_DEVICES",
            "CUDA_AVAIL",
        }:
            env.pop(key, None)
    return env


def _seed_git_repo(root: Path) -> None:
    """Minimal git repo with current routing files; features via symlink."""
    (root / "tools").mkdir(parents=True)
    (root / "dataset").mkdir(parents=True)
    shutil.copy2(REPO / "prepare.py", root / "prepare.py")
    shutil.copy2(REPO / "rnaomni.env", root / "rnaomni.env")
    shutil.copy2(REPO / "tools" / "user_env.py", root / "tools" / "user_env.py")
    (root / "rnaomni").symlink_to(REPO / "rnaomni")
    (root / "features").symlink_to(REPO / "features")
    (root / "train.py").write_text("# smoke marker\nprint('lane')\n", encoding="utf-8")
    (root / "README.md").write_text("parallel smoke repo\n", encoding="utf-8")
    (root / ".gitignore").write_text(
        "\n".join(
            [
                ".env",
                "workspace/",
                "results.tsv",
                "orchestrator_state.json",
                "run.log",
                "worktrees/",
                ".parallel/",
                "__pycache__/",
                "",
            ]
        ),
        encoding="utf-8",
    )
    subprocess.run(["git", "init"], cwd=str(root), check=True, capture_output=True)
    subprocess.run(
        ["git", "add", "prepare.py", "rnaomni.env", "rnaomni", "tools", "train.py", "README.md", ".gitignore"],
        cwd=str(root),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "add", "features"],
        cwd=str(root),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "smoke seed"],
        cwd=str(root),
        check=True,
        capture_output=True,
        env={**os.environ, "GIT_AUTHOR_NAME": "smoke", "GIT_AUTHOR_EMAIL": "smoke@test",
             "GIT_COMMITTER_NAME": "smoke", "GIT_COMMITTER_EMAIL": "smoke@test"},
    )


def _run_in_lane(worktree: Path, spec: LaneSpec, gpu: int, code: str) -> subprocess.CompletedProcess[str]:
    env = merge_lane_env(_base_env(), spec, gpu)
    env["PYTHONPATH"] = str(worktree) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(worktree),
        env=env,
        text=True,
        capture_output=True,
    )


PROBE = r"""
import os, prepare
from pathlib import Path
from rnaomni.results_io import ensure_results_tsv, results_tsv_path, LEGACY_RESULTS_TSV
info = prepare.apply_dataset_route()
path = ensure_results_tsv("classification")
print("CSV", info["csv_name"])
print("ROOT", Path(info["compare_root"]).name)
print("STEM_FILE", path.name)
print("ACTIVE", os.path.relpath(path, Path('.').resolve()))
print("ROOT_POINTER", int(LEGACY_RESULTS_TSV.exists() or LEGACY_RESULTS_TSV.is_symlink()))
print("CUDA", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("USE", int(prepare.USE_COMPARE_SPLITS))
"""


def test_gpu_rules() -> None:
    assert assign_gpus(2) == [0, 0]
    assert assign_gpus(1, gpus=[3, 4]) == [3]
    assert assign_gpus(3) == [0, 0, 0]
    assert assign_gpus(2, requested=[0, 0]) == [0, 0]
    assert assign_gpus(4, gpus=[0, 1, 2, 3]) == [0, 1, 2, 3]
    assert pick_next_gpu([0], 0) == (0, 1)
    assert pick_next_gpu([0, 1], 0) == (0, 1)
    assert pick_next_gpu([0, 1], 2) == (0, 3)
    assert pick_next_gpu([0, 1], 1, requested=3) == (3, 1)
    saved = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    try:
        assert parse_cuda_visible_devices() == [2, 3]
        assert parse_cuda_visible_devices("0,1") == [0, 1]
    finally:
        if saved is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved
    print("gpu rules OK")


def test_expected_stems() -> None:
    assert expected_stem(NOISE_200, explore=True) == "compare_Class_88fam_200%noise"
    assert expected_stem(NOISE_0, explore=False) == "compare_Class_88fam_0%noise"
    env0 = lane_env(LaneSpec("cls_200noise", NOISE_200), 0)
    env1 = lane_env(LaneSpec("cls_0noise", NOISE_0), 1)
    assert env0["RNAOMNI_DATA_MODE"] == "full"
    assert "COMPARE_FULL" not in env0
    env0 = lane_env(LaneSpec("cls_200noise", NOISE_200), 0)
    env1 = lane_env(LaneSpec("cls_0noise", NOISE_0), 1)
    assert env0["RNAOMNI_COMPARE_ROOT"] != env1["RNAOMNI_COMPARE_ROOT"]
    assert env0["CUDA_VISIBLE_DEVICES"] == "0"
    assert env1["CUDA_VISIBLE_DEVICES"] == "1"
    print("stems/env OK")


def test_worktrees() -> None:
    lanes = [
        LaneSpec("cls_200noise", NOISE_200, explore=True),
        LaneSpec("cls_0noise", NOISE_0, explore=True),
    ]
    with tempfile.TemporaryDirectory(prefix="rnaomni_par_") as tmp:
        git_root = Path(tmp) / "repo"
        git_root.mkdir()
        _seed_git_repo(git_root)
        batch_id = "smoke1"
        payload = provision_batch(
            git_root,
            batch_id=batch_id,
            lanes=lanes,
            data_root=REPO,
            gpus=(0, 1),
        )
        try:
            rows = payload["lanes"]
            assert len(rows) == 2
            stems = []
            heads = []
            for spec, row, gpu in zip(lanes, rows, [0, 1]):
                wt = Path(row["worktree"])
                assert (wt / "dataset" / "forcompare").is_symlink()
                assert (wt / "dataset" / "forcompare" / Path(spec.compare_root).name / "train.csv").is_file()
                for rel_dir, required_name in L4_WEIGHT_DIRS:
                    linked = wt / rel_dir
                    assert linked.is_symlink(), linked
                    assert (linked / required_name).is_file(), linked / required_name
                    assert linked.resolve() == (REPO / rel_dir).resolve()
                assert (wt / ".env").is_file()
                proc = _run_in_lane(wt, spec, gpu, PROBE)
                assert proc.returncode == 0, proc.stdout + proc.stderr
                parsed = dict(
                    line.split(" ", 1) for line in proc.stdout.strip().splitlines() if " " in line
                )
                assert parsed["CSV"] == expected_stem(spec.compare_root, explore=True)
                assert parsed["CUDA"] == str(gpu)
                assert parsed["USE"] == "1"
                assert parsed["ROOT_POINTER"] == "0"
                assert parsed["ACTIVE"].startswith("workspace/compare_")
                assert str(REPO) not in parsed["ACTIVE"]
                stems.append(parsed["CSV"])
                head = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=str(wt), text=True
                ).strip()
                heads.append(head)

            assert stems[0] != stems[1], stems
            wt0 = Path(rows[0]["worktree"])
            wt1 = Path(rows[1]["worktree"])
            marker = wt0 / "train.py"
            marker.write_text("# lane0 edit\n", encoding="utf-8")
            subprocess.run(["git", "add", "train.py"], cwd=str(wt0), check=True, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", "lane0"],
                cwd=str(wt0),
                check=True,
                capture_output=True,
                env={**os.environ, "GIT_AUTHOR_NAME": "smoke", "GIT_AUTHOR_EMAIL": "smoke@test",
                     "GIT_COMMITTER_NAME": "smoke", "GIT_COMMITTER_EMAIL": "smoke@test"},
            )
            head0_after = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(wt0), text=True
            ).strip()
            head1_after = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(wt1), text=True
            ).strip()
            assert head0_after != heads[0]
            assert head1_after == heads[1]
            subprocess.run(
                ["git", "reset", "--hard", heads[0]],
                cwd=str(wt0),
                check=True,
                capture_output=True,
            )
            head1_post_reset = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(wt1), text=True
            ).strip()
            assert head1_post_reset == heads[1]
            print("worktrees OK:", stems)
        finally:
            teardown_batch(git_root, batch_id)


def test_l4_weight_links_missing() -> None:
    with tempfile.TemporaryDirectory(prefix="rnaomni_l4_") as tmp:
        root = Path(tmp)
        wt = root / "wt"
        wt.mkdir()
        try:
            link_l4_weights(root, wt)
        except ParallelConfigError as err:
            assert "missing L4 weight" in str(err)
            assert "no download" in str(err)
        else:
            raise AssertionError("expected ParallelConfigError")
    print("l4 missing weight OK")


def test_parent_overlay_does_not_hijack() -> None:
    """Launcher env beats a parent 200%noise overlay."""
    spec = LaneSpec("cls_0noise", NOISE_0, explore=True)
    base = _base_env()
    base["RNAOMNI_COMPARE_ROOT"] = str(REPO / NOISE_200)
    merged = merge_lane_env(base, spec, 1)
    assert merged["RNAOMNI_COMPARE_ROOT"] == NOISE_0
    assert merged["CUDA_VISIBLE_DEVICES"] == "1"
    print("overlay hijack blocked OK")


def main() -> int:
    missing = [
        REPO / p / "train.csv"
        for p in (NOISE_200, NOISE_0)
    ]
    for path in missing:
        assert path.is_file(), f"need {path}"
    for rel_dir, required_name in L4_WEIGHT_DIRS:
        path = REPO / rel_dir / required_name
        assert path.is_file(), f"need {path}"
    test_gpu_rules()
    test_expected_stems()
    test_l4_weight_links_missing()
    test_parent_overlay_does_not_hijack()
    test_worktrees()
    print("smoke_parallel_multidataset: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
