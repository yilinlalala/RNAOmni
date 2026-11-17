#!/usr/bin/env python3
"""Submit one dataset job per invocation.

  python -m rnaomni.parallel --compare-root dataset/forcompare/Class_88fam_200%noise

GPU comes from CUDA_VISIBLE_DEVICES or --gpu. Unset → GPU 0.
Running-lane cap: ``RNAOMNI_MAX_RUNNING_LANES`` (default 2). ``0`` / negative =
unlimited. At cap this process waits for a slot, or ``--no-wait`` refuses.
GPU occupancy is still not checked.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from collections.abc import MutableMapping
from pathlib import Path

from tools.parallel_lanes import (
    DEFAULT_GPUS,
    LaneSpec,
    ParallelConfigError,
    dashboard_path,
    expected_stem,
    global_lock_path,
    merge_lane_env,
    parallel_dir,
    parse_cuda_visible_devices,
    pick_next_gpu,
    pool_dashboard_path,
    pool_path,
    provision_batch,
    slug_dataset_id,
    teardown_batch,
    update_manifest,
)
from rnaomni.paths import REPO
from rnaomni.threads import fill_thread_env
from tools.user_env import load_user_env

load_user_env(REPO)

DASHBOARD_HEADER = (
    "dataset_id\tstem\ttask_type\tbranch\tgpu\tstatus\t"
    "best_metric\tbest_commit\tn_keep\tK_train\tstop_reason\trun_dir\n"
)

# Not a leaderboard. Rows stay in launch order.
DASHBOARD_NOTE = (
    "# dashboard only — do not rank best_metric across dataset_id / task_type\n"
)


DEFAULT_MAX_RUNNING_LANES = 2
DEFAULT_LANE_CAP_POLL_SEC = 1.0


class LockBusy(RuntimeError):
    """Pool lock is held (only used with non-blocking acquire)."""


class LaneCapFull(RuntimeError):
    """Running-lane cap reached and ``--no-wait`` was set."""


class PoolLock:
    """Short exclusive flock on <git_root>/.parallel/LOCK.

    Held only while allocating or freeing a GPU, never for the overnight loop.
    """

    def __init__(self, git_root: Path):
        self.git_root = git_root
        self.path = global_lock_path(git_root)
        self._fd = None

    def acquire(self, *, blocking: bool = True) -> None:
        import fcntl

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = open(self.path, "a+", encoding="utf-8")
        flags = fcntl.LOCK_EX if blocking else (fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            fcntl.flock(self._fd.fileno(), flags)
        except BlockingIOError as err:
            self._fd.close()
            self._fd = None
            raise LockBusy(f"pool lock busy ({self.path})") from err

    def release(self) -> None:
        import fcntl

        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd.fileno(), fcntl.LOCK_UN)
        finally:
            self._fd.close()
            self._fd = None

    def __enter__(self) -> "PoolLock":
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def _empty_pool(gpus: list[int]) -> dict:
    return {"gpus": list(gpus), "jobs": [], "rr_cursor": 0}


def load_pool(git_root: Path, gpus: list[int]) -> dict:
    path = pool_path(git_root)
    if not path.is_file():
        return _empty_pool(gpus)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return _empty_pool(gpus)
    if not isinstance(data, dict):
        return _empty_pool(gpus)
    data.setdefault("jobs", [])
    data.setdefault("gpus", list(gpus))
    data.setdefault("rr_cursor", 0)
    return data


def save_pool(git_root: Path, pool: dict) -> None:
    path = pool_path(git_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pool, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def reap_pool(pool: dict) -> None:
    for job in pool.get("jobs") or []:
        if job.get("status") != "running":
            continue
        if pid_alive(job.get("pid")):
            continue
        job["status"] = "dead"
        job["pid"] = None
        job["stop_reason"] = job.get("stop_reason") or "pid_gone"


def running_jobs(pool: dict) -> list[dict]:
    return [j for j in (pool.get("jobs") or []) if j.get("status") == "running"]


def resolved_max_running_lanes(env: MutableMapping[str, str] | None = None) -> int | None:
    """Return the running-job cap, or None for unlimited.

    Unset / blank → 2. ``0`` / negative → unlimited (2026-08-13 oversubscribe).
    """
    src = os.environ if env is None else env
    raw = src.get("RNAOMNI_MAX_RUNNING_LANES")
    if raw is None or str(raw).strip() == "":
        return DEFAULT_MAX_RUNNING_LANES
    n = int(str(raw).strip())
    if n <= 0:
        return None
    return n


def lane_cap_poll_sec(env: MutableMapping[str, str] | None = None) -> float:
    src = os.environ if env is None else env
    raw = src.get("RNAOMNI_LANE_CAP_POLL_SEC")
    if raw is None or str(raw).strip() == "":
        return DEFAULT_LANE_CAP_POLL_SEC
    return max(0.05, float(str(raw).strip()))


def _cap_message(n_run: int, cap: int) -> str:
    return (
        f"running lanes at cap ({n_run}/{cap}); "
        "wait (default) or python -m rnaomni.parallel --status"
    )


def _reap_running_count(git_root: Path, gpus: list[int]) -> tuple[dict, int]:
    pool = load_pool(git_root, gpus)
    reap_pool(pool)
    save_pool(git_root, pool)
    return pool, len(running_jobs(pool))


def wait_for_lane_slot(
    git_root: Path,
    gpus: list[int],
    *,
    no_wait: bool = False,
) -> int:
    """Block until running < cap (or return immediately if unlimited).

    Releases the pool lock while sleeping so finish_job can reap. ``no_wait``
    refuses instead of blocking.
    """
    announced = False
    poll = lane_cap_poll_sec()
    while True:
        cap = resolved_max_running_lanes()
        if cap is None:
            return 0
        with PoolLock(git_root):
            _pool, n_run = _reap_running_count(git_root, gpus)
            if n_run < cap:
                return n_run
            if no_wait:
                raise LaneCapFull(_cap_message(n_run, cap))
        if not announced:
            print(
                f"lane cap: running {n_run}/{cap} — waiting for a slot "
                "(python -m rnaomni.parallel --status)",
                flush=True,
            )
            announced = True
        time.sleep(poll)


def _parse_gpu_list(raw: str) -> list[int]:
    parts = [p.strip() for p in str(raw).replace(";", ",").split(",")]
    return [int(p) for p in parts if p]


def load_batch_spec(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = _load_yaml_mapping(path, text)
    if not isinstance(data, dict):
        raise ParallelConfigError(f"batch spec is not a mapping: {path}")
    if not isinstance(data.get("lanes"), list):
        data = _parse_batch_yaml(text)
    return data


def _load_yaml_mapping(path: Path, text: str) -> dict:
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        return loaded if isinstance(loaded, dict) else {}
    except ImportError:
        return _parse_batch_yaml(text)


def _parse_scalar(value: str):
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        value = value[1:-1]
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in ("null", "~", ""):
        return None
    if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
        return int(value)
    return value


def _parse_batch_yaml(text: str) -> dict:
    """Minimal batch-file parser (no PyYAML required)."""
    data: dict = {"lanes": []}
    current: dict | None = None
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        s = raw.strip()
        if s.startswith("- "):
            current = {}
            data["lanes"].append(current)
            rest = s[2:].strip()
            if ":" in rest:
                key, val = rest.split(":", 1)
                current[key.strip()] = _parse_scalar(val)
            continue
        if indent > 0 and current is not None and ":" in s:
            key, val = s.split(":", 1)
            current[key.strip()] = _parse_scalar(val)
            continue
        if ":" not in s:
            continue
        key, val = s.split(":", 1)
        key = key.strip()
        val = val.strip()
        if key == "lanes":
            continue
        if key in ("gpus", "cuda_avail", "cuda_visible_devices"):
            inner = val.strip().lstrip("[").rstrip("]")
            data["gpus"] = [int(x.strip()) for x in inner.split(",") if x.strip()]
        else:
            data[key] = _parse_scalar(val)
    return data


def specs_from_payload(data: dict) -> tuple[list[LaneSpec], list[int], str | None]:
    lanes_raw = data.get("lanes") or []
    if not lanes_raw:
        raise ParallelConfigError("batch spec has no lanes")
    specs: list[LaneSpec] = []
    for row in lanes_raw:
        if not isinstance(row, dict):
            raise ParallelConfigError(f"lane is not a mapping: {row!r}")
        specs.append(
            LaneSpec(
                dataset_id=str(row["dataset_id"]),
                compare_root=str(row["compare_root"]),
                explore=bool(row.get("explore", True)),
                gpu=int(row["gpu"]) if row.get("gpu") is not None else None,
                active_csv=str(row["active_csv"]) if row.get("active_csv") else None,
                fold=int(row["fold"]) if row.get("fold") is not None else None,
            )
        )
    raw_gpus = data.get("cuda_visible_devices", data.get("cuda_avail", data.get("gpus")))
    gpus: list[int] | None
    if raw_gpus is None:
        gpus = None
    elif isinstance(raw_gpus, str):
        gpus = _parse_gpu_list(raw_gpus)
    else:
        gpus = [int(g) for g in raw_gpus]
    batch_id = data.get("batch_id")
    return specs, gpus, str(batch_id) if batch_id else None


def default_batch_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def explore_for_compare_root(
    compare_root: str, *, formal: bool, fold: int | None = None
) -> bool:
    """Deprecated. Campaigns start full; sampling is orchestrator-owned."""
    return False


def _kill_process_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            return
    deadline = time.time() + 8.0
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            return
        time.sleep(0.1)
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _lane_log(worktree: Path) -> Path:
    log_dir = worktree / "workspace"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "launcher_orchestrator.stdout.log"


def _lane_state(worktree: Path) -> Path:
    return worktree / "workspace" / "orchestrator_state.json"


def spawn_lane(
    *,
    git_root: Path,
    worktree: Path,
    spec: LaneSpec,
    gpu: int,
    batch_id: str,
    extra_args: list[str],
) -> "subprocess.Popen[str]":
    import subprocess

    orch = worktree / "orchestrator.py"
    if not orch.is_file():
        raise ParallelConfigError(f"missing {orch}")
    env = merge_lane_env(os.environ.copy(), spec, gpu)
    env["PYTHONPATH"] = str(worktree) + os.pathsep + env.get("PYTHONPATH", "")
    env["RNAOMNI_DATASET_ID"] = spec.dataset_id
    env["RNAOMNI_BATCH_ID"] = batch_id
    fill_thread_env(env)
    cmd = [
        sys.executable,
        str(orch),
        "--log-file",
        str(worktree / "workspace" / "orchestrator.log"),
        *extra_args,
    ]
    log = _lane_log(worktree)
    handle = log.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(worktree),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    proc._launcher_log = handle  # type: ignore[attr-defined]
    return proc


def _close_lane_log(proc: object) -> None:
    handle = getattr(proc, "_launcher_log", None)
    if handle is not None:
        try:
            handle.close()
        except OSError:
            pass


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _task_type_from_results(path: Path) -> str:
    if not path.is_file():
        return ""
    first = path.read_text(encoding="utf-8").splitlines()[:1]
    if not first:
        return ""
    if "val_accuracy" in first[0]:
        return "classification"
    if "val_spearman" in first[0] or "val_RMSE" in first[0] or "\tRMSE\t" in first[0]:
        return "regression"
    return ""


def _newest_run_dir(worktree: Path) -> Path | None:
    runs = worktree / "workspace" / "runs"
    if not runs.is_dir():
        return None
    dirs = [p for p in runs.iterdir() if p.is_dir() and p.name != "INDEX.tsv"]
    if not dirs:
        return None
    return max(dirs, key=lambda p: p.stat().st_mtime)


def archive_lane(
    worktree: Path,
    *,
    batch_id: str,
    dataset_id: str,
    stop_reason: str,
    env: dict[str, str],
) -> Path | None:
    import subprocess

    archive_py = worktree / "archive_run.py"
    tag = f"{batch_id}__{dataset_id}"
    run_env = dict(env)
    run_env["PYTHONPATH"] = str(worktree) + os.pathsep + run_env.get("PYTHONPATH", "")
    if archive_py.is_file():
        cmd = [sys.executable, str(archive_py), "--tag", tag, "--stop-reason", stop_reason]
    else:
        cmd = [
            sys.executable,
            "-m",
            "rnaomni.archive",
            "--tag",
            tag,
            "--stop-reason",
            stop_reason,
        ]
    proc = subprocess.run(
        cmd,
        cwd=str(worktree),
        env=run_env,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        return _newest_run_dir(worktree)
    return _newest_run_dir(worktree)


def dashboard_row(lane: dict, worktree: Path) -> dict[str, str]:
    state = _read_json(_lane_state(worktree))
    stem = lane.get("stem") or ""
    results = worktree / "workspace" / f"{stem}.tsv"
    if not results.is_file():
        results = worktree / "results.tsv"
    task_type = _task_type_from_results(results if results.is_file() else Path())
    meta = {}
    run_dir = lane.get("run_dir") or ""
    if run_dir:
        meta_path = Path(run_dir) / "meta.json"
        if not meta_path.is_file() and not Path(run_dir).is_absolute():
            meta_path = worktree / run_dir / "meta.json"
        meta = _read_json(meta_path)
    best_metric = meta.get("best_metric", "")
    if best_metric == "" and state.get("best") is not None:
        best_metric = state.get("best")
    best_commit = meta.get("best_commit") or ""
    n_keep = meta.get("n_keep", "")
    stop_reason = (
        lane.get("stop_reason")
        or meta.get("stop_reason")
        or state.get("stop_reason")
        or ""
    )
    k_train = state.get("K_train", state.get("k_train", ""))
    return {
        "dataset_id": str(lane.get("dataset_id") or ""),
        "stem": stem,
        "task_type": str(meta.get("task_type") or task_type),
        "branch": str(lane.get("branch") or ""),
        "gpu": str(lane.get("gpu") if lane.get("gpu") is not None else ""),
        "status": str(lane.get("status") or ""),
        "best_metric": "" if best_metric in (None, "") else str(best_metric),
        "best_commit": str(best_commit or ""),
        "n_keep": "" if n_keep in (None, "") else str(n_keep),
        "K_train": "" if k_train in (None, "") else str(k_train),
        "stop_reason": str(stop_reason or ""),
        "run_dir": str(run_dir or ""),
    }


def write_dashboard(git_root: Path, batch_id: str, payload: dict) -> Path:
    path = dashboard_path(git_root, batch_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [DASHBOARD_NOTE, DASHBOARD_HEADER]
    for lane in payload.get("lanes") or []:
        wt = Path(lane["worktree"])
        row = dashboard_row(lane, wt)
        lines.append(
            "\t".join(
                [
                    row["dataset_id"],
                    row["stem"],
                    row["task_type"],
                    row["branch"],
                    row["gpu"],
                    row["status"],
                    row["best_metric"],
                    row["best_commit"],
                    row["n_keep"],
                    row["K_train"],
                    row["stop_reason"],
                    row["run_dir"],
                ]
            )
            + "\n"
        )
    path.write_text("".join(lines), encoding="utf-8")
    copy = parallel_dir(git_root, batch_id) / "dashboard.tsv"
    copy.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return path


def write_pool_dashboard(git_root: Path, pool: dict) -> Path:
    path = pool_dashboard_path(git_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [DASHBOARD_NOTE, DASHBOARD_HEADER]
    for job in pool.get("jobs") or []:
        wt = Path(job.get("worktree") or "")
        row = dashboard_row(job, wt) if wt else {
            "dataset_id": str(job.get("dataset_id") or ""),
            "stem": str(job.get("stem") or ""),
            "task_type": "",
            "branch": str(job.get("branch") or ""),
            "gpu": str(job.get("gpu") if job.get("gpu") is not None else ""),
            "status": str(job.get("status") or ""),
            "best_metric": "",
            "best_commit": "",
            "n_keep": "",
            "K_train": "",
            "stop_reason": str(job.get("stop_reason") or ""),
            "run_dir": str(job.get("run_dir") or ""),
        }
        lines.append(
            "\t".join(
                [
                    row["dataset_id"],
                    row["stem"],
                    row["task_type"],
                    row["branch"],
                    row["gpu"],
                    row["status"],
                    row["best_metric"],
                    row["best_commit"],
                    row["n_keep"],
                    row["K_train"],
                    row["stop_reason"],
                    row["run_dir"],
                ]
            )
            + "\n"
        )
    path.write_text("".join(lines), encoding="utf-8")
    return path


def _assert_lane_root(worktree: Path, spec: LaneSpec, gpu: int) -> None:
    import subprocess

    env = merge_lane_env(os.environ.copy(), spec, gpu)
    env["PYTHONPATH"] = str(worktree) + os.pathsep + env.get("PYTHONPATH", "")
    code = (
        "import os, prepare; "
        "info = prepare.apply_dataset_route(); "
        "print(info['csv_name']); "
        "print(os.path.basename(info['compare_root']))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(worktree),
        env=env,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise ParallelConfigError(
            f"lane {spec.dataset_id} route probe failed: {proc.stderr or proc.stdout}"
        )
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    want_stem = expected_stem(spec.compare_root, explore=spec.explore)
    if lines[0] != want_stem:
        raise ParallelConfigError(
            f"lane {spec.dataset_id} stem {lines[0]!r} != {want_stem!r}"
        )
    want_root = Path(spec.compare_root).name
    if lines[1] != want_root:
        raise ParallelConfigError(
            f"lane {spec.dataset_id} root {lines[1]!r} != {want_root!r}"
        )


def make_job_id(dataset_id: str) -> str:
    base = f"{time.strftime('%Y%m%d_%H%M%S')}_{dataset_id}"
    return base[:64]


def start_job(
    *,
    git_root: Path,
    spec: LaneSpec,
    gpus: list[int],
    extra_args: list[str],
    data_root: Path | None = None,
    start_ref: str = "HEAD",
    job_id: str | None = None,
    requested_gpu: int | None = None,
    no_wait: bool = False,
) -> dict:
    """Provision one worktree and spawn orchestrator. Does not wait for the job.

    Blocks (default) or refuses (``no_wait``) while the pool already has
    ``RNAOMNI_MAX_RUNNING_LANES`` jobs with status==running.
    """
    git_root = git_root.resolve()
    data_root = (data_root or git_root).resolve()
    while True:
        wait_for_lane_slot(git_root, gpus, no_wait=no_wait)
        with PoolLock(git_root):
            pool = load_pool(git_root, gpus)
            reap_pool(pool)
            cap = resolved_max_running_lanes()
            n_run = len(running_jobs(pool))
            if cap is not None and n_run >= cap:
                save_pool(git_root, pool)
                if no_wait:
                    raise LaneCapFull(_cap_message(n_run, cap))
                continue
            avail = list(gpus) or list(DEFAULT_GPUS)
            pool["gpus"] = avail
            cursor = int(pool.get("rr_cursor") or 0)
            gpu, next_cursor = pick_next_gpu(avail, cursor, requested=requested_gpu)
            if requested_gpu is None:
                pool["rr_cursor"] = next_cursor
            jid = job_id or make_job_id(spec.dataset_id)
            if parallel_dir(git_root, jid).exists():
                jid = f"{jid}_{os.getpid()}"[:64]
            pinned = LaneSpec(
                dataset_id=spec.dataset_id,
                compare_root=spec.compare_root,
                explore=spec.explore,
                gpu=gpu,
                active_csv=spec.active_csv,
                fold=spec.fold,
            )
            payload = provision_batch(
                git_root,
                batch_id=jid,
                lanes=[pinned],
                data_root=data_root,
                gpus=avail,
                start_ref=start_ref,
            )
            lane = payload["lanes"][0]
            wt = Path(lane["worktree"])
            try:
                _assert_lane_root(wt, pinned, gpu)
                print(
                    f"submit {pinned.dataset_id} job={jid} gpu={gpu} "
                    f"root={pinned.compare_root} stem={lane['stem']}",
                    flush=True,
                )
                proc = spawn_lane(
                    git_root=git_root,
                    worktree=wt,
                    spec=pinned,
                    gpu=gpu,
                    batch_id=jid,
                    extra_args=extra_args,
                )
            except Exception:
                teardown_batch(git_root, jid)
                raise
            lane["pid"] = proc.pid
            lane["status"] = "running"
            lane["job_id"] = jid
            update_manifest(git_root, jid, payload)
            pool["jobs"].append(dict(lane))
            save_pool(git_root, pool)
            write_pool_dashboard(git_root, pool)
            write_dashboard(git_root, jid, payload)
        return {
            "job_id": jid,
            "spec": pinned,
            "lane": lane,
            "proc": proc,
            "payload": payload,
            "gpu": gpu,
        }


def finish_job(
    *,
    git_root: Path,
    live: dict,
    gpus: list[int],
    cleanup: bool = False,
) -> dict:
    """Wait for one submitted job and archive. Does not reserve or free GPUs."""
    proc = live["proc"]
    spec = live["spec"]
    jid = live["job_id"]
    lane = live["lane"]
    payload = live["payload"]
    rc = proc.wait()
    _close_lane_log(proc)
    wt = Path(lane["worktree"])
    lane["pid"] = None
    lane["exit_code"] = rc
    if rc == 0:
        lane["status"] = "ok"
        stop_reason = (
            _read_json(_lane_state(wt)).get("stop_reason") or "stopped"
        )
        lane["stop_reason"] = stop_reason
        run_dir = archive_lane(
            wt,
            batch_id=jid,
            dataset_id=spec.dataset_id,
            stop_reason=str(stop_reason),
            env=merge_lane_env(os.environ.copy(), spec, int(lane["gpu"])),
        )
        lane["run_dir"] = str(run_dir) if run_dir else ""
    else:
        lane["status"] = "failed"
        lane["stop_reason"] = f"orchestrator_exit_{rc}"
    payload["lanes"] = [lane]
    update_manifest(git_root, jid, payload)
    write_dashboard(git_root, jid, payload)
    with PoolLock(git_root):
        pool = load_pool(git_root, gpus)
        for job in pool.get("jobs") or []:
            if job.get("job_id") == jid or (
                job.get("worktree") == lane.get("worktree") and job.get("status") == "running"
            ):
                job.update(lane)
                break
        save_pool(git_root, pool)
        dash = write_pool_dashboard(git_root, pool)
    print(f"job {jid} status={lane['status']} gpu={lane['gpu']} dashboard={dash}", flush=True)
    if cleanup:
        teardown_batch(git_root, jid)
    payload["failed"] = rc != 0
    return payload


def _kill_in_flight(lives: list[dict]) -> None:
    for live in lives:
        proc = live.get("proc")
        pid = getattr(proc, "pid", None) if proc is not None else None
        if pid and proc.poll() is None:  # type: ignore[attr-defined]
            _kill_process_group(pid)
        _close_lane_log(proc)


def _first_finished_index(lives: list[dict]) -> int | None:
    for i, live in enumerate(lives):
        proc = live.get("proc")
        if proc is not None and proc.poll() is not None:
            return i
    return None


def run_batch(
    *,
    git_root: Path,
    batch_id: str,
    specs: list[LaneSpec],
    gpus: list[int],
    extra_args: list[str],
    data_root: Path | None = None,
    start_ref: str = "HEAD",
    dry_run: bool = False,
    cleanup: bool = False,
    no_wait: bool = False,
) -> dict:
    """Start jobs under the running-lane cap, then wait. No cross-job kill."""
    if dry_run:
        assigned_preview = {
            "batch_id": batch_id,
            "gpus": gpus,
            "max_running_lanes": resolved_max_running_lanes(),
            "lanes": [
                {
                    "dataset_id": s.dataset_id,
                    "compare_root": s.compare_root,
                    "explore": s.explore,
                    "stem": expected_stem(s.compare_root, explore=s.explore),
                }
                for s in specs
            ],
        }
        print(json.dumps(assigned_preview, indent=2))
        return assigned_preview

    pending = list(specs)
    in_flight: list[dict] = []
    payloads: list[dict] = []
    failed = False
    try:
        while pending or in_flight:
            cap = resolved_max_running_lanes()
            while pending:
                if cap is not None and len(in_flight) >= cap:
                    break
                spec = pending[0]
                in_flight.append(
                    start_job(
                        git_root=git_root,
                        spec=spec,
                        gpus=gpus,
                        extra_args=extra_args,
                        data_root=data_root,
                        start_ref=start_ref,
                        requested_gpu=spec.gpu,
                        no_wait=no_wait,
                    )
                )
                pending.pop(0)
            if pending and no_wait and cap is not None and len(in_flight) >= cap:
                raise LaneCapFull(
                    _cap_message(len(in_flight), cap)
                    + f"; {len(pending)} job(s) not started"
                )
            if not in_flight:
                break
            idx = _first_finished_index(in_flight)
            while idx is None:
                time.sleep(min(0.2, lane_cap_poll_sec()))
                idx = _first_finished_index(in_flight)
            live = in_flight.pop(idx)
            payloads.append(
                finish_job(
                    git_root=git_root,
                    live=live,
                    gpus=gpus,
                    cleanup=cleanup,
                )
            )
            failed = failed or bool(payloads[-1].get("failed"))
    except Exception:
        _kill_in_flight(in_flight)
        raise
    return {"batch_id": batch_id, "lanes": [p["lanes"][0] for p in payloads], "failed": failed}


def print_status(git_root: Path, gpus: list[int]) -> None:
    pool = load_pool(git_root, gpus)
    reap_pool(pool)
    live = running_jobs(pool)
    cap = resolved_max_running_lanes()
    cap_s = "unlimited" if cap is None else str(cap)
    print(
        f"running {len(live)}/{cap_s}  CUDA_VISIBLE_DEVICES={pool.get('gpus') or gpus}"
    )
    for job in pool.get("jobs") or []:
        print(
            f"{job.get('status')}\tgpu={job.get('gpu')}\t{job.get('dataset_id')}\t"
            f"{job.get('stem')}\tjob={job.get('job_id')}\tpid={job.get('pid')}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Submit one dataset job per call. GPU from CUDA_VISIBLE_DEVICES or --gpu; "
            "default GPU 0. Running-lane cap RNAOMNI_MAX_RUNNING_LANES "
            f"(default {DEFAULT_MAX_RUNNING_LANES}; 0 = unlimited). "
            "No GPU occupancy check."
        )
    )
    p.add_argument(
        "--compare-root",
        default=None,
        help="Submit this dataset as one job (e.g. dataset/forcompare/Class_88fam_0%%noise).",
    )
    p.add_argument(
        "--dataset-id",
        default=None,
        help="Optional slug (default: folder name with unsafe chars stripped).",
    )
    p.add_argument(
        "--formal",
        action="store_true",
        help="Deprecated no-op. Campaigns always start full; orchestrator may switch to sample.",
    )
    p.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Pin a single foldN for fold-layout compare roots. Default: all foldN dirs (e.g. 0–9). Ignored when the root is flat.",
    )
    p.add_argument(
        "--no-wait",
        action="store_true",
        help="If the running-lane cap is full, refuse this submit instead of waiting.",
    )
    p.add_argument(
        "--max-running-lanes",
        type=int,
        default=None,
        help=(
            f"Override RNAOMNI_MAX_RUNNING_LANES for this process "
            f"(default {DEFAULT_MAX_RUNNING_LANES}; 0 = unlimited)."
        ),
    )
    p.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Pin this physical GPU id (CUDA_VISIBLE_DEVICES). Does not check if the card is empty.",
    )
    p.add_argument(
        "--status",
        action="store_true",
        help="Print the GPU job pool and exit.",
    )
    p.add_argument(
        "--spec",
        type=Path,
        default=None,
        help="Optional YAML/JSON: start several jobs now (each job one id from CUDA_VISIBLE_DEVICES).",
    )
    p.add_argument(
        "--batch-id",
        default=None,
        help="Job id for a single submit (default: timestamp_datasetid).",
    )
    p.add_argument(
        "--cuda-visible-devices",
        "--gpus",
        dest="cuda_visible_devices",
        default=None,
        metavar="IDS",
        help="Same as env CUDA_VISIBLE_DEVICES. Default: 0. Multiple ids are cycled.",
    )
    p.add_argument(
        "--git-root",
        type=Path,
        default=REPO,
        help="Repo that owns worktrees/ and .parallel/ (default: this checkout).",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Tree that has dataset/forcompare (default: --git-root).",
    )
    p.add_argument(
        "--start-ref",
        default="HEAD",
        help="git ref for worktree add (default HEAD).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the job that would be submitted; do not start orchestrator.",
    )
    p.add_argument(
        "--cleanup",
        action="store_true",
        help="After this job stops, remove its worktree/branch.",
    )
    p.add_argument(
        "--cleanup-only",
        action="store_true",
        help="Only teardown --batch-id; do not launch.",
    )
    p.add_argument(
        "orchestrator_args",
        nargs=argparse.REMAINDER,
        help="Passed through to orchestrator.py (put after --).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.max_running_lanes is not None:
        os.environ["RNAOMNI_MAX_RUNNING_LANES"] = str(args.max_running_lanes)
    git_root = args.git_root.resolve()
    extra = list(args.orchestrator_args or [])
    if extra and extra[0] == "--":
        extra = extra[1:]
    gpus = parse_cuda_visible_devices(args.cuda_visible_devices)
    data_root = args.data_root.resolve() if args.data_root else git_root

    if args.status:
        print_status(git_root, gpus)
        return 0

    if args.cleanup_only:
        if not args.batch_id:
            print("error: --cleanup-only requires --batch-id", file=sys.stderr)
            return 2
        teardown_batch(git_root, args.batch_id)
        print(f"cleaned {args.batch_id}")
        return 0

    try:
        if args.spec:
            data = load_batch_spec(args.spec.resolve())
            specs, spec_gpus, spec_batch = specs_from_payload(data)
            env_or_cli = bool(
                args.cuda_visible_devices or os.environ.get("CUDA_VISIBLE_DEVICES")
            )
            if spec_gpus is not None and not env_or_cli:
                gpus = spec_gpus
            payload = run_batch(
                git_root=git_root,
                batch_id=args.batch_id or spec_batch or default_batch_id(),
                specs=specs,
                gpus=gpus,
                extra_args=extra,
                data_root=data_root,
                start_ref=args.start_ref,
                dry_run=args.dry_run,
                cleanup=args.cleanup,
                no_wait=bool(args.no_wait),
            )
            return 1 if payload.get("failed") else 0

        if not args.compare_root:
            print(
                "error: submit one dataset with --compare-root "
                "(or pass --spec / --status)",
                file=sys.stderr,
            )
            return 2

        if args.formal:
            print(
                "warning: --formal is a no-op; campaigns start full "
                "(orchestrator may switch to sample on encode timeout)",
                file=sys.stderr,
            )

        spec = LaneSpec(
            dataset_id=args.dataset_id or slug_dataset_id(args.compare_root),
            compare_root=args.compare_root,
            explore=explore_for_compare_root(
                args.compare_root, formal=args.formal, fold=args.fold
            ),
            gpu=args.gpu,
            fold=args.fold,
        )
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "dataset_id": spec.dataset_id,
                        "compare_root": spec.compare_root,
                        "explore": spec.explore,
                        "fold": spec.fold,
                        "stem": expected_stem(spec.compare_root, explore=spec.explore),
                        "requested_gpu": args.gpu,
                        "max_running_lanes": resolved_max_running_lanes(),
                        "no_wait": bool(args.no_wait),
                    },
                    indent=2,
                )
            )
            return 0
        live = start_job(
            git_root=git_root,
            spec=spec,
            gpus=gpus,
            extra_args=extra,
            data_root=data_root,
            start_ref=args.start_ref,
            job_id=args.batch_id,
            requested_gpu=args.gpu,
            no_wait=bool(args.no_wait),
        )
        payload = finish_job(
            git_root=git_root,
            live=live,
            gpus=gpus,
            cleanup=args.cleanup,
        )
        return 1 if payload.get("failed") else 0
    except LaneCapFull as err:
        print(f"error: {err}", file=sys.stderr)
        return 4
    except LockBusy as err:
        print(f"error: {err}", file=sys.stderr)
        return 3
    except ParallelConfigError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
