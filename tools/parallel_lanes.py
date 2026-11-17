"""Multi-dataset parallel: one worktree per job.

GPU pool is CUDA_VISIBLE_DEVICES (NVIDIA). Unset → GPU 0 for every job.
Launcher / process env wins over rnaomni.env and any .env overlay.
Running-lane cap is ``RNAOMNI_MAX_RUNNING_LANES`` (see rnaomni.parallel).
GPU occupancy is not checked (OOM is still the backstop if you oversubscribe).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_GPUS = (0,)
DATASET_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
BATCH_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

# First-batch whitelist (plan). dataset_id is a branch-safe slug; compare_root
# keeps the on-disk folder name (may contain '%').
FIRST_BATCH_LANES = (
    {
        "dataset_id": "cls_200noise",
        "compare_root": "dataset/forcompare/Class_88fam_200%noise",
        "explore": True,
    },
    {
        "dataset_id": "cls_0noise",
        "compare_root": "dataset/forcompare/Class_88fam_0%noise",
        "explore": True,
    },
)


class ParallelConfigError(ValueError):
    """Bad lane / GPU / batch request."""


@dataclass(frozen=True)
class LaneSpec:
    dataset_id: str
    compare_root: str
    explore: bool = True
    gpu: int | None = None
    active_csv: str | None = None
    fold: int | None = None

    def __post_init__(self) -> None:
        if not DATASET_ID_RE.match(self.dataset_id):
            raise ParallelConfigError(
                f"dataset_id {self.dataset_id!r} is not a branch-safe slug"
            )
        root = self.compare_root.strip()
        if not root:
            raise ParallelConfigError("compare_root is empty")
        object.__setattr__(self, "compare_root", root)


def expected_stem(compare_root: str, *, explore: bool = True) -> str:
    """Results stem for a compare root. ``explore`` is ignored (no ``_explore`` suffix)."""
    return f"compare_{Path(compare_root.rstrip('/')).name}"


def slug_dataset_id(compare_root: str) -> str:
    """Branch-safe id from a compare-root folder name."""
    name = Path(compare_root.rstrip("/")).name
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", name)
    slug = re.sub(r"_+", "_", slug).strip("_")
    if not slug:
        raise ParallelConfigError("empty dataset_id slug")
    if slug[0].isdigit():
        slug = f"ds_{slug}"
    return slug[:48]


def parse_cuda_visible_devices(*raws: str | None) -> list[int]:
    """GPU list. CLI args win, then env CUDA_VISIBLE_DEVICES, else [0]."""
    candidates = [*(raws or ()), os.environ.get("CUDA_VISIBLE_DEVICES")]
    for raw in candidates:
        if raw is None or not str(raw).strip():
            continue
        parts = [p.strip() for p in str(raw).replace(";", ",").split(",") if p.strip()]
        ids: list[int] = []
        for part in parts:
            try:
                ids.append(int(part))
            except ValueError as err:
                raise ParallelConfigError(
                    f"CUDA_VISIBLE_DEVICES must be integer ids, not {part!r}"
                ) from err
        if ids:
            return ids
    return list(DEFAULT_GPUS)


def parse_cuda_avail(*raws: str | None) -> list[int]:
    """Alias for parse_cuda_visible_devices."""
    return parse_cuda_visible_devices(*raws)


def pick_next_gpu(
    gpus: list[int] | tuple[int, ...],
    cursor: int = 0,
    *,
    requested: int | None = None,
) -> tuple[int, int]:
    """Return (gpu, next_cursor). Pin skips the cycle. No occupancy check."""
    pool = list(gpus) or list(DEFAULT_GPUS)
    if requested is not None:
        return int(requested), cursor
    gpu = pool[int(cursor) % len(pool)]
    return gpu, int(cursor) + 1


def pool_path(git_root: Path) -> Path:
    return Path(git_root) / ".parallel" / "pool.json"


def assign_gpus(
    n_lanes: int,
    *,
    gpus: list[int] | tuple[int, ...] | None = None,
    requested: list[int | None] | None = None,
) -> list[int]:
    """Map N lanes onto CUDA_VISIBLE_DEVICES. Reuse is allowed."""
    if n_lanes < 1:
        raise ParallelConfigError("need at least one lane")
    pool = list(DEFAULT_GPUS if gpus is None else gpus)
    if not pool:
        raise ParallelConfigError("empty GPU list")
    if requested is None:
        requested = [None] * n_lanes
    if len(requested) != n_lanes:
        raise ParallelConfigError("requested GPU list length != n_lanes")
    assigned: list[int] = []
    cursor = 0
    for req in requested:
        if req is None:
            gpu, cursor = pick_next_gpu(pool, cursor)
        else:
            gpu, cursor = pick_next_gpu(pool, cursor, requested=int(req))
        assigned.append(gpu)
    return assigned


def lane_env(spec: LaneSpec, gpu: int) -> dict[str, str]:
    """Process env injected by the launcher. Must beat rnaomni.env / .env."""
    env = {
        "RNAOMNI_COMPARE_ROOT": spec.compare_root,
        "RNAOMNI_USE_COMPARE_SPLITS": "1",
        "CUDA_VISIBLE_DEVICES": str(gpu),
    }
    if spec.active_csv:
        env["RNAOMNI_ACTIVE_CSV"] = spec.active_csv
    if spec.fold is not None:
        env["RNAOMNI_COMPARE_FOLD_IDS"] = str(int(spec.fold))
    env["RNAOMNI_DATA_MODE"] = "full"
    env.pop("COMPARE_FULL", None)
    env.pop("COMPARE_NO_SAMPLE", None)
    return env


# Only dataset-route / GPU keys. Do not strip RNAOMNI_CONDA_ENV or other table keys.
_ROUTE_ENV_KEYS = {
    "RNAOMNI_COMPARE_ROOT",
    "RNAOMNI_USE_COMPARE_SPLITS",
    "RNAOMNI_ACTIVE_CSV",
    "RNAOMNI_COMPARE_FOLD_IDS",
    "RNAOMNI_COMPARE_VALID_NAME",
    "COMPARE_ROOT",
    "USE_COMPARE_SPLITS",
    "COMPARE_FULL",
    "COMPARE_NO_SAMPLE",
    "RNAOMNI_DATA_MODE",
    "CUDA_VISIBLE_DEVICES",
}


def merge_lane_env(base: dict[str, str], spec: LaneSpec, gpu: int) -> dict[str, str]:
    """Copy a parent env, drop inherited dataset route keys, apply this lane."""
    out = dict(base)
    for key in _ROUTE_ENV_KEYS:
        out.pop(key, None)
    injected = lane_env(spec, gpu)
    for key, value in injected.items():
        if value == "":
            out.pop(key, None)
        else:
            out[key] = value
    return out


def parallel_dir(git_root: Path, batch_id: str) -> Path:
    return git_root / ".parallel" / batch_id


def worktree_dir(git_root: Path, batch_id: str, dataset_id: str) -> Path:
    return git_root / "worktrees" / batch_id / dataset_id


def lane_branch(batch_id: str, dataset_id: str) -> str:
    return f"RNAclaw/{batch_id}/{dataset_id}"


def _git(git_root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(git_root),
        text=True,
        capture_output=True,
        check=check,
    )


# Registry load paths (gitignore). Parallel worktrees share these dirs; do not copy.
L4_WEIGHT_DIRS = (
    ("tools/RNA-FM/pretrained", "RNA-FM_pretrained.pth"),
    ("tools/RiNALMo/weights", "rinalmo_micro_pretrained.pt"),
    ("tools/Evo/model", "model.safetensors.index.json"),
    ("tools/Evo2/weights", "evo2_7b.pt"),
)


def _symlink_shared_dir(src: Path, dest: Path, *, label: str) -> Path:
    """Share an untracked directory via symlink; do not copy."""
    src = src.resolve()
    if not src.is_dir():
        raise ParallelConfigError(f"missing {label}: {src}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink() or dest.exists():
        if dest.is_symlink() and dest.resolve() == src:
            return dest
        raise ParallelConfigError(f"{label} already exists: {dest}")
    dest.symlink_to(src)
    return dest


# Untracked dirs under dataset/ that every lane worktree should see.
# forcompare is required; others are linked only when present on data_root.
SHARED_DATASET_DIRS = ("forcompare", "RNA task")


def link_compare_data(data_root: Path, worktree: Path) -> list[Path]:
    """Share untracked dataset dirs via symlink; do not copy."""
    linked: list[Path] = []
    for name in SHARED_DATASET_DIRS:
        src = data_root / "dataset" / name
        dest = worktree / "dataset" / name
        if name == "forcompare":
            linked.append(
                _symlink_shared_dir(src, dest, label="compare data")
            )
        elif src.is_dir():
            linked.append(
                _symlink_shared_dir(src, dest, label=f"dataset/{name}")
            )
    return linked


def link_l4_weights(src_root: Path, worktree: Path) -> list[Path]:
    """Share ignored L4 checkpoints via symlink; do not copy or download."""
    linked: list[Path] = []
    for rel_dir, required_name in L4_WEIGHT_DIRS:
        src = src_root / rel_dir
        required = src / required_name
        if not required.is_file():
            raise ParallelConfigError(
                f"missing L4 weight: {required.resolve()} "
                f"(no download; place the file at {rel_dir}/{required_name})"
            )
        dest = worktree / rel_dir
        linked.append(_symlink_shared_dir(src, dest, label=rel_dir))
    return linked


def write_lane_overlay(worktree: Path, spec: LaneSpec) -> Path:
    """Lane-local .env (no secrets). Process env still wins if already set."""
    # Quote string values so paths with spaces (e.g. dataset/RNA task/...) source cleanly.
    lines = [
        "# Generated by tools/parallel_lanes.py — lane overlay, not the repo table.",
        f'RNAOMNI_COMPARE_ROOT="{spec.compare_root}"',
        "RNAOMNI_USE_COMPARE_SPLITS=1",
    ]
    if spec.active_csv:
        lines.append(f'RNAOMNI_ACTIVE_CSV="{spec.active_csv}"')
    if spec.fold is not None:
        lines.append(f"RNAOMNI_COMPARE_FOLD_IDS={int(spec.fold)}")
    lines.append("RNAOMNI_DATA_MODE=full")
    path = worktree / ".env"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def save_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def provision_batch(
    git_root: Path,
    *,
    batch_id: str,
    lanes: list[LaneSpec],
    data_root: Path | None = None,
    gpus: list[int] | tuple[int, ...] | None = None,
    start_ref: str = "HEAD",
) -> dict:
    """Create one worktree per lane. Does not start training."""
    git_root = git_root.resolve()
    if not BATCH_ID_RE.match(batch_id):
        raise ParallelConfigError(f"bad batch_id: {batch_id!r}")
    ids = [ln.dataset_id for ln in lanes]
    if len(ids) != len(set(ids)):
        raise ParallelConfigError(f"duplicate dataset_id in lanes: {ids}")
    assigned = assign_gpus(
        len(lanes),
        gpus=gpus,
        requested=[ln.gpu for ln in lanes],
    )
    data_root = (data_root or git_root).resolve()
    meta_dir = parallel_dir(git_root, batch_id)
    if meta_dir.exists():
        raise ParallelConfigError(f"batch already exists: {meta_dir}")
    meta_dir.mkdir(parents=True, exist_ok=False)

    created: list[dict] = []
    try:
        for spec, gpu in zip(lanes, assigned):
            branch = lane_branch(batch_id, spec.dataset_id)
            wt = worktree_dir(git_root, batch_id, spec.dataset_id)
            wt.parent.mkdir(parents=True, exist_ok=True)
            proc = _git(
                git_root,
                "worktree",
                "add",
                "-b",
                branch,
                str(wt),
                start_ref,
                check=False,
            )
            if proc.returncode != 0:
                raise ParallelConfigError(
                    f"git worktree add failed for {spec.dataset_id}: "
                    f"{proc.stderr.strip() or proc.stdout.strip()}"
                )
            link_compare_data(data_root, wt)
            link_l4_weights(data_root, wt)
            write_lane_overlay(wt, spec)
            created.append(
                {
                    "dataset_id": spec.dataset_id,
                    "compare_root": spec.compare_root,
                    "explore": spec.explore,
                    "gpu": gpu,
                    "branch": branch,
                    "worktree": str(wt),
                    "stem": expected_stem(spec.compare_root, explore=spec.explore),
                    "pid": None,
                    "status": "provisioned",
                    "cuda_visible_devices": str(gpu),
                }
            )
    except Exception:
        _cleanup_created(git_root, created)
        if meta_dir.exists():
            for leftover in meta_dir.glob("*"):
                leftover.unlink()
            meta_dir.rmdir()
        raise

    payload = {
        "batch_id": batch_id,
        "gpus": list(DEFAULT_GPUS if gpus is None else gpus),
        "start_ref": start_ref,
        "lanes": created,
    }
    save_manifest(meta_dir / "manifest.json", payload)
    return payload


def _cleanup_created(git_root: Path, created: list[dict]) -> None:
    for row in reversed(created):
        wt = Path(row["worktree"])
        _git(git_root, "worktree", "remove", "--force", str(wt), check=False)
        _git(git_root, "branch", "-D", row["branch"], check=False)


def teardown_batch(git_root: Path, batch_id: str) -> None:
    """Remove worktrees, lane branches, and .parallel/<batch_id>."""
    git_root = git_root.resolve()
    manifest_path = parallel_dir(git_root, batch_id) / "manifest.json"
    if manifest_path.is_file():
        payload = load_manifest(manifest_path)
        _cleanup_created(git_root, list(payload.get("lanes") or []))
    meta = parallel_dir(git_root, batch_id)
    if meta.is_dir():
        for leftover in meta.glob("*"):
            leftover.unlink()
        meta.rmdir()
    wt_batch = git_root / "worktrees" / batch_id
    if wt_batch.is_dir():
        # worktree remove should have emptied it; drop leftover dirs.
        for child in sorted(wt_batch.rglob("*"), reverse=True):
            if child.is_symlink() or child.is_file():
                child.unlink()
            elif child.is_dir():
                try:
                    child.rmdir()
                except OSError:
                    pass
        try:
            wt_batch.rmdir()
        except OSError:
            pass


def global_lock_path(git_root: Path) -> Path:
    return git_root / ".parallel" / "LOCK"


def update_manifest(git_root: Path, batch_id: str, payload: dict) -> None:
    save_manifest(parallel_dir(git_root, batch_id) / "manifest.json", payload)


def dashboard_path(git_root: Path, batch_id: str) -> Path:
    return git_root / "workspace" / "batches" / batch_id / "dashboard.tsv"


def pool_dashboard_path(git_root: Path) -> Path:
    return git_root / "workspace" / "batches" / "pool" / "dashboard.tsv"


def first_batch_specs() -> list[LaneSpec]:
    return [LaneSpec(**row) for row in FIRST_BATCH_LANES]


def main() -> int:
    """Print the frozen Phase 1 defaults (no provisioning)."""
    print(
        json.dumps(
            {
                "gpus": list(DEFAULT_GPUS),
                "lanes": [asdict(s) for s in first_batch_specs()],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
