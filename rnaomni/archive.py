#!/usr/bin/env python3
"""Archive overnight search outputs under workspace/runs/<timestamp>_<csv_stem>/.

Copies the active results TSV, features/active.yaml, orchestrator.log (if present),
and HEAD train.py into a timestamped directory; writes meta.json and appends
workspace/runs/INDEX.tsv.

Who calls this:
    Standalone ``orchestrator.py`` (no ``RNAOMNI_BATCH_ID``): ``_archive_on_stop``
    at search stop.
    Parallel lane: orchestrator skips; ``rnaomni.parallel.archive_lane`` runs this
    once after the lane process exits (``--tag {batch_id}__{dataset_id}``).

Usage:
    python -m rnaomni.archive
    python -m rnaomni.archive --tag 20260729_225500 --stop-reason patience
    python -m rnaomni.archive --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from prepare import (
    ACTIVE_CSV,
    USE_COMPARE_SPLITS,
    clean_dataset,
    data_mode,
    load_dataset,
    resolve_compare_per_class_eval,
    resolve_compare_per_class_train,
    resolve_sample_row_cap_eval,
    resolve_sample_row_cap_train,
    sampling_on,
)
from rnaomni.results_io import (
    RESULTS_HEADER_CLASSIFICATION,
    RESULTS_HEADER_REGRESSION,
    active_dataset_label as results_active_dataset_label,
    results_tsv_path,
)
from features.pipeline import RECIPE_PATH
from rnaomni.paths import (
    INDEX_PATH,
    LEGACY_RESULTS_TSV,
    ORCH_LOG,
    REPO,
    RUNS_DIR,
    STATE_PATH,
)

TRAIN_PY = REPO / "train.py"
BRANCH_PREFIX = "RNAclaw/"

INDEX_HEADER = (
    "run_dir\ttimestamp\tcsv_name\ttask_type\tbranch\thead\t"
    "n_keep\tbest_metric\tbest_commit\tstop_reason\tstatus\n"
)


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=REPO,
        stderr=subprocess.DEVNULL,
        text=True,
    ).strip()


def _safe_git(*args: str) -> str | None:
    try:
        return _git(*args)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def detect_task_type(csv_name: str = ACTIVE_CSV) -> str:
    try:
        if USE_COMPARE_SPLITS:
            from prepare import load_compare_splits_dataframe, infer_task_type

            df = load_compare_splits_dataframe(per_class=1, quiet=True)
            return infer_task_type(df["predicted_target"].to_numpy())
        df = load_dataset(csv_name)
        _, _, task_type = clean_dataset(df)
        return task_type
    except Exception:
        return "regression"


def active_dataset_label() -> str:
    return results_active_dataset_label()


def csv_stem(csv_name: str = ACTIVE_CSV) -> str:
    return Path(csv_name).stem


def infer_tag_from_branch() -> str | None:
    branch = _safe_git("rev-parse", "--abbrev-ref", "HEAD")
    if branch and branch.startswith(BRANCH_PREFIX):
        return branch[len(BRANCH_PREFIX) :]
    return None


def resolve_run_tag(tag: str | None = None) -> str:
    if tag:
        return tag
    from_branch = infer_tag_from_branch()
    if from_branch:
        return from_branch
    return time.strftime("%Y%m%d_%H%M%S")


def run_dirname(tag: str, csv_name: str = ACTIVE_CSV) -> str:
    # Prefer {tag}_{stem}; if tag already embeds stem or is a plain timestamp, still append stem
    # when tag looks like YYYYMMDD_HHMMSS only.
    # Lane branches are RNAclaw/<batch>/<dataset_id> — flatten slashes for a single dir.
    tag = tag.replace("/", "__")
    stem = csv_stem(csv_name)
    if tag.endswith(f"_{stem}") or tag == stem:
        return tag
    return f"{tag}_{stem}"


def parse_results(results_path: Path | None = None) -> dict:
    """Parse results.tsv for keep stats and header kind."""
    if results_path is None:
        results_path = results_tsv_path()
    if not results_path.is_file():
        raise FileNotFoundError(f"results.tsv not found: {results_path}")

    text = results_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines:
        raise ValueError("results.tsv is empty")

    header = lines[0] + "\n"
    if header == RESULTS_HEADER_CLASSIFICATION:
        higher_better = True
        task_hint = "classification"
    elif header == RESULTS_HEADER_REGRESSION:
        higher_better = False
        task_hint = "regression"
    else:
        # Tolerate slight header drift; infer from column names.
        cols = lines[0].split("\t")
        if "val_accuracy" in cols:
            higher_better = True
            task_hint = "classification"
        else:
            higher_better = False
            task_hint = "regression"

    data_rows = [ln for ln in lines[1:] if ln.strip()]
    if not data_rows:
        raise ValueError("results.tsv has header only (no experiment rows)")

    n_keep = 0
    best_commit: str | None = None
    best_metric: float | None = None
    best_desc = ""
    n_rows = 0
    n_discard = 0
    n_crash = 0

    for line in data_rows:
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        n_rows += 1
        commit = parts[0]
        val_s = parts[1]
        status = parts[-2]
        desc = parts[-1]
        if status == "keep":
            n_keep += 1
        elif status == "discard":
            n_discard += 1
        elif status == "crash":
            n_crash += 1
        if status != "keep":
            continue
        try:
            val = float(val_s)
        except ValueError:
            continue
        if (not higher_better) and val <= 0:
            continue
        if best_metric is None:
            best_metric = val
            best_commit = commit
            best_desc = desc
        elif higher_better and val > best_metric:
            best_metric = val
            best_commit = commit
            best_desc = desc
        elif (not higher_better) and val < best_metric:
            best_metric = val
            best_commit = commit
            best_desc = desc

    return {
        "task_hint": task_hint,
        "higher_better": higher_better,
        "n_rows": n_rows,
        "n_keep": n_keep,
        "n_discard": n_discard,
        "n_crash": n_crash,
        "best_commit": best_commit,
        "best_metric": best_metric,
        "best_description": best_desc,
    }


def archive_run(
    stop_reason: str | None = None,
    tag: str | None = None,
    dry_run: bool = False,
    csv_name: str | None = None,
) -> Path:
    """Snapshot search outputs; return destination directory path."""
    if csv_name is None:
        csv_name = active_dataset_label()
    task_type = detect_task_type(csv_name)
    active_results = results_tsv_path()
    if not active_results.is_file() and LEGACY_RESULTS_TSV.is_file():
        active_results = LEGACY_RESULTS_TSV
    stats = parse_results(active_results)
    run_tag = resolve_run_tag(tag)
    dirname = run_dirname(run_tag, csv_name)
    out_dir = RUNS_DIR / dirname
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    branch = _safe_git("rev-parse", "--abbrev-ref", "HEAD") or ""
    head = _safe_git("rev-parse", "HEAD") or ""
    head_short = head[:7] if head else ""

    meta = {
        "run_dir": dirname,
        "timestamp": timestamp,
        "tag": run_tag,
        "csv_name": csv_name,
        "task_type": task_type,
        "branch": branch,
        "head": head,
        "head_short": head_short,
        "stop_reason": stop_reason or "",
        "n_rows": stats["n_rows"],
        "n_keep": stats["n_keep"],
        "n_discard": stats["n_discard"],
        "n_crash": stats["n_crash"],
        "best_commit": stats["best_commit"],
        "best_metric": stats["best_metric"],
        "best_description": stats["best_description"],
        "status": "ok" if stats["n_keep"] > 0 else "no_keep",
        "data_mode": data_mode(),
        "paths": {
            "results_tsv": "results.tsv",
            "results_source": str(active_results.relative_to(REPO)),
            "recipe": "active.yaml",
            "orchestrator_log": "orchestrator.log" if ORCH_LOG.is_file() else None,
            "train_snapshot": "train.py.snapshot",
            "meta": "meta.json",
        },
    }
    if sampling_on():
        meta["K_train"] = resolve_compare_per_class_train()
        meta["K_eval"] = resolve_compare_per_class_eval()
        meta["sample_row_cap_train"] = resolve_sample_row_cap_train()
        meta["sample_row_cap_eval"] = resolve_sample_row_cap_eval()
    if STATE_PATH.is_file():
        try:
            saved = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            saved = None
        if isinstance(saved, dict):
            if saved.get("data_mode") in {"full", "sample"}:
                meta["data_mode"] = saved["data_mode"]
            for key in ("K_train", "K_eval", "k_train", "k_eval"):
                if saved.get(key) not in (None, ""):
                    meta[key if key.startswith("K") else key.replace("k_", "K_")] = saved[key]

    if dry_run:
        print(f"[dry-run] would archive to {out_dir}")
        print(json.dumps(meta, indent=2, ensure_ascii=False))
        return out_dir

    if out_dir.exists():
        # Avoid clobber: append a disambiguating suffix.
        suffix = time.strftime("%H%M%S")
        out_dir = RUNS_DIR / f"{dirname}_{suffix}"
        meta["run_dir"] = out_dir.name
        dirname = out_dir.name

    out_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(active_results, out_dir / "results.tsv")
    if RECIPE_PATH.is_file():
        shutil.copy2(RECIPE_PATH, out_dir / "active.yaml")
    if ORCH_LOG.is_file():
        shutil.copy2(ORCH_LOG, out_dir / "orchestrator.log")
    if TRAIN_PY.is_file():
        shutil.copy2(TRAIN_PY, out_dir / "train.py.snapshot")

    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not INDEX_PATH.is_file()
    with INDEX_PATH.open("a", encoding="utf-8") as f:
        if write_header:
            f.write(INDEX_HEADER)
        best_metric_s = (
            f"{stats['best_metric']:.6f}" if stats["best_metric"] is not None else ""
        )
        best_commit_s = stats["best_commit"] or ""
        f.write(
            f"{dirname}\t{timestamp}\t{csv_name}\t{task_type}\t{branch}\t{head_short}\t"
            f"{stats['n_keep']}\t{best_metric_s}\t{best_commit_s}\t"
            f"{stop_reason or ''}\t{meta['status']}\n"
        )

    print(f"archived: {out_dir}")
    return out_dir


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Archive overnight search outputs under workspace/runs/."
    )
    p.add_argument(
        "--tag",
        default=None,
        help="Run tag (default: RNAclaw/<tag> branch suffix, else local timestamp).",
    )
    p.add_argument(
        "--stop-reason",
        default=None,
        help="Optional stop reason (e.g. patience, max_iterations).",
    )
    p.add_argument(
        "--csv",
        default=None,
        help="CSV name recorded in meta (default: compare_* on the external route, else ACTIVE_CSV).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned archive without writing files.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        archive_run(
            stop_reason=args.stop_reason,
            tag=args.tag,
            dry_run=args.dry_run,
            csv_name=args.csv,
        )
    except (FileNotFoundError, ValueError) as err:
        print(f"archive_run failed: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
