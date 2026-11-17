"""Per-dataset results paths and header management.

Active log: workspace/<dataset_stem>.tsv
Run archives: workspace/runs/<tag>_<stem>/ (+ INDEX.tsv)
Repo-root results.tsv is not written (legacy symlink is dropped if present).

Layout matches RNAClaw ``results/`` (TSV columns, runs archive contents);
only the root directory name is ``workspace/``.

Path constants are defined in ``rnaomni.paths`` and re-exported here.
"""

from __future__ import annotations

import time
from pathlib import Path

from rnaomni.paths import (  # noqa: F401 — INDEX_PATH / RUNS_DIR re-exported
    INDEX_PATH,
    LEGACY_RESULTS_TSV,
    REPO,
    RESULTS_DIR,
    RUNS_DIR,
)

RESULTS_HEADER_REGRESSION = (
    "commit\tval_spearman\ttest_RMSE\ttest_R2\ttest_spearman\ttest_pearson\t"
    "memory_gb\tdata_mode\tstatus\tdescription\n"
)
RESULTS_HEADER_CLASSIFICATION = (
    "commit\tval_accuracy\ttest_accuracy\ttest_macro_f1\t"
    "memory_gb\tdata_mode\tstatus\tdescription\n"
)
OLD_RESULTS_HEADER = "commit\tRMSE\tmemory_gb\tstatus\tdescription\n"
OLD_RESULTS_HEADER_REGRESSION_NO_MODE = (
    "commit\tval_RMSE\ttest_RMSE\ttest_R2\ttest_spearman\ttest_pearson\t"
    "memory_gb\tstatus\tdescription\n"
)
OLD_RESULTS_HEADER_CLASSIFICATION_NO_MODE = (
    "commit\tval_accuracy\ttest_accuracy\ttest_macro_f1\t"
    "memory_gb\tstatus\tdescription\n"
)


def active_dataset_label() -> str:
    from prepare import ACTIVE_CSV, USE_COMPARE_SPLITS, compare_active_csv_name

    if USE_COMPARE_SPLITS:
        return compare_active_csv_name()
    return ACTIVE_CSV


def active_results_stem() -> str:
    return Path(active_dataset_label()).stem


def results_tsv_path() -> Path:
    return RESULTS_DIR / f"{active_results_stem()}.tsv"


def results_tsv_rel() -> str:
    """Repo-relative path of the active results file (for prompts / logs)."""
    path = results_tsv_path()
    try:
        return path.relative_to(REPO).as_posix()
    except ValueError:
        return path.as_posix()


def results_header_for_task(task_type: str) -> str:
    if task_type == "classification":
        return RESULTS_HEADER_CLASSIFICATION
    return RESULTS_HEADER_REGRESSION


def task_type_from_header(header_line: str) -> str | None:
    line = header_line.strip()
    if not line:
        return None
    cols = line.split("\t")
    if "val_accuracy" in cols:
        return "classification"
    if (
        "val_spearman" in cols
        or "val_RMSE" in cols
        or cols == ["commit", "RMSE", "memory_gb", "status", "description"]
    ):
        return "regression"
    return None


def drop_root_results_pointer() -> None:
    """Remove leftover repo-root results.tsv (symlink or empty file)."""
    legacy = LEGACY_RESULTS_TSV
    if legacy.is_symlink():
        legacy.unlink()
        return
    if legacy.exists() and legacy.stat().st_size == 0:
        legacy.unlink()


def refresh_results_symlink(target: Path | None = None) -> Path:
    """Deprecated: root results.tsv is not maintained. Drops leftover pointer."""
    drop_root_results_pointer()
    return (target or results_tsv_path()).resolve()


def _migrate_legacy_plain_file(current_task_type: str) -> None:
    """Move a plain (non-symlink) results.tsv into workspace/ once."""
    legacy = LEGACY_RESULTS_TSV
    if not legacy.exists() or legacy.is_symlink():
        return
    if legacy.stat().st_size == 0:
        legacy.unlink()
        return

    text = legacy.read_text(encoding="utf-8")
    first = text.splitlines()[0] if text.strip() else ""
    header_task = task_type_from_header(first)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if header_task == current_task_type:
        dest = results_tsv_path()
    else:
        dest = RESULTS_DIR / f"_legacy_{time.strftime('%Y%m%d_%H%M%S')}.tsv"

    if dest.exists() and dest.stat().st_size > 0:
        # Keep both; don't clobber an existing per-dataset log.
        dest = RESULTS_DIR / f"_legacy_{time.strftime('%Y%m%d_%H%M%S')}.tsv"

    dest.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    legacy.unlink()


def _insert_data_mode_column(text_lines: list[str], *, default_mode: str = "full") -> str:
    """Insert data_mode before status on a header+rows that lack it."""
    if not text_lines:
        return ""
    header = text_lines[0].rstrip("\n")
    cols = header.split("\t")
    if "data_mode" in cols:
        return "".join(
            ln if ln.endswith("\n") else ln + "\n" for ln in text_lines
        )
    try:
        status_idx = cols.index("status")
    except ValueError:
        return "".join(
            ln if ln.endswith("\n") else ln + "\n" for ln in text_lines
        )
    cols.insert(status_idx, "data_mode")
    out = ["\t".join(cols) + "\n"]
    for line in text_lines[1:]:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < status_idx:
            parts = parts + [""] * (status_idx - len(parts))
        parts.insert(status_idx, default_mode)
        out.append("\t".join(parts) + "\n")
    return "".join(out)


def ensure_results_tsv(task_type: str) -> Path:
    """
    Ensure workspace/<stem>.tsv exists with the correct task header.
    Does not write repo-root results.tsv. Wrong-schema history for this stem is dropped.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _migrate_legacy_plain_file(task_type)
    drop_root_results_pointer()

    path = results_tsv_path()
    header = results_header_for_task(task_type)

    if not path.is_file() or path.stat().st_size == 0:
        path.write_text(header, encoding="utf-8")
    else:
        existing = path.read_text(encoding="utf-8").splitlines()
        existing_header = (existing[0] + "\n") if existing else ""
        if existing_header == header:
            pass
        elif existing_header.strip() in {
            OLD_RESULTS_HEADER_REGRESSION_NO_MODE.strip(),
            OLD_RESULTS_HEADER_CLASSIFICATION_NO_MODE.strip(),
        }:
            path.write_text(_insert_data_mode_column(existing), encoding="utf-8")
        elif (
            task_type == "regression"
            and existing_header.strip() == OLD_RESULTS_HEADER.strip()
        ):
            migrated = [header]
            for line in existing[1:]:
                parts = line.split("\t")
                if len(parts) == 5:
                    commit, val_rmse, memory_gb, status, description = parts
                    migrated.append(
                        f"{commit}\t{val_rmse}\tNA\tNA\tNA\tNA\t"
                        f"{memory_gb}\tfull\t{status}\t{description}\n"
                    )
                else:
                    migrated.append(line if line.endswith("\n") else line + "\n")
            path.write_text("".join(migrated), encoding="utf-8")
        else:
            # Different task schema for this dataset — start fresh.
            path.write_text(header, encoding="utf-8")

    return path
