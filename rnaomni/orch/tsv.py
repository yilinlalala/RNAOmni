"""Results.tsv read/rewrite, crash digest, and encode-timeout log probe."""

from __future__ import annotations

import re
from pathlib import Path

from prepare import ENCODE_TIMEOUT_EXIT, data_mode
from rnaomni.results_io import (
    ensure_results_tsv,
    results_header_for_task,
    results_tsv_path,
)
from rnaomni.orch.gitops import git_head
from rnaomni.paths import RUN_LOG

CRASH_MARK = " || crash: "
CRASH_DIGEST_MAX = 360


def is_sample_row(row: dict[str, str]) -> bool:
    return (row.get("description") or "").strip().lower().startswith("[sample]")


def rewrite_latest_results_status(new_status: str, *, note: str | None = None) -> bool:
    """Rewrite status (and optional description note) on the last results.tsv data row."""
    return rewrite_latest_results_fields(status=new_status, note=note)


def rewrite_latest_results_fields(
    *,
    status: str | None = None,
    description: str | None = None,
    note: str | None = None,
) -> bool:
    """Rewrite status and/or description on the last results.tsv data row."""
    path = results_tsv_path()
    if not path.is_file():
        return False
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    if len(lines) < 2:
        return False
    headers = lines[0].rstrip("\n").split("\t")
    try:
        status_idx = headers.index("status")
        desc_idx = headers.index("description")
    except ValueError:
        return False
    last = lines[-1]
    ends_nl = last.endswith("\n")
    parts = last.rstrip("\n").split("\t")
    if len(parts) <= max(status_idx, desc_idx):
        parts = parts + [""] * (max(status_idx, desc_idx) + 1 - len(parts))
    if status is not None:
        parts[status_idx] = status
    if description is not None:
        parts[desc_idx] = description.replace("\t", " ")
    elif note:
        desc = parts[desc_idx]
        if note not in desc:
            parts[desc_idx] = f"{desc} ({note})" if desc else note
    lines[-1] = "\t".join(parts) + ("\n" if ends_nl else "")
    path.write_text("".join(lines), encoding="utf-8")
    return True


def append_orchestrator_crash_row(description: str) -> None:
    """Write a crash row when train.py was killed in encode (no Python atexit)."""
    path = results_tsv_path()
    if not path.is_file():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return
    headers = lines[0].split("\t")
    commit = (git_head() or "???????")[:7]
    values = {h: "" for h in headers}
    values["commit"] = commit
    values["data_mode"] = data_mode()
    values["status"] = "crash"
    values["description"] = description.replace("\t", " ")
    for h in headers:
        if h in {"commit", "data_mode", "status", "description"}:
            continue
        if h == "memory_gb":
            values[h] = "0.0"
        else:
            values[h] = "0.000000"
    with path.open("a", encoding="utf-8") as f:
        f.write("\t".join(values[h] for h in headers) + "\n")


def summarize_run_log(path: Path | None = None, *, max_chars: int = CRASH_DIGEST_MAX) -> str:
    """One-line exception digest from run.log. Never returns a traceback dump."""
    log_path = path if path is not None else RUN_LOG
    if not log_path.is_file():
        return "no run.log"
    raw = log_path.read_text(encoding="utf-8", errors="replace")
    if not raw.strip():
        return "empty run.log"
    idx = raw.rfind("Traceback (most recent call last):")
    blob = raw[idx:] if idx >= 0 else raw[-2500:]
    file_line = ""
    exc_line = ""
    for ln in blob.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("File "):
            file_line = s
        if re.search(r"(Error|Exception|KeyboardInterrupt)\b", s) and not s.startswith(
            "File "
        ):
            exc_line = s
    parts = [p for p in (exc_line, file_line) if p]
    text = " | ".join(parts) if parts else " ".join(
        ln.strip() for ln in blob.splitlines() if ln.strip()
    )[-max_chars:]
    text = re.sub(r"[\t\r\n]+", " ", text).strip()
    text = text.replace("{", "(").replace("}", ")")
    if len(text) > max_chars:
        text = text[: max_chars - 1] + "…"
    return text or "unparsed crash"


def attach_crash_digest_to_latest_row(digest: str) -> bool:
    """Append a single-line crash digest to the latest TSV description."""
    row = latest_results_row()
    if row is None:
        return False
    desc = (row.get("description") or "").replace("\t", " ").strip()
    if CRASH_MARK in desc:
        desc = desc.split(CRASH_MARK, 1)[0].rstrip()
    clean = re.sub(r"[\t\r\n]+", " ", digest).strip()[:CRASH_DIGEST_MAX]
    new_desc = f"{desc}{CRASH_MARK}{clean}" if desc else f"crash:{clean}"
    return rewrite_latest_results_fields(status="crash", description=new_desc)


def format_crash_prompt_block(rows: list[dict[str, str]] | None = None) -> str:
    """At most one short crash block. Empty unless the latest non-sample row is crash."""
    seq = rows if rows is not None else read_results_rows()
    for row in reversed(seq):
        if is_sample_row(row):
            continue
        status = (row.get("status") or "").strip()
        if status in ("keep", "discard"):
            return ""
        if status != "crash":
            continue
        desc = row.get("description") or ""
        commit = (row.get("commit") or "")[:7]
        if CRASH_MARK in desc:
            digest = desc.split(CRASH_MARK, 1)[1].strip()
        else:
            digest = "train crashed (no digest on row)"
        digest = re.sub(r"[\t\r\n]+", " ", digest).strip()[:CRASH_DIGEST_MAX]
        digest = digest.replace("{", "(").replace("}", ")")
        return (
            "Last train crash (do not repeat; do not read run.log):\n"
            f"{commit} {digest}\n"
            "ONE new edit that avoids this error, then STOP.\n"
        )
    return ""


def is_encode_timeout(train_exit: int) -> bool:
    if int(train_exit) == int(ENCODE_TIMEOUT_EXIT):
        return True
    log_path = RUN_LOG
    if not log_path.is_file():
        return False
    tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    return "ENCODE_TIMEOUT" in tail or "ENCODE_FAIL" in tail


def read_results_rows() -> list[dict[str, str]]:
    path = results_tsv_path()
    if not path.is_file():
        return []
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) < 2:
        return []
    headers = lines[0].split("\t")
    rows: list[dict[str, str]] = []
    for line in lines[1:]:
        values = line.split("\t")
        if len(values) < len(headers):
            values = values + [""] * (len(headers) - len(values))
        rows.append(dict(zip(headers, values[: len(headers)])))
    return rows


def latest_results_row() -> dict[str, str] | None:
    rows = read_results_rows()
    return rows[-1] if rows else None


def parse_metric(row: dict[str, str], metric_name: str) -> float | None:
    raw = row.get(metric_name)
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def improvement_delta(
    metric: float,
    best: float,
    *,
    higher_is_better: bool,
) -> float:
    if higher_is_better:
        return metric - best
    return best - metric


def results_tsv_ready() -> bool:
    """True when active per-dataset results file has the correct task header."""
    from rnaomni.orch.prompts import detect_task_type

    task_type = detect_task_type()
    path = ensure_results_tsv(task_type)
    if not path.is_file() or path.stat().st_size == 0:
        return False
    header = path.read_text(encoding="utf-8").splitlines()[0] + "\n"
    return header == results_header_for_task(task_type)
