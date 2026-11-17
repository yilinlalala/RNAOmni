"""Search state, checkpoint, sample-expand, and [sample] row parsing."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
from dataclasses import dataclass, field

from prepare import (
    apply_compare_sampling_env,
    apply_data_mode,
    resolve_compare_per_class_eval,
    resolve_compare_per_class_train,
    resolve_sample_row_cap_eval,
    resolve_sample_row_cap_train,
)
from rnaomni.orch.tsv import (
    append_orchestrator_crash_row,
    improvement_delta,
    is_encode_timeout,
    is_sample_row,
    latest_results_row,
    parse_metric,
    read_results_rows,
    rewrite_latest_results_fields,
)
from rnaomni.paths import STATE_PATH
from tools.user_env import env_float, env_int

DEFAULT_MAX_ITERATIONS = 100
DEFAULT_MIN_ITERATIONS = env_int("MIN_ITERATIONS", 10)
DEFAULT_MAX_AGENT_FAILURES = 3
DEFAULT_PATIENCE = env_int("PATIENCE", 15)
DEFAULT_MIN_DELTA_CLASSIFICATION = env_float("MIN_DELTA_CLASSIFICATION", 3e-4)
DEFAULT_MIN_DELTA_REGRESSION = env_float("MIN_DELTA_REGRESSION", 1e-3)
DEFAULT_EFF_WINDOW = 5
DEFAULT_SAMPLE_EXPAND_PATIENCE = env_int("SAMPLE_EXPAND_PATIENCE", 3)
DEFAULT_SAMPLE_EXPAND_FACTOR = 2
DEFAULT_K_TRAIN_CAP = 500
DEFAULT_FRAMEWORK = "baseline"  # as-is train.py; not on FRAMEWORK_MENU
FRAMEWORK_MENU: tuple[str, ...] = (
    "pool_mlp",
    "attn_pool_mlp",
    "resnet1d_pool",
    "cls_mlp",
    "tf_mean",
    "tf_cls",
    "tf_attn",
    "tf_conv",
    "max_pool_mlp",
    "gated_pool_mlp",
    "gru_pool",
    "mean_deep_mlp",
)
DEFAULT_FRAMEWORK_PATIENCE = env_int("FRAMEWORK_PATIENCE", 10)

if DEFAULT_FRAMEWORK in FRAMEWORK_MENU:
    raise RuntimeError(
        f"DEFAULT_FRAMEWORK={DEFAULT_FRAMEWORK!r} must not be on FRAMEWORK_MENU"
    )

_SAMPLE_DESC_RE = re.compile(
    r"\[sample\].*?K_train\s+(\d+)\s*->\s*(\d+).*?K_eval\s*=\s*(\d+).*?kept\s*=\s*(yes|no)",
    re.IGNORECASE | re.DOTALL,
)
_FRAMEWORK_NAME_RE = re.compile(r"framework=([a-z0-9_]+)", re.IGNORECASE)
_FRAMEWORK_BASE_RE = re.compile(r"\bframework_base\b", re.IGNORECASE)
_CAUSE_RE = re.compile(r"\bcause=([a-z0-9_]+)\b", re.IGNORECASE)
CAUSE_TOKENS: tuple[str, ...] = ("head", "feature", "knob", "sample", "crash")
_CAUSE_CLOSED = frozenset(CAUSE_TOKENS)


def format_sample_description(
    *,
    scale: float,
    old_k: int,
    trial_k: int,
    k_eval: int,
    kept: bool,
) -> str:
    scale_s = (
        f"{scale:.0f}"
        if float(scale).is_integer()
        else f"{scale:.4g}".rstrip("0").rstrip(".")
    )
    kept_s = "yes" if kept else "no"
    return (
        f"[sample] data sampled to ×{scale_s}"
        f" (K_train {int(old_k)}->{int(trial_k)}, K_eval={int(k_eval)}, kept={kept_s})"
    )


def parse_sample_description(description: str) -> dict[str, object] | None:
    text = (description or "").strip()
    if not text.lower().startswith("[sample]"):
        return None
    match = _SAMPLE_DESC_RE.search(text)
    if not match:
        return None
    old_k = int(match.group(1))
    trial_k = int(match.group(2))
    k_eval = int(match.group(3))
    kept = match.group(4).lower() == "yes"
    return {
        "old_k": old_k,
        "trial_k": trial_k,
        "k_eval": k_eval,
        "kept": kept,
    }


def parse_round_kind(description: str) -> dict[str, object]:
    """Parse TSV description prefixes for feature/model/sample and framework tags."""
    text = (description or "").strip()
    lower = text.lower()
    if lower.startswith("[sample]"):
        kind = "sample"
    elif lower.startswith("[feature]"):
        kind = "feature"
    elif lower.startswith("[model]"):
        kind = "model"
    else:
        kind = "other"
    name = None
    match = _FRAMEWORK_NAME_RE.search(text)
    if match:
        name = match.group(1).lower()
    cause = None
    cause_match = _CAUSE_RE.search(text)
    if cause_match:
        token = cause_match.group(1).lower()
        if token in _CAUSE_CLOSED:
            cause = token
    return {
        "kind": kind,
        "framework": name,
        "framework_base": bool(_FRAMEWORK_BASE_RE.search(text)),
        "cause": cause,
    }


def infer_cause(parsed: dict[str, object], status: str) -> str:
    """Resolve cause= token. Missing tokens are inferred; campaigns stay green."""
    if (status or "").strip() == "crash":
        return "crash"
    written = parsed.get("cause")
    written_s = str(written).lower() if written else ""
    # cause=head without a real do(H) is a knob; does not draw a named head.
    if written_s == "head" and not parsed.get("framework_base"):
        return "knob"
    if written_s in _CAUSE_CLOSED:
        return written_s
    kind = parsed.get("kind")
    if kind == "sample":
        return "sample"
    if kind == "feature":
        return "feature"
    if kind == "model" and parsed.get("framework_base"):
        return "head"
    return "knob"


def format_cause_block(state: object | None) -> str:
    """Short bottleneck card for Iter. Empty-safe; never dumps TSV or run.log."""
    last_do = getattr(state, "last_do", None) or "-"
    last_result = getattr(state, "last_result", None) or "-"
    fail_head = int(getattr(state, "fail_head", 0) or 0)
    fail_feature = int(getattr(state, "fail_feature", 0) or 0)
    fail_knob = int(getattr(state, "fail_knob", 0) or 0)
    fail_sample = int(getattr(state, "fail_sample", 0) or 0)
    raw_sup = getattr(state, "suppressed", None) or []
    if isinstance(raw_sup, str):
        sup_s = raw_sup
    else:
        sup_s = ",".join(str(x) for x in raw_sup)
    if hasattr(state, "next_do_token"):
        next_do = str(state.next_do_token() or "")
    else:
        next_do = ""
    if next_do == "feature":
        write_line = "MUST [feature] this round. Write cause=feature. One layer only."
    elif next_do:
        write_line = f"Write cause={next_do}. One layer only."
    else:
        write_line = "Write cause=feature or cause=knob. One layer only."
    return (
        f"Bottleneck: last_do={last_do} result={last_result} "
        f"fail[head]={fail_head} fail[feature]={fail_feature} "
        f"fail[knob]={fail_knob} fail[sample]={fail_sample} "
        f"suppressed=({sup_s})\n"
        f"next_do={next_do}\n"
        f"{write_line}\n"
    )


def framework_rng_seed(run_tag: str | None = None) -> int:
    """Stable seed from campaign tag (BATCH_ID / branch). Same tag → same draws."""
    tag = (run_tag or os.environ.get("RNAOMNI_BATCH_ID") or "").strip()
    if not tag:
        tag = DEFAULT_FRAMEWORK
    return int(hashlib.sha256(tag.encode("utf-8")).hexdigest()[:8], 16)


def framework_base_lifts_global(
    metric: float,
    best_before: float | None,
    *,
    min_delta: float,
    higher_is_better: bool,
) -> bool:
    """True when a framework_base shot beats the pre-switch global best."""
    if best_before is None:
        return True
    return (
        improvement_delta(metric, best_before, higher_is_better=higher_is_better)
        >= min_delta
    )


def is_framework_base_round(
    parsed: dict[str, object], state: "SearchState | None"
) -> bool:
    """First shot of a named framework (TSV keep even if worse than global).

    A later inner round on an already-tried but non-active head (flat base that
    did not become the working tree) is not a new first shot.
    """
    if parsed.get("kind") != "model":
        return False
    if parsed.get("framework_base"):
        return True
    name = parsed.get("framework")
    if not name or state is None:
        return False
    token = str(name)
    tried = getattr(state, "frameworks_tried", None) or []
    if token in tried and token != (state.active_framework or ""):
        return False
    return token != state.active_framework


@dataclass
class SearchCheckpoint:
    """Persisted across orchestrator process restarts."""

    iteration: int = 0
    rows_seen: int = 0
    stop_reason: str | None = None
    sample_on: bool = False
    k_train0: int | None = None
    k_train: int | None = None
    k_eval: int | None = None
    sample_scale: float = 1.0
    sample_expand_done: bool = False
    sample_exhausted: bool = False
    sample_plateau: bool = False
    sample_expand_factor: int = DEFAULT_SAMPLE_EXPAND_FACTOR
    sample_expand_patience: int = DEFAULT_SAMPLE_EXPAND_PATIENCE
    k_train_cap: int = DEFAULT_K_TRAIN_CAP
    data_mode: str = "full"
    sample_grow_blocked: bool = False
    sample_wallclock_retry: bool = False
    sample_wallclock_used: bool = False
    next_framework: str | None = None

    def apply_sample_fields_to(self, state: "SearchState") -> None:
        """Overlay expand knobs / K / data_mode. Call before TSV replay for mode."""
        if self.sample_expand_factor >= 1:
            state.sample_expand_factor = int(self.sample_expand_factor)
        if self.k_train_cap >= 1:
            state.k_train_cap = int(self.k_train_cap)
        if self.k_train0 is not None:
            state.k_train0 = int(self.k_train0)
        if self.sample_exhausted:
            state.sample_exhausted = True
        if self.data_mode in {"full", "sample"}:
            state.data_mode = self.data_mode
            state.sample_on = self.data_mode == "sample" or self.sample_on
        state.sample_grow_blocked = bool(self.sample_grow_blocked)
        state.sample_wallclock_retry = bool(self.sample_wallclock_retry)
        state.sample_wallclock_used = bool(self.sample_wallclock_used)
        if self.k_train is not None:
            state.k_train = int(self.k_train)
        if self.k_eval is not None:
            state.k_eval = int(self.k_eval)
        state.sync_sampling_env()

    def apply_framework_fields_to(self, state: "SearchState") -> None:
        """Restore a sticky next_framework after TSV replay (only if still plateaued)."""
        nxt = (self.next_framework or "").strip()
        if not nxt or not state.framework_plateaued():
            return
        if nxt in state.remaining_frameworks():
            state.next_framework = nxt


def load_search_checkpoint(*, rows_seen: int, valid_rounds: int) -> SearchCheckpoint:
    """Load iteration budget; invalidate if results.tsv was reset (fewer rows)."""
    floor = SearchCheckpoint(iteration=valid_rounds, rows_seen=rows_seen)
    if not STATE_PATH.is_file():
        return floor
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        logging.warning("Ignoring corrupt %s: %s", STATE_PATH.name, err)
        return floor

    try:
        saved_rows = int(data.get("rows_seen", 0))
        saved_iter = int(data.get("iteration", 0))
    except (TypeError, ValueError) as err:
        logging.warning("Ignoring malformed %s: %s", STATE_PATH.name, err)
        return floor

    if saved_rows > rows_seen:
        logging.info(
            "Search checkpoint invalidated (results.tsv reset: "
            "saved_rows=%s current_rows=%s)",
            saved_rows,
            rows_seen,
        )
        clear_search_checkpoint()
        return floor

    stop_reason = data.get("stop_reason") or None
    if isinstance(stop_reason, str):
        stop_reason = stop_reason.strip() or None
    else:
        stop_reason = None

    def _opt_int(*keys: str) -> int | None:
        for key in keys:
            raw = data.get(key)
            if raw is None or raw == "":
                continue
            try:
                return int(raw)
            except (TypeError, ValueError):
                continue
        return None

    def _opt_float(key: str) -> float | None:
        raw = data.get(key)
        if raw is None or raw == "":
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    return SearchCheckpoint(
        iteration=max(saved_iter, valid_rounds),
        rows_seen=rows_seen,
        stop_reason=stop_reason,
        sample_on=bool(data.get("sample_on", False)),
        k_train0=_opt_int("K_train0", "k_train0"),
        k_train=_opt_int("K_train", "k_train"),
        k_eval=_opt_int("K_eval", "k_eval"),
        sample_scale=_opt_float("sample_scale") or 1.0,
        sample_expand_done=bool(data.get("sample_expand_done", False)),
        sample_exhausted=bool(data.get("sample_exhausted", False)),
        sample_plateau=bool(data.get("sample_plateau", False)),
        sample_expand_factor=int(
            data.get("sample_expand_factor", DEFAULT_SAMPLE_EXPAND_FACTOR)
        ),
        sample_expand_patience=int(
            data.get("sample_expand_patience", DEFAULT_SAMPLE_EXPAND_PATIENCE)
        ),
        k_train_cap=int(data.get("k_train_cap", DEFAULT_K_TRAIN_CAP)),
        data_mode=(
            str(data.get("data_mode") or "").strip()
            if str(data.get("data_mode") or "").strip() in {"full", "sample"}
            else ("sample" if data.get("sample_on") else "full")
        ),
        sample_grow_blocked=bool(data.get("sample_grow_blocked", False)),
        sample_wallclock_retry=bool(data.get("sample_wallclock_retry", False)),
        sample_wallclock_used=bool(data.get("sample_wallclock_used", False)),
        next_framework=(
            str(data.get("next_framework") or "").strip() or None
        ),
    )


def save_search_checkpoint(
    *,
    iteration: int,
    rows_seen: int,
    valid_rounds: int,
    no_improve_streak: int,
    best: float | None,
    stop_reason: str | None = None,
    state: "SearchState | None" = None,
) -> None:
    payload: dict[str, object] = {
        "iteration": iteration,
        "rows_seen": rows_seen,
        "valid_rounds": valid_rounds,
        "no_improve_streak": no_improve_streak,
        "best": best,
        "stop_reason": stop_reason,
    }
    if state is not None:
        payload.update(
            {
                "sample_on": state.sample_on,
                "K_train0": state.k_train0,
                "K_train": state.k_train,
                "K_eval": state.k_eval,
                "sample_scale": state.sample_scale,
                "sample_expand_done": state.sample_expand_done,
                "sample_exhausted": state.sample_exhausted,
                "sample_plateau": state.sample_plateau,
                "sample_expand_factor": state.sample_expand_factor,
                "sample_expand_patience": state.sample_expand_patience,
                "k_train_cap": state.k_train_cap,
                "data_mode": state.data_mode,
                "sample_grow_blocked": state.sample_grow_blocked,
                "sample_wallclock_retry": state.sample_wallclock_retry,
                "sample_wallclock_used": state.sample_wallclock_used,
                "next_framework": state.next_framework,
                "best_commit": state.best_commit,
                "active_framework": state.active_framework,
            }
        )
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def clear_search_checkpoint() -> None:
    try:
        STATE_PATH.unlink(missing_ok=True)
    except OSError as err:
        logging.warning("Failed to clear %s: %s", STATE_PATH.name, err)


@dataclass
class SearchState:
    """Orchestrator-owned search progress (shared best / min_delta with keep)."""

    decision_metric: str
    higher_is_better: bool
    min_delta: float
    patience: int
    min_iterations: int
    max_iterations: int
    max_agent_failures: int
    eff_window: int

    best: float | None = None
    best_commit: str | None = None
    no_improve_streak: int = 0
    valid_rounds: int = 0
    consecutive_agent_failures: int = 0
    rows_seen: int = 0
    best_history: list[float] = field(default_factory=list)
    last_delta: float | None = None
    stop_reason: str | None = None

    # Subsample: start full; orchestrator switches to sample on encode timeout.
    sample_on: bool = False
    k_train0: int | None = None
    k_train: int | None = None
    k_eval: int | None = None
    sample_scale: float = 1.0
    sample_expand_done: bool = False
    sample_exhausted: bool = False
    sample_plateau: bool = False
    sample_expand_factor: int = DEFAULT_SAMPLE_EXPAND_FACTOR
    sample_expand_patience: int = DEFAULT_SAMPLE_EXPAND_PATIENCE
    k_train_cap: int = DEFAULT_K_TRAIN_CAP
    data_mode: str = "full"
    sample_grow_blocked: bool = False
    sample_wallclock_retry: bool = False
    sample_wallclock_used: bool = False

    # Named prediction-head framework (16.1). Prompt injection of next_framework is 16.2.
    active_framework: str = DEFAULT_FRAMEWORK
    framework_head: str | None = None
    framework_best: float | None = None
    framework_best_commit: str | None = None
    framework_no_improve_streak: int = 0
    frameworks_tried: list[str] = field(default_factory=lambda: [DEFAULT_FRAMEWORK])
    next_framework: str | None = None
    framework_patience: int = DEFAULT_FRAMEWORK_PATIENCE

    # Bottleneck card (C0): last process-layer intervention + consecutive ineffective counts.
    last_do: str | None = None
    last_result: str | None = None
    fail_head: int = 0
    fail_feature: int = 0
    fail_knob: int = 0
    fail_sample: int = 0
    suppressed: list[str] = field(default_factory=list)
    last_non_sample_crash: bool = False

    def configure_sampling_from_env(self) -> None:
        apply_data_mode("full")
        self.data_mode = "full"
        self.sample_on = False
        self.k_train = None
        self.k_eval = None
        self.k_train0 = None
        self.sample_scale = 1.0
        self.sample_expand_done = False
        self.sample_exhausted = False
        self.sample_plateau = False
        self.sample_grow_blocked = False
        self.sample_wallclock_retry = False
        self.sample_wallclock_used = False

    def switch_to_sample(self, task_type: str) -> None:
        """Full encode timed out: subsample, reset best, keep HEAD."""
        self.data_mode = "sample"
        self.sample_on = True
        self.best = None
        self.best_commit = None
        self.no_improve_streak = 0
        self.valid_rounds = 0
        self.best_history = []
        self.last_delta = None
        if task_type == "classification":
            self.k_train = resolve_compare_per_class_train() or 10
            self.k_eval = resolve_compare_per_class_eval() or 10
            self.k_train_cap = DEFAULT_K_TRAIN_CAP
        else:
            self.k_train = resolve_sample_row_cap_train()
            self.k_eval = resolve_sample_row_cap_eval()
            self.k_train_cap = max(int(self.k_train) * 64, 2000)
        self.k_train0 = self.k_train
        self.sample_scale = 1.0
        self.sample_exhausted = False
        self.sample_plateau = False
        self.sample_grow_blocked = False
        self.sample_wallclock_retry = False
        self.sample_wallclock_used = False
        self._reset_framework_tracking(keep_active=True)
        self.sync_sampling_env()
        logging.info(
            "Switched to sample data_mode K_train=%s K_eval=%s (best reset)",
            self.k_train,
            self.k_eval,
        )

    def sync_sampling_env(self) -> None:
        apply_data_mode(self.data_mode)
        if self.k_train is None or self.k_eval is None:
            return
        apply_compare_sampling_env(k_train=int(self.k_train), k_eval=int(self.k_eval))

    def trial_k_train(self) -> int | None:
        if self.k_train is None:
            return None
        return min(int(self.k_train) * int(self.sample_expand_factor), int(self.k_train_cap))

    def can_grow_sample(self) -> bool:
        if not self.sample_on or self.k_train is None:
            return False
        if self.sample_exhausted:
            return False
        if self.sample_grow_blocked and not self.sample_wallclock_retry:
            return False
        trial = self.trial_k_train()
        return trial is not None and trial > int(self.k_train)

    def should_expand_sample(self) -> bool:
        if not self.sample_on or self.sample_exhausted or not self.can_grow_sample():
            return False
        if self.sample_wallclock_retry:
            return True
        return self.no_improve_streak >= self.sample_expand_patience

    def eff_window_value(self) -> float | None:
        w = self.eff_window
        if w <= 0 or len(self.best_history) < w + 1:
            return None
        old = self.best_history[-(w + 1)]
        new = self.best_history[-1]
        if self.higher_is_better:
            return (new - old) / w
        return (old - new) / w

    def log_status(self, *, iteration: int, extra: str = "") -> None:
        best_s = f"{self.best:.6f}" if self.best is not None else "None"
        delta_s = f"{self.last_delta:.6g}" if self.last_delta is not None else "n/a"
        eff = self.eff_window_value()
        eff_s = f"{eff:.6g}" if eff is not None else "n/a"
        sample_s = (
            f"data_mode={self.data_mode} sample_on={self.sample_on} "
            f"K_train={self.k_train} K_eval={self.k_eval} "
            f"scale={self.sample_scale:g} expand_done={self.sample_expand_done} "
            f"exhausted={self.sample_exhausted} grow_blocked={self.sample_grow_blocked} "
            f"wallclock_retry={self.sample_wallclock_retry}"
        )
        fw_best = (
            f"{self.framework_best:.6f}" if self.framework_best is not None else "None"
        )
        fw_s = (
            f"framework={self.active_framework} fw_best={fw_best} "
            f"fw_streak={self.framework_no_improve_streak}/{self.framework_patience} "
            f"fw_tried={','.join(self.frameworks_tried)} next={self.next_framework or '-'} "
            f"last_do={self.last_do or '-'} result={self.last_result or '-'} "
            f"suppressed={','.join(self.suppressed) or '-'}"
        )
        msg = (
            f"Search: iteration={iteration} valid_rounds={self.valid_rounds} "
            f"best={best_s} delta={delta_s} no_improve_streak={self.no_improve_streak} "
            f"eff_window={eff_s} agent_fail_streak={self.consecutive_agent_failures} "
            f"{sample_s} {fw_s}"
        )
        if extra:
            msg = f"{msg} {extra}"
        logging.info(msg)

    def sync_from_tsv(self) -> None:
        """Replay keep/discard rows so patience / best survive process restart."""
        saved_sample_on = self.sample_on
        saved_k0 = self.k_train0
        saved_k_train = self.k_train
        saved_k_eval = self.k_eval
        saved_factor = self.sample_expand_factor
        saved_patience = self.sample_expand_patience
        saved_cap = self.k_train_cap

        self.best = None
        self.best_commit = None
        self.no_improve_streak = 0
        self.valid_rounds = 0
        self.best_history = []
        self.last_delta = None
        self.rows_seen = 0
        self.sample_expand_done = False
        self.sample_exhausted = False
        self.sample_plateau = False
        self.sample_on = saved_sample_on
        self.k_train0 = saved_k0
        self.k_train = saved_k_train
        self.k_eval = saved_k_eval
        self.sample_scale = 1.0
        self.sample_expand_factor = saved_factor
        self.sample_expand_patience = saved_patience
        self.k_train_cap = saved_cap
        self._reset_framework_tracking()

        self.ingest_new_rows()
        self.sync_sampling_env()
        logging.info(
            "Restored search from results.tsv: rows=%s valid_rounds=%s best=%s "
            "no_improve_streak=%s exhausted=%s plateau=%s K_train=%s",
            self.rows_seen,
            self.valid_rounds,
            f"{self.best:.6f}" if self.best is not None else "None",
            self.no_improve_streak,
            self.sample_exhausted,
            self.sample_plateau,
            self.k_train,
        )

    def ingest_new_rows(self) -> int:
        """Process new results.tsv rows since last sync. Returns count ingested."""
        rows = read_results_rows()
        new_rows = rows[self.rows_seen :]
        self.rows_seen = len(rows)
        ingested = 0
        for row in new_rows:
            status = (row.get("status") or "").strip()
            row_mode = (row.get("data_mode") or "full").strip() or "full"
            if row_mode != (self.data_mode or "full"):
                logging.info(
                    "Skip %s row while data_mode=%s",
                    row_mode,
                    self.data_mode,
                )
                continue
            if status == "crash":
                self._apply_crash_round(row)
                continue
            if status not in ("keep", "discard"):
                logging.info(
                    "Ignoring non-decision row status=%s description=%s",
                    status,
                    row.get("description", ""),
                )
                continue
            metric = parse_metric(row, self.decision_metric)
            if metric is None:
                logging.warning("Skip row with unparseable %s: %s", self.decision_metric, row)
                continue
            if is_sample_row(row):
                self._apply_sample_round(metric=metric, status=status, row=row)
            else:
                self._apply_valid_round(metric=metric, status=status, row=row)
            ingested += 1
        return ingested

    def _reset_framework_tracking(self, *, keep_active: bool = False) -> None:
        """Drop tried heads. Empty TSV starts at baseline; sample switch keeps the current name."""
        active = self.active_framework if keep_active else DEFAULT_FRAMEWORK
        active = active or DEFAULT_FRAMEWORK
        self.active_framework = active
        self.framework_head = None
        self.framework_best = None
        self.framework_best_commit = None
        self.framework_no_improve_streak = 0
        self.frameworks_tried = [active]
        self.next_framework = None
        self._reset_cause_tracking()

    def _reenable_frameworks(self, *, metric: float, commit: str | None) -> None:
        """Feature/sample keep: old heads may be retried on the new context."""
        active = self.active_framework or DEFAULT_FRAMEWORK
        self.active_framework = active
        self.frameworks_tried = [active]
        self.framework_no_improve_streak = 0
        self.next_framework = None
        self.framework_best = metric
        self.framework_best_commit = commit
        self.framework_head = commit
        self._clear_cause_fails()
        logging.info(
            "Re-enabled frameworks after context keep: active=%s tried=%s",
            active,
            ",".join(self.frameworks_tried),
        )

    def remaining_frameworks(self) -> list[str]:
        tried = set(self.frameworks_tried)
        return [name for name in FRAMEWORK_MENU if name not in tried]

    def framework_plateaued(self) -> bool:
        return self.framework_no_improve_streak >= int(self.framework_patience)

    def ensure_next_framework(
        self, run_tag: str | None = None, *, note_head: str | None = None
    ) -> str | None:
        """Sticky next_framework. N2: note name if legal; else plateau uniform draw."""
        if "head" in self.suppressed:
            if self.next_framework is not None:
                logging.info(
                    "Head suppressed; skip next_framework draw (active=%s)",
                    self.active_framework,
                )
            self.next_framework = None
            return None
        remaining = self.remaining_frameworks()
        if self.next_framework and self.next_framework in remaining:
            return self.next_framework
        note_name = (note_head or "").strip()
        if note_name and note_name in remaining and int(self.valid_rounds or 0) > 0:
            self.next_framework = note_name
            logging.info(
                "N2 note head: next=%s (active=%s remaining=%s)",
                note_name,
                self.active_framework,
                ",".join(remaining),
            )
            return note_name
        if not self.framework_plateaued():
            self.next_framework = None
            return None
        if not remaining:
            self.next_framework = None
            logging.info(
                "Framework plateau on %s but menu empty; stay put",
                self.active_framework,
            )
            return None
        pool = sorted(remaining)
        pick = random.Random(framework_rng_seed(run_tag)).choice(pool)
        self.next_framework = pick
        logging.info(
            "Framework plateau: next=%s (active=%s remaining=%s)",
            pick,
            self.active_framework,
            ",".join(pool),
        )
        return pick

    def framework_reset_commit(self) -> str | None:
        return self.framework_best_commit or self.framework_head

    def decision_baseline(
        self, parsed: dict[str, object], global_best: float | None
    ) -> float | None:
        """Keep/discard reference for this round. None = first shot, always keep."""
        if is_framework_base_round(parsed, self):
            return None
        if parsed.get("kind") == "model" and self.framework_best is not None:
            return self.framework_best
        return global_best

    def _mark_framework_tried(self, name: str) -> None:
        if name not in self.frameworks_tried:
            self.frameworks_tried.append(name)

    def _reset_cause_tracking(self) -> None:
        self.last_do = None
        self.last_result = None
        self.last_non_sample_crash = False
        self._clear_cause_fails()

    def _clear_cause_fails(self) -> None:
        self.fail_head = 0
        self.fail_feature = 0
        self.fail_knob = 0
        self.fail_sample = 0
        self.suppressed = []

    def _suppress(self, layer: str) -> None:
        if layer not in self.suppressed:
            self.suppressed.append(layer)

    def _unsuppress(self, layer: str) -> None:
        self.suppressed = [x for x in self.suppressed if x != layer]

    def next_do_token(self) -> str:
        """C0 vetoes: crash, sample, head draw, head-suppressed plateau."""
        if self.last_non_sample_crash:
            return "crash"
        if self.should_expand_sample():
            return "sample"
        head_suppressed = "head" in self.suppressed
        nxt = (self.next_framework or "").strip()
        if nxt and not head_suppressed:
            return "head"
        if self.framework_plateaued() and not head_suppressed:
            return "head"
        if self.framework_plateaued() and head_suppressed:
            return "feature"
        return ""

    def _apply_crash_round(self, row: dict[str, str]) -> None:
        parsed = parse_round_kind(row.get("description") or "")
        self._apply_cause_round(
            parsed=parsed, status="crash", metric=None, best_before=None
        )

    def _apply_cause_round(
        self,
        *,
        parsed: dict[str, object],
        status: str,
        metric: float | None,
        best_before: float | None,
        kept: bool | None = None,
    ) -> None:
        cause = infer_cause(parsed, status)
        if status != "crash" and is_framework_base_round(parsed, self):
            cause = "head"
        kind = parsed.get("kind")
        if status == "crash":
            self.last_do = "crash"
            self.last_result = "crash"
            if kind != "sample":
                self.last_non_sample_crash = True
            return

        if kind != "sample":
            self.last_non_sample_crash = False
        self.last_do = cause

        if cause == "head":
            lifted = False
            if best_before is None:
                lifted = True
            elif metric is not None:
                delta = improvement_delta(
                    metric, best_before, higher_is_better=self.higher_is_better
                )
                lifted = delta >= self.min_delta
            if lifted:
                self.fail_head = 0
                self._unsuppress("head")
                self.last_result = "keep"
            else:
                self.fail_head += 1
                self._suppress("head")
                self.last_result = "flat"
            return

        if cause == "feature":
            if status == "keep":
                self.last_result = "keep"
            else:
                self.last_result = "discard"
                self.fail_feature += 1
            return

        if cause == "knob":
            if status == "keep":
                self.last_result = "keep"
                self.fail_knob = 0
            else:
                self.last_result = "discard"
                self.fail_knob += 1
            return

        if cause == "sample":
            is_kept = bool(kept) if kept is not None else status == "keep"
            if is_kept:
                self.last_result = "keep"
            else:
                self.last_result = "discard"
                self.fail_sample += 1
            return

        self.last_result = "keep" if status == "keep" else "discard"

    def _apply_framework_round(
        self,
        *,
        metric: float,
        status: str,
        parsed: dict[str, object],
        commit: str | None,
        best_before: float | None = None,
    ) -> None:
        kind = parsed.get("kind")
        if kind in {"feature", "model"}:
            self._apply_cause_round(
                parsed=parsed,
                status=status,
                metric=metric,
                best_before=best_before,
            )
        if kind == "feature":
            if status == "keep":
                self._reenable_frameworks(metric=metric, commit=commit)
            return
        if kind != "model":
            return

        name = str(parsed.get("framework") or self.active_framework or DEFAULT_FRAMEWORK)
        if is_framework_base_round(parsed, self):
            self._mark_framework_tried(name)
            self.next_framework = None
            lifted = framework_base_lifts_global(
                metric,
                best_before,
                min_delta=self.min_delta,
                higher_is_better=self.higher_is_better,
            )
            if not lifted:
                logging.info(
                    "Framework base %s flat vs global; keep active=%s (tried, not working tree)",
                    name,
                    self.active_framework,
                )
                return
            self.active_framework = name
            self.framework_head = commit
            self.framework_best = metric
            self.framework_best_commit = commit
            self.framework_no_improve_streak = 0
            logging.info(
                "Framework base %s at %s metric=%.6f (not counted toward fw streak)",
                name,
                (commit or "?")[:7],
                metric,
            )
            return

        if name != (self.active_framework or DEFAULT_FRAMEWORK):
            logging.info(
                "Skip fw tracking for non-active head %s (active=%s)",
                name,
                self.active_framework,
            )
            return

        self.active_framework = name
        self._mark_framework_tried(name)
        if self.framework_best is None:
            self.framework_best = metric
            if status == "keep":
                self.framework_best_commit = commit
                self.framework_head = self.framework_head or commit
                self.framework_no_improve_streak = 0
            else:
                self.framework_no_improve_streak = 1
            return

        delta = improvement_delta(
            metric, self.framework_best, higher_is_better=self.higher_is_better
        )
        significant = delta >= self.min_delta
        if status == "keep" and significant:
            self.framework_best = metric
            self.framework_best_commit = commit
            self.framework_no_improve_streak = 0
            logging.info(
                "Framework keep %s: metric=%.6f delta=%.6g; fw streak reset",
                name,
                metric,
                delta,
            )
            return
        self.framework_no_improve_streak += 1
        logging.info(
            "Framework no-improve %s: status=%s metric=%.6f fw_streak=%s/%s",
            name,
            status,
            metric,
            self.framework_no_improve_streak,
            self.framework_patience,
        )

    def _apply_sample_round(
        self, *, metric: float, status: str, row: dict[str, str]
    ) -> None:
        """Apply a [sample] expand row: update K from kept=; discard exhausts data axis."""
        self.sample_on = True
        parsed = parse_sample_description(row.get("description") or "")
        kept_flag = status == "keep"
        if parsed is not None:
            old_k = int(parsed["old_k"])  # type: ignore[arg-type]
            trial_k = int(parsed["trial_k"])  # type: ignore[arg-type]
            k_eval = int(parsed["k_eval"])  # type: ignore[arg-type]
            kept = bool(parsed["kept"])
            kept_flag = kept
            self.k_eval = k_eval
            if self.k_train0 is None:
                self.k_train0 = old_k
            if kept:
                self.k_train = trial_k
                self.sample_scale = (
                    float(trial_k) / float(self.k_train0) if self.k_train0 else 1.0
                )
            else:
                self.k_train = old_k
                self.sample_scale = (
                    float(old_k) / float(self.k_train0) if self.k_train0 else 1.0
                )
        self.sync_sampling_env()

        commit = (row.get("commit") or "").strip() or None
        self.valid_rounds += 1
        if self.best is None:
            self.best = metric
            self.best_commit = commit
            self.last_delta = 0.0
        else:
            self.last_delta = improvement_delta(
                metric, self.best, higher_is_better=self.higher_is_better
            )
            if status == "keep" and self.last_delta >= self.min_delta:
                self.best = metric
                self.best_commit = commit
        self.best_history.append(self.best)
        self.sample_expand_done = False
        self._apply_cause_round(
            parsed=parse_round_kind(row.get("description") or ""),
            status=status,
            metric=metric,
            best_before=None,
            kept=kept_flag,
        )
        if status == "keep":
            self.no_improve_streak = 0
            self.sample_exhausted = False
            self.sample_plateau = False
            self._reenable_frameworks(metric=metric, commit=commit)
        else:
            # Model-axis: stop expanding K, do not stop the whole search.
            self.sample_exhausted = True
            self.sample_plateau = False
        logging.info(
            "Sample-expand row ingested: status=%s metric=%.6f best=%.6f "
            "K_train=%s scale=%g exhausted=%s plateau=%s streak=%s",
            status,
            metric,
            self.best,
            self.k_train,
            self.sample_scale,
            self.sample_exhausted,
            self.sample_plateau,
            self.no_improve_streak,
        )

    def _apply_valid_round(
        self, *, metric: float, status: str, row: dict[str, str]
    ) -> None:
        parsed = parse_round_kind(row.get("description") or "")
        commit = (row.get("commit") or "").strip() or None
        best_before = self.best
        self.valid_rounds += 1
        if self.best is None:
            self.best = metric
            self.best_commit = commit
            self.last_delta = 0.0
            self.no_improve_streak = 0 if status == "keep" else 1
            self.best_history.append(self.best)
            logging.info(
                "Established best %s=%.6f from first valid round (status=%s)",
                self.decision_metric,
                self.best,
                status,
            )
            self._apply_framework_round(
                metric=metric,
                status=status,
                parsed=parsed,
                commit=commit,
                best_before=best_before,
            )
            return

        delta = improvement_delta(
            metric, self.best, higher_is_better=self.higher_is_better
        )
        self.last_delta = delta
        significant = delta >= self.min_delta

        if status == "keep" and significant:
            self.best = metric
            self.best_commit = commit
            self.no_improve_streak = 0
            logging.info(
                "Significant keep: %s=%.6f delta=%.6g (>= min_delta=%.6g); streak reset",
                self.decision_metric,
                metric,
                delta,
                self.min_delta,
            )
        else:
            if status == "keep" and not significant:
                logging.warning(
                    "Keep below min_delta: %s=%.6f delta=%.6g < %.6g "
                    "(treated as no-improve for stop; best unchanged at %.6f)",
                    self.decision_metric,
                    metric,
                    delta,
                    self.min_delta,
                    self.best,
                )
            self.no_improve_streak += 1

        self.best_history.append(self.best)
        self._apply_framework_round(
            metric=metric,
            status=status,
            parsed=parsed,
            commit=commit,
            best_before=best_before,
        )

    def note_agent_outcome(self, code: int) -> None:
        if code == 0:
            self.consecutive_agent_failures = 0
        else:
            self.consecutive_agent_failures += 1
            logging.warning(
                "Agent failure streak=%s/%s (exit_code=%s)",
                self.consecutive_agent_failures,
                self.max_agent_failures,
                code,
            )

    def should_stop(self, iteration: int) -> str | None:
        if self.consecutive_agent_failures >= self.max_agent_failures:
            return (
                f"max_agent_failures "
                f"({self.consecutive_agent_failures}>={self.max_agent_failures})"
            )
        if iteration >= self.max_iterations:
            return f"max_iterations ({iteration}>={self.max_iterations})"
        if self.should_expand_sample():
            return None
        if (
            self.valid_rounds >= self.min_iterations
            and self.no_improve_streak >= self.patience
        ):
            return (
                f"patience "
                f"(no_improve_streak={self.no_improve_streak}>={self.patience}, "
                f"valid_rounds={self.valid_rounds}>={self.min_iterations}, "
                f"min_delta={self.min_delta:g})"
            )
        return None


def run_sample_expand_round(state: SearchState) -> str:
    """Expand K_train ×factor, retrain current best HEAD only, keep or roll back K.

    Does not replay earlier keep/discard commits. Never git-reset.
    Returns final status used on the results row (keep|discard|crash|missing).
    """
    if state.k_train is None or state.k_eval is None or state.k_train0 is None:
        state.sample_exhausted = True
        logging.warning("Sample expand skipped: missing K_train/K_eval/K_train0")
        return "missing"

    baseline_best = state.best
    old_k = int(state.k_train)
    trial_k = state.trial_k_train()
    if trial_k is None or trial_k <= old_k:
        state.sample_exhausted = True
        logging.info("Sample expand skipped: cannot grow K_train=%s", old_k)
        return "missing"

    scale = float(trial_k) / float(state.k_train0)
    logging.info(
        "Sample expand: trying K_train %s -> %s (×%.4g vs K_train0=%s); "
        "baseline_best=%s",
        old_k,
        trial_k,
        scale,
        state.k_train0,
        f"{baseline_best:.6f}" if baseline_best is not None else "None",
    )

    # Next expand only after SAMPLE_EXPAND_PATIENCE more non-improves, and
    # only if this round keeps. Model discard exhausts the sample axis.
    state.sample_expand_done = False
    if state.sample_wallclock_retry:
        state.sample_wallclock_retry = False
        state.sample_wallclock_used = True
    apply_compare_sampling_env(k_train=trial_k, k_eval=int(state.k_eval))
    state.k_train = trial_k

    from rnaomni.orch.loop import run_train_external

    train_exit = run_train_external()
    if is_encode_timeout(train_exit):
        row = latest_results_row()
        if row is None or (row.get("status") or "").strip() != "crash":
            append_orchestrator_crash_row("ENCODE_TIMEOUT")
        desc = format_sample_description(
            scale=scale,
            old_k=old_k,
            trial_k=trial_k,
            k_eval=int(state.k_eval),
            kept=False,
        )
        rewrite_latest_results_fields(status="crash", description=desc)
        state.k_train = old_k
        state.sample_scale = float(old_k) / float(state.k_train0)
        state.sync_sampling_env()
        state.sample_grow_blocked = True
        state.sample_exhausted = False
        state.sample_plateau = False
        logging.info(
            "Sample expand encode timeout → rollback K_train=%s grow_blocked (no git reset)",
            old_k,
        )
        state.rows_seen = len(read_results_rows())
        return "crash"

    row = latest_results_row()
    if row is None:
        logging.warning("Sample expand: no results row after train; rolling back K")
        state.k_train = old_k
        state.sample_scale = float(old_k) / float(state.k_train0)
        state.sync_sampling_env()
        state.no_improve_streak = 0
        return "missing"

    metric = parse_metric(row, state.decision_metric)
    status_raw = (row.get("status") or "").strip()

    if status_raw == "crash" or metric is None or train_exit != 0:
        desc = format_sample_description(
            scale=scale,
            old_k=old_k,
            trial_k=trial_k,
            k_eval=int(state.k_eval),
            kept=False,
        )
        rewrite_latest_results_fields(status="crash", description=desc)
        state.k_train = old_k
        state.sample_scale = float(old_k) / float(state.k_train0)
        state.sync_sampling_env()
        # Crash does not clear streak or stop immediately; lock further expands.
        state.sample_exhausted = True
        state.sample_plateau = False
        logging.info("Sample expand crash/fail → rollback K_train=%s (no git reset)", old_k)
        # Do not ingest crash into valid_rounds via normal path; leave row as crash.
        state.rows_seen = len(read_results_rows())
        return "crash"

    if baseline_best is None:
        significant = True
        delta = 0.0
    else:
        delta = improvement_delta(
            metric, baseline_best, higher_is_better=state.higher_is_better
        )
        significant = delta >= state.min_delta

    kept = bool(significant)
    if kept:
        final_status = "keep"
        state.k_train = trial_k
        state.sample_scale = scale
        state.sample_grow_blocked = False
        logging.info(
            "Sample expand KEEP: %s=%.6f delta=%s (>= min_delta=%g); "
            "retain K_train=%s",
            state.decision_metric,
            metric,
            f"{delta:.6g}" if baseline_best is not None else "n/a",
            state.min_delta,
            trial_k,
        )
    else:
        final_status = "discard"
        state.k_train = old_k
        state.sample_scale = float(old_k) / float(state.k_train0)
        logging.info(
            "Sample expand DISCARD: %s=%.6f delta=%.6g < min_delta=%g; "
            "rollback K_train=%s (no git reset)",
            state.decision_metric,
            metric,
            delta,
            state.min_delta,
            old_k,
        )

    state.sync_sampling_env()
    desc = format_sample_description(
        scale=scale,
        old_k=old_k,
        trial_k=trial_k,
        k_eval=int(state.k_eval),
        kept=kept,
    )
    rewrite_latest_results_fields(status=final_status, description=desc)

    # Ingest the rewritten row (keep: reset streak; discard: freeze sample axis only).
    ingested = state.ingest_new_rows()
    if ingested == 0:
        logging.warning("Sample expand row was not ingested after rewrite")
        if final_status == "keep":
            state.no_improve_streak = 0
            state.sample_exhausted = False
            state.sample_plateau = False
        else:
            state.sample_exhausted = True
            state.sample_plateau = False
    return final_status


