"""CLI, agent turn, external train.py, and the overnight main loop."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import rnaomni.agent_backend as agent_backend
from rnaomni.agent_backend import (
    EXIT_TRANSPORT_DEAD,
    prompt as agent_prompt,
    resolve_backend,
    resolve_model,
)
from prepare import (
    ACTIVE_CSV,
    COMPARE_ROOT,
    USE_COMPARE_SPLITS,
    compare_dataset_ready,
    data_mode,
    dataset_route_prompt_block,
)
from rnaomni.results_io import ensure_results_tsv
from rnaomni.orch.gitops import (
    BRANCH_PREFIX,
    git_branch,
    git_head,
    git_reset_hard,
    head_unchanged,
)
from rnaomni.orch.prompts import (
    PROMPT_SETUP_TEMPLATE,
    build_iter_prompt,
    build_prompt,
    decision_spec,
    default_min_delta,
    detect_task_type,
)
from rnaomni.orch.note import (
    apply_sample_expand_to_note,
    ensure_opening_note,
    legal_note_head_name,
    n1_must_action,
    release_crash_repair,
    revise_campaign_after_status,
)
from rnaomni.orch.search import (
    DEFAULT_EFF_WINDOW,
    DEFAULT_MAX_AGENT_FAILURES,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MIN_DELTA_CLASSIFICATION,
    DEFAULT_MIN_DELTA_REGRESSION,
    DEFAULT_MIN_ITERATIONS,
    DEFAULT_PATIENCE,
    DEFAULT_SAMPLE_EXPAND_PATIENCE,
    SearchState,
    framework_base_lifts_global,
    is_framework_base_round,
    load_search_checkpoint,
    parse_round_kind,
    run_sample_expand_round,
    save_search_checkpoint,
)
from rnaomni.orch.tsv import (
    append_orchestrator_crash_row,
    attach_crash_digest_to_latest_row,
    improvement_delta,
    is_encode_timeout,
    latest_results_row,
    parse_metric,
    read_results_rows,
    rewrite_latest_results_status,
    summarize_run_log,
)
from rnaomni.paths import ORCH_LOG, REPO, RESULTS_DIR, RUN_LOG, STATE_PATH

EXIT_BRIDGE_DEAD = EXIT_TRANSPORT_DEAD

def setup_logging(log_file: Path | None, *, truncate: bool = False) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if truncate else "a"
        handlers.append(logging.FileHandler(log_file, mode=mode, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )


def run_train_external() -> int:
    """Run encode+train outside the SDK. Returns train.py exit code."""
    run_log = RUN_LOG
    run_log.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Starting external train.py (encode+train outside SDK)...")
    with run_log.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(
            [sys.executable, str(REPO / "train.py")],
            cwd=REPO,
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
    logging.info("External train.py finished: exit_code=%s", proc.returncode)
    return int(proc.returncode)


def _full_encode_established(state: SearchState | None) -> bool:
    """True when this campaign already completed at least one full-mode round."""
    if state is None:
        return False
    return state.best is not None or int(getattr(state, "valid_rounds", 0) or 0) > 0


def handle_empty_commit(state: SearchState) -> str:
    """Agent produced no new commit.

    Crash-repair empty commits release the crash veto so the next round can
    follow note.next instead of burning max_agent_failures. Ordinary empty
    commits still count as agent failures.

    Returns ``crash_release`` or ``agent_fail``.
    """
    if state.last_non_sample_crash:
        logging.warning(
            "Crash repair produced no commit; release crash veto "
            "(do not count toward max_agent_failures)"
        )
        state.last_non_sample_crash = False
        try:
            release_crash_repair(state)
        except Exception as err:
            logging.warning("Crash-release note update failed: %s", err)
        return "crash_release"
    state.note_agent_outcome(1)
    return "agent_fail"


def reconcile_after_train(
    *,
    pre_head: str,
    min_delta: float,
    best_before: float | None,
    decision_metric: str,
    higher_is_better: bool,
    train_exit: int,
    state: SearchState | None = None,
) -> str:
    """
    Apply min_delta keep rule and git-reset on discard/crash.

    Framework first shot (framework_base / new name): TSV stays keep as that
    head's baseline. If it does not lift the global best, reset the working
    tree to the global-best commit so later F/θ shots are not stuck on a
    worse head. In-framework [model] keep/discard uses the framework best
    (upgrade discard→keep when the shot beats that baseline).

    Full encode timeout: switch to sample only when no full round has
    succeeded yet (dataset too heavy). After an established full baseline,
    timeout is a per-shot crash and the campaign stays full.

    Returns the final status string used for logging (keep|discard|crash|missing|encode_timeout).
    """
    if is_encode_timeout(train_exit):
        stay_full = data_mode() == "full" and _full_encode_established(state)
        if data_mode() == "full" and not stay_full:
            logging.info("Encode timeout in full mode — no TSV row, will switch to sample")
            return "encode_timeout"
        row = latest_results_row()
        if row is None or (row.get("status") or "").strip() != "crash":
            append_orchestrator_crash_row("ENCODE_TIMEOUT")
        attach_crash_digest_to_latest_row(summarize_run_log())
        if stay_full:
            logging.info(
                "Encode timeout in full mode after established baseline → "
                "crash this shot, stay full, reset %s",
                pre_head[:7],
            )
        else:
            logging.info(
                "Encode timeout in sample mode → crash row + git reset %s",
                pre_head[:7],
            )
        git_reset_hard(pre_head)
        return "crash"

    row = latest_results_row()
    if row is None:
        logging.warning(
            "No results.tsv row after train (exit=%s); resetting to %s",
            train_exit,
            pre_head[:7],
        )
        git_reset_hard(pre_head)
        return "missing"

    status = (row.get("status") or "").strip()
    metric = parse_metric(row, decision_metric)
    parsed = parse_round_kind(row.get("description") or "")

    if status == "crash" or metric is None or train_exit != 0:
        if status != "crash":
            rewrite_latest_results_status("crash", note="train_exit_nonzero_or_bad_metric")
            status = "crash"
        attach_crash_digest_to_latest_row(summarize_run_log())
        logging.info("Crash/failed train → git reset --hard %s", pre_head[:7])
        git_reset_hard(pre_head)
        return status

    framework_base = is_framework_base_round(parsed, state)
    if framework_base:
        if status != "keep":
            rewrite_latest_results_status("keep", note="framework_base")
            status = "keep"
        lifted = framework_base_lifts_global(
            metric,
            best_before,
            min_delta=min_delta,
            higher_is_better=higher_is_better,
        )
        if not lifted:
            reset_to = pre_head
            if state is not None and state.best_commit:
                reset_to = str(state.best_commit)
            logging.info(
                "Framework base flat vs global best; TSV stays keep, reset HEAD → %s",
                reset_to[:7],
            )
            git_reset_hard(reset_to)
            return "keep"
        logging.info(
            "Framework base keep at HEAD %s (%s=%.6f); skip global reset",
            (git_head() or "")[:7],
            decision_metric,
            metric,
        )
        return "keep"

    baseline = best_before
    reset_to = pre_head
    if state is not None and parsed.get("kind") == "model":
        baseline = state.decision_baseline(parsed, best_before)
        reset_to = state.framework_reset_commit() or pre_head

    if baseline is None:
        significant = True
        delta = 0.0
    else:
        delta = improvement_delta(
            metric, baseline, higher_is_better=higher_is_better
        )
        significant = delta >= min_delta

    if significant:
        if status == "discard":
            rewrite_latest_results_status(
                "keep",
                note=f"vs_round_baseline delta={delta:.6g}",
            )
            logging.info(
                "Upgraded discard→keep (delta=%.6g >= min_delta=%g vs in-round baseline)",
                delta,
                min_delta,
            )
        logging.info(
            "Keep at HEAD %s (%s=%.6f delta=%s)",
            (git_head() or "")[:7],
            decision_metric,
            metric,
            f"{delta:.6g}" if baseline is not None else "n/a",
        )
        return "keep"

    if status == "keep":
        rewrite_latest_results_status(
            "discard",
            note=f"below min_delta={min_delta:g} delta={delta:.6g}",
        )
        logging.info(
            "Downgraded keep→discard (delta=%.6g < min_delta=%g); reset to %s",
            delta,
            min_delta,
            reset_to[:7],
        )
    else:
        logging.info("Discard → git reset --hard %s", reset_to[:7])
    git_reset_hard(reset_to)
    return "discard"


def maybe_grant_wallclock_retry(
    state: "SearchState", *, train_exit: int, status: str
) -> None:
    """After dropping a heavy feature, allow one sample-axis expand retry."""
    if status != "crash":
        return
    if not state.sample_on or not state.sample_grow_blocked:
        return
    if state.sample_wallclock_used or state.sample_wallclock_retry:
        return
    if not is_encode_timeout(train_exit):
        return
    state.sample_wallclock_retry = True
    logging.info(
        "Granted one sample-axis wall-clock expand retry after encode timeout"
    )


def dataset_ready() -> bool:
    if USE_COMPARE_SPLITS:
        return compare_dataset_ready()
    return (REPO / "dataset" / ACTIVE_CSV).is_file()


def feature_contract_ready() -> bool:
    from features.pipeline import RECIPE_PATH, REGISTRY_PATH

    return REGISTRY_PATH.is_file() and RECIPE_PATH.is_file()


def needs_setup() -> bool:
    branch = git_branch() or ""
    if not branch.startswith(BRANCH_PREFIX):
        return True
    if not dataset_ready():
        return True
    if not feature_contract_ready():
        return True
    # results.tsv is orchestrator-owned via ensure_results_tsv(); not a setup gate.
    return False


def log_latest_metrics() -> None:
    """Append key metrics from the latest results.tsv row to the orchestrator log."""
    row = latest_results_row()
    if row is None:
        logging.info("No results.tsv data row to log yet.")
        return
    logging.info("Metrics: %s", " ".join(f"{k}={v}" for k, v in row.items()))


def run_once(
    prompt: str,
    *,
    backend: str,
    model: str,
    api_key: str | None = None,
) -> int:
    logging.info(
        "Starting agent turn (backend=%s model=%s)",
        backend,
        model,
    )
    result = agent_prompt(
        prompt,
        cwd=REPO,
        model=model,
        backend=backend,
        api_key=api_key,
    )
    if result.transport_dead:
        logging.error(
            "Transport dead (backend=%s): %s — exit %s for process restart "
            "(not counted as agent failure)",
            backend,
            result.message,
            EXIT_BRIDGE_DEAD,
        )
        return EXIT_BRIDGE_DEAD

    logging.info("Agent finished: status=%s backend=%s", result.status, backend)
    if result.result:
        preview = result.result.strip().replace("\n", " ")
        if len(preview) > 500:
            preview = preview[:500] + "..."
        logging.info("Result preview: %s", preview)

    if result.status == "error":
        if result.message:
            logging.error("Agent error: %s", result.message[:500])
        return result.as_exit_code()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run RNAOmni experiments via agent_backend "
            "(default OpenCode; --backend cursor for Cursor SDK)."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=["cursor", "opencode"],
        default=None,
        help=(
            "Agent transport (default: AGENT_BACKEND or "
            f"{agent_backend.DEFAULT_BACKEND})."
        ),
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help=f"Hard cap on Iter rounds (1–{DEFAULT_MAX_ITERATIONS}; default {DEFAULT_MAX_ITERATIONS}). Setup excluded.",
    )
    parser.add_argument(
        "--min-iterations",
        type=int,
        default=DEFAULT_MIN_ITERATIONS,
        help=f"Minimum valid keep/discard rounds before patience may stop (default {DEFAULT_MIN_ITERATIONS}).",
    )
    parser.add_argument(
        "--max-agent-failures",
        type=int,
        default=DEFAULT_MAX_AGENT_FAILURES,
        help=f"Stop after this many consecutive SDK/agent failures (default {DEFAULT_MAX_AGENT_FAILURES}).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help=f"Stop after this many consecutive non-significant rounds once min-iterations met (default {DEFAULT_PATIENCE}).",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=None,
        help="Minimum metric improvement to keep AND to reset patience "
        f"(default: {DEFAULT_MIN_DELTA_CLASSIFICATION} classification / "
        f"{DEFAULT_MIN_DELTA_REGRESSION} regression).",
    )
    parser.add_argument(
        "--eff-window",
        type=int,
        default=DEFAULT_EFF_WINDOW,
        help=f"Window length for logged marginal-efficiency (default {DEFAULT_EFF_WINDOW}).",
    )
    parser.add_argument(
        "--sample-expand-patience",
        type=int,
        default=DEFAULT_SAMPLE_EXPAND_PATIENCE,
        help=(
            "When exploration sampling is on, expand K_train ×2 after this many "
            f"consecutive non-significant rounds (default {DEFAULT_SAMPLE_EXPAND_PATIENCE}). "
            "Keep → may expand again later. Discard → data axis exhausted, stop."
        ),
    )
    parser.add_argument(
        "--sleep-ok",
        type=float,
        default=10.0,
        help="Seconds to wait after a successful round before the next one.",
    )
    parser.add_argument(
        "--sleep-fail",
        type=float,
        default=60.0,
        help="Seconds to wait after startup failure before retrying.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model id for the selected backend "
            "(default: OPENCODE_MODEL / CURSOR_MODEL by backend)."
        ),
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=ORCH_LOG,
        help=(
            "Orchestrator log path (default: workspace/orchestrator.log). "
            "Truncated when starting a new campaign; appended on resume."
        ),
    )
    args = parser.parse_args()
    if args.max_iterations < 1 or args.max_iterations > DEFAULT_MAX_ITERATIONS:
        parser.error(
            f"--max-iterations must be in 1..{DEFAULT_MAX_ITERATIONS}, got {args.max_iterations}"
        )
    if args.min_iterations < 1:
        parser.error("--min-iterations must be >= 1")
    if args.min_iterations > args.max_iterations:
        parser.error("--min-iterations cannot exceed --max-iterations")
    if args.max_agent_failures < 1:
        parser.error("--max-agent-failures must be >= 1")
    if args.patience < 1:
        parser.error("--patience must be >= 1")
    if args.eff_window < 1:
        parser.error("--eff-window must be >= 1")
    if args.sample_expand_patience < 1:
        parser.error("--sample-expand-patience must be >= 1")
    if args.min_delta is not None and args.min_delta < 0:
        parser.error("--min-delta must be >= 0")
    return args


def _archive_on_stop(stop_reason: str) -> None:
    # Parallel lanes inject RNAOMNI_BATCH_ID; finish_job → archive_lane owns the
    # snapshot. Skipping here avoids a second archive_run that would create
    # workspace/runs/<tag>_<stem>_HHMMSS.
    if os.environ.get("RNAOMNI_BATCH_ID", "").strip():
        logging.info("parallel owns archive (RNAOMNI_BATCH_ID set); skip orchestrator archive_run")
        return
    try:
        from rnaomni.archive import archive_run

        out = archive_run(stop_reason=stop_reason)
        logging.info("Archived search outputs to %s", out)
    except Exception as err:
        logging.error("archive_run failed (search stop unchanged): %s", err)


def main() -> int:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(args.log_file, truncate=not STATE_PATH.is_file())
    from rnaomni.threads import apply_thread_limits

    n_threads = apply_thread_limits()
    if n_threads:
        logging.info("CPU threads capped at %s (RNAOMNI_CPU_THREADS / OMP)", n_threads)

    try:
        backend = resolve_backend(args.backend)
    except ValueError as err:
        logging.error("%s", err)
        return 1
    model = resolve_model(backend, args.model)
    api_key = os.environ.get("CURSOR_API_KEY")
    if backend == "cursor" and not api_key:
        logging.error("CURSOR_API_KEY is not set (required for --backend cursor).")
        return 1
    logging.info("Agent transport: backend=%s model=%s", backend, model)

    task_type = detect_task_type()
    decision_metric, higher_is_better, _ = decision_spec(task_type)
    min_delta = (
        args.min_delta if args.min_delta is not None else default_min_delta(task_type)
    )
    results_path = ensure_results_tsv(task_type)
    logging.info("Active results file: %s", results_path)

    try:
        prompt_setup = build_prompt(
            PROMPT_SETUP_TEMPLATE,
            dataset_route_block=dataset_route_prompt_block(),
        )
    except FileNotFoundError as err:
        logging.error("Skill registry resolution failed: %s", err)
        return 1

    if needs_setup():
        logging.info("Setup required; launching setup agent.")
        code = run_once(
            prompt_setup, backend=backend, model=model, api_key=api_key
        )
        if code != 0:
            return code
        if needs_setup():
            logging.error("Setup agent finished but setup checks still fail.")
            return 2
        logging.info("Setup complete.")
        # Dataset/contract may have changed during setup — refresh task + results.
        task_type = detect_task_type()
        decision_metric, higher_is_better, _ = decision_spec(task_type)
        if args.min_delta is None:
            min_delta = default_min_delta(task_type)
        results_path = ensure_results_tsv(task_type)
        logging.info("Post-setup results file: %s (task_type=%s)", results_path, task_type)

    state = SearchState(
        decision_metric=decision_metric,
        higher_is_better=higher_is_better,
        min_delta=min_delta,
        patience=args.patience,
        min_iterations=args.min_iterations,
        max_iterations=args.max_iterations,
        max_agent_failures=args.max_agent_failures,
        eff_window=args.eff_window,
        sample_expand_patience=args.sample_expand_patience,
    )
    state.configure_sampling_from_env()
    n_rows = len(read_results_rows())
    checkpoint = load_search_checkpoint(rows_seen=n_rows, valid_rounds=0)
    # Restore data_mode before TSV replay so keep/discard filter by current mode.
    checkpoint.apply_sample_fields_to(state)
    state.sync_from_tsv()
    checkpoint.apply_sample_fields_to(state)
    checkpoint.apply_framework_fields_to(state)
    iteration = max(int(checkpoint.iteration), int(state.valid_rounds))
    logging.info(
        "Search config: task_type=%s metric=%s min_delta=%g patience=%s "
        "min_iterations=%s max_iterations=%s max_agent_failures=%s eff_window=%s "
        "sample_on=%s K_train=%s K_eval=%s expand_patience=%s expand_done=%s "
        "initial_best=%s resumed_iteration=%s use_compare=%s compare_root=%s results=%s",
        task_type,
        decision_metric,
        min_delta,
        args.patience,
        args.min_iterations,
        args.max_iterations,
        args.max_agent_failures,
        args.eff_window,
        state.sample_on,
        state.k_train,
        state.k_eval,
        state.sample_expand_patience,
        state.sample_expand_done,
        f"{state.best:.6f}" if state.best is not None else "None",
        iteration,
        USE_COMPARE_SPLITS,
        COMPARE_ROOT,
        results_path.name,
    )
    try:
        ensure_opening_note(
            state,
            task_type=task_type,
            backend=backend,
            model=model,
            api_key=api_key,
        )
    except Exception as err:
        logging.warning("Opening campaign note failed (search continues): %s", err)
    if checkpoint.stop_reason:
        # Bridge-death false stops must not block resume; clear and continue.
        if "max_agent_failures" in checkpoint.stop_reason:
            logging.warning(
                "Clearing prior stop_reason=%s so search can resume after bridge death.",
                checkpoint.stop_reason,
            )
            save_search_checkpoint(
                iteration=checkpoint.iteration,
                rows_seen=state.rows_seen,
                valid_rounds=state.valid_rounds,
                no_improve_streak=state.no_improve_streak,
                best=state.best,
                stop_reason=None,
                state=state,
            )
        else:
            logging.info(
                "Prior search already stopped (%s); not continuing. "
                "Delete %s or re-init results.tsv to start a new search.",
                checkpoint.stop_reason,
                STATE_PATH.name,
            )
            return 0

    last_code = 0
    active_results_name = results_path.name
    while True:
        # Re-detect each round so COMPARE_ROOT / task switches stay in sync.
        task_type = detect_task_type()
        decision_metric, higher_is_better, _ = decision_spec(task_type)
        if args.min_delta is None:
            min_delta = default_min_delta(task_type)
        state.decision_metric = decision_metric
        state.higher_is_better = higher_is_better
        state.min_delta = min_delta
        results_path = ensure_results_tsv(task_type)
        if results_path.name != active_results_name:
            logging.info(
                "Results file switched %s → %s; replaying search state",
                active_results_name,
                results_path.name,
            )
            active_results_name = results_path.name
            state.configure_sampling_from_env()
            state.sync_from_tsv()

        # Phase 2: stagnated under sampling → expand K_train (current best only).
        if state.should_expand_sample():
            logging.info(
                "Triggering sample expand (no_improve_streak=%s>=%s); "
                "no agent this round.",
                state.no_improve_streak,
                state.sample_expand_patience,
            )
            expand_status = run_sample_expand_round(state)
            logging.info("Sample expand finished: status=%s", expand_status)
            try:
                apply_sample_expand_to_note(state, status=expand_status)
            except Exception as err:
                logging.warning("Sample-expand note update failed (search continues): %s", err)
            log_latest_metrics()
            state.log_status(iteration=iteration, extra="sample_expand")
            stop_reason = state.should_stop(iteration)
            save_search_checkpoint(
                iteration=iteration,
                rows_seen=state.rows_seen,
                valid_rounds=state.valid_rounds,
                no_improve_streak=state.no_improve_streak,
                best=state.best,
                stop_reason=stop_reason,
                state=state,
            )
            if stop_reason:
                state.stop_reason = stop_reason
                logging.info("Stopping search: stop_reason=%s", stop_reason)
                _archive_on_stop(stop_reason)
                return 0
            logging.info(
                "Sleeping %.1fs after sample expand before next round.",
                args.sleep_ok,
            )
            time.sleep(args.sleep_ok)
            continue

        stop_reason = state.should_stop(iteration)
        if stop_reason:
            state.stop_reason = stop_reason
            logging.info("Stopping search: stop_reason=%s", stop_reason)
            state.log_status(iteration=iteration, extra=f"stop_reason={stop_reason}")
            save_search_checkpoint(
                iteration=iteration,
                rows_seen=state.rows_seen,
                valid_rounds=state.valid_rounds,
                no_improve_streak=state.no_improve_streak,
                best=state.best,
                stop_reason=stop_reason,
                state=state,
            )
            _archive_on_stop(stop_reason)
            return 0

        iteration += 1
        logging.info(
            "Iteration %s starting (task_type=%s results=%s K_train=%s).",
            iteration,
            task_type,
            results_path.name,
            state.k_train,
        )
        run_tag = os.environ.get("RNAOMNI_BATCH_ID", "").strip() or (git_branch() or "")
        note_head = legal_note_head_name(state)
        state.ensure_next_framework(run_tag or None, note_head=note_head)
        if state.next_framework:
            target = state.best_commit
            head = git_head() or ""
            if target and head and not head.startswith(str(target)) and not str(target).startswith(head[:7]):
                logging.info(
                    "Framework switch: reset HEAD %s → global best %s before next=%s",
                    head[:7],
                    str(target)[:7],
                    state.next_framework,
                )
                git_reset_hard(str(target))
        must = n1_must_action(state=state)
        logging.info("N2 Iter MUST: %s note_head=%s", must, note_head)
        try:
            prompt_iter = build_iter_prompt(
                min_delta=min_delta, best=state.best, state=state
            )
        except FileNotFoundError as err:
            logging.error("Skill registry resolution failed: %s", err)
            return 1

        pre_head = git_head()
        if not pre_head:
            logging.error("Cannot resolve git HEAD before iteration.")
            return 1

        last_code = run_once(
            prompt_iter, backend=backend, model=model, api_key=api_key
        )

        if last_code == EXIT_BRIDGE_DEAD:
            # Edit-only session died; still ingest anything already on disk, then exit.
            ingested = state.ingest_new_rows()
            if ingested:
                logging.info(
                    "Ingested %s row(s) present before transport death.", ingested
                )
            log_latest_metrics()
            state.log_status(iteration=iteration, extra="transport_dead")
            save_search_checkpoint(
                iteration=iteration,
                rows_seen=state.rows_seen,
                valid_rounds=state.valid_rounds,
                no_improve_streak=state.no_improve_streak,
                best=state.best,
                stop_reason=None,
                state=state,
            )
            logging.error(
                "Exiting for orchestrator process restart (exit_code=%s). "
                "Re-run: python orchestrator.py",
                EXIT_BRIDGE_DEAD,
            )
            return EXIT_BRIDGE_DEAD

        if last_code != 0:
            state.note_agent_outcome(last_code)
            log_latest_metrics()
            state.log_status(iteration=iteration)
            stop_reason = state.should_stop(iteration)
            save_search_checkpoint(
                iteration=iteration,
                rows_seen=state.rows_seen,
                valid_rounds=state.valid_rounds,
                no_improve_streak=state.no_improve_streak,
                best=state.best,
                stop_reason=stop_reason,
                state=state,
            )
            if stop_reason:
                state.stop_reason = stop_reason
                logging.info("Stopping search: stop_reason=%s", stop_reason)
                _archive_on_stop(stop_reason)
                return last_code
            logging.info(
                "Sleeping %.1fs before next round (exit_code=%s).",
                args.sleep_fail,
                last_code,
            )
            time.sleep(args.sleep_fail)
            continue

        if head_unchanged(pre_head):
            # Sample expand already retrained current best. Do not replay HEAD.
            empty = handle_empty_commit(state)
            if empty == "crash_release":
                logging.warning(
                    "Agent produced no new commit (HEAD still %s); "
                    "crash veto released, skip train.",
                    pre_head[:7],
                )
                extra = "crash_repair_no_commit"
            else:
                logging.warning(
                    "Agent produced no new commit (HEAD still %s); skip train.",
                    pre_head[:7],
                )
                extra = "no_commit_skip_train"
            log_latest_metrics()
            state.log_status(iteration=iteration, extra=extra)
            stop_reason = state.should_stop(iteration)
            save_search_checkpoint(
                iteration=iteration,
                rows_seen=state.rows_seen,
                valid_rounds=state.valid_rounds,
                no_improve_streak=state.no_improve_streak,
                best=state.best,
                stop_reason=stop_reason,
                state=state,
            )
            if stop_reason:
                state.stop_reason = stop_reason
                logging.info("Stopping search: stop_reason=%s", stop_reason)
                _archive_on_stop(stop_reason)
                return 1
            logging.info(
                "Sleeping %.1fs before next round (no new commit).",
                args.sleep_fail,
            )
            time.sleep(args.sleep_fail)
            continue

        # Agent edit+commit succeeded — run encode+train outside SDK.
        best_before = state.best
        train_exit = run_train_external()
        final_status = reconcile_after_train(
            pre_head=pre_head,
            min_delta=min_delta,
            best_before=best_before,
            decision_metric=decision_metric,
            higher_is_better=higher_is_better,
            train_exit=train_exit,
            state=state,
        )
        if final_status == "encode_timeout":
            state.switch_to_sample(task_type)
            logging.info("Retrying same HEAD after switch to sample")
            train_exit = run_train_external()
            final_status = reconcile_after_train(
                pre_head=pre_head,
                min_delta=min_delta,
                best_before=state.best,
                decision_metric=decision_metric,
                higher_is_better=higher_is_better,
                train_exit=train_exit,
                state=state,
            )
        maybe_grant_wallclock_retry(
            state, train_exit=train_exit, status=final_status
        )
        logging.info("Post-train reconcile status=%s", final_status)

        state.note_agent_outcome(0)
        ingested = state.ingest_new_rows()
        if ingested == 0:
            logging.warning(
                "No new keep/discard row ingested after external train "
                "(status=%s train_exit=%s)",
                final_status,
                train_exit,
            )
        try:
            revise_campaign_after_status(
                state,
                status=final_status,
                backend=backend,
                model=model,
                api_key=api_key,
            )
        except Exception as err:
            logging.warning("Campaign note revision failed (search continues): %s", err)
        log_latest_metrics()
        state.log_status(iteration=iteration)

        stop_reason = state.should_stop(iteration)
        save_search_checkpoint(
            iteration=iteration,
            rows_seen=state.rows_seen,
            valid_rounds=state.valid_rounds,
            no_improve_streak=state.no_improve_streak,
            best=state.best,
            stop_reason=stop_reason,
            state=state,
        )
        if stop_reason:
            state.stop_reason = stop_reason
            logging.info("Stopping search: stop_reason=%s", stop_reason)
            state.log_status(iteration=iteration, extra=f"stop_reason={stop_reason}")
            _archive_on_stop(stop_reason)
            return 0

        logging.info(
            "Sleeping %.1fs before next round (exit_code=0).", args.sleep_ok
        )
        time.sleep(args.sleep_ok)


