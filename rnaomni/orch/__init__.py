"""Overnight orchestrator package. Root ``orchestrator.py`` re-exports this."""

from __future__ import annotations

from rnaomni.results_io import (
    ensure_results_tsv,
    results_header_for_task,
    results_tsv_path,
)
from rnaomni.orch.gitops import (
    BRANCH_PREFIX,
    git_branch,
    git_head,
    git_reset_hard,
    head_unchanged,
)
from rnaomni.orch.loop import (
    EXIT_BRIDGE_DEAD,
    _archive_on_stop,
    dataset_ready,
    feature_contract_ready,
    handle_empty_commit,
    log_latest_metrics,
    main,
    maybe_grant_wallclock_retry,
    needs_setup,
    parse_args,
    reconcile_after_train,
    run_once,
    run_train_external,
    setup_logging,
)
from rnaomni.orch.prompts import (
    PROMPT_ITER_TEMPLATE,
    PROMPT_SETUP_TEMPLATE,
    build_iter_prompt,
    build_prompt,
    decision_spec,
    default_min_delta,
    detect_task_type,
    format_framework_block,
)
from rnaomni.orch.note import (
    apply_sample_expand_to_note,
    build_opening_prompt,
    collect_data_block,
    ensure_opening_note,
    format_note_block,
    legal_note_head_name,
    n1_must_action,
    release_crash_repair,
    revise_campaign_after_status,
)
from rnaomni.orch.search import (
    DEFAULT_EFF_WINDOW,
    DEFAULT_K_TRAIN_CAP,
    DEFAULT_MAX_AGENT_FAILURES,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MIN_DELTA_CLASSIFICATION,
    DEFAULT_MIN_DELTA_REGRESSION,
    DEFAULT_MIN_ITERATIONS,
    DEFAULT_PATIENCE,
    DEFAULT_SAMPLE_EXPAND_FACTOR,
    DEFAULT_SAMPLE_EXPAND_PATIENCE,
    DEFAULT_FRAMEWORK,
    DEFAULT_FRAMEWORK_PATIENCE,
    FRAMEWORK_MENU,
    SearchCheckpoint,
    SearchState,
    clear_search_checkpoint,
    format_cause_block,
    format_sample_description,
    framework_rng_seed,
    infer_cause,
    is_framework_base_round,
    framework_base_lifts_global,
    load_search_checkpoint,
    parse_round_kind,
    parse_sample_description,
    run_sample_expand_round,
    save_search_checkpoint,
)
from rnaomni.orch.tsv import (
    CRASH_DIGEST_MAX,
    CRASH_MARK,
    append_orchestrator_crash_row,
    attach_crash_digest_to_latest_row,
    format_crash_prompt_block,
    improvement_delta,
    is_encode_timeout,
    is_sample_row,
    latest_results_row,
    parse_metric,
    read_results_rows,
    results_tsv_ready,
    rewrite_latest_results_fields,
    rewrite_latest_results_status,
    summarize_run_log,
)
from rnaomni.paths import ORCH_LOG, REPO, RUN_LOG, STATE_PATH

# Keep tsv.results_tsv_path as the patch target for smoke tests.
import rnaomni.orch.tsv as tsv  # noqa: E402

tsv.results_tsv_path = results_tsv_path
