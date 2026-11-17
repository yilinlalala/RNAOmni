"""Data preparation package. Root ``prepare.py`` re-exports this."""

from __future__ import annotations

from features.pipeline import RECIPE_PATH
from rnaomni.paths import DATA_CARD_DIR, SPLIT_DIR
from rnaomni.prepare.constants import (
    ALPHABET,
    CLS_MMSEQS_COVERAGE,
    CLS_MMSEQS_MIN_SEQ_ID,
    CLS_TEST_FRAC,
    CLS_TRAIN_FRAC,
    CLS_VAL_FRAC,
    COMPARE_SPLIT_FILES,
    DATASET_DIR,
    DEFAULT_ACTIVE_CSV,
    DEFAULT_COMPARE_FOLD_IDS,
    DEFAULT_COMPARE_PER_CLASS_EVAL,
    DEFAULT_COMPARE_PER_CLASS_TRAIN,
    DEFAULT_COMPARE_ROOT,
    DEFAULT_COMPARE_VALID_NAME,
    DEFAULT_ENCODE_TIMEOUT_SEC,
    DEFAULT_SAMPLE_ROW_CAP_TRAIN,
    ENCODE_TIMEOUT_EXIT,
    FOLD_RANDOM_STATE,
    FOLD_TEST,
    FOLD_TRAIN,
    FOLD_VAL,
    LABEL_COL,
    MAX_SEQ_LEN,
    N_FOLDS,
    PROJECT_DIR,
    SEQUENCE_COL,
    TEST_FRACTION,
    TIME_BUDGET,
    search_fold_budget_sec,
    _env_bool_optional,
    _env_int_optional,
    _env_str,
    _env_truthy,
    _parse_fold_ids,
    _resolve_user_path,
)
from rnaomni.prepare.data import (
    _indices_for_split,
    encode_dataframe,
    gather_row_features,
    get_split_indices,
    iter_batches,
    load_prepared_data,
    make_dataloader,
)
from rnaomni.prepare.eval import (
    _macro_f1,
    evaluate_accuracy,
    evaluate_rmse,
    evaluate_spearman,
    evaluate_test_metrics,
)
from rnaomni.prepare.route import (
    apply_compare_sampling_env,
    apply_data_mode,
    apply_dataset_route,
    compare_active_csv_name,
    compare_dataset_ready,
    compare_full_data,
    compare_layout,
    discover_compare_fold_ids,
    compare_layout_error,
    compare_sampling_on,
    compare_split_paths,
    data_mode,
    dataset_route_prompt_block,
    encode_mode,
    encode_timeout_sec,
    peek_compare_task_type,
    resolve_compare_per_class_eval,
    resolve_compare_per_class_train,
    resolve_sample_row_cap_eval,
    resolve_sample_row_cap_train,
    sampling_on,
)
from rnaomni.prepare.sample import (
    _canonical_compare_frame,
    _row_cap_sample,
    _sample_main_csv_splits,
    _stratified_per_class_sample,
    get_split_indices_compare,
    load_compare_splits_dataframe,
)
from rnaomni.prepare.splits import (
    _csv_path,
    _csv_stem,
    _is_legal_sequence,
    _normalize_sequence,
    assign_classification_splits,
    assign_folds,
    build_data_card,
    classification_num_classes,
    clean_dataset,
    ensure_splits,
    infer_task_type,
    load_dataset,
    write_data_card,
)

# Import-time route (same as the old prepare.py module load).
apply_dataset_route()

_ROUTE_ATTRS = frozenset(
    {
        "ACTIVE_CSV",
        "COMPARE_FOLD_IDS",
        "COMPARE_PER_CLASS",
        "COMPARE_PER_CLASS_EVAL",
        "COMPARE_PER_CLASS_TRAIN",
        "COMPARE_ROOT",
        "COMPARE_ROW_LIMIT",
        "COMPARE_SAMPLE_SEED",
        "COMPARE_TASK_TYPE",
        "COMPARE_VALID_NAME",
        "USE_COMPARE_SPLITS",
    }
)
_DATA_ATTRS = frozenset({"INPUT_DIM", "_DATA_CACHE"})


def __getattr__(name: str):
    if name in _ROUTE_ATTRS:
        import rnaomni.prepare.route as route

        return getattr(route, name)
    if name in _DATA_ATTRS:
        import rnaomni.prepare.data as data

        return getattr(data, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _ROUTE_ATTRS | _DATA_ATTRS)
