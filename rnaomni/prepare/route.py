"""Dataset route, sampling env, and compare-layout helpers.

Mutable process state (ACTIVE_CSV, COMPARE_*, USE_COMPARE_SPLITS, …) lives here.
Other prep modules read these via ``from rnaomni.prepare import route as _route``.
"""

from __future__ import annotations

import os

import pandas as pd

from rnaomni.prepare.constants import (
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
    LABEL_COL,
    _env_bool_optional,
    _env_int_optional,
    _env_str,
    _parse_fold_ids,
    _resolve_user_path,
)

# Resolved by apply_dataset_route() (import + CLI). Do not hard-toggle.
ACTIVE_CSV = DEFAULT_ACTIVE_CSV
COMPARE_ROOT = DEFAULT_COMPARE_ROOT
COMPARE_FOLD_IDS = list(DEFAULT_COMPARE_FOLD_IDS)
COMPARE_VALID_NAME = DEFAULT_COMPARE_VALID_NAME
USE_COMPARE_SPLITS = False
COMPARE_TASK_TYPE: str | None = None

COMPARE_PER_CLASS = _env_int_optional("COMPARE_PER_CLASS")
COMPARE_PER_CLASS_TRAIN = _env_int_optional("COMPARE_PER_CLASS_TRAIN")
COMPARE_PER_CLASS_EVAL = _env_int_optional("COMPARE_PER_CLASS_EVAL")
COMPARE_SAMPLE_SEED = (
    int(os.environ["COMPARE_SAMPLE_SEED"])
    if os.environ.get("COMPARE_SAMPLE_SEED")
    else 0
)


def data_mode() -> str:
    """full (default) or sample. Orchestrator-owned via RNAOMNI_DATA_MODE."""
    raw = str(os.environ.get("RNAOMNI_DATA_MODE", "full")).strip().lower()
    return "sample" if raw == "sample" else "full"


def sampling_on() -> bool:
    return data_mode() == "sample"


def encode_timeout_sec() -> int:
    raw = os.environ.get("ENCODE_TIMEOUT_SEC")
    if raw is None or str(raw).strip() == "":
        return int(DEFAULT_ENCODE_TIMEOUT_SEC)
    return max(0, int(raw))


def encode_mode() -> str:
    """precompute (overnight train.py) or online (finalize 15.2A)."""
    raw = str(os.environ.get("RNAOMNI_ENCODE_MODE", "precompute")).strip().lower()
    return "online" if raw == "online" else "precompute"


def resolve_sample_row_cap_train() -> int:
    raw = _env_int_optional("SAMPLE_ROW_CAP_TRAIN")
    if raw is not None:
        return int(raw)
    return int(DEFAULT_SAMPLE_ROW_CAP_TRAIN)


def resolve_sample_row_cap_eval() -> int:
    raw = _env_int_optional("SAMPLE_ROW_CAP_EVAL")
    if raw is not None:
        return int(raw)
    return resolve_sample_row_cap_train()


def compare_full_data() -> bool:
    """True unless the orchestrator switched this process to sample mode."""
    return not sampling_on()


def resolve_compare_per_class_train() -> int | None:
    """Train per-class K: TRAIN → unified → EVAL → default 10 (unless formal)."""
    if compare_full_data():
        return None
    for name in (
        "COMPARE_PER_CLASS_TRAIN",
        "COMPARE_PER_CLASS",
        "COMPARE_PER_CLASS_EVAL",
    ):
        value = _env_int_optional(name)
        if value is not None:
            return value
    return int(DEFAULT_COMPARE_PER_CLASS_TRAIN)


def resolve_compare_per_class_eval() -> int | None:
    """Val/test per-class K: EVAL → unified → TRAIN → default 10 (unless formal)."""
    if compare_full_data():
        return None
    for name in (
        "COMPARE_PER_CLASS_EVAL",
        "COMPARE_PER_CLASS",
        "COMPARE_PER_CLASS_TRAIN",
    ):
        value = _env_int_optional(name)
        if value is not None:
            return value
    return int(DEFAULT_COMPARE_PER_CLASS_EVAL)


def apply_data_mode(mode: str) -> None:
    text = str(mode).strip().lower()
    if text not in {"full", "sample"}:
        raise ValueError(f"data_mode must be full|sample, got {mode!r}")
    os.environ["RNAOMNI_DATA_MODE"] = text


def apply_compare_sampling_env(*, k_train: int, k_eval: int) -> None:
    """Pin train/eval caps into env for this process and child train.py runs."""
    global COMPARE_PER_CLASS, COMPARE_PER_CLASS_TRAIN, COMPARE_PER_CLASS_EVAL
    k_train_i = int(k_train)
    k_eval_i = int(k_eval)
    if k_train_i < 1 or k_eval_i < 1:
        raise ValueError(f"K_train/K_eval must be >= 1, got {k_train_i}/{k_eval_i}")
    os.environ["COMPARE_PER_CLASS_TRAIN"] = str(k_train_i)
    os.environ["COMPARE_PER_CLASS_EVAL"] = str(k_eval_i)
    os.environ["SAMPLE_ROW_CAP_TRAIN"] = str(k_train_i)
    os.environ["SAMPLE_ROW_CAP_EVAL"] = str(k_eval_i)
    os.environ.pop("COMPARE_PER_CLASS", None)
    os.environ.pop("COMPARE_FULL", None)
    os.environ.pop("COMPARE_NO_SAMPLE", None)
    apply_data_mode("sample")
    COMPARE_PER_CLASS = None
    COMPARE_PER_CLASS_TRAIN = k_train_i
    COMPARE_PER_CLASS_EVAL = k_eval_i


def compare_sampling_on() -> bool:
    """True when RNAOMNI_DATA_MODE=sample (HEK and COMPARE, cls and reg)."""
    return sampling_on()


# Deprecated: COMPARE_ROW_LIMIT used df.head(N) and broke 88-way labels. Ignored.
if os.environ.get("COMPARE_ROW_LIMIT") and not compare_sampling_on():
    print(
        "WARNING: COMPARE_ROW_LIMIT is deprecated (was unsafe head-N). "
        "Ignoring it. Sampling is RNAOMNI_DATA_MODE=sample (orchestrator-owned)."
    )
COMPARE_ROW_LIMIT = None  # kept name for older call sites; always unused


def discover_compare_fold_ids(compare_root: str) -> list[int]:
    """Return sorted fold ids if ``foldN/`` dirs exist, else ``[0]`` for a flat layout."""
    if not compare_root or not os.path.isdir(compare_root):
        return list(DEFAULT_COMPARE_FOLD_IDS)
    ids: list[int] = []
    try:
        names = os.listdir(compare_root)
    except OSError:
        return list(DEFAULT_COMPARE_FOLD_IDS)
    for name in names:
        if not name.startswith("fold"):
            continue
        suffix = name[4:]
        if suffix.isdigit() and os.path.isdir(os.path.join(compare_root, name)):
            ids.append(int(suffix))
    return sorted(ids) if ids else list(DEFAULT_COMPARE_FOLD_IDS)


def compare_layout() -> str:
    """Return 'fold' if fold{id}/ exists for the first fold, else 'flat' (CSVs at root)."""
    if not COMPARE_FOLD_IDS:
        return "flat"
    fold_dir = os.path.join(COMPARE_ROOT, f"fold{COMPARE_FOLD_IDS[0]}")
    return "fold" if os.path.isdir(fold_dir) else "flat"


def _compare_split_dir(fold_id: int) -> str:
    """fold{N}/ if that directory exists, else the compare root (flat train/val/test)."""
    fold_dir = os.path.join(COMPARE_ROOT, f"fold{int(fold_id)}")
    return fold_dir if os.path.isdir(fold_dir) else COMPARE_ROOT


def compare_split_paths(fold_id: int) -> dict[str, str]:
    root = _compare_split_dir(fold_id)
    return {
        "train": os.path.join(root, "train.csv"),
        "val": os.path.join(root, "val.csv"),
        "test": os.path.join(root, "test.csv"),
    }


def _compare_split_filenames(split_dir: str) -> set[str]:
    if not os.path.isdir(split_dir):
        return set()
    return {name for name in os.listdir(split_dir) if os.path.isfile(os.path.join(split_dir, name))}


def compare_layout_error(split_dir: str) -> str:
    found = sorted(_compare_split_filenames(split_dir))
    return (
        "COMPARE splits must be exactly train.csv, val.csv, test.csv under "
        f"{split_dir}; found {found}"
    )


def compare_dataset_ready() -> bool:
    if not COMPARE_FOLD_IDS:
        return False
    return all(
        _compare_split_filenames(_compare_split_dir(int(fid))) == set(COMPARE_SPLIT_FILES)
        for fid in COMPARE_FOLD_IDS
    )


def apply_dataset_route(
    *,
    compare_root: str | None = None,
    active_csv: str | None = None,
    use_compare: bool | None = None,
    fold_ids=None,
    valid_name: str | None = None,
    require_splits: bool = False,
) -> dict:
    """Pick external pre-split files vs main CSV pipeline.

    External route only if a split directory is **named** (``compare_root`` /
    ``RNAOMNI_COMPARE_ROOT`` / ``COMPARE_ROOT``) or ``use_compare=True`` /
    ``RNAOMNI_USE_COMPARE_SPLITS=1``. Presence of files under the default
    compare directory does not hijack the main CSV route.
    Named root that is not exactly train.csv/val.csv/test.csv (flat at root,
    or ``fold{N}/``) raises. Force main: ``use_compare=False`` / ``--no-compare``
    / ``--csv`` without ``--compare-root``.
    """
    global ACTIVE_CSV, COMPARE_ROOT, COMPARE_FOLD_IDS, COMPARE_VALID_NAME
    global USE_COMPARE_SPLITS, COMPARE_TASK_TYPE

    csv_raw = active_csv or _env_str("RNAOMNI_ACTIVE_CSV")
    ACTIVE_CSV = os.path.basename(csv_raw) if csv_raw else DEFAULT_ACTIVE_CSV

    root_raw = compare_root or _env_str("RNAOMNI_COMPARE_ROOT", "COMPARE_ROOT")
    user_named_root = bool(compare_root) or bool(root_raw)
    COMPARE_ROOT = _resolve_user_path(root_raw) if root_raw else DEFAULT_COMPARE_ROOT

    env_folds = _env_str("RNAOMNI_COMPARE_FOLD_IDS", "COMPARE_FOLD_IDS")
    explicit_folds = fold_ids if fold_ids is not None else env_folds
    if explicit_folds is None:
        COMPARE_FOLD_IDS = discover_compare_fold_ids(COMPARE_ROOT)
    else:
        COMPARE_FOLD_IDS = _parse_fold_ids(explicit_folds)
    COMPARE_VALID_NAME = (
        valid_name
        or _env_str("RNAOMNI_COMPARE_VALID_NAME", "COMPARE_VALID_NAME")
        or DEFAULT_COMPARE_VALID_NAME
    )

    explicit = (
        use_compare
        if use_compare is not None
        else _env_bool_optional("RNAOMNI_USE_COMPARE_SPLITS", "USE_COMPARE_SPLITS")
    )
    # Named root (CLI / overlay env) wins over rnaomni.env's USE_COMPARE_SPLITS=0.
    # CLI --no-compare / --csv (use_compare=False) still forces main.
    if use_compare is False:
        want_external = False
    elif user_named_root:
        want_external = True
    else:
        want_external = explicit is True

    ready = compare_dataset_ready() if want_external else False
    if not want_external:
        USE_COMPARE_SPLITS = False
        COMPARE_TASK_TYPE = None
    elif not ready:
        bad = next(
            (
                fid
                for fid in COMPARE_FOLD_IDS
                if _compare_split_filenames(_compare_split_dir(int(fid)))
                != set(COMPARE_SPLIT_FILES)
            ),
            COMPARE_FOLD_IDS[0] if COMPARE_FOLD_IDS else None,
        )
        split_dir = _compare_split_dir(int(bad)) if bad is not None else COMPARE_ROOT
        raise FileNotFoundError(compare_layout_error(split_dir))
    else:
        USE_COMPARE_SPLITS = True
        COMPARE_TASK_TYPE = peek_compare_task_type()

    import rnaomni.prepare.data as _data

    _data._DATA_CACHE = None
    return {
        "use_compare_splits": bool(USE_COMPARE_SPLITS),
        "compare_root": COMPARE_ROOT,
        "active_csv": ACTIVE_CSV,
        "fold_ids": list(COMPARE_FOLD_IDS),
        "valid_name": COMPARE_VALID_NAME,
        "csv_name": compare_active_csv_name() if USE_COMPARE_SPLITS else ACTIVE_CSV,
        "task_type": COMPARE_TASK_TYPE,
        "layout": compare_layout() if USE_COMPARE_SPLITS else None,
    }


def dataset_route_prompt_block() -> str:
    """Setup-prompt fragment: verify the active route's files, not a hardcoded set."""
    if USE_COMPARE_SPLITS:
        if compare_layout() == "fold":
            ids = ",".join(str(i) for i in COMPARE_FOLD_IDS)
            return (
                f"External pre-split route is ON with {len(COMPARE_FOLD_IDS)} folds "
                f"[{ids}] under {COMPARE_ROOT}. Each foldN/ has train.csv, val.csv, "
                f"test.csv. Do NOT require dataset/{ACTIVE_CSV} for this run."
            )
        paths = compare_split_paths(COMPARE_FOLD_IDS[0])
        return (
            "External pre-split route is ON. Verify these files exist: "
            f"{paths['train']}, {paths['val']}, {paths['test']}. "
            f"Do NOT require dataset/{ACTIVE_CSV} for this run."
        )
    csv_path = os.path.join(DATASET_DIR, ACTIVE_CSV)
    return (
        "Main CSV route is ON. Verify "
        f"{csv_path} exists. Do not require forcompare train/val/test."
    )


def compare_active_csv_name() -> str:
    """Synthetic csv_name for data cards / archive on the external-split route."""
    return f"compare_{os.path.basename(COMPARE_ROOT)}"


def peek_compare_task_type(
    compare_root: str | None = None, *, fold_id: int | None = None
) -> str:
    """Infer classification vs regression from train.csv (no encode)."""
    from rnaomni.prepare.sample import _canonical_compare_frame
    from rnaomni.prepare.splits import infer_task_type

    saved_root = COMPARE_ROOT
    saved_folds = list(COMPARE_FOLD_IDS)
    try:
        if compare_root:
            globals()["COMPARE_ROOT"] = _resolve_user_path(compare_root)
        fid = int(saved_folds[0] if fold_id is None else fold_id)
        if fold_id is not None:
            globals()["COMPARE_FOLD_IDS"] = [fid]
        paths = compare_split_paths(fid)
        train = paths["train"]
        if not os.path.isfile(train):
            return "regression"
        df = pd.read_csv(train, nrows=4096)
        part = _canonical_compare_frame(df, train)
        return infer_task_type(part[LABEL_COL].to_numpy())
    finally:
        globals()["COMPARE_ROOT"] = saved_root
        globals()["COMPARE_FOLD_IDS"] = saved_folds
