"""Immutable prep constants and env helpers. Numeric defaults from rnaomni.env."""

from __future__ import annotations

import os

from rnaomni.paths import REPO
from tools.user_env import env_float, env_int, load_user_env

load_user_env(str(REPO))

MAX_SEQ_LEN = env_int("MAX_SEQ_LEN", 1024)
TIME_BUDGET = env_int("TIME_BUDGET", 60)
N_FOLDS = env_int("N_FOLDS", 5)
TEST_FRACTION = env_float("TEST_FRACTION", 0.1)
FOLD_TEST = -1
FOLD_VAL = 0
FOLD_TRAIN = 1
ALPHABET = "ACGU"
FOLD_RANDOM_STATE = 42

SEQUENCE_COL = "input_seq"
LABEL_COL = "predicted_target"
COMPARE_SPLIT_FILES = ("train.csv", "val.csv", "test.csv")

CLS_TRAIN_FRAC = env_float("CLS_TRAIN_FRAC", 0.7)
CLS_VAL_FRAC = env_float("CLS_VAL_FRAC", 0.2)
CLS_TEST_FRAC = env_float("CLS_TEST_FRAC", 0.1)
CLS_MMSEQS_MIN_SEQ_ID = env_float("CLS_MMSEQS_MIN_SEQ_ID", 0.8)
CLS_MMSEQS_COVERAGE = env_float("CLS_MMSEQS_COVERAGE", 0.8)

PROJECT_DIR = str(REPO)
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")

DEFAULT_ACTIVE_CSV = "HEK_te_prediction_task.csv"
DEFAULT_COMPARE_ROOT = os.path.join(DATASET_DIR, "forcompare", "Class_88fam_200%noise")
# Fallback when a compare root has no foldN/ directories (flat train/val/test).
# Fold-layout roots (fold0..fold9) default to every foldN found on disk.
DEFAULT_COMPARE_FOLD_IDS = [0]
DEFAULT_COMPARE_VALID_NAME = "val.csv"

DEFAULT_COMPARE_PER_CLASS_TRAIN = env_int("COMPARE_PER_CLASS_TRAIN", 10)
DEFAULT_COMPARE_PER_CLASS_EVAL = env_int("COMPARE_PER_CLASS_EVAL", 10)
DEFAULT_SAMPLE_ROW_CAP_TRAIN = env_int("SAMPLE_ROW_CAP_TRAIN", 1000)
DEFAULT_ENCODE_TIMEOUT_SEC = env_int("ENCODE_TIMEOUT_SEC", 5400)
ENCODE_TIMEOUT_EXIT = 124

_FORMAL_TOKENS = frozenset({"full", "none", "off", "all", "false"})
_TRUTHY = frozenset({"1", "true", "yes", "on", "full"})
_FALSY = frozenset({"0", "false", "no", "off"})


def _env_truthy(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in _TRUTHY


def _env_str(*names: str) -> str | None:
    for name in names:
        raw = os.environ.get(name)
        if raw is not None and str(raw).strip() != "":
            return str(raw).strip()
    return None


def _env_bool_optional(*names: str) -> bool | None:
    """Return True/False if any name is set, else None (auto)."""
    for name in names:
        raw = os.environ.get(name)
        if raw is None or str(raw).strip() == "":
            continue
        text = str(raw).strip().lower()
        if text in _FALSY:
            return False
        if text in _TRUTHY:
            return True
        raise ValueError(f"{name}={raw!r} is not a boolean")
    return None


def _resolve_user_path(raw: str) -> str:
    if os.path.isabs(raw):
        return os.path.abspath(raw)
    for base in (PROJECT_DIR, DATASET_DIR):
        cand = os.path.abspath(os.path.join(base, raw))
        if os.path.isdir(cand) or os.path.isfile(cand):
            return cand
    return os.path.abspath(os.path.join(PROJECT_DIR, raw))


def _parse_fold_ids(raw) -> list[int]:
    if raw is None:
        return list(DEFAULT_COMPARE_FOLD_IDS)
    if isinstance(raw, (list, tuple)):
        return [int(x) for x in raw]
    parts = [p.strip() for p in str(raw).replace(";", ",").split(",")]
    ids = [int(p) for p in parts if p]
    return ids if ids else list(DEFAULT_COMPARE_FOLD_IDS)


def search_fold_budget_sec(n_cv_folds: int = 1, *, use_compare_splits: bool = False) -> float:
    """Seconds of search training allowed for one CV fold.

    Compare pre-split route: ``TIME_BUDGET`` is per fold (60s × 10 folds by default).
    Main CSV route: ``TIME_BUDGET`` is the total across internal CV folds.
    """
    n = max(int(n_cv_folds), 1)
    if use_compare_splits:
        return float(TIME_BUDGET)
    return float(TIME_BUDGET) / n


def _env_int_optional(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return None
    text = str(raw).strip().lower()
    if text in _FORMAL_TOKENS:
        return None
    return int(raw)
