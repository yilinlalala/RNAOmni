"""A+D / row-cap sampling and external compare-split loading."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

import rnaomni.prepare.route as _route
from rnaomni.prepare.constants import (
    FOLD_TEST,
    FOLD_TRAIN,
    FOLD_VAL,
    LABEL_COL,
    SEQUENCE_COL,
)

_SPLIT_SEED_OFFSET = {"train": 1, "val": 2, "test": 3}


def _label_class_set(series: pd.Series) -> set[int]:
    y = pd.to_numeric(series, errors="coerce")
    y = y[y.notna()]
    if y.empty:
        return set()
    return set(y.astype(np.int64).tolist())


def _assert_class_coverage(
    *,
    split_name: str,
    before: set[int],
    after: set[int],
) -> None:
    """Classification hard rule: stratified sample must keep every pre-sample class."""
    missing = before - after
    if missing:
        preview = sorted(missing)[:20]
        more = "" if len(missing) <= 20 else f" (+{len(missing) - 20} more)"
        raise ValueError(
            f"class coverage broken in split={split_name!r}: "
            f"missing {len(missing)} classes {preview}{more}"
        )


def _stratified_per_class_sample(
    df: pd.DataFrame, *, per_class: int, seed: int, split_name: str
) -> pd.DataFrame:
    """Take up to ``per_class`` rows per label; rare classes keep all rows."""
    if per_class < 1:
        raise ValueError(f"per_class must be >= 1, got {per_class}")
    y = pd.to_numeric(df[LABEL_COL], errors="coerce")
    work = df.loc[y.notna()].copy()
    if work.empty:
        return work.reset_index(drop=True)
    work["_y"] = y.loc[work.index].astype(np.int64)
    offset = _SPLIT_SEED_OFFSET.get(split_name, 9)
    rng = np.random.default_rng(int(seed) * 1009 + int(offset))
    parts = []
    for cls in sorted(work["_y"].unique()):
        g = work.loc[work["_y"] == cls]
        take = min(int(per_class), len(g))
        chosen = rng.choice(g.index.to_numpy(), size=take, replace=False)
        parts.append(work.loc[chosen])
    out = pd.concat(parts, axis=0).drop(columns=["_y"])
    return out.sort_index().reset_index(drop=True)


def _row_cap_sample(
    df: pd.DataFrame, *, cap: int, seed: int, split_name: str
) -> pd.DataFrame:
    """Unstratified row cap (regression). Keep all rows if n <= cap."""
    if cap < 1:
        raise ValueError(f"row cap must be >= 1, got {cap}")
    if len(df) <= int(cap):
        return df.reset_index(drop=True)
    offset = _SPLIT_SEED_OFFSET.get(split_name, 9)
    rng = np.random.default_rng(int(seed) * 1009 + int(offset))
    chosen = rng.choice(df.index.to_numpy(), size=int(cap), replace=False)
    return df.loc[chosen].sort_index().reset_index(drop=True)


def _sample_main_csv_splits(df: pd.DataFrame, task_type: str) -> pd.DataFrame:
    """Subsample HEK/main-CSV folds after ensure_splits."""
    if not _route.sampling_on() or "fold" not in df.columns:
        return df
    seed = int(_route.COMPARE_SAMPLE_SEED)
    parts = []
    for fold_id, g in df.groupby("fold", sort=True):
        fold_id = int(fold_id)
        if fold_id == FOLD_TEST:
            split_name = "test"
            cap_k = _route.resolve_compare_per_class_eval()
            cap_n = _route.resolve_sample_row_cap_eval()
        elif fold_id == FOLD_VAL or (
            task_type == "classification" and fold_id == FOLD_VAL
        ):
            split_name = "val"
            cap_k = _route.resolve_compare_per_class_eval()
            cap_n = _route.resolve_sample_row_cap_eval()
        else:
            split_name = "train"
            cap_k = _route.resolve_compare_per_class_train()
            cap_n = _route.resolve_sample_row_cap_train()
        if task_type == "classification":
            k = 10 if cap_k is None else int(cap_k)
            classes_before = _label_class_set(g[LABEL_COL])
            sampled = _stratified_per_class_sample(
                g, per_class=k, seed=seed, split_name=split_name
            )
            _assert_class_coverage(
                split_name=f"fold{fold_id}",
                before=classes_before,
                after=_label_class_set(sampled[LABEL_COL]),
            )
        else:
            sampled = _row_cap_sample(g, cap=int(cap_n), seed=seed, split_name=split_name)
        parts.append(sampled)
    if not parts:
        return df.reset_index(drop=True)
    return pd.concat(parts, axis=0).sort_index().reset_index(drop=True)


def _canonical_compare_frame(df: pd.DataFrame, path: str) -> pd.DataFrame:
    """Require standardized input_seq / predicted_target on compare CSVs."""
    missing = [c for c in (SEQUENCE_COL, LABEL_COL) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Compare CSV {path} must have {SEQUENCE_COL} and {LABEL_COL}; missing {missing}"
        )
    part = df[[SEQUENCE_COL, LABEL_COL]].copy()
    part[SEQUENCE_COL] = (
        part[SEQUENCE_COL].astype(str).str.upper().str.replace("T", "U", regex=False)
    )
    part[LABEL_COL] = pd.to_numeric(part[LABEL_COL], errors="coerce")
    part = part.dropna(subset=[SEQUENCE_COL, LABEL_COL])
    return part[part[SEQUENCE_COL].str.len() > 0].reset_index(drop=True)


def load_compare_splits_dataframe(
    fold_ids=None,
    per_class=None,
    per_class_train=None,
    per_class_eval=None,
    row_limit=None,
    *,
    quiet: bool = False,
) -> pd.DataFrame:
    """Load external compare folds with no collapse/cleaning.

    Sample when ``RNAOMNI_DATA_MODE=sample`` (orchestrator-owned):
      - classification: scheme A+D, ``per_class`` or env K_train / K_eval (default 10)
      - regression: unstratified row cap (SAMPLE_ROW_CAP_TRAIN / EVAL, default 1000)
      - explicit ``per_class=`` still samples even in full mode (task-type peeks)

    Classification: after stratified sample, every class present in the split
    before sampling must still be present (cover-all-classes).

    ``row_limit`` is deprecated (old head-N); ignored with a warning if passed.
    ``quiet=True`` suppresses the external-split summary line (for task-type
    probes that intentionally use a tiny ``per_class`` preview).
    """
    fold_ids = list(_route.COMPARE_FOLD_IDS if fold_ids is None else fold_ids)
    if row_limit is not None:
        print(
            "WARNING: load_compare_splits_dataframe(row_limit=...) is deprecated; "
            "ignoring head-N. Pass per_class=K or COMPARE_PER_CLASS_TRAIN/EVAL."
        )
    if per_class is not None:
        k_train = k_eval = int(per_class)
    else:
        k_train = (
            int(per_class_train)
            if per_class_train is not None
            else _route.resolve_compare_per_class_train()
        )
        k_eval = (
            int(per_class_eval)
            if per_class_eval is not None
            else _route.resolve_compare_per_class_eval()
        )
    sample_seed = int(_route.COMPARE_SAMPLE_SEED)
    if not fold_ids:
        raise ValueError("COMPARE_FOLD_IDS is empty")

    task_type = _route.COMPARE_TASK_TYPE or _route.peek_compare_task_type()
    row_cap_train = None
    row_cap_eval = None
    explicit_k = per_class is not None or per_class_train is not None or per_class_eval is not None
    if not _route.sampling_on() and not explicit_k:
        k_train = None
        k_eval = None
    elif _route.sampling_on() and task_type == "regression" and not explicit_k:
        k_train = None
        k_eval = None
        row_cap_train = _route.resolve_sample_row_cap_train()
        row_cap_eval = _route.resolve_sample_row_cap_eval()

    pieces = []
    split_stats = []
    for fid in fold_ids:
        paths = _route.compare_split_paths(fid)
        for missing in [p for p in paths.values() if not os.path.isfile(p)]:
            raise FileNotFoundError(f"Compare split missing: {missing}")
        for split_name, path in paths.items():
            df = pd.read_csv(path)
            part = _canonical_compare_frame(df, path)
            k = k_train if split_name == "train" else k_eval
            ncap = row_cap_train if split_name == "train" else row_cap_eval
            if k is not None:
                classes_before = _label_class_set(part[LABEL_COL])
                part = _stratified_per_class_sample(
                    part, per_class=int(k), seed=sample_seed, split_name=split_name
                )
                _assert_class_coverage(
                    split_name=split_name,
                    before=classes_before,
                    after=_label_class_set(part[LABEL_COL]),
                )
            elif ncap is not None:
                part = _row_cap_sample(
                    part, cap=int(ncap), seed=sample_seed, split_name=split_name
                )
            n_classes = int(part[LABEL_COL].nunique())
            vc = part[LABEL_COL].value_counts()
            split_stats.append(
                f"{split_name}:rows={len(part)},classes={n_classes},"
                f"K={k},min_count={int(vc.min()) if len(vc) else 0}"
            )
            part["compare_fold"] = int(fid)
            part["split"] = split_name
            pieces.append(part)

    out = pd.concat(pieces, ignore_index=True)
    if len(fold_ids) == 1:
        fold_col = np.full(len(out), FOLD_TRAIN, dtype=np.int64)
        fold_col[out["split"].to_numpy() == "val"] = FOLD_VAL
        fold_col[out["split"].to_numpy() == "test"] = FOLD_TEST
        out["fold"] = fold_col
    else:
        out["fold"] = out["compare_fold"].astype(np.int64)
    if not quiet:
        print(
            f"Data: external splits loaded {_route.COMPARE_ROOT} folds={fold_ids} "
            f"rows={len(out)} K_train={k_train} K_eval={k_eval} "
            f"seed={sample_seed} stem={_route.compare_active_csv_name()} "
            f"counts={out['split'].value_counts().to_dict()} "
            f"[{'; '.join(split_stats)}]"
        )
    return out


def get_split_indices_compare(fold_id, split, compare_folds, split_names):
    assert split in ("train", "val", "test")
    cf = np.asarray(compare_folds)
    sn = np.asarray(split_names)
    return np.where((cf == int(fold_id)) & (sn == split))[0]
