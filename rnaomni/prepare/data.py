"""Recipe encode, prepared-data memo, and dataloaders."""

from __future__ import annotations

import os
import threading

import numpy as np
import torch

import rnaomni.prepare.route as _route
from features.pipeline import (
    encode_dataframe_from_recipe,
    encode_sequence_batch,
    load_active_recipe,
    load_feature_registry,
    recipe_cache_key,
    resolve_input_dim,
    unique_sequence_index,
)
from rnaomni.prepare.constants import (
    ALPHABET,
    CLS_MMSEQS_COVERAGE,
    CLS_MMSEQS_MIN_SEQ_ID,
    CLS_TEST_FRAC,
    CLS_TRAIN_FRAC,
    CLS_VAL_FRAC,
    ENCODE_TIMEOUT_EXIT,
    FOLD_TEST,
    FOLD_TRAIN,
    FOLD_VAL,
    LABEL_COL,
    MAX_SEQ_LEN,
    N_FOLDS,
    SEQUENCE_COL,
)
from rnaomni.prepare.sample import (
    _sample_main_csv_splits,
    get_split_indices_compare,
    load_compare_splits_dataframe,
)
from rnaomni.prepare.splits import (
    build_data_card,
    classification_num_classes,
    clean_dataset,
    ensure_splits,
    infer_task_type,
    load_dataset,
    write_data_card,
)

# Default input dim for L0-only recipe; overwritten by load_prepared_data().
INPUT_DIM = 4
_DATA_CACHE = None
# Finalize online path: uid -> (feat (L_i, C) float32, length).
_ONLINE_UNIQUE_CACHE: dict[int, tuple[np.ndarray, int]] = {}


def encode_dataframe(df, alphabet=ALPHABET, max_len=MAX_SEQ_LEN, task_type="regression"):
    """Encode unique sequences via features/active.yaml (+ registry).

    Returns x (U,L,C), lengths (U,), labels (N,), meta with row_to_uid.
    If ENCODE_TIMEOUT_SEC > 0, a watchdog kills this process with
    ENCODE_TIMEOUT_EXIT when encode exceeds the deadline (C builders included).
    """
    timeout = _route.encode_timeout_sec()
    stop = threading.Event()

    def _watchdog() -> None:
        if timeout <= 0:
            return
        if not stop.wait(timeout):
            print("ENCODE_TIMEOUT", flush=True)
            os._exit(ENCODE_TIMEOUT_EXIT)

    watcher = None
    if timeout > 0:
        watcher = threading.Thread(target=_watchdog, daemon=True)
        watcher.start()
    try:
        x, lengths, labels, meta = encode_dataframe_from_recipe(
            df,
            sequence_col=SEQUENCE_COL,
            label_col=LABEL_COL,
            max_seq_len=max_len,
            alphabet=alphabet,
            label_dtype=np.int64 if task_type == "classification" else np.float32,
        )
        return x, lengths, labels, meta
    finally:
        stop.set()


def load_prepared_data(csv_name=None):
    """
    Load dataset, clean rows, ensure splits exist, encode features for the active
    recipe, and write a data card. Features are computed at call time (no disk
    cache). Within one process, the same recipe/csv/encode_mode is memoized so
    fold dataloaders reuse tensors.

    ``RNAOMNI_ENCODE_MODE=online`` (finalize 15.2A) skips the unique table
    encode; ``gather_row_features`` builds missing uids on the batch path.
    Overnight ``train.py`` stays precompute.
    Returns dict with tensors and metadata including input_dim and task_type.
    """
    global _DATA_CACHE, INPUT_DIM, _ONLINE_UNIQUE_CACHE

    if csv_name is None:
        csv_name = _route.ACTIVE_CSV

    recipe = load_active_recipe()
    registry = load_feature_registry()
    input_dim = resolve_input_dim(recipe, registry, alphabet=ALPHABET)
    recipe_id = recipe.get("recipe_id")
    k_train = _route.resolve_compare_per_class_train() if _route.sampling_on() else None
    k_eval = _route.resolve_compare_per_class_eval() if _route.sampling_on() else None
    row_cap_train = _route.resolve_sample_row_cap_train() if _route.sampling_on() else None
    row_cap_eval = _route.resolve_sample_row_cap_eval() if _route.sampling_on() else None
    compare_key = (
        (
            "compare",
            tuple(_route.COMPARE_FOLD_IDS),
            k_train,
            k_eval,
            row_cap_train,
            row_cap_eval,
            _route.COMPARE_SAMPLE_SEED,
            _route.COMPARE_ROOT,
            _route.data_mode(),
        )
        if _route.USE_COMPARE_SPLITS
        else ("csv", csv_name, _route.data_mode(), k_train, k_eval, row_cap_train, row_cap_eval)
    )
    memo_key = (
        compare_key,
        recipe_cache_key(recipe),
        input_dim,
        _route.encode_mode(),
    )

    if _DATA_CACHE is not None and _DATA_CACHE.get("memo_key") == memo_key:
        return _DATA_CACHE

    if _route.USE_COMPARE_SPLITS:
        df = load_compare_splits_dataframe()
        task_type = infer_task_type(df[LABEL_COL].to_numpy())
        filter_stats = {
            "n_raw": len(df),
            "n_illegal_base": 0,
            "n_nonfinite_label": 0,
            "n_seq_groups_averaged": 0,
            "n_rows_collapsed": 0,
            "n_kept": len(df),
            "alphabet": ALPHABET,
            "task_type": task_type,
            "external_splits": True,
            "compare_root": _route.COMPARE_ROOT,
            "compare_layout": _route.compare_layout(),
            "compare_fold_ids": list(_route.COMPARE_FOLD_IDS),
            "compare_valid_name": _route.COMPARE_VALID_NAME,
            "compare_per_class": _route.COMPARE_PER_CLASS,
            "compare_per_class_train": k_train,
            "compare_per_class_eval": k_eval,
            "compare_sample_seed": _route.COMPARE_SAMPLE_SEED,
            "compare_sampling_on": _route.sampling_on(),
            "data_mode": _route.data_mode(),
        }
        print(
            f"Data: external splits no-clean load kept={len(df)} task={task_type} "
            f"(duplicates across splits kept as-is; "
            f"K_train={k_train} K_eval={k_eval} seed={_route.COMPARE_SAMPLE_SEED} "
            f"stem={_route.compare_active_csv_name()})"
        )
        csv_name = _route.compare_active_csv_name()
    else:
        df_raw = load_dataset(csv_name)
        df, filter_stats, task_type = clean_dataset(df_raw)
        print(
            f"Data: cleaned {filter_stats['n_raw']} -> {filter_stats['n_kept']} "
            f"(task={task_type}, illegal={filter_stats['n_illegal_base']}, "
            f"seq_groups={filter_stats['n_seq_groups_averaged']}, "
            f"rows_collapsed={filter_stats['n_rows_collapsed']}, "
            f"nonfinite_label={filter_stats['n_nonfinite_label']})"
        )
        df = ensure_splits(df, task_type, csv_name=csv_name)
        if _route.sampling_on():
            n_before = len(df)
            df = _sample_main_csv_splits(df, task_type)
            print(
                f"Data: sampled {n_before} -> {len(df)} "
                f"(data_mode=sample task={task_type} "
                f"K_train={k_train} K_eval={k_eval} "
                f"row_cap_train={row_cap_train} row_cap_eval={row_cap_eval})"
            )

    num_classes = classification_num_classes(df[LABEL_COL]) if task_type == "classification" else None
    extra = {}
    if task_type == "classification":
        extra = {
            "split_protocol": {
                "method": "within_class_mmseqs_cluster",
                "min_seq_id": CLS_MMSEQS_MIN_SEQ_ID,
                "coverage": CLS_MMSEQS_COVERAGE,
                "train_frac": CLS_TRAIN_FRAC,
                "val_frac": CLS_VAL_FRAC,
                "test_frac": CLS_TEST_FRAC,
            },
            "num_classes": num_classes,
            "decision_metric": "val_accuracy",
            "decision_higher_is_better": True,
        }
    else:
        extra = {
            "decision_metric": "val_spearman",
            "decision_higher_is_better": True,
        }

    if _route.USE_COMPARE_SPLITS:
        extra["split_protocol"] = {
            "method": "external_pre_split",
            "root": _route.COMPARE_ROOT,
            "layout": _route.compare_layout(),
            "valid_name": _route.COMPARE_VALID_NAME,
            "fold_ids": list(_route.COMPARE_FOLD_IDS),
            "per_class": _route.COMPARE_PER_CLASS,
            "per_class_train": k_train,
            "per_class_eval": k_eval,
            "sample_seed": _route.COMPARE_SAMPLE_SEED,
            "results_stem": _route.compare_active_csv_name(),
        }

    data_card = build_data_card(
        df, filter_stats, csv_name=csv_name, task_type=task_type, extra=extra
    )
    data_card_path = write_data_card(data_card, csv_name=csv_name)
    print(f"Data: wrote data card -> {data_card_path}")

    if _route.USE_COMPARE_SPLITS:
        n_cv_folds = len(_route.COMPARE_FOLD_IDS)
    else:
        n_cv_folds = 1 if task_type == "classification" else N_FOLDS

    unique_seqs, row_to_uid_np = unique_sequence_index(df[SEQUENCE_COL].astype(str).tolist())
    n_rows = len(df)
    n_unique = len(unique_seqs)
    row_to_uid = torch.from_numpy(np.asarray(row_to_uid_np, dtype=np.int64))
    if task_type == "classification":
        labels = torch.from_numpy(df[LABEL_COL].to_numpy().astype(np.int64))
    else:
        labels = torch.from_numpy(df[LABEL_COL].to_numpy().astype(np.float32))
    enabled = [item["name"] for item in recipe["features"] if item.get("enabled", True)]
    fusion = recipe.get("fusion", "concat")

    _ONLINE_UNIQUE_CACHE = {}
    online = _route.encode_mode() == "online"
    x = None
    lengths = None
    store_len = None
    if online:
        print(
            f"Data: online encode (skip precompute) n_unique={n_unique} "
            f"n_rows={n_rows} recipe_id={recipe_id} input_dim={input_dim} "
            f"features={enabled}",
            flush=True,
        )
        INPUT_DIM = int(input_dim)
    else:
        x_np, lengths_np, labels_np, feature_meta = encode_dataframe(df, task_type=task_type)
        if feature_meta["input_dim"] != input_dim:
            raise RuntimeError("input_dim mismatch after encode")
        if int(feature_meta["n_rows"]) != n_rows:
            raise RuntimeError(f"encode n_rows={feature_meta['n_rows']} != df rows={n_rows}")
        x = torch.from_numpy(x_np)
        lengths = torch.from_numpy(lengths_np)
        row_to_uid = torch.from_numpy(np.asarray(feature_meta["row_to_uid"], dtype=np.int64))
        if task_type == "classification":
            labels = torch.from_numpy(labels_np.astype(np.int64))
        else:
            labels = torch.from_numpy(labels_np.astype(np.float32))
        store_len = int(x.shape[1])
        n_unique = int(feature_meta["n_unique"])
        enabled = feature_meta.get("enabled_features")
        fusion = feature_meta.get("fusion")
        print(
            f"Data: recipe_id={recipe_id} input_dim={input_dim} "
            f"features={enabled} n_unique={n_unique} n_rows={n_rows} "
            f"store_len={store_len}"
        )
        INPUT_DIM = int(x.shape[-1])
        if INPUT_DIM != input_dim:
            raise RuntimeError(f"Tensor C={INPUT_DIM} != recipe input_dim={input_dim}")
        if int(x.shape[0]) != n_unique:
            raise RuntimeError(f"x_unique U={x.shape[0]} != n_unique={n_unique}")

    _DATA_CACHE = {
        "df": df,
        "x": x,
        "lengths": lengths,
        "labels": labels,
        "row_to_uid": row_to_uid,
        "unique_seqs": unique_seqs,
        "n_rows": n_rows,
        "n_unique": n_unique,
        "store_len": store_len,
        "csv_name": csv_name,
        "input_dim": INPUT_DIM,
        "task_type": task_type,
        "num_classes": num_classes,
        "n_cv_folds": n_cv_folds,
        "decision_metric": extra["decision_metric"],
        "decision_higher_is_better": extra["decision_higher_is_better"],
        "feature_recipe_id": recipe_id,
        "enabled_features": enabled,
        "fusion": fusion,
        "filter_stats": filter_stats,
        "data_card": data_card,
        "data_card_path": data_card_path,
        "memo_key": memo_key,
        "use_compare_splits": bool(_route.USE_COMPARE_SPLITS),
        "encode_mode": "online" if online else "precompute",
        "recipe": recipe,
        "registry": registry,
    }
    return _DATA_CACHE


def get_split_indices(fold_id, split, folds, compare_folds=None, split_names=None):
    """Return row indices for train, val, or test."""
    assert split in ("train", "val", "test")
    if _route.USE_COMPARE_SPLITS and compare_folds is not None and split_names is not None:
        return get_split_indices_compare(fold_id, split, compare_folds, split_names)
    if _route.USE_COMPARE_SPLITS and len(_route.COMPARE_FOLD_IDS) == 1:
        folds = np.asarray(folds)
        if split == "test":
            return np.where(folds == FOLD_TEST)[0]
        if split == "train":
            return np.where(folds == FOLD_TRAIN)[0]
        return np.where(folds == FOLD_VAL)[0]
    folds = np.asarray(folds)
    if split == "test":
        return np.where(folds == FOLD_TEST)[0]
    if split == "train":
        return np.where((folds != fold_id) & (folds >= 0))[0]
    return np.where(folds == fold_id)[0]


def _indices_for_split(data, fold_id, split):
    """Resolve train/val/test row indices (external-split aware)."""
    df = data["df"]
    if (
        data.get("use_compare_splits")
        and "compare_fold" in df.columns
        and len(_route.COMPARE_FOLD_IDS) > 1
    ):
        return get_split_indices(
            _route.COMPARE_FOLD_IDS[fold_id],
            split,
            df["fold"].to_numpy(),
            compare_folds=df["compare_fold"].to_numpy(),
            split_names=df["split"].to_numpy(),
        )
    return get_split_indices(fold_id, split, df["fold"].to_numpy())


def _as_row_index(row_idx) -> torch.Tensor:
    if isinstance(row_idx, torch.Tensor):
        return row_idx.to(dtype=torch.long)
    return torch.as_tensor(row_idx, dtype=torch.long)


def _gather_online_features(data, uid: torch.Tensor):
    """Encode missing unique ids, then pad this row-batch to its own max length."""
    uids = [int(v) for v in uid.detach().cpu().tolist()]
    missing: list[int] = []
    seen: set[int] = set()
    for u in uids:
        if u not in _ONLINE_UNIQUE_CACHE and u not in seen:
            missing.append(u)
            seen.add(u)
    if missing:
        seqs = [data["unique_seqs"][u] for u in missing]
        x_np, lengths_np = encode_sequence_batch(
            seqs,
            max_seq_len=MAX_SEQ_LEN,
            alphabet=ALPHABET,
            recipe=data.get("recipe"),
            registry=data.get("registry"),
        )
        channel_dim = int(x_np.shape[-1])
        expected = int(data["input_dim"])
        if channel_dim != expected:
            raise RuntimeError(f"online encode C={channel_dim} != input_dim={expected}")
        for i, u in enumerate(missing):
            length_i = int(lengths_np[i])
            feat = np.array(x_np[i, :length_i, :], dtype=np.float32, copy=True)
            _ONLINE_UNIQUE_CACHE[u] = (feat, length_i)

    feats = []
    lens: list[int] = []
    for u in uids:
        feat, length_i = _ONLINE_UNIQUE_CACHE[u]
        feats.append(feat)
        lens.append(length_i)
    batch_len = max(lens) if lens else 0
    channel_dim = int(data["input_dim"])
    x = np.zeros((len(uids), batch_len, channel_dim), dtype=np.float32)
    for i, feat in enumerate(feats):
        x[i, : lens[i], :] = feat
    return torch.from_numpy(x), torch.as_tensor(lens, dtype=torch.long)


def gather_row_features(data, row_idx):
    """Unique-table x/lengths for row indices. Labels stay per-row (not gathered here).

    Online mode (finalize 15.2A): ``data['x']`` is None; missing uids are encoded
    via ``encode_sequence_batch`` and cached per process.
    """
    idx = _as_row_index(row_idx)
    uid = data["row_to_uid"].index_select(0, idx)
    if data.get("encode_mode") == "online" or data.get("x") is None:
        return _gather_online_features(data, uid)
    x = data["x"].index_select(0, uid)
    lengths = data["lengths"].index_select(0, uid)
    return x, lengths


def make_dataloader(fold_id, split, batch_size, csv_name=None):
    """
    Infinite dataloader for one CV / fixed split.

    Yields (x, y, lengths, epoch):
      x:       (B, L, C) float32 on CUDA  — C = active recipe input_dim
      y:       (B,) float32 (regression) or int64 (classification) on CUDA
      lengths: (B,) int64 on CUDA — valid positions before padding
      epoch:   int, increments each full pass over the split
    """
    assert split in ("train", "val")
    if csv_name is None:
        csv_name = _route.ACTIVE_CSV
    data = load_prepared_data(csv_name)
    n_cv = int(data.get("n_cv_folds", N_FOLDS))
    assert 0 <= fold_id < n_cv

    indices = _indices_for_split(data, fold_id, split)
    if len(indices) == 0:
        raise ValueError(f"No samples for fold={fold_id}, split={split!r}")

    y_all = data["labels"]

    epoch = 0
    while True:
        epoch += 1
        order = np.random.permutation(indices) if split == "train" else indices
        for start in range(0, len(order), batch_size):
            batch_idx = order[start : start + batch_size]
            idx_t = torch.as_tensor(batch_idx, dtype=torch.long)
            x, lengths = gather_row_features(data, idx_t)
            x = x.pin_memory().to(device="cuda", non_blocking=True)
            y = y_all.index_select(0, idx_t).pin_memory().to(device="cuda", non_blocking=True)
            lengths = lengths.pin_memory().to(device="cuda", non_blocking=True)
            yield x, y, lengths, epoch


def iter_batches(fold_id, split, batch_size, csv_name=None):
    """Finite iterator over all batches in a split (for evaluation)."""
    if csv_name is None:
        csv_name = _route.ACTIVE_CSV
    data = load_prepared_data(csv_name)
    indices = _indices_for_split(data, fold_id, split)
    y_all = data["labels"]

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        idx_t = torch.as_tensor(batch_idx, dtype=torch.long)
        x, lengths = gather_row_features(data, idx_t)
        x = x.to(device="cuda", non_blocking=True)
        y = y_all.index_select(0, idx_t).to(device="cuda", non_blocking=True)
        lengths = lengths.to(device="cuda", non_blocking=True)
        yield x, y, lengths
