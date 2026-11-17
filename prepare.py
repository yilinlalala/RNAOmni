"""
Data preparation for RNAOmni experiments.
Loads CSV, cleans rows, assigns splits, builds features from
features/active.yaml at call time, writes a data card, and exposes dataloaders.

Task type is inferred from predicted_target:
  - integer-valued labels with small cardinality -> classification
  - otherwise -> regression

Regression splits: test holdout + CV (fractions / n_folds in rnaomni.env).
Classification splits: within-class MMseqs clustering, then cluster-aware
  train/val/test (identity and fractions in rnaomni.env).

Orchestration boundary (path H / S1):
  - Humans may edit the recipe/registry wiring in this file.
  - Agents must NOT modify prepare.py, rnaomni/prepare/, or rnaomni.env (splits, metrics, TIME_BUDGET).
  - Agents may edit features/active.yaml (whitelist keys only) and train.py.
  - No disk feature cache: train/load_prepared_data encodes from the active recipe.

Dataset route (see rnaomni.env; local overlay `.env`):
  - Main CSV (repo default): ACTIVE_CSV from rnaomni.env (HEK_te_prediction_task.csv),
    clean + assign splits.
  - External pre-split: only when a split directory is named
    (RNAOMNI_COMPARE_ROOT / --compare-root) and train/val/test exist.
  Resolution: apply_dataset_route() at import (after load_user_env) and CLI.
    Disk files under the default compare dir do not auto-hijack the main route.

Usage (conda myenv3.11):
    python prepare.py
    python prepare.py --csv Rfam_88families_classification.csv
    python prepare.py --compare-root dataset/forcompare/Class_88fam_200%noise
    python prepare.py --skip-encode
"""

from __future__ import annotations

import argparse
import sys
import types


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


class _PrepareModule(types.ModuleType):
    """Forward reads/writes of route and cache state into rnaomni.prepare."""

    def __getattr__(self, name: str):
        import rnaomni.prepare as pkg

        return getattr(pkg, name)

    def __setattr__(self, name: str, value):
        if name in _ROUTE_ATTRS:
            import rnaomni.prepare.route as route

            setattr(route, name, value)
            return
        if name in _DATA_ATTRS:
            import rnaomni.prepare.data as data

            setattr(data, name, value)
            return
        super().__setattr__(name, value)

    def __dir__(self):
        import rnaomni.prepare as pkg

        return sorted(set(super().__dir__()) | set(dir(pkg)))


def _cli() -> None:
    from rnaomni.prepare import (
        ACTIVE_CSV,
        ALPHABET,
        COMPARE_FOLD_IDS,
        COMPARE_PER_CLASS,
        COMPARE_PER_CLASS_EVAL,
        COMPARE_PER_CLASS_TRAIN,
        COMPARE_ROOT,
        COMPARE_SAMPLE_SEED,
        DATASET_DIR,
        FOLD_TEST,
        LABEL_COL,
        MAX_SEQ_LEN,
        N_FOLDS,
        RECIPE_PATH,
        SEQUENCE_COL,
        USE_COMPARE_SPLITS,
        apply_dataset_route,
        build_data_card,
        classification_num_classes,
        clean_dataset,
        compare_active_csv_name,
        compare_sampling_on,
        ensure_splits,
        infer_task_type,
        load_compare_splits_dataframe,
        load_dataset,
        load_prepared_data,
        resolve_compare_per_class_eval,
        resolve_compare_per_class_train,
        write_data_card,
    )

    parser = argparse.ArgumentParser(
        description="Clean data, assign splits, write data card, smoke-test encoding"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="CSV filename under dataset/ (forces main pipeline if --compare-root is omitted)",
    )
    parser.add_argument(
        "--compare-root",
        type=str,
        default=None,
        help="Directory with exactly train.csv, val.csv, test.csv (or foldN/ of those)",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Pin a single fold id under compare-root. Default: all foldN dirs. Ignored for flat layouts.",
    )
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="Force main CSV pipeline even if pre-split files exist",
    )
    parser.add_argument(
        "--skip-encode",
        action="store_true",
        help="Only clean/split/data-card (skip feature encoding smoke test)",
    )
    args = parser.parse_args()

    if args.no_compare:
        apply_dataset_route(active_csv=args.csv, use_compare=False)
    elif args.compare_root:
        apply_dataset_route(
            compare_root=args.compare_root,
            active_csv=args.csv,
            use_compare=True,
            require_splits=True,
            fold_ids=[args.fold] if args.fold is not None else None,
        )
    elif args.csv:
        apply_dataset_route(active_csv=args.csv, use_compare=False)

    from rnaomni.prepare import (
        ACTIVE_CSV as _active_csv,
        COMPARE_FOLD_IDS as _fold_ids,
        COMPARE_PER_CLASS as _per_class,
        COMPARE_PER_CLASS_EVAL as _per_class_eval,
        COMPARE_PER_CLASS_TRAIN as _per_class_train,
        COMPARE_ROOT as _compare_root,
        COMPARE_SAMPLE_SEED as _sample_seed,
        USE_COMPARE_SPLITS as _use_compare,
    )

    csv_name = args.csv or _active_csv
    print(f"Dataset directory: {DATASET_DIR}")
    print(f"Route: {'external_pre_split' if _use_compare else 'main_csv'}")
    print(f"Active CSV: {csv_name}")
    print(f"USE_COMPARE_SPLITS: {_use_compare}")
    if _use_compare:
        print(f"COMPARE_ROOT: {_compare_root}")
        print(f"COMPARE_FOLD_IDS: {_fold_ids}")
        print(f"COMPARE_PER_CLASS: {_per_class}")
        print(
            f"COMPARE_PER_CLASS_TRAIN: {_per_class_train} -> {resolve_compare_per_class_train()}"
        )
        print(
            f"COMPARE_PER_CLASS_EVAL: {_per_class_eval} -> {resolve_compare_per_class_eval()}"
        )
        print(f"COMPARE_SAMPLE_SEED: {_sample_seed}")
        print(f"compare stem: {compare_active_csv_name()}")
    print(f"Active recipe: {RECIPE_PATH}")
    print(f"max_seq_len: {MAX_SEQ_LEN}")
    print()

    if args.skip_encode:
        if _use_compare:
            df = load_compare_splits_dataframe()
            task_type = infer_task_type(df[LABEL_COL].to_numpy())
            k_train = resolve_compare_per_class_train()
            k_eval = resolve_compare_per_class_eval()
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
                "compare_root": _compare_root,
                "compare_fold_ids": list(_fold_ids),
                "compare_per_class": _per_class,
                "compare_per_class_train": k_train,
                "compare_per_class_eval": k_eval,
                "compare_sample_seed": _sample_seed,
                "compare_sampling_on": compare_sampling_on(),
                "skip_encode": True,
            }
            csv_name = compare_active_csv_name()
            extra = {
                "decision_metric": "val_spearman" if task_type == "regression" else "val_accuracy",
                "decision_higher_is_better": True,
                "split_protocol": {
                    "method": "external_pre_split",
                    "root": _compare_root,
                    "fold_ids": list(_fold_ids),
                    "per_class": _per_class,
                    "per_class_train": k_train,
                    "per_class_eval": k_eval,
                    "sample_seed": _sample_seed,
                    "results_stem": csv_name,
                },
            }
            data_card = build_data_card(
                df, filter_stats, csv_name=csv_name, task_type=task_type, extra=extra
            )
            data_card_path = write_data_card(data_card, csv_name=csv_name)
            print(f"Data: external splits skip-encode kept={len(df)} task={task_type}")
            print(f"Data: wrote data card -> {data_card_path}")
            data = {
                "df": df,
                "task_type": task_type,
                "num_classes": None,
                "filter_stats": filter_stats,
                "data_card_path": data_card_path,
                "enabled_features": None,
                "input_dim": None,
                "feature_recipe_id": None,
                "use_compare_splits": True,
                "n_cv_folds": len(_fold_ids),
            }
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
            num_classes = (
                classification_num_classes(df[LABEL_COL])
                if task_type == "classification"
                else None
            )
            extra = {"num_classes": num_classes} if num_classes is not None else None
            data_card = build_data_card(
                df, filter_stats, csv_name=csv_name, task_type=task_type, extra=extra
            )
            data_card_path = write_data_card(data_card, csv_name=csv_name)
            print(f"Data: wrote data card -> {data_card_path}")
            data = {
                "df": df,
                "task_type": task_type,
                "num_classes": num_classes,
                "filter_stats": filter_stats,
                "data_card_path": data_card_path,
                "enabled_features": None,
                "input_dim": None,
                "feature_recipe_id": None,
                "use_compare_splits": False,
                "n_cv_folds": 1 if task_type == "classification" else N_FOLDS,
            }
    else:
        data = load_prepared_data(csv_name=csv_name)

    df = data["df"]
    task_type = data.get("task_type") or infer_task_type(df[LABEL_COL])
    seq_lens = df[SEQUENCE_COL].str.len()
    print(f"Task: {task_type}")
    if data.get("use_compare_splits"):
        print(f"Rows: {len(df)} (external pre-split files)")
        print(f"Splits: {df['split'].value_counts().to_dict()}")
        print(f"CV folds: {data.get('n_cv_folds')} (COMPARE_FOLD_IDS={_fold_ids})")
    else:
        n_test = int((df["fold"] == FOLD_TEST).sum())
        n_cv = int((df["fold"] >= 0).sum())
        print(f"Rows: {len(df)} (test: {n_test}, train/val pool: {n_cv})")
    print(
        f"Sequence length: min={seq_lens.min()}, max={seq_lens.max()}, "
        f"median={seq_lens.median():.0f}"
    )
    if task_type == "classification":
        print(
            f"Label ({LABEL_COL}): classes={data.get('num_classes')} "
            f"min={int(df[LABEL_COL].min())} max={int(df[LABEL_COL].max())}"
        )
        if "split" in df.columns and not data.get("use_compare_splits"):
            print(f"Splits: {df['split'].value_counts().to_dict()}")
    else:
        print(
            f"Label ({LABEL_COL}): min={df[LABEL_COL].min():.4f}, "
            f"max={df[LABEL_COL].max():.4f}"
        )
        if not data.get("use_compare_splits"):
            cv_counts = df.loc[df["fold"] >= 0, "fold"].value_counts().sort_index().to_dict()
            print(f"CV folds: {N_FOLDS} (counts: {cv_counts})")
    if data.get("enabled_features") is not None:
        print(
            f"Features: {data['enabled_features']}  "
            f"input_dim={data['input_dim']}  recipe_id={data['feature_recipe_id']}"
        )
        if data.get("n_unique") is not None:
            extra = f"Unique sequences: {data['n_unique']} (from {data.get('n_rows', len(df))} rows)"
            if data.get("store_len") is not None:
                extra += f" store_len={data['store_len']}"
            print(extra)
    print(f"Data card: {data.get('data_card_path')}")
    print()
    print("Done! Ready to train (features are re-encoded on each train.py start).")


if __name__ == "__main__":
    from rnaomni.threads import apply_thread_limits

    apply_thread_limits()
    _cli()
else:
    import rnaomni.prepare as _pkg  # noqa: F401 — import-time apply_dataset_route()

    _facade = _PrepareModule(__name__, __doc__)
    _facade.__file__ = __file__
    _facade.__package__ = __package__
    sys.modules[__name__] = _facade
