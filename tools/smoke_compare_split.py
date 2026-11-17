#!/usr/bin/env python3
"""Small smoke test for the external pre-split route (Class_88fam_200%noise).

Usage (conda myenv3.11):
  python tools/smoke_compare_split.py                  # full data, load only
  RNAOMNI_DATA_MODE=sample python tools/smoke_compare_split.py --steps 0
  COMPARE_PER_CLASS_TRAIN=10 COMPARE_PER_CLASS_EVAL=5 RNAOMNI_DATA_MODE=sample python tools/smoke_compare_split.py --steps 0
"""

from __future__ import annotations

import argparse
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ["RNAOMNI_COMPARE_ROOT"] = os.path.join(
    REPO, "dataset", "forcompare", "Class_88fam_200%noise"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke external pre-split load + short train")
    parser.add_argument("--steps", type=int, default=0, help="Train steps on fold0 (0=load only)")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    import prepare
    from prepare import (
        USE_COMPARE_SPLITS,
        COMPARE_ROOT,
        COMPARE_FOLD_IDS,
        LABEL_COL,
        compare_active_csv_name,
        compare_sampling_on,
        evaluate_accuracy,
        evaluate_rmse,
        load_compare_splits_dataframe,
        load_prepared_data,
        make_dataloader,
        classification_num_classes,
        resolve_compare_per_class_eval,
        resolve_compare_per_class_train,
    )

    if not USE_COMPARE_SPLITS:
        print(
            "USE_COMPARE_SPLITS is False — set RNAOMNI_COMPARE_ROOT to a "
            "directory with train.csv/val.csv/test.csv (or drop --no-compare)."
        )
        return 1

    prepare._DATA_CACHE = None
    k_train = resolve_compare_per_class_train()
    k_eval = resolve_compare_per_class_eval()
    stem = compare_active_csv_name()
    print(f"COMPARE_ROOT={COMPARE_ROOT}")
    print(f"COMPARE_FOLD_IDS={COMPARE_FOLD_IDS}")
    print(f"COMPARE_PER_CLASS={prepare.COMPARE_PER_CLASS}")
    print(f"K_train={k_train} K_eval={k_eval} sampling_on={compare_sampling_on()}")
    print(f"COMPARE_SAMPLE_SEED={prepare.COMPARE_SAMPLE_SEED}")
    print(f"stem={stem}")

    assert not stem.endswith("_explore"), stem
    assert "_pc" not in stem, stem

    # Reproducibility: same per_class/seed → identical row counts twice.
    df_a = load_compare_splits_dataframe()
    df_b = load_compare_splits_dataframe()
    assert len(df_a) == len(df_b)
    assert list(df_a[LABEL_COL]) == list(df_b[LABEL_COL])
    for split_name, g in df_a.groupby("split"):
        n_cls = int(g[LABEL_COL].nunique())
        assert n_cls == 88, f"{split_name} classes={n_cls}"
        if compare_sampling_on():
            k = k_train if split_name == "train" else k_eval
            assert k is not None
            vc = g[LABEL_COL].value_counts()
            assert int(vc.max()) <= int(k), f"{split_name} max_count={int(vc.max())} > K={k}"
    n_classes = classification_num_classes(df_a[LABEL_COL])
    assert n_classes == 88, n_classes
    print(f"stratified smoke: rows={len(df_a)} num_classes={n_classes} (reproducible OK)")

    # Asymmetric A+D path (explicit args; does not depend on env).
    df_asym = load_compare_splits_dataframe(per_class_train=5, per_class_eval=3, quiet=True)
    for split_name, g in df_asym.groupby("split"):
        assert int(g[LABEL_COL].nunique()) == 88, split_name
        k = 5 if split_name == "train" else 3
        assert int(g[LABEL_COL].value_counts().max()) <= k
    print(
        f"asymmetric smoke: rows={len(df_asym)} "
        f"counts={df_asym['split'].value_counts().to_dict()}"
    )

    if args.steps <= 0:
        print("smoke_compare_split: OK (load only)")
        return 0

    t0 = time.time()
    data = load_prepared_data()
    print(f"load+encode seconds: {time.time() - t0:.2f}")
    print(
        f"task={data['task_type']} input_dim={data['input_dim']} "
        f"n_cv_folds={data['n_cv_folds']} n_rows={len(data['df'])} "
        f"num_classes={data.get('num_classes')}"
    )
    print(f"split counts: {data['df']['split'].value_counts().to_dict()}")
    print(f"features: {data['enabled_features']}")

    assert data["n_cv_folds"] == len(COMPARE_FOLD_IDS)
    assert data["use_compare_splits"] is True
    if data["task_type"] == "classification":
        assert int(data["num_classes"]) == 88

    fold_id = COMPARE_FOLD_IDS[0]
    loader_fold = 0 if len(COMPARE_FOLD_IDS) == 1 else fold_id

    if args.steps <= 0:
        print("smoke_compare_split: OK (load only)")
        return 0

    if not __import__("torch").cuda.is_available():
        print("CUDA not available — skip train steps")
        return 0

    import torch
    import torch.nn.functional as F
    from train import PooledMLP

    num_outputs = int(data["num_classes"]) if data["task_type"] == "classification" else 1
    model = PooledMLP(
        input_dim=data["input_dim"], num_outputs=num_outputs
    ).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loader = make_dataloader(loader_fold, "train", args.batch_size)

    model.train()
    t1 = time.time()
    for step, (x, y, lengths, epoch) in enumerate(loader):
        if step >= args.steps:
            break
        pred = model(x, lengths)
        if data["task_type"] == "classification":
            loss = F.cross_entropy(pred, y.long())
        else:
            if pred.ndim > 1:
                pred = pred.squeeze(-1)
            loss = F.mse_loss(pred, y.float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        print(f"step {step} loss={loss.item():.6f} epoch={epoch}")

    if data["task_type"] == "classification":
        acc = evaluate_accuracy(model, args.batch_size, fold_id=loader_fold)
        print(f"val_accuracy (fold {loader_fold}): {acc:.6f}")
    else:
        rmse = evaluate_rmse(model, args.batch_size, fold_id=loader_fold)
        print(f"val_RMSE (fold {loader_fold}): {rmse:.6f}")
    print(f"train+eval seconds: {time.time() - t1:.2f}")
    print("smoke_compare_split: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
