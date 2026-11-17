"""Fixed evaluation metrics. Agents must not change these."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import rnaomni.prepare.route as _route
from rnaomni.prepare.constants import N_FOLDS
from rnaomni.prepare.data import iter_batches, load_prepared_data


def _val_fold_ids(fold_id):
    fold_ids = range(N_FOLDS) if fold_id is None else [fold_id]
    if _route.USE_COMPARE_SPLITS and fold_id is None:
        fold_ids = list(range(len(_route.COMPARE_FOLD_IDS)))
    return list(fold_ids)


@torch.no_grad()
def _val_pred_target_tensors(model, batch_size, fold_id=None, csv_name=None):
    """Return (preds, targets) float tensors for each requested val fold."""
    if csv_name is None:
        csv_name = _route.ACTIVE_CSV
    was_training = model.training
    model.eval()
    pairs = []
    for fid in _val_fold_ids(fold_id):
        preds = []
        targets = []
        for x, y, lengths in iter_batches(fid, "val", batch_size, csv_name=csv_name):
            pred = model(x, lengths)
            if pred.ndim > 1:
                pred = pred.squeeze(-1)
            preds.append(pred.float())
            targets.append(y.float())
        pairs.append((torch.cat(preds), torch.cat(targets)))
    if was_training:
        model.train()
    return pairs


@torch.no_grad()
def evaluate_rmse(model, batch_size, fold_id=None, csv_name=None):
    """
    Root mean squared error on validation splits (regression).

    If fold_id is None, returns the mean val RMSE across all folds.
    Model: forward(x, lengths) -> (B,) predictions.
    """
    rmses = []
    for preds, targets in _val_pred_target_tensors(
        model, batch_size, fold_id=fold_id, csv_name=csv_name
    ):
        rmses.append(torch.sqrt(F.mse_loss(preds, targets)).item())
    return sum(rmses) / len(rmses)


@torch.no_grad()
def evaluate_spearman(model, batch_size, fold_id=None, csv_name=None):
    """
    Spearman rank correlation on validation splits (regression; higher better).

    If fold_id is None, returns the mean val Spearman across all folds.
    Degenerate folds (constant preds/targets) contribute 0.0.
    """
    scores = []
    for preds, targets in _val_pred_target_tensors(
        model, batch_size, fold_id=fold_id, csv_name=csv_name
    ):
        rho = float(
            pd.Series(preds.detach().cpu().numpy()).corr(
                pd.Series(targets.detach().cpu().numpy()), method="spearman"
            )
        )
        if not np.isfinite(rho):
            rho = 0.0
        scores.append(rho)
    return sum(scores) / len(scores)


@torch.no_grad()
def evaluate_accuracy(model, batch_size, fold_id=0, csv_name=None):
    """
    Classification accuracy on the validation split (fold_id default 0).
    Model: forward(x, lengths) -> (B, num_classes) logits.
    """
    if csv_name is None:
        csv_name = _route.ACTIVE_CSV
    was_training = model.training
    model.eval()

    correct = 0
    total = 0
    for x, y, lengths in iter_batches(fold_id, "val", batch_size, csv_name=csv_name):
        logits = model(x, lengths)
        if logits.ndim == 1:
            raise RuntimeError("Classification model must return (B, num_classes) logits")
        pred = logits.argmax(dim=-1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())

    if was_training:
        model.train()
    return correct / max(total, 1)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    f1s = []
    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return float(np.mean(f1s)) if f1s else float("nan")


def _paired_compare_test(data, models) -> bool:
    n = int(data.get("n_cv_folds", 1))
    return bool(data.get("use_compare_splits")) and n > 1 and len(models) > 1


def _test_fold_groups(models, data):
    """Yield (fold_id, models_for_that_test).

    Compare k-fold: model i on fold i test (no cross-fold ensemble).
    CSV route / single compare fold: ensemble all models on fold-0 test
    (HEK global holdout).
    """
    if _paired_compare_test(data, models):
        n = min(len(models), int(data.get("n_cv_folds", len(models))))
        for fold_id in range(n):
            yield fold_id, [models[fold_id]]
        return
    yield 0, list(models)


@torch.no_grad()
def _ensemble_test_arrays(models, fold_id, batch_size, csv_name, task_type):
    """Ensemble ``models`` on one test split. Returns (preds, targets) numpy."""
    for model in models:
        model.eval()
    if task_type == "classification":
        prob_sum = None
        targets = None
        for model_idx, model in enumerate(models):
            batch_probs = []
            batch_targets = []
            for x, y, lengths in iter_batches(fold_id, "test", batch_size, csv_name=csv_name):
                logits = model(x, lengths)
                probs = F.softmax(logits.float(), dim=-1).cpu()
                batch_probs.append(probs)
                if model_idx == 0:
                    batch_targets.append(y.long().cpu())
            model_probs = torch.cat(batch_probs, dim=0)
            prob_sum = model_probs if prob_sum is None else prob_sum + model_probs
            if model_idx == 0:
                targets = torch.cat(batch_targets, dim=0)
        probs = (prob_sum / len(models)).numpy()
        return probs.argmax(axis=-1), targets.numpy()

    pred_sum = None
    targets = None
    for model_idx, model in enumerate(models):
        batch_preds = []
        batch_targets = []
        for x, y, lengths in iter_batches(fold_id, "test", batch_size, csv_name=csv_name):
            pred = model(x, lengths).float().cpu()
            if pred.ndim > 1:
                pred = pred.squeeze(-1)
            batch_preds.append(pred)
            if model_idx == 0:
                batch_targets.append(y.float().cpu())
        model_preds = torch.cat(batch_preds)
        pred_sum = model_preds if pred_sum is None else pred_sum + model_preds
        if model_idx == 0:
            targets = torch.cat(batch_targets)
    return (pred_sum / len(models)).numpy(), targets.numpy()


def _regression_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    ss_res = float(np.sum((targets - preds) ** 2))
    ss_tot = float(np.sum((targets - targets.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    pred_s = pd.Series(preds)
    target_s = pd.Series(targets)
    return {
        "rmse": rmse,
        "r2": r2,
        "spearman": float(pred_s.corr(target_s, method="spearman")),
        "pearson": float(pred_s.corr(target_s, method="pearson")),
    }


@torch.no_grad()
def evaluate_test_metrics(models, batch_size, csv_name=None):
    """
    Test-set metrics. Behavior depends on inferred task_type in prepared data.

    Compare k-fold: each model is scored on its own test split, then metrics
    are computed on the concatenated predictions (true 10-fold test).
    CSV / single-fold compare: ensemble mean on fold-0 test (HEK holdout).

    Regression: RMSE, R², Spearman, Pearson.
    Classification: accuracy, macro_F1.
    """
    if csv_name is None:
        csv_name = _route.ACTIVE_CSV
    if not models:
        raise ValueError("models must be non-empty")

    data = load_prepared_data(csv_name)
    task_type = data["task_type"]

    pred_chunks = []
    target_chunks = []
    for fold_id, fold_models in _test_fold_groups(models, data):
        preds, targets = _ensemble_test_arrays(
            fold_models, fold_id, batch_size, csv_name, task_type
        )
        pred_chunks.append(preds)
        target_chunks.append(targets)

    preds = np.concatenate(pred_chunks)
    targets = np.concatenate(target_chunks)

    if task_type == "classification":
        num_classes = int(data["num_classes"])
        acc = float(np.mean(preds == targets))
        return {
            "accuracy": acc,
            "macro_f1": _macro_f1(targets, preds, num_classes),
        }

    return _regression_metrics(preds, targets)
