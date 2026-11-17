#!/usr/bin/env python3
"""
RNAOmni finalize: re-train the best search commit, save checkpoints, export test predictions.

Usage:
    python -m rnaomni.finalize --commit auto
    python -m rnaomni.finalize --config finalize.json --run-tag 20260621_131515
    python -m rnaomni.finalize --commit 98b2dbb --mult 100 --output-dir workspace/runs/<tag>/finalize
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import ModuleType

try:
    sys.stdout.reconfigure(line_buffering=True)
except (AttributeError, OSError):
    pass

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

from rnaomni.threads import apply_thread_limits

apply_thread_limits()  # OMP/MKL/OPENBLAS before numpy/torch load

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

apply_thread_limits()  # torch.set_num_threads after import

from prepare import (
    ACTIVE_CSV,
    INPUT_DIM,
    LABEL_COL,
    MAX_SEQ_LEN,
    N_FOLDS,
    SEQUENCE_COL,
    USE_COMPARE_SPLITS,
    _indices_for_split,
    compare_active_csv_name,
    encode_mode,
    encode_timeout_sec,
    evaluate_accuracy,
    evaluate_spearman,
    evaluate_test_metrics,
    gather_row_features,
    load_prepared_data,
    make_dataloader,
    search_fold_budget_sec,
)
from rnaomni.results_io import (
    RESULTS_HEADER_CLASSIFICATION,
    RESULTS_HEADER_REGRESSION,
    results_tsv_path,
)
from rnaomni.paths import REPO, RUNS_DIR

# Importing this module must not flip overnight train.py onto the online path.
_FINALIZE_RUNTIME_CONFIGURED = False


def configure_finalize_runtime() -> None:
    """15.2 B+A: kill encode wall; Dataset encodes on the batch path. Force, do not setdefault."""
    global _FINALIZE_RUNTIME_CONFIGURED
    os.environ["ENCODE_TIMEOUT_SEC"] = "0"
    os.environ["RNAOMNI_ENCODE_MODE"] = "online"
    if not _FINALIZE_RUNTIME_CONFIGURED:
        print(
            "Finalize runtime: ENCODE_TIMEOUT_SEC=0 RNAOMNI_ENCODE_MODE=online",
            flush=True,
        )
        _FINALIZE_RUNTIME_CONFIGURED = True


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _peak_vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated()) / 1024.0 / 1024.0


def _reset_peak_vram() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def resource_payload(
    *,
    training_seconds: float,
    train_wall_s: float,
    test_eval_seconds: float,
    total_seconds: float,
    peak_train_vram_mb: float,
    peak_test_vram_mb: float,
) -> dict:
    """Train/test time + VRAM for a full finalize run (written next to metrics.json)."""
    peak_vram_mb = max(float(peak_train_vram_mb), float(peak_test_vram_mb))
    return {
        "training_seconds": float(training_seconds),
        "gpu_train_s": float(training_seconds),
        "train_wall_s": float(train_wall_s),
        "test_eval_seconds": float(test_eval_seconds),
        "total_seconds": float(total_seconds),
        "peak_train_vram_mb": float(peak_train_vram_mb),
        "peak_test_vram_mb": float(peak_test_vram_mb),
        "peak_vram_mb": peak_vram_mb,
    }


def finalize_split_indices(data: dict, fold_id: int, split: str):
    """Row indices for finalize (wrapper over prepare._indices_for_split)."""
    return _indices_for_split(data, fold_id, split)


def finalize_default_csv_name() -> str:
    """Active dataset label for finalize when --csv / config omit csv_name."""
    if USE_COMPARE_SPLITS:
        return compare_active_csv_name()
    return ACTIVE_CSV


# Fallback defaults only; CLI / config file override these.
DEFAULTS = {
    "mult": 100.0,
    "early_stop": False,
    "patience": 5,
    "min_delta_ratio": 0.002,
    "checkpoint_mode": "best_val",
    "target_epochs": 100,
    "probe_steps": 20,
}


@dataclass
class FinalizeConfig:
    commit: str = "auto"
    run_tag: str | None = None
    output_dir: str | None = None
    csv_name: str = field(default_factory=finalize_default_csv_name)
    mult: float = DEFAULTS["mult"]
    fold_budget_sec: float | None = None
    early_stop: bool = DEFAULTS["early_stop"]
    patience: int = DEFAULTS["patience"]
    min_delta: float | None = None
    min_delta_ratio: float = DEFAULTS["min_delta_ratio"]
    eval_every: int | None = None
    checkpoint_mode: str = DEFAULTS["checkpoint_mode"]
    target_epochs: int = DEFAULTS["target_epochs"]
    seed: int = 42
    resolved: dict = field(default_factory=dict)


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=REPO,
        stderr=subprocess.DEVNULL,
        text=True,
    ).strip()


def infer_run_tag() -> str:
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    prefix = "RNAclaw/"
    if branch.startswith(prefix):
        return branch[len(prefix) :]
    return time.strftime("%Y%m%d_%H%M%S")


def load_config_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as err:
            raise SystemExit(f"PyYAML is required for {path}: {err}") from err
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise SystemExit(f"Config file must contain a JSON/YAML object: {path}")
    return data


def select_best_commit(results_path: Path | None = None) -> tuple[str, float, str]:
    if results_path is None:
        results_path = results_tsv_path()
    if not results_path.is_file():
        raise SystemExit(f"results.tsv not found: {results_path}")

    best_commit = None
    best_val = None
    best_desc = ""

    with results_path.open(encoding="utf-8") as f:
        header = f.readline()
        if header == RESULTS_HEADER_CLASSIFICATION:
            higher_better = True
        elif header == RESULTS_HEADER_REGRESSION:
            higher_better = True
        else:
            raise SystemExit(
                "Unexpected results.tsv header; expected regression or classification format."
            )

        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            commit = parts[0]
            val_s = parts[1]
            status = parts[-2]
            desc = parts[-1]
            if status != "keep":
                continue
            try:
                val = float(val_s)
            except ValueError:
                continue
            if not math.isfinite(val):
                continue
            if best_val is None:
                best_val = val
                best_commit = commit
                best_desc = desc
            elif higher_better and val > best_val:
                best_val = val
                best_commit = commit
                best_desc = desc
            elif (not higher_better) and val < best_val:
                best_val = val
                best_commit = commit
                best_desc = desc

    if best_commit is None or best_val is None:
        raise SystemExit("No successful keep rows found in results.tsv.")

    full_commit = _git("rev-parse", best_commit)
    return full_commit, best_val, best_desc


def resolve_commit(commit: str) -> tuple[str, float | None, str]:
    if commit == "auto":
        return select_best_commit()

    full_commit = _git("rev-parse", commit)
    search_val = None
    desc = ""
    results_path = results_tsv_path()
    if results_path.is_file():
        with results_path.open(encoding="utf-8") as f:
            next(f, None)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 9 and parts[0].startswith(commit[:7]):
                    try:
                        search_val = float(parts[1])
                    except ValueError:
                        pass
                    desc = parts[8]
                    break
    return full_commit, search_val, desc


def resolve_predictor_class(train_mod: ModuleType) -> type | None:
    """Prefer ``Predictor``; fall back to ``PooledMLP`` for older search snapshots."""
    cls = getattr(train_mod, "Predictor", None)
    if cls is not None:
        return cls
    return getattr(train_mod, "PooledMLP", None)


def load_train_module(commit: str) -> ModuleType:
    source = subprocess.check_output(
        ["git", "show", f"{commit}:train.py"],
        cwd=REPO,
        stderr=subprocess.PIPE,
        text=True,
    )
    module = ModuleType(f"train_snapshot_{commit[:7]}")
    module.__file__ = f"{commit}:train.py"
    sys.modules[module.__name__] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)  # noqa: S102
    module._source = source
    if resolve_predictor_class(module) is None:
        raise SystemExit(
            f"train.py at {commit[:7]} has neither Predictor nor PooledMLP."
        )
    return module


def _train_step_loss(train_mod: ModuleType, task_type: str, pred, y):
    """Use the searched train.py loss, not a hardcoded search-baseline MSE."""
    if task_type == "classification":
        ls = float(getattr(train_mod, "LABEL_SMOOTHING", 0.0) or 0.0)
        if ls:
            return F.cross_entropy(pred, y.long(), label_smoothing=ls)
        return F.cross_entropy(pred, y.long())
    src = getattr(train_mod, "_source", "") or ""
    if "smooth_l1_loss" in src:
        match = re.search(r"smooth_l1_loss\([^)]*beta\s*=\s*([0-9eE.+-]+)", src)
        beta = float(match.group(1)) if match else 1.0
        return F.smooth_l1_loss(pred, y, beta=beta)
    return F.mse_loss(pred, y)


def _finalize_lr_scale(
    *,
    gpu_train_s: float,
    warmup_duration: float,
    fold_budget: float,
    min_lr_ratio: float,
) -> float:
    """Warmup is 10% of the finalize fold GPU budget; cosine over the remainder."""
    if warmup_duration > 0 and gpu_train_s < warmup_duration:
        return gpu_train_s / warmup_duration
    if fold_budget <= warmup_duration:
        return 1.0
    progress = (gpu_train_s - warmup_duration) / (fold_budget - warmup_duration)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    floor = min(max(float(min_lr_ratio), 0.0), 1.0)
    return floor + (1.0 - floor) * cosine


def _model_kwargs(train_mod: ModuleType, data: dict | None = None) -> dict:
    cls = resolve_predictor_class(train_mod)
    if cls is None:
        raise SystemExit("train.py snapshot has neither Predictor nor PooledMLP.")
    sig = inspect.signature(cls.__init__)
    num_outputs = 1
    if data is not None and data.get("task_type") == "classification":
        num_outputs = int(data["num_classes"])
    attr_map = {
        "input_dim": (data or {}).get("input_dim", INPUT_DIM),
        "hidden_dim": getattr(train_mod, "HIDDEN_DIM", 256),
        "dropout": getattr(train_mod, "DROPOUT", 0.1),
        "conv_channels": getattr(train_mod, "CONV_CHANNELS", None),
        "num_outputs": num_outputs,
    }
    kwargs: dict = {}
    for pname, param in sig.parameters.items():
        if pname == "self":
            continue
        if pname in attr_map and attr_map[pname] is not None:
            kwargs[pname] = attr_map[pname]
        elif param.default is not inspect.Parameter.empty:
            kwargs[pname] = param.default
        else:
            raise SystemExit(
                f"Cannot infer Predictor argument {pname!r} from train.py snapshot."
            )
    return kwargs


def build_model(train_mod: ModuleType, device: torch.device, data: dict | None = None) -> torch.nn.Module:
    cls = resolve_predictor_class(train_mod)
    if cls is None:
        raise SystemExit("train.py snapshot has neither Predictor nor PooledMLP.")
    return cls(**_model_kwargs(train_mod, data)).to(device)


def _resolved_float(resolved: dict, key: str, default: float) -> float:
    """Read a float from cfg.resolved; 0.0 is valid and must not fall back."""
    if key not in resolved or resolved[key] is None:
        return float(default)
    return float(resolved[key])


def clamp_fold_budget(desired: float, budget_min: float, budget_max: float) -> float:
    lo = float(budget_min)
    hi = float(budget_max)
    if hi < lo:
        lo, hi = hi, lo
    return min(hi, max(lo, float(desired)))


def resolve_probed_fold_budget(
    *,
    gpu_s: float,
    n_steps: int,
    steps_per_epoch: int,
    target_epochs: int,
    budget_min: float,
    budget_max: float,
) -> dict:
    """Map a short GPU probe onto K epochs, then cap at mult×search wall-clock.

    Search TIME_BUDGET is wall-clock (loader/encode included). Finalize
    budgets are GPU-step seconds. Do not use search wall-clock as a GPU
    floor — that over-trains fast folds (e.g. 16s probe forced to 60s).
    """
    spe = max(int(steps_per_epoch), 1)
    k = max(int(target_epochs), 1)
    lo = float(budget_min)
    hi = float(budget_max)
    if n_steps <= 0 or gpu_s <= 0:
        sec_per_step = 0.0
        epoch_gpu_s = 0.0
        desired = hi
    else:
        sec_per_step = float(gpu_s) / float(n_steps)
        epoch_gpu_s = sec_per_step * spe
        desired = epoch_gpu_s * k
    chosen = clamp_fold_budget(desired, lo, hi)
    return {
        "fold_budget_sec": chosen,
        "sec_per_step": sec_per_step,
        "epoch_gpu_s": epoch_gpu_s,
        "desired_gpu_s": desired,
        "steps_per_epoch": spe,
        "target_epochs": k,
        "budget_min_sec": min(lo, hi),
        "budget_max_sec": max(lo, hi),
    }


def steps_per_epoch(fold_id: int, batch_size: int, csv_name: str) -> int:
    data = load_prepared_data(csv_name)
    indices = finalize_split_indices(data, fold_id, "train")
    return max(1, math.ceil(len(indices) / batch_size))


def resolve_config(args: argparse.Namespace, file_cfg: dict | None) -> FinalizeConfig:
    configure_finalize_runtime()
    merged: dict = dict(DEFAULTS)
    if file_cfg:
        merged.update({k: v for k, v in file_cfg.items() if v is not None})
    for key in (
        "commit",
        "run_tag",
        "output_dir",
        "csv_name",
        "mult",
        "fold_budget_sec",
        "early_stop",
        "patience",
        "min_delta",
        "min_delta_ratio",
        "eval_every",
        "checkpoint_mode",
        "target_epochs",
        "seed",
    ):
        val = getattr(args, key, None)
        if val is not None:
            merged[key] = val

    cfg = FinalizeConfig(
        commit=str(merged.get("commit", "auto")),
        run_tag=merged.get("run_tag"),
        output_dir=merged.get("output_dir"),
        csv_name=str(merged.get("csv_name") or finalize_default_csv_name()),
        mult=float(merged.get("mult", DEFAULTS["mult"])),
        fold_budget_sec=merged.get("fold_budget_sec"),
        early_stop=bool(merged.get("early_stop", DEFAULTS["early_stop"])),
        patience=int(merged.get("patience", DEFAULTS["patience"])),
        min_delta=merged.get("min_delta"),
        min_delta_ratio=float(merged.get("min_delta_ratio", DEFAULTS["min_delta_ratio"])),
        eval_every=merged.get("eval_every"),
        checkpoint_mode=str(merged.get("checkpoint_mode", DEFAULTS["checkpoint_mode"])),
        target_epochs=int(merged.get("target_epochs", DEFAULTS["target_epochs"])),
        seed=int(merged.get("seed", 42)),
    )

    if cfg.checkpoint_mode not in {"best_val", "last"}:
        raise SystemExit("checkpoint_mode must be 'best_val' or 'last'.")

    full_commit, search_val, desc = resolve_commit(cfg.commit)
    cfg.commit = full_commit

    train_mod = load_train_module(full_commit)
    batch_size = int(getattr(train_mod, "BATCH_SIZE", 256))
    data = load_prepared_data(csv_name=cfg.csv_name)
    task_type = data["task_type"]
    n_cv_folds = int(data.get("n_cv_folds", N_FOLDS))
    search_fold_budget = search_fold_budget_sec(
        n_cv_folds, use_compare_splits=bool(data.get("use_compare_splits"))
    )
    # Search TIME_BUDGET is wall-clock; it is not a GPU-second floor.
    budget_min = 0.0
    budget_max = float(search_fold_budget) * float(cfg.mult)
    user_fold_budget = merged.get("fold_budget_sec")
    if user_fold_budget is not None:
        cfg.fold_budget_sec = float(user_fold_budget)
        budget_mode = "fixed"
    else:
        cfg.fold_budget_sec = budget_max
        budget_mode = "probe"

    run_tag = cfg.run_tag or infer_run_tag()
    cfg.run_tag = run_tag
    if cfg.output_dir is None:
        cfg.output_dir = str(RUNS_DIR / run_tag / "finalize")

    if cfg.min_delta is None:
        baseline = search_val if search_val is not None else (0.01 if task_type == "classification" else 1.0)
        cfg.min_delta = cfg.min_delta_ratio * abs(baseline)

    cfg.resolved = {
        "search_fold_budget_sec": search_fold_budget,
        "fold_budget_sec": cfg.fold_budget_sec,
        "budget_mode": budget_mode,
        "budget_min_sec": budget_min,
        "budget_max_sec": budget_max,
        "target_epochs": int(cfg.target_epochs),
        "probe_steps": int(DEFAULTS["probe_steps"]),
        "search_val_metric": search_val,
        "search_description": desc,
        "batch_size": batch_size,
        "steps_per_epoch_fold0": steps_per_epoch(0, batch_size, cfg.csv_name),
        "commit_short": full_commit[:7],
        "task_type": task_type,
        "n_cv_folds": n_cv_folds,
        "num_classes": data.get("num_classes"),
        "higher_is_better": True,
    }
    if cfg.eval_every is None:
        cfg.eval_every = max(int(cfg.resolved["steps_per_epoch_fold0"]), 1)
    cfg.resolved["eval_mode"] = "epoch"
    cfg.resolved["eval_every"] = int(cfg.eval_every)
    # Small-data epochs can be 8 steps; do not eval every epoch across a 6000s horizon.
    cfg.resolved["eval_min_gpu_s"] = float(search_fold_budget)

    return cfg, train_mod


def train_one_fold(
    fold_id: int,
    train_mod: ModuleType,
    cfg: FinalizeConfig,
    device: torch.device,
    data: dict,
) -> dict:
    batch_size = cfg.resolved["batch_size"]
    search_fold_budget = float(cfg.resolved["search_fold_budget_sec"])
    budget_min = _resolved_float(cfg.resolved, "budget_min_sec", 0.0)
    budget_max = _resolved_float(
        cfg.resolved, "budget_max_sec", search_fold_budget * cfg.mult
    )
    budget_mode = str(cfg.resolved.get("budget_mode") or "probe")
    target_epochs = int(cfg.resolved.get("target_epochs") or cfg.target_epochs)
    probe_steps = max(int(cfg.resolved.get("probe_steps") or DEFAULTS["probe_steps"]), 1)
    fold_budget = float(cfg.fold_budget_sec)
    lr = float(getattr(train_mod, "LR", 1e-3))
    weight_decay = float(getattr(train_mod, "WEIGHT_DECAY", 1e-4))
    ema_decay = float(getattr(train_mod, "EMA_DECAY", 0.999))
    warmup_fraction = float(getattr(train_mod, "WARMUP_FRACTION", 0.1))
    min_lr_ratio = float(getattr(train_mod, "MIN_LR", 0.0) or 0.0)
    task_type = data["task_type"]
    higher_is_better = True
    spe = steps_per_epoch(fold_id, batch_size, cfg.csv_name)
    eval_every = max(int(cfg.eval_every or spe), 1)
    eval_min_gpu_s = float(cfg.resolved.get("eval_min_gpu_s") or search_fold_budget)
    last_eval_gpu_s = 0.0
    budget_probe = {
        "fold_budget_sec": fold_budget,
        "sec_per_step": 0.0,
        "epoch_gpu_s": 0.0,
        "desired_gpu_s": fold_budget,
        "steps_per_epoch": spe,
        "target_epochs": target_epochs,
        "budget_min_sec": budget_min,
        "budget_max_sec": budget_max,
    }
    budget_resolved = budget_mode == "fixed"

    model = build_model(train_mod, device, data)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ema = train_mod.EMA(model, ema_decay)

    train_loader = make_dataloader(fold_id, "train", batch_size, csv_name=cfg.csv_name)
    x, y, lengths, epoch = next(train_loader)

    fold_start = time.time()
    step = 0
    smooth_loss = 0.0
    gpu_train_s = 0.0
    wall_s = 0.0
    evals_without_improve = 0
    best_val = -math.inf if higher_is_better else math.inf
    best_step = 0
    best_epoch = 0
    best_state: dict | None = None
    stopped_by = "time_cap"

    def _eval_val(m):
        if task_type == "classification":
            return evaluate_accuracy(m, batch_size, fold_id=fold_id, csv_name=cfg.csv_name)
        return evaluate_spearman(m, batch_size, fold_id=fold_id, csv_name=cfg.csv_name)

    def _is_improved(val, best):
        if higher_is_better:
            return val > best + cfg.min_delta
        return val + cfg.min_delta < best

    while True:
        wall_s = time.time() - fold_start
        warmup_duration = fold_budget * warmup_fraction
        lr_scale = _finalize_lr_scale(
            gpu_train_s=gpu_train_s,
            warmup_duration=warmup_duration,
            fold_budget=fold_budget,
            min_lr_ratio=min_lr_ratio,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * lr_scale

        _cuda_sync(device)
        t0 = time.time()
        pred = model(x, lengths)
        loss = _train_step_loss(train_mod, task_type, pred, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        ema.update(model)
        _cuda_sync(device)
        gpu_train_s += time.time() - t0

        loss_f = loss.item()
        if math.isnan(loss_f) or loss_f > 1e4:
            raise RuntimeError(f"fold {fold_id} failed: loss={loss_f}")

        x, y, lengths, epoch = next(train_loader)

        ema_beta = 0.9
        smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * loss_f
        debiased = smooth_loss / (1 - ema_beta ** (step + 1))
        remaining = max(0.0, fold_budget - gpu_train_s)
        print(
            f"\rfold {fold_id} step {step:05d} | loss: {debiased:.6f} | "
            f"epoch: {epoch} | gpu_train_s: {gpu_train_s:.1f}/{fold_budget:.0f}s | "
            f"wall_s: {wall_s:.0f}s remaining: {remaining:.0f}s    ",
            end="",
            flush=True,
        )

        step += 1

        if not budget_resolved and step >= probe_steps:
            budget_probe = resolve_probed_fold_budget(
                gpu_s=gpu_train_s,
                n_steps=step,
                steps_per_epoch=spe,
                target_epochs=target_epochs,
                budget_min=budget_min,
                budget_max=budget_max,
            )
            fold_budget = float(budget_probe["fold_budget_sec"])
            eval_min_gpu_s = min(search_fold_budget, max(fold_budget * 0.1, 1e-6))
            budget_resolved = True
            print(
                f"\nfold {fold_id} budget: {fold_budget:.1f}s GPU "
                f"(T_epoch={budget_probe['epoch_gpu_s']:.4f}s × {target_epochs}, "
                f"desired={budget_probe['desired_gpu_s']:.1f}s, "
                f"clamped to [{budget_min:.0f}, {budget_max:.0f}])"
            )

        at_epoch = step > 10 and eval_every > 0 and (step % eval_every == 0)
        past_eval_floor = (gpu_train_s - last_eval_gpu_s) >= eval_min_gpu_s
        should_eval = at_epoch and past_eval_floor
        if should_eval:
            last_eval_gpu_s = gpu_train_s
            model.load_state_dict(ema.shadow)
            val_metric = _eval_val(model)
            model.train()
            metric_name = "val_accuracy" if task_type == "classification" else "val_spearman"
            print(
                f"\nfold {fold_id} step {step:05d} | {metric_name}: {val_metric:.6f} | "
                f"best: {best_val:.6f} | patience: {evals_without_improve}/{cfg.patience}"
            )
            if _is_improved(val_metric, best_val):
                best_val = val_metric
                best_step = step
                best_epoch = epoch
                best_state = ema.state_dict()
                evals_without_improve = 0
            else:
                evals_without_improve += 1
                if cfg.early_stop and evals_without_improve >= cfg.patience:
                    stopped_by = "early_stop"
                    break

        if step > 10 and gpu_train_s >= fold_budget:
            break

    print()
    wall_s = time.time() - fold_start

    if best_state is None:
        best_state = ema.state_dict()
        model.load_state_dict(ema.shadow)
        best_val = _eval_val(model)
        best_step = step
        best_epoch = epoch
    elif cfg.checkpoint_mode == "last":
        best_state = ema.state_dict()
        model.load_state_dict(ema.shadow)
        best_val = _eval_val(model)
        stopped_by = "time_cap" if stopped_by != "early_stop" else stopped_by
    else:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    final_val = _eval_val(model)
    metric_name = "val_accuracy" if task_type == "classification" else "val_spearman"
    print(f"fold {fold_id} final {metric_name}: {final_val:.6f} (stopped_by={stopped_by})")

    del model, optimizer, train_loader
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "fold_id": fold_id,
        "state_dict": best_state,
        "val_metric": final_val,
        "best_val_metric": best_val,
        # backward-compatible keys
        "val_rmse": final_val,
        "best_val_rmse": best_val,
        "steps": step,
        "best_step": best_step,
        "best_epoch": best_epoch,
        "training_seconds": gpu_train_s,
        "gpu_train_s": gpu_train_s,
        "wall_s": wall_s,
        "peak_train_vram_mb": _peak_vram_mb(),
        "stopped_by": stopped_by,
        "fold_budget_sec": fold_budget,
        "epoch_gpu_s": budget_probe["epoch_gpu_s"],
        "desired_gpu_s": budget_probe["desired_gpu_s"],
        "sec_per_step": budget_probe["sec_per_step"],
    }


@torch.no_grad()
def collect_test_predictions(
    models: list[torch.nn.Module],
    batch_size: int,
    csv_name: str,
) -> pd.DataFrame:
    data = load_prepared_data(csv_name)
    df = data["df"]
    task_type = data["task_type"]
    n_cv = int(data.get("n_cv_folds", 1))
    paired = bool(data.get("use_compare_splits")) and n_cv > 1 and len(models) > 1
    if paired:
        groups = [(i, [models[i]]) for i in range(min(len(models), n_cv))]
    else:
        groups = [(0, list(models))]

    for model in models:
        model.eval()

    rows: list[dict] = []
    for cv_fold, fold_models in groups:
        indices = finalize_split_indices(data, cv_fold, "test")
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            idx_t = torch.as_tensor(batch_idx, dtype=torch.long)
            x, lengths = gather_row_features(data, idx_t)
            x = x.to(device="cuda", non_blocking=True)
            y = data["labels"].index_select(0, idx_t)
            lengths = lengths.to(device="cuda", non_blocking=True)

            if task_type == "classification":
                prob_sum = None
                for model in fold_models:
                    logits = model(x, lengths).float()
                    probs = F.softmax(logits, dim=-1).cpu()
                    prob_sum = probs if prob_sum is None else prob_sum + probs
                probs = (prob_sum / len(fold_models)).numpy()
                preds = probs.argmax(axis=-1)
                for local_i, row_idx in enumerate(batch_idx):
                    rec = {
                        "row_index": int(row_idx),
                        "input_seq": df.iloc[int(row_idx)][SEQUENCE_COL],
                        "predicted_target": int(y[local_i].item()),
                        "prediction": int(preds[local_i]),
                        "fold": int(df.iloc[int(row_idx)]["fold"]),
                        "cv_fold": int(cv_fold),
                    }
                    if "compare_fold" in df.columns:
                        rec["compare_fold"] = int(df.iloc[int(row_idx)]["compare_fold"])
                    rows.append(rec)
            else:
                pred_sum = None
                for model in fold_models:
                    batch_pred = model(x, lengths).float().cpu()
                    if batch_pred.ndim > 1:
                        batch_pred = batch_pred.squeeze(-1)
                    pred_sum = batch_pred if pred_sum is None else pred_sum + batch_pred
                preds = (pred_sum / len(fold_models)).numpy()

                for local_i, row_idx in enumerate(batch_idx):
                    rec = {
                        "row_index": int(row_idx),
                        "input_seq": df.iloc[int(row_idx)][SEQUENCE_COL],
                        "predicted_target": float(y[local_i].item()),
                        "prediction": float(preds[local_i]),
                        "fold": int(df.iloc[int(row_idx)]["fold"]),
                        "cv_fold": int(cv_fold),
                    }
                    if "compare_fold" in df.columns:
                        rec["compare_fold"] = int(df.iloc[int(row_idx)]["compare_fold"])
                    rows.append(rec)

    return pd.DataFrame(rows)


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def export_train_snapshot(commit: str, out_dir: Path) -> None:
    source = subprocess.check_output(
        ["git", "show", f"{commit}:train.py"],
        cwd=REPO,
        text=True,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train.py.snapshot").write_text(source, encoding="utf-8")


def run_finalize(cfg: FinalizeConfig, train_mod: ModuleType) -> dict:
    configure_finalize_runtime()
    out_dir = Path(cfg.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)

    export_train_snapshot(cfg.commit, out_dir)

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    device = torch.device("cuda")

    data = load_prepared_data(csv_name=cfg.csv_name)
    task_type = data["task_type"]
    n_cv_folds = int(cfg.resolved.get("n_cv_folds", data.get("n_cv_folds", N_FOLDS)))
    print(f"Finalize commit: {cfg.commit[:7]}")
    print(f"Task: {task_type} | Output dir: {out_dir}")
    search_fold_budget = float(cfg.resolved["search_fold_budget_sec"])
    warmup_fraction = float(getattr(train_mod, "WARMUP_FRACTION", 0.1))
    budget_mode = str(cfg.resolved.get("budget_mode") or "probe")
    budget_min = _resolved_float(cfg.resolved, "budget_min_sec", 0.0)
    budget_max = _resolved_float(
        cfg.resolved, "budget_max_sec", float(cfg.fold_budget_sec)
    )
    if budget_mode == "fixed":
        print(
            f"Fold budget: {cfg.fold_budget_sec:.1f}s GPU-step (fixed --fold-budget-sec; "
            f"search per-fold wall {search_fold_budget:.1f}s)"
        )
        print(
            f"Warmup: {cfg.fold_budget_sec * warmup_fraction:.1f}s GPU-step "
            f"(finalize fold budget × WARMUP_FRACTION={warmup_fraction:g})"
        )
    else:
        print(
            f"Fold budget: probe {cfg.target_epochs} epochs, "
            f"clamp [{budget_min:.0f}, {budget_max:.0f}]s GPU "
            f"(no search-wall GPU floor; cap {cfg.mult:g}× search wall {search_fold_budget:.0f}s)"
        )
        print(
            f"Warmup: {warmup_fraction:g} of the probed fold budget "
            f"(WARMUP_FRACTION={warmup_fraction:g})"
        )
    print(f"Encode: mode={encode_mode()} timeout_sec={encode_timeout_sec()}")
    print(
        f"Eval: every {cfg.eval_every} steps (1 epoch) and "
        f"≥{cfg.resolved['eval_min_gpu_s']:.0f}s GPU since last eval "
        f"(save {cfg.checkpoint_mode}, does not stop)"
    )
    print(f"Early stop: {cfg.early_stop} | patience: {cfg.patience} | min_delta: {cfg.min_delta:.6f}")

    t_start = time.time()
    fold_results = []
    fold_state_dicts = []

    for fold_id in range(n_cv_folds):
        print(f"\n=== Fold {fold_id}/{n_cv_folds - 1} ===")
        _reset_peak_vram()
        result = train_one_fold(fold_id, train_mod, cfg, device, data)
        result["peak_train_vram_mb"] = float(result.get("peak_train_vram_mb") or _peak_vram_mb())
        fold_results.append(result)
        fold_state_dicts.append(result["state_dict"])

        payload = {
            "fold_id": fold_id,
            "git_commit": cfg.commit[:7],
            "csv_name": cfg.csv_name,
            "task_type": task_type,
            "state_dict": result["state_dict"],
            "val_metric": result["val_metric"],
            "best_val_metric": result["best_val_metric"],
            "steps": result["steps"],
            "best_step": result["best_step"],
            "best_epoch": result["best_epoch"],
            "training_seconds": result["training_seconds"],
            "gpu_train_s": result.get("gpu_train_s", result["training_seconds"]),
            "wall_s": result.get("wall_s"),
            "peak_train_vram_mb": result.get("peak_train_vram_mb"),
            "stopped_by": result["stopped_by"],
            "hyperparams": {
                "model": _model_kwargs(train_mod, data),
                "batch_size": cfg.resolved["batch_size"],
                "lr": getattr(train_mod, "LR", 1e-3),
                "weight_decay": getattr(train_mod, "WEIGHT_DECAY", 1e-4),
                "ema_decay": getattr(train_mod, "EMA_DECAY", 0.999),
                "warmup_fraction": getattr(train_mod, "WARMUP_FRACTION", 0.1),
            },
        }
        save_checkpoint(ckpt_dir / f"fold_{fold_id}.pt", payload)

    val_metric = sum(r["val_metric"] for r in fold_results) / len(fold_results)

    if bool(data.get("use_compare_splits")) and n_cv_folds > 1:
        print("\n=== Test eval (paired per-fold) ===")
    else:
        print("\n=== Test ensemble eval ===")
    _reset_peak_vram()
    t_test_start = time.time()
    test_models = []
    for state_dict in fold_state_dicts:
        test_model = build_model(train_mod, device, data)
        test_model.load_state_dict({k: v.to(device) for k, v in state_dict.items()})
        test_models.append(test_model)

    test_metrics = evaluate_test_metrics(test_models, cfg.resolved["batch_size"], csv_name=cfg.csv_name)
    predictions = collect_test_predictions(test_models, cfg.resolved["batch_size"], csv_name=cfg.csv_name)
    test_eval_seconds = time.time() - t_test_start
    peak_test_vram_mb = _peak_vram_mb()

    predictions.to_csv(out_dir / "test_predictions.csv", index=False)

    total_seconds = time.time() - t_start
    training_seconds = sum(r["training_seconds"] for r in fold_results)
    train_wall_s = sum(float(r.get("wall_s") or 0.0) for r in fold_results)
    peak_train_vram_mb = max(
        (float(r.get("peak_train_vram_mb") or 0.0) for r in fold_results),
        default=0.0,
    )
    resource = resource_payload(
        training_seconds=training_seconds,
        train_wall_s=train_wall_s,
        test_eval_seconds=test_eval_seconds,
        total_seconds=total_seconds,
        peak_train_vram_mb=peak_train_vram_mb,
        peak_test_vram_mb=peak_test_vram_mb,
    )

    fold_resource_keys = {
        "training_seconds": lambda r: r["training_seconds"],
        "gpu_train_s": lambda r: r.get("gpu_train_s", r["training_seconds"]),
        "wall_s": lambda r: r.get("wall_s"),
        "peak_train_vram_mb": lambda r: r.get("peak_train_vram_mb"),
        "stopped_by": lambda r: r["stopped_by"],
        "fold_budget_sec": lambda r: r.get("fold_budget_sec"),
        "epoch_gpu_s": lambda r: r.get("epoch_gpu_s"),
        "desired_gpu_s": lambda r: r.get("desired_gpu_s"),
    }

    if task_type == "classification":
        metrics = {
            "val_accuracy": val_metric,
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            **resource,
            "folds": [
                {
                    "fold_id": r["fold_id"],
                    "val_accuracy": r["val_metric"],
                    "best_val_accuracy": r["best_val_metric"],
                    "steps": r["steps"],
                    "best_step": r["best_step"],
                    "best_epoch": r["best_epoch"],
                    **{k: fn(r) for k, fn in fold_resource_keys.items()},
                }
                for r in fold_results
            ],
        }
        metrics_summary = {
            "val_accuracy": val_metric,
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
        }
    else:
        metrics = {
            "val_spearman": val_metric,
            "test_rmse": test_metrics["rmse"],
            "test_r2": test_metrics["r2"],
            "test_spearman": test_metrics["spearman"],
            "test_pearson": test_metrics["pearson"],
            **resource,
            "folds": [
                {
                    "fold_id": r["fold_id"],
                    "val_spearman": r["val_metric"],
                    "best_val_spearman": r["best_val_metric"],
                    "steps": r["steps"],
                    "best_step": r["best_step"],
                    "best_epoch": r["best_epoch"],
                    **{k: fn(r) for k, fn in fold_resource_keys.items()},
                }
                for r in fold_results
            ],
        }
        metrics_summary = {
            "val_spearman": val_metric,
            "test_rmse": test_metrics["rmse"],
            "test_r2": test_metrics["r2"],
            "test_spearman": test_metrics["spearman"],
            "test_pearson": test_metrics["pearson"],
        }

    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "resource.json", resource)

    manifest = {
        "run_tag": cfg.run_tag,
        "git_commit": cfg.commit,
        "git_commit_short": cfg.commit[:7],
        "csv_name": cfg.csv_name,
        "task_type": task_type,
        "search_val_metric": cfg.resolved.get("search_val_metric"),
        "search_description": cfg.resolved.get("search_description"),
        "config": {
            k: v
            for k, v in asdict(cfg).items()
            if k != "resolved"
        },
        "resolved": cfg.resolved,
        "metrics_summary": metrics_summary,
        "resource": resource,
        "artifacts": {
            "checkpoints_dir": str(ckpt_dir),
            "metrics_json": str(out_dir / "metrics.json"),
            "resource_json": str(out_dir / "resource.json"),
            "test_predictions_csv": str(out_dir / "test_predictions.csv"),
            "train_snapshot": str(out_dir / "train.py.snapshot"),
        },
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(out_dir / "manifest.json", manifest)

    print("---")
    if task_type == "classification":
        print(f"val_accuracy:     {val_metric:.6f}")
        print(f"test_accuracy:    {test_metrics['accuracy']:.6f}")
        print(f"test_macro_f1:    {test_metrics['macro_f1']:.6f}")
    else:
        print(f"val_spearman:     {val_metric:.6f}")
        print(f"test_RMSE:        {test_metrics['rmse']:.6f}")
        print(f"test_R2:          {test_metrics['r2']:.6f}")
        print(f"test_spearman:    {test_metrics['spearman']:.6f}")
        print(f"test_pearson:     {test_metrics['pearson']:.6f}")
    print(f"training_seconds: {resource['training_seconds']:.1f} (GPU-step)")
    print(f"train_wall_s:     {resource['train_wall_s']:.1f}")
    print(f"test_eval_seconds:{resource['test_eval_seconds']:.1f}")
    print(f"total_seconds:    {resource['total_seconds']:.1f}")
    print(f"peak_train_vram_mb:{resource['peak_train_vram_mb']:.1f}")
    print(f"peak_test_vram_mb: {resource['peak_test_vram_mb']:.1f}")
    print(f"peak_vram_mb:     {resource['peak_vram_mb']:.1f}")
    print(f"artifacts:        {out_dir}")

    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finalize RNAOmni search: re-train best commit and export artifacts.")
    parser.add_argument("--config", type=Path, help="Optional JSON/YAML config file.")
    parser.add_argument("--commit", default=None, help="Git commit hash or 'auto' (default: auto).")
    parser.add_argument("--run-tag", default=None, help="Run tag for workspace/runs/<tag>/finalize (default: from branch).")
    parser.add_argument("--output-dir", default=None, help="Output directory (overrides workspace/runs/<tag>/finalize).")
    parser.add_argument(
        "--csv",
        dest="csv_name",
        default=None,
        help="Dataset CSV under dataset/ (default: compare_* on the external route, else ACTIVE_CSV).",
    )
    parser.add_argument(
        "--mult",
        type=float,
        default=None,
        help="Max finalize GPU-sec / search wall-clock ratio when probing (default 100). Search TIME_BUDGET is a cap, not a GPU floor.",
    )
    parser.add_argument(
        "--fold-budget-sec",
        type=float,
        default=None,
        help="Lock per-fold GPU-step budget in seconds (skips epoch probe). Excludes loader/encode/eval.",
    )
    parser.add_argument(
        "--target-epochs",
        type=int,
        default=None,
        help="Probed horizon in data passes (default 100). Capped by --mult × search wall-clock, not floored by it.",
    )
    parser.add_argument("--early-stop", dest="early_stop", action="store_true", default=None)
    parser.add_argument("--no-early-stop", dest="early_stop", action="store_false")
    parser.add_argument("--patience", type=int, default=None, help="Used only with --early-stop: stop after this many evals without val improvement.")
    parser.add_argument("--min-delta", type=float, default=None, help="Absolute val RMSE improvement threshold.")
    parser.add_argument("--min-delta-ratio", type=float, default=None, help="Relative threshold vs search val RMSE.")
    parser.add_argument(
        "--eval-every",
        type=int,
        default=None,
        help="Val eval frequency in optimizer steps. Default: one epoch, also gated by a GPU-time floor after the probe.",
    )
    parser.add_argument(
        "--checkpoint-mode",
        choices=["best_val", "last"],
        default=None,
        help="Save best val checkpoint per fold (default) or last EMA weights.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: 42).")
    return parser


def main() -> int:
    configure_finalize_runtime()
    parser = build_arg_parser()
    args = parser.parse_args()
    print(
        f"CPU threads: {torch.get_num_threads()} (RNAOMNI_CPU_THREADS / OMP)",
        flush=True,
    )

    file_cfg = load_config_file(args.config) if args.config else None
    cfg, train_mod = resolve_config(args, file_cfg)

    try:
        run_finalize(cfg, train_mod)
    except Exception as err:
        print(f"Finalize failed: {err}", file=sys.stderr)
        raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
