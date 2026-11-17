"""
RNAOmni training script. Single-GPU, single-file.
Usage: python train.py  # conda myenv3.11; uv: uv run train.py

Search baseline (iter 1): PooledMLP on s1_l0_onehot (onehot, input_dim from recipe).

Task type is inferred in prepare.py from predicted_target.
  regression  -> MSE, scalar head, val_spearman (higher better)
  classification -> CrossEntropy, head size = data["num_classes"]
                   (from predicted_target max_id+1; never hardcode),
                   val_accuracy (higher better)
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from rnaomni.threads import apply_thread_limits

apply_thread_limits()  # OMP/MKL/OPENBLAS before torch loads

import gc
import math
import subprocess
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

apply_thread_limits()  # torch.set_num_threads after import

from prepare import (
    MAX_SEQ_LEN,
    TIME_BUDGET,
    N_FOLDS,
    ENCODE_TIMEOUT_EXIT,
    data_mode,
    make_dataloader,
    evaluate_spearman,
    evaluate_accuracy,
    evaluate_test_metrics,
    load_prepared_data,
    search_fold_budget_sec,
)
from rnaomni.results_io import ensure_results_tsv, results_tsv_path

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PooledMLP(nn.Module):
    """Conv1d motif extractor, mean/max-pool, then MLP -> num_outputs."""

    def __init__(self, input_dim, hidden_dim=256, dropout=0.1, conv_channels=256, num_outputs=1):
        super().__init__()
        self.input_dim = input_dim
        self.num_outputs = int(num_outputs)
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, conv_channels, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(conv_channels, input_dim, kernel_size=7, padding=3),
            nn.ReLU(),
        )
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_outputs),
        )

    def forward(self, x, lengths):
        # x: (B, L, C), lengths: (B,)
        B, L, C = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        mask = (pos < lengths.unsqueeze(1)).unsqueeze(-1).to(x.dtype)
        xc = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = x + xc * mask
        denom = lengths.clamp(min=1).unsqueeze(-1).to(x.dtype)
        mean_pooled = (x * mask).sum(dim=1) / denom
        neg_inf = torch.finfo(x.dtype).min
        max_pooled = x.masked_fill(mask == 0, neg_inf).max(dim=1).values
        pooled = torch.cat([mean_pooled, max_pooled], dim=-1)
        out = self.net(pooled)
        if self.num_outputs == 1:
            return out.squeeze(-1)
        return out


def _tf_nhead(width: int) -> int:
    return 4 if int(width) % 4 == 0 else 1


def _tf_encoder(width: int, dropout: float, n_layers: int = 2) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=width,
        nhead=_tf_nhead(width),
        dim_feedforward=2 * width,
        dropout=dropout,
        batch_first=True,
        activation="gelu",
        norm_first=True,
    )
    try:
        return nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
    except TypeError:
        return nn.TransformerEncoder(layer, num_layers=n_layers)


def _length_pad_mask(lengths, seqlen, *, device):
    pos = torch.arange(seqlen, device=device).unsqueeze(0)
    return pos >= lengths.clamp(min=1).unsqueeze(1)


def _stem_to_width(input_dim: int, width: int) -> nn.Module:
    return nn.Identity() if int(input_dim) == int(width) else nn.Linear(int(input_dim), int(width))


def _out_mlp(in_dim: int, width: int, dropout: float, num_outputs: int, *, extra_layers: int = 0):
    layers = [nn.Linear(in_dim, width), nn.ReLU(), nn.Dropout(dropout)]
    for _ in range(extra_layers):
        layers.extend([nn.Linear(width, width), nn.ReLU(), nn.Dropout(dropout)])
    layers.append(nn.Linear(width, num_outputs))
    return nn.Sequential(*layers)


def _masked_mean(hidden, pad, lengths):
    keep = (~pad).unsqueeze(-1).to(hidden.dtype)
    return (hidden * keep).sum(dim=1) / lengths.clamp(min=1).unsqueeze(-1).to(hidden.dtype)


def _squeeze_out(out, num_outputs: int):
    if int(num_outputs) == 1:
        return out.squeeze(-1)
    return out


class TfMean(nn.Module):
    """tf_mean: stem → WIDTH TransformerEncoder (≤2 layers) → mean-pool → MLP."""

    def __init__(self, input_dim, hidden_dim=256, dropout=0.1, num_outputs=1):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_outputs = int(num_outputs)
        width = self.hidden_dim
        self.stem = _stem_to_width(self.input_dim, width)
        self.encoder = _tf_encoder(width, dropout, n_layers=2)
        self.net = _out_mlp(width, width, dropout, self.num_outputs)

    def forward(self, x, lengths):
        hidden = self.stem(x)
        pad = _length_pad_mask(lengths, hidden.size(1), device=hidden.device)
        hidden = self.encoder(hidden, src_key_padding_mask=pad)
        pooled = _masked_mean(hidden, pad, lengths)
        return _squeeze_out(self.net(pooled), self.num_outputs)


class TfCls(nn.Module):
    """tf_cls: stem → CLS + WIDTH TransformerEncoder (≤2 layers) → CLS → MLP."""

    def __init__(self, input_dim, hidden_dim=256, dropout=0.1, num_outputs=1):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_outputs = int(num_outputs)
        width = self.hidden_dim
        self.stem = _stem_to_width(self.input_dim, width)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.encoder = _tf_encoder(width, dropout, n_layers=2)
        self.net = _out_mlp(width, width, dropout, self.num_outputs)

    def forward(self, x, lengths):
        hidden = self.stem(x)
        bsz = hidden.size(0)
        cls = self.cls_token.expand(bsz, -1, -1)
        hidden = torch.cat([cls, hidden], dim=1)
        pad = _length_pad_mask(lengths, hidden.size(1) - 1, device=hidden.device)
        cls_pad = torch.zeros(bsz, 1, dtype=torch.bool, device=hidden.device)
        hidden = self.encoder(hidden, src_key_padding_mask=torch.cat([cls_pad, pad], dim=1))
        return _squeeze_out(self.net(hidden[:, 0]), self.num_outputs)


class TfAttn(nn.Module):
    """tf_attn: stem → WIDTH encoder → attn-pool → MLP."""

    def __init__(self, input_dim, hidden_dim=256, dropout=0.1, num_outputs=1):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_outputs = int(num_outputs)
        width = self.hidden_dim
        self.stem = _stem_to_width(self.input_dim, width)
        self.encoder = _tf_encoder(width, dropout, n_layers=2)
        self.attn_proj = nn.Linear(width, 1)
        self.net = _out_mlp(width, width, dropout, self.num_outputs)

    def forward(self, x, lengths):
        hidden = self.stem(x)
        pad = _length_pad_mask(lengths, hidden.size(1), device=hidden.device)
        hidden = self.encoder(hidden, src_key_padding_mask=pad)
        logits = self.attn_proj(hidden).squeeze(-1)
        logits = logits.masked_fill(pad, torch.finfo(logits.dtype).min)
        attn = torch.softmax(logits, dim=-1)
        pooled = (hidden * attn.unsqueeze(-1)).sum(dim=1)
        return _squeeze_out(self.net(pooled), self.num_outputs)


class TfConv(nn.Module):
    """tf_conv: stem → k=7 conv residual → 1-layer encoder → mean-pool → MLP."""

    def __init__(self, input_dim, hidden_dim=256, dropout=0.1, num_outputs=1):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_outputs = int(num_outputs)
        width = self.hidden_dim
        self.stem = _stem_to_width(self.input_dim, width)
        self.conv = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=7, padding=3),
            nn.ReLU(),
        )
        self.encoder = _tf_encoder(width, dropout, n_layers=1)
        self.net = _out_mlp(width, width, dropout, self.num_outputs)

    def forward(self, x, lengths):
        hidden = self.stem(x)
        pad = _length_pad_mask(lengths, hidden.size(1), device=hidden.device)
        keep = (~pad).unsqueeze(-1).to(hidden.dtype)
        hidden = hidden + self.conv(hidden.transpose(1, 2)).transpose(1, 2) * keep
        hidden = self.encoder(hidden, src_key_padding_mask=pad)
        pooled = _masked_mean(hidden, pad, lengths)
        return _squeeze_out(self.net(pooled), self.num_outputs)


class MaxPoolMLP(nn.Module):
    """max_pool_mlp: stem → max-pool → MLP (no conv)."""

    def __init__(self, input_dim, hidden_dim=256, dropout=0.1, num_outputs=1):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_outputs = int(num_outputs)
        width = self.hidden_dim
        self.stem = _stem_to_width(self.input_dim, width)
        self.net = _out_mlp(width, width, dropout, self.num_outputs)

    def forward(self, x, lengths):
        hidden = self.stem(x)
        pad = _length_pad_mask(lengths, hidden.size(1), device=hidden.device)
        neg_inf = torch.finfo(hidden.dtype).min
        pooled = hidden.masked_fill(pad.unsqueeze(-1), neg_inf).max(dim=1).values
        return _squeeze_out(self.net(pooled), self.num_outputs)


class GatedPoolMLP(nn.Module):
    """gated_pool_mlp: stem → per-position gate → softmax weighted sum → MLP."""

    def __init__(self, input_dim, hidden_dim=256, dropout=0.1, num_outputs=1):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_outputs = int(num_outputs)
        width = self.hidden_dim
        self.stem = _stem_to_width(self.input_dim, width)
        self.gate = nn.Linear(width, 1)
        self.net = _out_mlp(width, width, dropout, self.num_outputs)

    def forward(self, x, lengths):
        hidden = self.stem(x)
        pad = _length_pad_mask(lengths, hidden.size(1), device=hidden.device)
        logits = self.gate(hidden).squeeze(-1)
        logits = logits.masked_fill(pad, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        pooled = (hidden * weights.unsqueeze(-1)).sum(dim=1)
        return _squeeze_out(self.net(pooled), self.num_outputs)


class GruPool(nn.Module):
    """gru_pool: stem → 1-layer bidirectional GRU → mean-pool → MLP."""

    def __init__(self, input_dim, hidden_dim=256, dropout=0.1, num_outputs=1):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_outputs = int(num_outputs)
        width = self.hidden_dim
        self.stem = _stem_to_width(self.input_dim, width)
        self.gru = nn.GRU(
            input_size=width,
            hidden_size=width,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        self.net = _out_mlp(2 * width, width, dropout, self.num_outputs)

    def forward(self, x, lengths):
        hidden = self.stem(x)
        lengths_cpu = lengths.detach().cpu().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            hidden, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=hidden.size(1)
        )
        pad = _length_pad_mask(lengths, out.size(1), device=out.device)
        pooled = _masked_mean(out, pad, lengths)
        return _squeeze_out(self.net(pooled), self.num_outputs)


class MeanDeepMLP(nn.Module):
    """mean_deep_mlp: stem → mean-pool → deeper MLP (no conv / no encoder)."""

    def __init__(self, input_dim, hidden_dim=256, dropout=0.1, num_outputs=1):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_outputs = int(num_outputs)
        width = self.hidden_dim
        self.stem = _stem_to_width(self.input_dim, width)
        self.net = _out_mlp(width, width, dropout, self.num_outputs, extra_layers=2)

    def forward(self, x, lengths):
        hidden = self.stem(x)
        pad = _length_pad_mask(lengths, hidden.size(1), device=hidden.device)
        pooled = _masked_mean(hidden, pad, lengths)
        return _squeeze_out(self.net(pooled), self.num_outputs)


# Overnight / finalize instantiate this name. Swap the alias to change the head;
# keep PooledMLP so older snapshots still finalize.
# Opening campaign tags this as framework=baseline (not a menu name).
# Named switch: set Predictor to the class for next_framework (classes above).
Predictor = PooledMLP

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# In-framework knobs while staying on the named head: HIDDEN_DIM / LR /
# BATCH_SIZE / DROPOUT / LABEL_SMOOTHING / EMA_DECAY. WIDTH for tf_* should
# stay divisible by 4 (nhead=4). Switch Predictor only when next_framework
# is set (named classes already defined above).
# ---------------------------------------------------------------------------

HIDDEN_DIM = 384  # smaller head → more steps in fixed TIME_BUDGET on 77-dim s3 recipe
DROPOUT = 0.1
BATCH_SIZE = 256
LR = 2e-3
WEIGHT_DECAY = 1e-4
EMA_DECAY = 0.999
WARMUP_FRACTION = 0.1
LABEL_SMOOTHING = 0.05  # soften 88-way family targets to reduce overconfident logits


class EMA:
    """Exponential moving average of model weights for more stable eval."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)

    def state_dict(self):
        return {k: v.detach().cpu() for k, v in self.shadow.items()}

# ---------------------------------------------------------------------------
# Results logging
# ---------------------------------------------------------------------------

def _git_info():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short=7", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        description = subprocess.check_output(
            ["git", "log", "-1", "--format=%s"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return commit, description.replace("\t", " ")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "???????", "unknown"


def _best_decision_value(task_type: str):
    """Best prior decision metric for the current data_mode, or None."""
    path = results_tsv_path()
    if not path.is_file():
        return None
    mode = data_mode()
    best = None
    with path.open(encoding="utf-8") as f:
        header = next(f, "")
        cols = header.rstrip("\n").split("\t")
        mode_idx = cols.index("data_mode") if "data_mode" in cols else None
        status_idx = cols.index("status") if "status" in cols else None
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            if mode_idx is not None:
                if mode_idx >= len(parts) or (parts[mode_idx] or "full") != mode:
                    continue
            if status_idx is not None and status_idx < len(parts):
                if (parts[status_idx] or "").strip() == "crash":
                    continue
            try:
                value = float(parts[1])
            except ValueError:
                continue
            if not math.isfinite(value):
                continue
            if best is None or value > best:
                best = value
    return best


def _format_metric(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{value:.6f}"


def _append_result(task_type, decision_value, test_metrics, memory_gb, status, description=None):
    commit, git_desc = _git_info()
    if description is None:
        description = git_desc
    path = ensure_results_tsv(task_type)
    with path.open("a", encoding="utf-8") as f:
        if task_type == "classification":
            f.write(
                f"{commit}\t{decision_value:.6f}\t"
                f"{_format_metric(test_metrics.get('accuracy'))}\t"
                f"{_format_metric(test_metrics.get('macro_f1'))}\t"
                f"{memory_gb:.1f}\t{data_mode()}\t{status}\t{description}\n"
            )
        else:
            f.write(
                f"{commit}\t{decision_value:.6f}\t"
                f"{_format_metric(test_metrics.get('rmse'))}\t"
                f"{_format_metric(test_metrics.get('r2'))}\t"
                f"{_format_metric(test_metrics.get('spearman'))}\t"
                f"{_format_metric(test_metrics.get('pearson'))}\t"
                f"{memory_gb:.1f}\t{data_mode()}\t{status}\t{description}\n"
            )


def log_run_result(task_type, decision_value, test_metrics, peak_vram_mb):
    memory_gb = peak_vram_mb / 1024
    best = _best_decision_value(task_type)
    status = "keep" if best is None or decision_value > best else "discard"
    _append_result(task_type, decision_value, test_metrics, memory_gb, status)


def log_crash_result(task_type="regression"):
    if task_type == "classification":
        crash_metrics = {"accuracy": 0.0, "macro_f1": 0.0}
    else:
        crash_metrics = {"rmse": 0.0, "r2": 0.0, "spearman": 0.0, "pearson": 0.0}
    _append_result(task_type, 0.0, crash_metrics, 0.0, "crash")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    device = torch.device("cuda")

    try:
        data = load_prepared_data()
    except SystemExit:
        raise
    except Exception:
        print("ENCODE_FAIL", flush=True)
        raise SystemExit(ENCODE_TIMEOUT_EXIT)

    input_dim = int(data["input_dim"])
    task_type = data["task_type"]
    n_cv_folds = int(data.get("n_cv_folds", N_FOLDS))
    if task_type == "classification":
        if data.get("num_classes") is None:
            raise RuntimeError("classification requires data['num_classes'] from predicted_target")
        num_outputs = int(data["num_classes"])
    else:
        num_outputs = 1

    print(
        f"CPU threads: {torch.get_num_threads()} "
        f"(RNAOMNI_CPU_THREADS / OMP)",
        flush=True,
    )
    print(
        f"Task: {task_type}, input_dim: {input_dim}, max_seq_len: {MAX_SEQ_LEN}, "
        f"cv_folds: {n_cv_folds}, num_outputs: {num_outputs}"
        + (f" (from predicted_target)" if task_type == "classification" else "")
        + f", recipe_id: {data.get('feature_recipe_id')}, features: {data.get('enabled_features')}"
    )
    use_compare = bool(data.get("use_compare_splits"))
    fold_budget = search_fold_budget_sec(n_cv_folds, use_compare_splits=use_compare)
    if use_compare:
        print(
            f"Time budget: {fold_budget:.0f}s per fold × {n_cv_folds} folds "
            f"({fold_budget * n_cv_folds:.0f}s train)"
        )
    else:
        print(f"Time budget: {TIME_BUDGET}s ({fold_budget:.0f}s per fold)")
    fold_metrics = []
    fold_state_dicts = []
    total_training_time = 0
    total_steps = 0

    for fold_id in range(n_cv_folds):
        print(f"\n=== Fold {fold_id}/{n_cv_folds - 1} ===")
        model = Predictor(
            input_dim=input_dim,
            hidden_dim=HIDDEN_DIM,
            dropout=DROPOUT,
            num_outputs=num_outputs,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        ema = EMA(model, EMA_DECAY)

        train_loader = make_dataloader(fold_id, "train", BATCH_SIZE)
        x, y, lengths, epoch = next(train_loader)

        fold_start = time.time()
        step = 0
        smooth_loss = 0.0

        while True:
            torch.cuda.synchronize()
            t0 = time.time()

            elapsed = time.time() - fold_start
            warmup_duration = fold_budget * WARMUP_FRACTION
            if warmup_duration > 0 and elapsed < warmup_duration:
                lr_scale = elapsed / warmup_duration
            elif fold_budget > warmup_duration:
                progress = (elapsed - warmup_duration) / (fold_budget - warmup_duration)
                lr_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
            else:
                lr_scale = 1.0
            for param_group in optimizer.param_groups:
                param_group["lr"] = LR * lr_scale

            pred = model(x, lengths)
            if task_type == "classification":
                loss = F.cross_entropy(pred, y.long(), label_smoothing=LABEL_SMOOTHING)
            else:
                loss = F.mse_loss(pred, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            ema.update(model)

            loss_f = loss.item()
            if math.isnan(loss_f) or loss_f > 1e4:
                print("FAIL")
                raise SystemExit(1)

            x, y, lengths, epoch = next(train_loader)

            torch.cuda.synchronize()
            dt = time.time() - t0
            if step > 10:
                total_training_time += dt

            ema_beta = 0.9
            smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * loss_f
            debiased = smooth_loss / (1 - ema_beta ** (step + 1))
            remaining = max(0.0, fold_budget - elapsed)
            print(
                f"\rfold {fold_id} step {step:05d} | loss: {debiased:.6f} | "
                f"epoch: {epoch} | fold_remaining: {remaining:.0f}s    ",
                end="",
                flush=True,
            )
            step += 1
            total_steps += 1

            if step > 10 and elapsed >= fold_budget:
                break

        print()
        model.load_state_dict(ema.shadow)
        if task_type == "classification":
            fold_metric = evaluate_accuracy(model, BATCH_SIZE, fold_id=fold_id)
            print(f"fold {fold_id} val accuracy: {fold_metric:.6f}")
        else:
            fold_metric = evaluate_spearman(model, BATCH_SIZE, fold_id=fold_id)
            print(f"fold {fold_id} val Spearman: {fold_metric:.6f}")

        fold_metrics.append(fold_metric)
        fold_state_dicts.append(ema.state_dict())
        del model, optimizer, train_loader
        gc.collect()
        torch.cuda.empty_cache()

    decision_value = sum(fold_metrics) / len(fold_metrics)

    if use_compare and n_cv_folds > 1:
        print("\n=== Test eval (paired per-fold, not counted in training time) ===")
    else:
        print("\n=== Test ensemble eval (not counted in training time) ===")
    t_test_start = time.time()
    test_models = []
    for state_dict in fold_state_dicts:
        test_model = Predictor(
            input_dim=input_dim,
            hidden_dim=HIDDEN_DIM,
            dropout=DROPOUT,
            num_outputs=num_outputs,
        ).to(device)
        test_model.load_state_dict(state_dict)
        test_models.append(test_model)
    test_metrics = evaluate_test_metrics(test_models, BATCH_SIZE)
    del test_models, fold_state_dicts
    gc.collect()
    torch.cuda.empty_cache()
    test_eval_seconds = time.time() - t_test_start

    num_params = sum(
        p.numel()
        for p in Predictor(
            input_dim=input_dim, hidden_dim=HIDDEN_DIM, num_outputs=num_outputs
        ).parameters()
    )
    t_end = time.time()
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    print("---")
    if task_type == "classification":
        print(f"val_accuracy:     {decision_value:.6f}")
        print(f"test_accuracy:    {test_metrics['accuracy']:.6f}")
        print(f"test_macro_f1:    {test_metrics['macro_f1']:.6f}")
    else:
        print(f"val_spearman:     {decision_value:.6f}")
        print(f"test_RMSE:        {test_metrics['rmse']:.6f}")
        print(f"test_R2:          {test_metrics['r2']:.6f}")
        print(f"test_spearman:    {test_metrics['spearman']:.6f}")
        print(f"test_pearson:     {test_metrics['pearson']:.6f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"test_eval_seconds:{test_eval_seconds:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_steps:        {total_steps}")
    print(f"num_params_M:     {num_params / 1e6:.3f}")
    print(f"hidden_dim:       {HIDDEN_DIM}")

    log_run_result(task_type, decision_value, test_metrics, peak_vram_mb)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        if exc.code == ENCODE_TIMEOUT_EXIT:
            raise
        if exc.code not in (None, 0):
            task = "regression"
            try:
                task = load_prepared_data().get("task_type", "regression")
            except Exception:
                pass
            log_crash_result(task)
        raise
    except Exception:
        task = "regression"
        try:
            task = load_prepared_data().get("task_type", "regression")
        except Exception:
            pass
        log_crash_result(task)
        raise
