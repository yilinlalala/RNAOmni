#!/usr/bin/env python3
"""Smoke 15.2: finalize runtime (B) + GPU-timer contract (C) + online gather (A).

Does not run train_one_fold (needs CUDA). Isolation: train.py must not set
RNAOMNI_ENCODE_MODE.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# finalize.apply_thread_limits must run before a second torch import path.
from rnaomni.finalize import configure_finalize_runtime

import numpy as np
import torch

from features.pipeline import unique_sequence_index
from rnaomni.prepare import data as prep_data
from rnaomni.prepare.data import gather_row_features
from rnaomni.prepare.route import encode_mode, encode_timeout_sec


def _onehot_recipe() -> dict:
    return {
        "recipe_id": "smoke_finalize_online",
        "fusion": "concat",
        "features": [{"name": "onehot", "enabled": True}],
    }


def test_default_is_precompute() -> None:
    os.environ.pop("RNAOMNI_ENCODE_MODE", None)
    assert encode_mode() == "precompute", encode_mode()
    print("default encode_mode=precompute OK")


def test_configure_finalize_runtime() -> None:
    os.environ["ENCODE_TIMEOUT_SEC"] = "2160"
    os.environ.pop("RNAOMNI_ENCODE_MODE", None)
    configure_finalize_runtime()
    assert os.environ["ENCODE_TIMEOUT_SEC"] == "0"
    assert os.environ["RNAOMNI_ENCODE_MODE"] == "online"
    assert encode_timeout_sec() == 0
    assert encode_mode() == "online"
    print("configure_finalize_runtime B+A env OK")


def test_train_py_does_not_set_online() -> None:
    tree = ast.parse((REPO / "train.py").read_text(encoding="utf-8"))
    assigned = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute):
            pass
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Attribute)
                    and target.value.attr == "environ"
                    and isinstance(target.slice, ast.Constant)
                ):
                    assigned.append(str(target.slice.value))
    assert "RNAOMNI_ENCODE_MODE" not in assigned, assigned
    src = (REPO / "train.py").read_text(encoding="utf-8")
    assert "RNAOMNI_ENCODE_MODE" not in src
    assert "configure_finalize_runtime" not in src
    print("train.py isolation OK")


def test_gpu_timer_excludes_loader() -> None:
    src = (REPO / "rnaomni/finalize.py").read_text(encoding="utf-8")
    assert "gpu_train_s" in src
    assert "gpu_train_s += time.time() - t0" in src
    assert "if step > 10 and gpu_train_s >= fold_budget" in src
    assert "def resolve_probed_fold_budget" in src
    assert "def clamp_fold_budget" in src
    assert 'budget_mode = "probe"' in src
    assert "warmup_duration = fold_budget * warmup_fraction" in src
    assert "warmup_duration = search_fold_budget * warmup_fraction" not in src
    assert 'cfg.resolved["eval_mode"] = "epoch"' in src
    assert "cfg.eval_every = max(int(cfg.resolved[\"steps_per_epoch_fold0\"]), 1)" in src
    assert "budget_min = float(search_fold_budget)" not in src
    assert "budget_min = 0.0" in src
    assert "def _resolved_float" in src
    assert '"early_stop": False' in src
    assert "if cfg.early_stop and evals_without_improve >= cfg.patience:" in src
    assert "def _finalize_lr_scale" in src
    assert "def _train_step_loss" in src
    assert "peak_train_vram_mb" in src
    assert "peak_test_vram_mb" in src
    assert "reset_peak_memory_stats" in src
    assert 'write_json(out_dir / "resource.json"' in src
    train_fn = src.split("def train_one_fold", 1)[1].split("\n@torch.no_grad()", 1)[0]
    sync_then_next = "x, y, lengths, epoch = next(train_loader)" in train_fn
    assert sync_then_next
    gpu_add_at = train_fn.find("gpu_train_s += time.time() - t0")
    next_at = train_fn.find("x, y, lengths, epoch = next(train_loader)")
    # First next() is the prefetch before the loop; the in-loop next must be after GPU add.
    in_loop_next = train_fn.rfind("x, y, lengths, epoch = next(train_loader)")
    assert gpu_add_at < in_loop_next, "in-loop next(loader) must sit outside GPU timing"
    assert "elapsed = time.time() - fold_start" not in train_fn
    print("GPU-only timer contract C OK")


def test_resource_payload() -> None:
    from rnaomni.finalize import resource_payload

    payload = resource_payload(
        training_seconds=715.1,
        train_wall_s=3659.4,
        test_eval_seconds=918.7,
        total_seconds=4584.2,
        peak_train_vram_mb=172.7,
        peak_test_vram_mb=80.0,
    )
    assert payload["gpu_train_s"] == 715.1
    assert payload["peak_vram_mb"] == 172.7
    assert payload["peak_train_vram_mb"] == 172.7
    assert payload["peak_test_vram_mb"] == 80.0
    print("resource_payload OK")


def test_probe_budget_clamp() -> None:
    from rnaomni.finalize import resolve_probed_fold_budget

    # Search wall-clock is not a GPU floor: a 4s probe stays 4s.
    muscle = resolve_probed_fold_budget(
        gpu_s=0.1,
        n_steps=20,
        steps_per_epoch=8,
        target_epochs=100,
        budget_min=0.0,
        budget_max=6000.0,
    )
    assert abs(muscle["desired_gpu_s"] - 4.0) < 1e-9, muscle
    assert abs(muscle["fold_budget_sec"] - 4.0) < 1e-9, muscle

    mid = resolve_probed_fold_budget(
        gpu_s=0.1,
        n_steps=20,
        steps_per_epoch=1500,
        target_epochs=100,
        budget_min=0.0,
        budget_max=6000.0,
    )
    assert abs(mid["fold_budget_sec"] - 750.0) < 1e-9, mid

    huge = resolve_probed_fold_budget(
        gpu_s=0.1,
        n_steps=20,
        steps_per_epoch=20000,
        target_epochs=100,
        budget_min=0.0,
        budget_max=6000.0,
    )
    assert huge["fold_budget_sec"] == 6000.0

    # An explicit GPU min still clamps up if a caller passes one.
    floored = resolve_probed_fold_budget(
        gpu_s=0.1,
        n_steps=20,
        steps_per_epoch=8,
        target_epochs=100,
        budget_min=60.0,
        budget_max=6000.0,
    )
    assert floored["fold_budget_sec"] == 60.0
    print("probe budget clamp OK")


def test_online_gather_pad_and_cache() -> None:
    prep_data._ONLINE_UNIQUE_CACHE = {}
    seqs = ["AC", "ACGUACGU", "AC"]
    unique_seqs, row_to_uid = unique_sequence_index(seqs)
    assert unique_seqs == ["AC", "ACGUACGU"]
    data = {
        "x": None,
        "encode_mode": "online",
        "unique_seqs": unique_seqs,
        "row_to_uid": torch.from_numpy(np.asarray(row_to_uid, dtype=np.int64)),
        "input_dim": 4,
        "recipe": _onehot_recipe(),
        "registry": None,
    }
    calls = {"n": 0}
    orig = prep_data.encode_sequence_batch

    def _counting(sequences, **kwargs):
        calls["n"] += 1
        return orig(sequences, **kwargs)

    prep_data.encode_sequence_batch = _counting
    try:
        x, lengths = gather_row_features(data, [0, 1, 2])
        assert x.shape == (3, 8, 4), x.shape
        np.testing.assert_array_equal(lengths.numpy(), [2, 8, 2])
        np.testing.assert_allclose(x[0, :2], x[2, :2])
        assert torch.count_nonzero(x[0, 2:]) == 0
        assert calls["n"] == 1
        x2, lengths2 = gather_row_features(data, [0, 2])
        assert x2.shape == (2, 2, 4), x2.shape
        np.testing.assert_array_equal(lengths2.numpy(), [2, 2])
        assert calls["n"] == 1
    finally:
        prep_data.encode_sequence_batch = orig
        prep_data._ONLINE_UNIQUE_CACHE = {}
    print("online gather pad + uid cache OK")


def main() -> int:
    test_default_is_precompute()
    test_configure_finalize_runtime()
    test_train_py_does_not_set_online()
    test_gpu_timer_excludes_loader()
    test_resource_payload()
    test_probe_budget_clamp()
    test_online_gather_pad_and_cache()
    print("smoke_finalize_online: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
