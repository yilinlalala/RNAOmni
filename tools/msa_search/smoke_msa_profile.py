#!/usr/bin/env python3
"""Smoke: dataset-scoped hits → profile → msa_profile.build."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from features.builders import msa_profile
from features.pipeline import load_active_recipe, load_feature_registry, resolve_input_dim
from tools.msa_search.assets import AssetRoot, load_hit_json, unique_sequences_from_csv
from tools.msa_search.contract import (
    DIM,
    dataset_id_from_csv,
    resolve_dataset_asset_dir,
    seq_hash,
)
from tools.msa_search.extract_profile import profile_from_hit_record
from tools.msa_search.run_precompute import _run_profiles


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=Path("HEK_te_prediction_task.csv"))
    p.add_argument("--limit", type=int, default=1)
    p.add_argument("--force-profile", action="store_true")
    args = p.parse_args()

    csv_path = args.csv
    if not csv_path.is_file():
        csv_path = ROOT / "dataset" / csv_path.name
    dataset_id = dataset_id_from_csv(csv_path)
    out = resolve_dataset_asset_dir(repo_root=ROOT, dataset_id=dataset_id)
    root = AssetRoot(out)
    root.ensure_layout()
    seqs = unique_sequences_from_csv(csv_path, limit=args.limit)
    if not seqs:
        print("no seqs")
        return 1

    from types import SimpleNamespace

    ns = SimpleNamespace(
        force=args.force_profile,
        top_k=200,
        dataset_id=dataset_id,
    )
    fails = _run_profiles(ns, root, seqs)
    if fails:
        print("profile stage failed; need hits for this dataset:", out)
        return 1

    h = seq_hash(seqs[0])
    rec = load_hit_json(root, h)
    prof, meta = profile_from_hit_record(rec, top_k=200)
    assert prof.shape[1] == DIM

    arr = msa_profile.build(
        seqs,
        max_len=128,
        params={"dataset_id": dataset_id, "asset_root": "artifacts/msa"},
    )
    assert arr.shape == (len(seqs), 128, DIM)
    L = min(len(seqs[0]), 128)
    assert np.allclose(arr[0, :L], prof[:L])

    # Also resolve via ACTIVE_CSV path
    arr2 = msa_profile.build(seqs, max_len=128, params={})
    assert np.allclose(arr2[0, :L], arr[0, :L])

    reg = load_feature_registry()
    assert "msa_profile" in {e["name"] for e in reg["features"]}
    _ = resolve_input_dim(load_active_recipe(), reg)
    print(
        "smoke ok",
        f"dataset_id={dataset_id}",
        f"out={out}",
        f"L={prof.shape[0]} hits_used={meta['n_hits_used']} n_eff={meta['n_eff']:.2f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
