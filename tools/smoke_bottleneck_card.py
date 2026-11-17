#!/usr/bin/env python3
"""C0: cause= parse/infer, bottleneck card, head suppress (no train.py).

Usage:
  python tools/smoke_bottleneck_card.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

os.environ["RNAOMNI_COMPARE_ROOT"] = str(
    REPO / "dataset" / "forcompare" / "Class_88fam_200%noise"
)


def _state(**kwargs):
    import orchestrator as orch

    defaults = dict(
        decision_metric="val_accuracy",
        higher_is_better=True,
        min_delta=0.001,
        patience=15,
        min_iterations=10,
        max_iterations=100,
        max_agent_failures=3,
        eff_window=5,
        sample_expand_patience=3,
        framework_patience=10,
    )
    defaults.update(kwargs)
    return orch.SearchState(**defaults)


def _cls_row(commit, val, status, desc, mode="full") -> str:
    return (
        f"{commit}\t{val:.6f}\t{val:.6f}\t{val:.6f}\t0.1\t{mode}\t{status}\t{desc}\n"
    )


def _patch_tsv(orch, path):
    import rnaomni.orch.tsv as tsv

    orig = tsv.results_tsv_path
    tsv.results_tsv_path = lambda: path  # type: ignore[assignment]
    orch.results_tsv_path = lambda: path  # type: ignore[assignment]
    return orig


def _unpatch_tsv(orch, orig):
    import rnaomni.orch.tsv as tsv

    tsv.results_tsv_path = orig  # type: ignore[assignment]
    orch.results_tsv_path = orig  # type: ignore[assignment]


def main() -> int:
    import orchestrator as orch
    from rnaomni.results_io import RESULTS_HEADER_CLASSIFICATION

    parsed = orch.parse_round_kind(
        "[model] framework=attn_pool_mlp cause=knob hidden_dim"
    )
    assert parsed["cause"] == "knob", parsed
    parsed_missing = orch.parse_round_kind("[model] framework=baseline lr tweak")
    assert parsed_missing["cause"] is None
    assert orch.infer_cause(parsed_missing, "keep") == "knob"
    parsed_fb = orch.parse_round_kind(
        "[model] framework=attn_pool_mlp framework_base cause=head first"
    )
    assert parsed_fb["cause"] == "head" and parsed_fb["framework_base"] is True
    assert orch.infer_cause(parsed_fb, "keep") == "head"
    parsed_feat = orch.parse_round_kind("[feature] kmer k=4")
    assert parsed_feat["cause"] is None
    assert orch.infer_cause(parsed_feat, "keep") == "feature"
    parsed_sample = orch.parse_round_kind(
        "[sample] 数据采样到 ×2（K_train 10->20，K_eval=10，kept=yes）"
    )
    assert orch.infer_cause(parsed_sample, "keep") == "sample"
    crash_p = orch.parse_round_kind("[model] framework=baseline boom")
    assert orch.infer_cause(crash_p, "crash") == "crash"
    print("parse/infer cause OK")

    fake_head = orch.parse_round_kind(
        "[model] framework=baseline cause=head I pick attn_pool_mlp"
    )
    assert orch.infer_cause(fake_head, "discard") == "knob"
    print("cause=head without framework_base infers knob OK")

    with tempfile.TemporaryDirectory() as tmp:
        fake = Path(tmp) / "fake_head.tsv"
        lines = [RESULTS_HEADER_CLASSIFICATION]
        lines.append(
            _cls_row("a000001", 0.90, "keep", "[model] framework=baseline")
        )
        for i in range(10):
            lines.append(
                _cls_row(
                    f"d{i:07d}",
                    0.90,
                    "discard",
                    f"[model] framework=baseline cause=head fake{i}",
                )
            )
        fake.write_text("".join(lines), encoding="utf-8")
        orig = _patch_tsv(orch, fake)
        try:
            st = _state()
            st.sync_from_tsv()
            assert st.last_do == "knob", st.last_do
            assert st.fail_knob == 10, st.fail_knob
            assert st.fail_head == 0
            assert "head" not in st.suppressed
            nxt = st.ensure_next_framework("campaign-aaa")
            assert nxt in orch.FRAMEWORK_MENU, nxt
            print("no next_framework + cause=head stays knob, still draws OK:", nxt)
        finally:
            _unpatch_tsv(orch, orig)

    with tempfile.TemporaryDirectory() as tmp:
        fake = Path(tmp) / "flat.tsv"
        fake.write_text(
            RESULTS_HEADER_CLASSIFICATION
            + _cls_row("a000001", 0.90, "keep", "[model] framework=baseline")
            + _cls_row(
                "b000001",
                0.50,
                "keep",
                "[model] framework=attn_pool_mlp framework_base cause=head first",
            ),
            encoding="utf-8",
        )
        orig = _patch_tsv(orch, fake)
        try:
            st = _state()
            st.sync_from_tsv()
            assert st.last_do == "head", st.last_do
            assert st.last_result == "flat", st.last_result
            assert st.fail_head == 1
            assert "head" in st.suppressed
            assert st.active_framework == "baseline", st.active_framework
            assert "attn_pool_mlp" in st.frameworks_tried
            assert abs(st.best - 0.90) < 1e-9
            print("framework_base below prior best → flat + suppress head, keep working tree OK")
        finally:
            _unpatch_tsv(orch, orig)

    with tempfile.TemporaryDirectory() as tmp:
        fake = Path(tmp) / "lift.tsv"
        fake.write_text(
            RESULTS_HEADER_CLASSIFICATION
            + _cls_row("a000001", 0.90, "keep", "[model] framework=baseline")
            + _cls_row(
                "b000001",
                0.92,
                "keep",
                "[model] framework=attn_pool_mlp framework_base cause=head first",
            ),
            encoding="utf-8",
        )
        orig = _patch_tsv(orch, fake)
        try:
            st = _state()
            st.sync_from_tsv()
            assert st.last_do == "head"
            assert st.last_result == "keep"
            assert st.fail_head == 0
            assert "head" not in st.suppressed
            assert abs(st.best - 0.92) < 1e-9
            print("framework_base lifts global ≥ min_delta → no suppress OK")
        finally:
            _unpatch_tsv(orch, orig)

    with tempfile.TemporaryDirectory() as tmp:
        fake = Path(tmp) / "suppress_plateau.tsv"
        lines = [RESULTS_HEADER_CLASSIFICATION]
        lines.append(
            _cls_row("a000001", 0.90, "keep", "[model] framework=baseline")
        )
        for i in range(10):
            lines.append(
                _cls_row(
                    f"d{i:07d}",
                    0.90,
                    "discard",
                    f"[model] framework=baseline d{i}",
                )
            )
        lines.append(
            _cls_row(
                "b000001",
                0.50,
                "keep",
                "[model] framework=attn_pool_mlp framework_base first shot",
            )
        )
        fake.write_text("".join(lines), encoding="utf-8")
        orig = _patch_tsv(orch, fake)
        try:
            st = _state()
            st.sync_from_tsv()
            assert "head" in st.suppressed, st.suppressed
            assert st.active_framework == "baseline", st.active_framework
            assert st.framework_plateaued(), st.framework_no_improve_streak
            nxt = st.ensure_next_framework("campaign-aaa")
            assert nxt is None
            assert st.next_framework is None
            block = orch.format_cause_block(st)
            assert "next_do=feature" in block, block
            assert "suppressed=(head)" in block
            print("suppress + fw_streak>=10 → no draw, next_do=feature OK")
        finally:
            _unpatch_tsv(orch, orig)

    with tempfile.TemporaryDirectory() as tmp:
        fake = Path(tmp) / "feature_keep.tsv"
        lines = [RESULTS_HEADER_CLASSIFICATION]
        lines.append(
            _cls_row("a000001", 0.90, "keep", "[model] framework=baseline")
        )
        lines.append(
            _cls_row(
                "b000001",
                0.50,
                "keep",
                "[model] framework=attn_pool_mlp framework_base first",
            )
        )
        for i in range(3):
            lines.append(
                _cls_row(
                    f"d{i:07d}",
                    0.50,
                    "discard",
                    f"[model] framework=attn_pool_mlp d{i}",
                )
            )
        lines.append(_cls_row("c000001", 0.91, "keep", "[feature] kmer k=4"))
        fake.write_text("".join(lines), encoding="utf-8")
        orig = _patch_tsv(orch, fake)
        try:
            st = _state()
            st.sync_from_tsv()
            assert st.suppressed == [], st.suppressed
            assert st.fail_head == 0 and st.fail_knob == 0 and st.fail_feature == 0
            assert st.active_framework == "baseline", st.active_framework
            assert st.frameworks_tried == ["baseline"], st.frameworks_tried
            assert "attn_pool_mlp" in st.remaining_frameworks()
            print("feature keep clears suppress + fail, tried=[active] OK")
        finally:
            _unpatch_tsv(orch, orig)

    st_card = _state()
    block0 = orch.format_cause_block(st_card)
    assert len(block0) < 400, len(block0)
    assert "run.log" not in block0
    assert block0.startswith("Bottleneck:")
    assert "fail[head]=0" in block0
    assert "next_do=" in block0
    print("card string <400, no run.log OK:", repr(block0))

    with tempfile.TemporaryDirectory() as tmp:
        fake = Path(tmp) / "old_tsv.tsv"
        fake.write_text(
            RESULTS_HEADER_CLASSIFICATION
            + _cls_row("a000001", 0.90, "keep", "[model] framework=baseline")
            + _cls_row(
                "b000001",
                0.50,
                "keep",
                "[model] framework=attn_pool_mlp framework_base first shot",
            ),
            encoding="utf-8",
        )
        orig = _patch_tsv(orch, fake)
        try:
            st = _state()
            st.sync_from_tsv()
            assert st.last_do == "head" and st.last_result == "flat"
            assert "head" in st.suppressed
            st.framework_no_improve_streak = 10
            assert st.ensure_next_framework("campaign-aaa") is None
            print("old TSV without cause= infers + suppresses like tagged OK")
        finally:
            _unpatch_tsv(orch, orig)

    with tempfile.TemporaryDirectory() as tmp:
        fake = Path(tmp) / "crash.tsv"
        fake.write_text(
            RESULTS_HEADER_CLASSIFICATION
            + _cls_row("a000001", 0.90, "keep", "[model] framework=baseline")
            + _cls_row(
                "c000001",
                0.0,
                "crash",
                "[model] framework=baseline boom || crash: RuntimeError: x",
            ),
            encoding="utf-8",
        )
        orig = _patch_tsv(orch, fake)
        try:
            st = _state()
            st.sync_from_tsv()
            assert st.last_do == "crash" and st.last_result == "crash"
            assert st.fail_knob == 0
            assert st.next_do_token() == "crash"
            block = orch.format_cause_block(st)
            assert "next_do=crash" in block
            print("crash row: no fail bump, next_do=crash OK")
        finally:
            _unpatch_tsv(orch, orig)

    with tempfile.TemporaryDirectory() as tmp:
        fake = Path(tmp) / "no_c1.tsv"
        lines = [RESULTS_HEADER_CLASSIFICATION]
        lines.append(
            _cls_row("a000001", 0.90, "keep", "[model] framework=baseline")
        )
        for i in range(3):
            lines.append(
                _cls_row(
                    f"d{i:07d}",
                    0.90,
                    "discard",
                    f"[model] framework=baseline knob{i}",
                )
            )
        fake.write_text("".join(lines), encoding="utf-8")
        orig = _patch_tsv(orch, fake)
        try:
            st = _state()
            st.sync_from_tsv()
            assert st.fail_knob == 3, st.fail_knob
            assert st.next_do_token() == "", st.next_do_token()
            import rnaomni.orch.note as note_mod

            old_note = note_mod.NOTE_PATH
            note_mod.NOTE_PATH = Path(tmp) / "no_note.json"
            try:
                fw = orch.format_framework_block(st)
                assert "MUST be [feature]" not in fw, fw
            finally:
                note_mod.NOTE_PATH = old_note
            print("fail[knob]>=3 does not lock layer (C1 removed) OK")
        finally:
            _unpatch_tsv(orch, orig)

    print("smoke_bottleneck_card: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
