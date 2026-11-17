#!/usr/bin/env python3
"""16.1: framework SearchState + reconcile reset targets (no train.py).

Usage:
  python tools/smoke_framework_search.py
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
        "[model] framework=attn_pool_mlp framework_base swap pooling"
    )
    assert parsed["kind"] == "model"
    assert parsed["framework"] == "attn_pool_mlp"
    assert parsed["framework_base"] is True
    parsed2 = orch.parse_round_kind("[feature] kmer k=4")
    assert parsed2["kind"] == "feature" and parsed2["framework_base"] is False
    print("parse_round_kind OK")

    assert orch.DEFAULT_FRAMEWORK == "baseline"
    assert "baseline" not in orch.FRAMEWORK_MENU
    st0 = _state()
    assert st0.active_framework == "baseline"
    assert st0.frameworks_tried == ["baseline"]
    remaining0 = st0.remaining_frameworks()
    assert "pool_mlp" in remaining0
    assert "tf_mean" in remaining0 and "tf_cls" in remaining0
    assert "tf_attn" in remaining0 and "gru_pool" in remaining0
    assert len(orch.FRAMEWORK_MENU) == 12
    assert set(remaining0) == set(orch.FRAMEWORK_MENU)
    st0.framework_no_improve_streak = 9
    assert st0.ensure_next_framework("campaign-aaa") is None
    st0.framework_no_improve_streak = 10
    pick0 = st0.ensure_next_framework("campaign-aaa")
    assert pick0 in orch.FRAMEWORK_MENU
    pick0b = st0.ensure_next_framework("campaign-aaa")
    assert pick0b == pick0
    print("B0 opening baseline remaining includes pool_mlp OK:", pick0)

    with tempfile.TemporaryDirectory() as tmp:
        fake = Path(tmp) / "compare_fw.tsv"
        lines = [RESULTS_HEADER_CLASSIFICATION]
        lines.append(
            _cls_row(
                "a000001",
                0.90,
                "keep",
                "[model] framework=pool_mlp baseline",
            )
        )
        for i in range(10):
            lines.append(
                _cls_row(
                    f"a0000{i+2:02d}",
                    0.90,
                    "discard",
                    f"[model] framework=pool_mlp d{i+1}",
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
        lines.append(
            _cls_row(
                "b000002",
                0.40,
                "discard",
                "[model] framework=attn_pool_mlp worse tweak",
            )
        )
        fake.write_text("".join(lines), encoding="utf-8")
        orig = _patch_tsv(orch, fake)
        try:
            st = _state()
            st.sync_from_tsv()
            assert st.active_framework == "pool_mlp", st.active_framework
            assert abs(st.best - 0.90) < 1e-9, st.best
            assert abs(st.framework_best - 0.90) < 1e-9, st.framework_best
            assert st.framework_head == "a000001"
            assert st.framework_best_commit == "a000001"
            assert st.framework_no_improve_streak == 10
            assert "pool_mlp" in st.frameworks_tried
            assert "attn_pool_mlp" in st.frameworks_tried
            assert st.framework_reset_commit() == "a000001"
            assert st.no_improve_streak >= 10
            print(
                "tsv A plateau → B base worse than A stays A (tried, not working tree) OK:",
                f"global={st.best} fw={st.framework_best} reset={st.framework_reset_commit()}",
            )
        finally:
            _unpatch_tsv(orch, orig)

    with tempfile.TemporaryDirectory() as tmp:
        fake = Path(tmp) / "compare_fw2.tsv"
        fake.write_text(
            RESULTS_HEADER_CLASSIFICATION
            + _cls_row("a000001", 0.90, "keep", "[model] framework=pool_mlp")
            + _cls_row(
                "b000001",
                0.91,
                "keep",
                "[model] framework=attn_pool_mlp framework_base",
            )
            + _cls_row("c000001", 0.92, "keep", "[feature] kmer k=4"),
            encoding="utf-8",
        )
        orig = _patch_tsv(orch, fake)
        try:
            st = _state()
            st.sync_from_tsv()
            assert st.active_framework == "attn_pool_mlp"
            assert st.frameworks_tried == ["attn_pool_mlp"], st.frameworks_tried
            assert "pool_mlp" in st.remaining_frameworks()
            assert st.framework_no_improve_streak == 0
            assert abs(st.framework_best - 0.92) < 1e-9
            print("feature keep re-enables old frameworks OK")
        finally:
            _unpatch_tsv(orch, orig)

        orig = _patch_tsv(orch, fake)
        try:
            st.sync_from_tsv()
            assert st.active_framework == "attn_pool_mlp"
            assert st.frameworks_tried == ["attn_pool_mlp"]
            print("sync_from_tsv twice is stable OK")
        finally:
            _unpatch_tsv(orch, orig)

    resets: list[str] = []

    def _fake_reset(commit: str) -> None:
        resets.append(commit)

    import rnaomni.orch.loop as loop

    orig_reset = loop.git_reset_hard
    loop.git_reset_hard = _fake_reset  # type: ignore[assignment]
    try:
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "row.tsv"
            fake.write_text(
                RESULTS_HEADER_CLASSIFICATION
                + _cls_row(
                    "b000001",
                    0.50,
                    "keep",
                    "[model] framework=attn_pool_mlp framework_base first",
                ),
                encoding="utf-8",
            )
            orig = _patch_tsv(orch, fake)
            try:
                st = _state()
                st.active_framework = "pool_mlp"
                st.best = 0.90
                st.best_commit = "a000001"
                st.framework_best = 0.90
                st.framework_best_commit = "a000001"
                st.framework_head = "a000001"
                status = orch.reconcile_after_train(
                    pre_head="a000001",
                    min_delta=0.001,
                    best_before=0.90,
                    decision_metric="val_accuracy",
                    higher_is_better=True,
                    train_exit=0,
                    state=st,
                )
                assert status == "keep", status
                assert resets == ["a000001"], resets
                print("reconcile: B first shot worse than A resets HEAD to global best OK")
            finally:
                _unpatch_tsv(orch, orig)

        resets.clear()
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "row.tsv"
            fake.write_text(
                RESULTS_HEADER_CLASSIFICATION
                + _cls_row(
                    "b000002",
                    0.40,
                    "discard",
                    "[model] framework=attn_pool_mlp tweak",
                ),
                encoding="utf-8",
            )
            orig = _patch_tsv(orch, fake)
            try:
                st = _state()
                st.active_framework = "attn_pool_mlp"
                st.best = 0.90
                st.framework_best = 0.50
                st.framework_best_commit = "b000001"
                st.framework_head = "b000001"
                status = orch.reconcile_after_train(
                    pre_head="b000001",
                    min_delta=0.001,
                    best_before=0.90,
                    decision_metric="val_accuracy",
                    higher_is_better=True,
                    train_exit=0,
                    state=st,
                )
                assert status == "discard", status
                assert resets == ["b000001"], resets
                print("reconcile: B inner discard resets to B not A OK")
            finally:
                _unpatch_tsv(orch, orig)

        resets.clear()
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "row.tsv"
            fake.write_text(
                RESULTS_HEADER_CLASSIFICATION
                + _cls_row(
                    "b000003",
                    0.55,
                    "keep",
                    "[model] framework=attn_pool_mlp better than B worse than A",
                ),
                encoding="utf-8",
            )
            orig = _patch_tsv(orch, fake)
            try:
                st = _state()
                st.active_framework = "attn_pool_mlp"
                st.best = 0.90
                st.framework_best = 0.50
                st.framework_best_commit = "b000001"
                st.framework_head = "b000001"
                status = orch.reconcile_after_train(
                    pre_head="b000001",
                    min_delta=0.001,
                    best_before=0.90,
                    decision_metric="val_accuracy",
                    higher_is_better=True,
                    train_exit=0,
                    state=st,
                )
                assert status == "keep", status
                assert resets == [], resets
                print("reconcile: B keep vs framework best (not global) OK")
            finally:
                _unpatch_tsv(orch, orig)
    finally:
        loop.git_reset_hard = orig_reset  # type: ignore[assignment]

    st = _state()
    st.framework_no_improve_streak = 10
    assert st.framework_plateaued() is True
    st.framework_no_improve_streak = 9
    assert st.framework_plateaued() is False
    print("FRAMEWORK_PATIENCE=10 plateau flag OK")

    st = _state()
    st.framework_no_improve_streak = 10
    st.frameworks_tried = ["pool_mlp"]
    pick_a = st.ensure_next_framework("campaign-aaa")
    pick_b = st.ensure_next_framework("campaign-aaa")
    assert pick_a == pick_b, (pick_a, pick_b)
    assert pick_a in orch.FRAMEWORK_MENU and pick_a != "pool_mlp"
    st2 = _state()
    st2.framework_no_improve_streak = 10
    st2.frameworks_tried = ["pool_mlp"]
    assert st2.ensure_next_framework("campaign-aaa") == pick_a
    st.next_framework = pick_a
    assert st.ensure_next_framework("other-tag") == pick_a
    block = orch.format_framework_block(st)
    assert f"next={pick_a}" in block and "SWITCH ROUND" in block
    assert f"framework={pick_a} framework_base" in block
    st3 = _state()
    assert st3.ensure_next_framework("campaign-aaa") is None
    stay = orch.format_framework_block(st3)
    assert "Stay on baseline" in stay and "next=(none)" in stay
    st4 = _state()
    st4.framework_no_improve_streak = 10
    st4.frameworks_tried = list(orch.FRAMEWORK_MENU)
    st4.active_framework = "cls_mlp"
    assert st4.remaining_frameworks() == []
    assert st4.ensure_next_framework("campaign-aaa") is None
    print("ensure_next_framework draw/sticky/empty OK:", pick_a)

    with tempfile.TemporaryDirectory() as tmp:
        fake = Path(tmp) / "plateau.tsv"
        lines = [RESULTS_HEADER_CLASSIFICATION]
        lines.append(_cls_row("a000001", 0.90, "keep", "[model] framework=pool_mlp"))
        for i in range(10):
            lines.append(
                _cls_row(f"d{i:07d}", 0.90, "discard", f"[model] framework=pool_mlp d{i}")
            )
        fake.write_text("".join(lines), encoding="utf-8")
        orig = _patch_tsv(orch, fake)
        try:
            st = _state()
            st.sync_from_tsv()
            assert st.framework_plateaued(), st.framework_no_improve_streak
            nxt = st.ensure_next_framework("campaign-aaa")
            assert nxt and nxt != "pool_mlp" and nxt not in st.frameworks_tried
            assert "next=" + nxt in orch.format_framework_block(st)
            print("tsv 10 discards → next_framework injected OK:", nxt)
        finally:
            _unpatch_tsv(orch, orig)

    with tempfile.TemporaryDirectory() as tmp:
        fake = Path(tmp) / "baseline_plateau.tsv"
        lines = [RESULTS_HEADER_CLASSIFICATION]
        lines.append(_cls_row("a000001", 0.90, "keep", "[model] framework=baseline"))
        for i in range(10):
            lines.append(
                _cls_row(
                    f"d{i:07d}",
                    0.90,
                    "discard",
                    f"[model] framework=baseline d{i}",
                )
            )
        fake.write_text("".join(lines), encoding="utf-8")
        orig = _patch_tsv(orch, fake)
        try:
            st = _state()
            st.sync_from_tsv()
            assert st.active_framework == "baseline", st.active_framework
            assert st.framework_plateaued(), st.framework_no_improve_streak
            nxt = st.ensure_next_framework("campaign-aaa")
            assert nxt in orch.FRAMEWORK_MENU
            assert nxt not in st.frameworks_tried
            assert "pool_mlp" in st.remaining_frameworks() or nxt == "pool_mlp"
            assert "next=" + nxt in orch.format_framework_block(st)
            print("tsv 10 baseline discards → next can be pool_mlp OK:", nxt)
        finally:
            _unpatch_tsv(orch, orig)

    st_n2 = _state()
    st_n2.valid_rounds = 1
    assert st_n2.ensure_next_framework("campaign-aaa", note_head="tf_mean") == "tf_mean"
    assert st_n2.next_framework == "tf_mean"
    assert not st_n2.framework_plateaued()
    # Sticky: a later different note name does not replace an unimplemented draw.
    assert (
        st_n2.ensure_next_framework("campaign-aaa", note_head="tf_attn") == "tf_mean"
    )
    print("N2 note head before plateau + sticky OK")

    st_open = _state()
    st_open.valid_rounds = 0
    assert st_open.ensure_next_framework("campaign-aaa", note_head="tf_mean") is None
    print("N2 opening valid_rounds=0 ignores note head OK")

    st_sup = _state()
    st_sup.valid_rounds = 1
    st_sup.suppressed = ["head"]
    assert st_sup.ensure_next_framework("campaign-aaa", note_head="tf_mean") is None
    print("N2 suppressed head ignores note name OK")

    st_tried = _state()
    st_tried.valid_rounds = 1
    st_tried.frameworks_tried = [orch.DEFAULT_FRAMEWORK, "tf_mean"]
    assert "tf_mean" not in st_tried.remaining_frameworks()
    assert st_tried.ensure_next_framework("campaign-aaa", note_head="tf_mean") is None
    print("N2 tried/off-menu note head dropped OK")

    print("smoke_framework_search: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
