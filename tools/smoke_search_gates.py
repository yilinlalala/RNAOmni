#!/usr/bin/env python3
"""Search-gate smokes: flat HEAD rollback, in-framework keep, full timeout,
crash-repair release, campaign-wide exclude, heavy L4 block (no train.py).

Usage:
  python tools/smoke_search_gates.py
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


def _fake_recipe(*, enabled=None, mode_unused=None):
    enabled = list(enabled or ["onehot", "rna_fm"])
    return {
        "recipe_id": "smoke_gates",
        "fusion": "concat",
        "enabled": enabled,
        "registered": [
            "onehot",
            "rna_fm",
            "rinalmo",
            "evo",
            "evo2",
            "struct_dotbracket",
        ],
        "tunable": {"rna_fm": ["repr_layer"]},
        "heavy": ["evo", "evo2"],
        "light_l4": ["rna_fm", "rinalmo"],
    }


def main() -> int:
    import orchestrator as orch
    import rnaomni.orch.loop as loop
    import rnaomni.orch.note as note_mod
    from rnaomni.results_io import RESULTS_HEADER_CLASSIFICATION

    orig_recipe = note_mod._recipe_snapshot
    note_mod._recipe_snapshot = _fake_recipe  # type: ignore[assignment]
    orig_note_path = note_mod.NOTE_PATH
    try:
        _run(orch, loop, note_mod, RESULTS_HEADER_CLASSIFICATION)
    finally:
        note_mod._recipe_snapshot = orig_recipe  # type: ignore[assignment]
        note_mod.NOTE_PATH = orig_note_path
    print("smoke_search_gates: OK")
    return 0


def _run(orch, loop, note_mod, header) -> None:
    resets: list[str] = []
    orig_reset = loop.git_reset_hard
    loop.git_reset_hard = lambda c: resets.append(c)  # type: ignore[assignment]
    orig_mode = loop.data_mode
    try:
        # 1. Flat framework_base: TSV keep, HEAD back to global best.
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "flat.tsv"
            fake.write_text(
                header
                + _cls_row(
                    "g7d3bde",
                    0.294,
                    "keep",
                    "[model] framework=gated_pool_mlp framework_base cause=head",
                ),
                encoding="utf-8",
            )
            orig = _patch_tsv(orch, fake)
            try:
                st = _state()
                st.active_framework = "attn_pool_mlp"
                st.best = 0.294
                st.best_commit = "7d3bdeb"
                st.framework_best = 0.294
                st.framework_best_commit = "7d3bdeb"
                st.frameworks_tried = ["baseline", "pool_mlp", "attn_pool_mlp"]
                resets.clear()
                status = orch.reconcile_after_train(
                    pre_head="7d3bdeb",
                    min_delta=0.001,
                    best_before=0.294,
                    decision_metric="val_accuracy",
                    higher_is_better=True,
                    train_exit=0,
                    state=st,
                )
                assert status == "keep", status
                assert resets == ["7d3bdeb"], resets
                row = orch.latest_results_row()
                assert (row or {}).get("status") == "keep"
                print("1 flat framework_base: TSV keep, reset to best_commit OK")
            finally:
                _unpatch_tsv(orch, orig)

        st_ing = _state()
        st_ing.active_framework = "attn_pool_mlp"
        st_ing.best = 0.294
        st_ing.frameworks_tried = ["attn_pool_mlp"]
        parsed = orch.parse_round_kind(
            "[model] framework=gated_pool_mlp framework_base cause=head"
        )
        st_ing._apply_framework_round(
            metric=0.193,
            status="keep",
            parsed=parsed,
            commit="c5f4d3b",
            best_before=0.294,
        )
        assert st_ing.active_framework == "attn_pool_mlp", st_ing.active_framework
        assert "gated_pool_mlp" in st_ing.frameworks_tried
        assert "head" in st_ing.suppressed
        print("1 ingest: flat head tried, working tree stays attn_pool OK")

        # 2. In-framework discard→keep vs framework_best (I12).
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "upgrade.tsv"
            fake.write_text(
                header
                + _cls_row(
                    "c000012",
                    0.204,
                    "discard",
                    "[model] framework=gated_pool_mlp cause=knob DROPOUT",
                ),
                encoding="utf-8",
            )
            orig = _patch_tsv(orch, fake)
            try:
                st = _state()
                st.active_framework = "gated_pool_mlp"
                st.best = 0.294
                st.framework_best = 0.193
                st.framework_best_commit = "c000009"
                st.framework_head = "c000009"
                resets.clear()
                status = orch.reconcile_after_train(
                    pre_head="c000009",
                    min_delta=0.001,
                    best_before=0.294,
                    decision_metric="val_accuracy",
                    higher_is_better=True,
                    train_exit=0,
                    state=st,
                )
                assert status == "keep", status
                assert resets == [], resets
                row = orch.latest_results_row()
                assert (row or {}).get("status") == "keep", row
                print("2 in-framework discard→keep vs framework_best OK")
            finally:
                _unpatch_tsv(orch, orig)

        # 3. Full encode timeout after an established keep stays full.
        loop.data_mode = lambda: "full"  # type: ignore[assignment]
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "timeout.tsv"
            fake.write_text(
                header
                + _cls_row("a000001", 0.294, "keep", "[model] framework=attn_pool_mlp"),
                encoding="utf-8",
            )
            orig = _patch_tsv(orch, fake)
            try:
                st = _state()
                st.best = 0.294
                st.best_commit = "a000001"
                st.valid_rounds = 8
                st.data_mode = "full"
                resets.clear()
                status = orch.reconcile_after_train(
                    pre_head="a000001",
                    min_delta=0.001,
                    best_before=0.294,
                    decision_metric="val_accuracy",
                    higher_is_better=True,
                    train_exit=124,
                    state=st,
                )
                assert status == "crash", status
                assert resets == ["a000001"], resets
                row = orch.latest_results_row()
                assert (row or {}).get("status") == "crash"
                print("3 established full timeout → crash, stay full OK")
            finally:
                _unpatch_tsv(orch, orig)

        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "first_to.tsv"
            fake.write_text(header, encoding="utf-8")
            orig = _patch_tsv(orch, fake)
            try:
                st = _state()
                assert st.best is None and st.valid_rounds == 0
                resets.clear()
                status = orch.reconcile_after_train(
                    pre_head="deadbee",
                    min_delta=0.001,
                    best_before=None,
                    decision_metric="val_accuracy",
                    higher_is_better=True,
                    train_exit=124,
                    state=st,
                )
                assert status == "encode_timeout", status
                assert resets == [], resets
                print("3 first full encode timeout → switch-to-sample signal OK")
            finally:
                _unpatch_tsv(orch, orig)

        # 4. Empty crash-repair commit releases veto; ordinary empty still fails.
        with tempfile.TemporaryDirectory() as tmp:
            note_mod.NOTE_PATH = Path(tmp) / "campaign_note.json"
            st = _state()
            st.last_non_sample_crash = True
            note = note_mod.empty_note()
            note["crash_events"] = [
                {
                    "name": "evo",
                    "kind": "feature",
                    "result": "crash",
                    "desc": "[feature] evo ENCODE_TIMEOUT",
                }
            ]
            note["failed_features"] = ["evo"]
            note["next"] = {
                "kind": "feature",
                "name": "rinalmo",
                "why": "avoid evo",
            }
            note_mod.save_note(note, note_mod.NOTE_PATH)
            assert orch.handle_empty_commit(st) == "crash_release"
            assert st.last_non_sample_crash is False
            assert st.consecutive_agent_failures == 0
            saved = note_mod.load_note(note_mod.NOTE_PATH)
            assert saved["next"]["name"] == "rinalmo", saved.get("next")
            legal = note_mod.collect_legal_actions(st, note=saved)
            assert any(
                x.get("kind") == "feature" and x.get("name") == "evo"
                for x in legal["exclude"]
            ), legal["exclude"]
            assert orch.handle_empty_commit(st) == "agent_fail"
            assert st.consecutive_agent_failures == 1
            print("4 crash-repair empty commit releases veto; second empty counts OK")

        # 5. Failed feature stays illegal; knobs are per-framework.
        grid = note_mod.empty_note()
        grid["next"] = {"kind": "feature", "name": "rinalmo", "why": "long_range"}
        note_mod.append_round_evidence(
            grid,
            cause="feature",
            result="discard",
            description="[feature] rinalmo L4",
        )
        assert "rinalmo" in grid["failed_features"]
        st = _state()
        legal = note_mod.collect_legal_actions(st, note=grid)
        assert (
            note_mod.validate_next(
                {"next": {"kind": "feature", "name": "rinalmo", "why": "again"}},
                legal,
            )
            is None
        )
        print("5 discarded feature stays in legal.exclude OK")

        knobs = note_mod.empty_note()
        knobs["next"] = {"kind": "knob", "name": "dropout", "why": "reg"}
        note_mod.append_round_evidence(
            knobs,
            cause="knob",
            result="discard",
            description="[model] framework=gated_pool_mlp cause=knob dropout",
            framework="gated_pool_mlp",
        )
        st_g = _state()
        st_g.active_framework = "gated_pool_mlp"
        legal_g = note_mod.collect_legal_actions(st_g, note=knobs)
        assert (
            note_mod.validate_next(
                {"next": {"kind": "knob", "name": "dropout", "why": "again"}},
                legal_g,
            )
            is None
        )
        st_a = _state()
        st_a.active_framework = "attn_pool_mlp"
        legal_a = note_mod.collect_legal_actions(st_a, note=knobs)
        assert (
            note_mod.validate_next(
                {"next": {"kind": "knob", "name": "dropout", "why": "on new head"}},
                legal_a,
            )
            is not None
        )
        print("5 knob exclude is per-framework OK")

        st_full = _state()
        st_full.data_mode = "full"
        legal_h = note_mod.collect_legal_actions(st_full)
        assert "evo" not in (legal_h.get("features_actionable") or [])
        assert any(
            x.get("name") == "evo" for x in legal_h.get("exclude") or []
        ), legal_h.get("exclude")
        assert (
            note_mod.validate_next(
                {"next": {"kind": "feature", "name": "evo", "why": "must"}},
                legal_h,
            )
            is None
        )
        st_s = _state()
        st_s.data_mode = "sample"
        legal_s = note_mod.collect_legal_actions(st_s)
        assert "evo" in (legal_s.get("features_actionable") or [])
        print("5 heavy evo/evo2 illegal in full with light L4 on; ok in sample OK")

        from rnaomni.orch.note import feature_is_heavy
        from features.pipeline import load_feature_registry

        registry = load_feature_registry()
        by_name = {e["name"]: e for e in registry["features"]}
        assert feature_is_heavy(by_name["evo"]) and feature_is_heavy(by_name["evo2"])
        assert not feature_is_heavy(by_name["rna_fm"])
        assert not feature_is_heavy(by_name["rinalmo"])
        print("5 registry heavy flags on evo/evo2 OK")
    finally:
        loop.git_reset_hard = orig_reset  # type: ignore[assignment]
        loop.data_mode = orig_mode  # type: ignore[assignment]


if __name__ == "__main__":
    raise SystemExit(main())
