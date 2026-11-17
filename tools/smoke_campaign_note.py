#!/usr/bin/env python3
"""N0: campaign note data block + opening prompt/parse (no live LLM).

Usage:
  python tools/smoke_campaign_note.py
"""

from __future__ import annotations

import json
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


def main() -> int:
    import orchestrator as orch
    from rnaomni.orch import note as note_mod

    st = _state()
    data = orch.collect_data_block(task_type="classification", state=st)
    assert data["task_type"] == "classification"
    assert "enabled" in data and isinstance(data["enabled"], list)
    assert "onehot" in data["enabled"] or data["registered"]
    assert "best" not in data and "val_accuracy" not in data
    print("collect_data_block OK: enabled=", data.get("enabled"))

    legal = note_mod.collect_legal_actions(st)
    prompt = orch.build_opening_prompt(data, legal)
    leaks = note_mod.opening_prompt_leaks(prompt)
    assert leaks == [], leaks
    assert "run.log" not in prompt
    assert "composition" in prompt and "legal" in prompt
    print("opening prompt has no score leak OK")

    parsed = note_mod.extract_json_object(
        'noise\n```json\n{"prior":{"suspect":["structure"],"reason":"long-ish RNA"},'
        '"next":{"kind":"feature","name":"struct_dotbracket","why":"structure"}}\n```\n'
    )
    assert parsed is not None
    prior = note_mod.parse_prior(parsed)
    assert prior["suspect"] == ["structure"], prior
    nxt = note_mod.validate_next(parsed, legal)
    assert nxt is not None and nxt["kind"] == "feature"
    print("parse prior + legal next OK:", nxt)

    bad = note_mod.validate_next(
        {"next": {"kind": "feature", "name": "not_a_feature", "why": "x"}},
        legal,
    )
    assert bad is None
    print("illegal next dropped OK")

    fake = note_mod.empty_note()
    fake["data"] = data
    fake["prior"] = prior
    fake["next"] = nxt
    block = orch.format_note_block(fake)
    assert len(block) < 800, len(block)
    assert "run.log" not in block
    assert "enabled=" in block
    assert "MUST this round" in block
    assert "struct_thermo" in block
    print("render <800 OK:", repr(block[:180]))

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "campaign_note.json"
        def _fail(_prompt: str):
            raise RuntimeError("no network")

        out = orch.ensure_opening_note(
            st,
            task_type="classification",
            completer=_fail,
            path=path,
            call_llm=True,
        )
        assert path.is_file()
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved["data"]["task_type"] == "classification"
        assert saved["data"].get("enabled")
        assert not (saved.get("prior") or {}).get("suspect")
        assert out["data"]["enabled"]
        print("API failure still writes data block OK")

        def _ok(_prompt: str) -> str:
            return json.dumps(
                {
                    "prior": {
                        "suspect": ["composition", "structure"],
                        "reason": "kmer-friendly families",
                    },
                    "next": {
                        "kind": "knob",
                        "name": "LR",
                        "why": "head already baseline",
                    },
                }
            )

        path2 = Path(tmp) / "note2.json"
        out2 = orch.ensure_opening_note(
            st,
            task_type="classification",
            completer=_ok,
            path=path2,
        )
        assert out2["prior"]["suspect"] == ["composition", "structure"]
        assert out2["next"]["kind"] == "knob" and out2["next"]["name"] == "LR"
        # Resume: prior set → do not overwrite even if completer would fail.
        out3 = orch.ensure_opening_note(
            st,
            task_type="classification",
            completer=_fail,
            path=path2,
        )
        assert out3["prior"]["suspect"] == ["composition", "structure"]
        print("opening LLM fill + resume skip OK")

        st_mid = _state()
        st_mid.valid_rounds = 4
        path3 = Path(tmp) / "note3.json"
        out4 = orch.ensure_opening_note(
            st_mid,
            task_type="classification",
            completer=_ok,
            path=path3,
        )
        assert not (out4.get("prior") or {}).get("suspect")
        print("mid-campaign skip opening LLM OK")

        def _head(_prompt: str) -> str:
            return json.dumps(
                {
                    "prior": {"suspect": ["structure"], "reason": "long RNA"},
                    "next": {"kind": "head", "name": "tf_mean", "why": "x"},
                }
            )

        path4 = Path(tmp) / "note4.json"
        out5 = orch.ensure_opening_note(
            st,
            task_type="classification",
            completer=_head,
            path=path4,
        )
        assert out5["prior"]["suspect"] == ["structure"]
        assert out5.get("next") is None, out5.get("next")
        print("opening next=head ignored (N1) OK")

    # Evidence append + fallback A.
    grid = note_mod.empty_note()
    grid["data"] = data
    grid["next"] = {"kind": "feature", "name": "struct_dotbracket", "why": "open"}
    note_mod.append_round_evidence(
        grid, cause="feature", result="discard", delta=-0.002, description="probe 1"
    )
    note_mod.append_round_evidence(
        grid, cause="feature", result="discard", delta=-0.001, description="probe 2"
    )
    assert len(grid["F"]["evidence"]) == 2, grid["F"]
    assert grid["F"]["evidence"][0]["desc"] == "probe 1"
    print("F evidence append keeps old rows OK")

    legal = note_mod.collect_legal_actions(st)
    kept = note_mod.apply_revision_reply(
        grid,
        {"next": {"kind": "feature", "name": "not_a_feature", "why": "x"}},
        legal,
    )
    assert kept is not None and kept["name"] == "struct_dotbracket", kept
    print("fallback A illegal next retains last legal OK")

    kept_head = note_mod.apply_revision_reply(
        grid,
        {"next": {"kind": "head", "name": "tf_mean", "why": "switch"}},
        legal,
    )
    assert kept_head is not None and kept_head["kind"] != "head"
    print("fallback A next=head ignored OK")

    st_n2 = _state()
    st_n2.valid_rounds = 1
    legal_n2 = note_mod.collect_legal_actions(st_n2)
    assert legal_n2.get("head_allowed") is True
    grid_n2 = note_mod.empty_note()
    grid_n2["next"] = {"kind": "feature", "name": "struct_dotbracket", "why": "open"}
    got_head = note_mod.apply_revision_reply(
        grid_n2,
        {"next": {"kind": "head", "name": "tf_mean", "why": "after first result"}},
        legal_n2,
    )
    assert got_head is not None and got_head["kind"] == "head", got_head
    assert got_head["name"] == "tf_mean"
    assert note_mod.legal_note_head_name(st_n2, grid_n2) == "tf_mean"
    print("N2 revision may pick remaining head after keep/discard OK")

    illegal_head = note_mod.apply_revision_reply(
        grid_n2,
        {"next": {"kind": "head", "name": "not_a_menu_head", "why": "x"}},
        legal_n2,
    )
    assert illegal_head is not None and illegal_head["name"] == "tf_mean", illegal_head
    print("N2 illegal head → fallback A OK")

    empty = note_mod.empty_note()
    none = note_mod.apply_revision_reply(
        empty,
        {"next": {"kind": "feature", "name": "not_a_feature", "why": "x"}},
        legal,
    )
    assert none is None
    print("fallback A no previous next → no MUST OK")

    legal_done = dict(legal)
    legal_done["features_actionable"] = [
        x for x in legal["features"] if x != "struct_dotbracket"
    ]
    legal_done["enabled"] = list(legal.get("enabled") or []) + ["struct_dotbracket"]
    legal_done["exclude"] = [{"kind": "feature", "name": "struct_dotbracket"}]
    assert (
        note_mod.validate_next(
            {"next": {"kind": "feature", "name": "struct_dotbracket", "why": "again"}},
            legal_done,
        )
        is None
    )
    stuck = note_mod.empty_note()
    stuck["next"] = {"kind": "feature", "name": "struct_dotbracket", "why": "open"}
    dropped = note_mod.apply_revision_reply(
        stuck,
        {"next": {"kind": "feature", "name": "struct_dotbracket", "why": "again"}},
        legal_done,
    )
    assert dropped is None, dropped
    assert stuck.get("next") is None
    print("already-on / just-tried feature is not a legal next OK")

    def _repeat_dotbracket(_prompt: str) -> str:
        return json.dumps(
            {
                "next": {
                    "kind": "feature",
                    "name": "struct_dotbracket",
                    "why": "again",
                }
            }
        )

    grid_rep = note_mod.empty_note()
    grid_rep["data"] = data
    grid_rep["next"] = {"kind": "feature", "name": "struct_dotbracket", "why": "open"}
    after_rep = note_mod.revise_after_round(
        grid_rep,
        cause="feature",
        result="keep",
        description="[feature] enable struct_dotbracket (cause=feature)",
        state=st,
        completer=_repeat_dotbracket,
    )
    nxt_rep = after_rep.get("next")
    assert nxt_rep is None or nxt_rep.get("name") != "struct_dotbracket", nxt_rep
    assert after_rep["data"].get("enabled") is not None
    rev_p = note_mod.build_revision_prompt(
        grid_rep,
        note_mod.collect_legal_actions(
            st, exclude=[{"kind": "feature", "name": "struct_dotbracket"}]
        ),
        cause="feature",
        result="keep",
        description="[feature] enable struct_dotbracket (cause=feature)",
    )
    assert "legal.exclude" in rev_p and "this_round" in rev_p
    print("keep revision cannot MUST the same feature again OK")

    enabled_now = note_mod.collect_legal_actions(st).get("enabled") or []
    for already in enabled_now:
        if already in (note_mod.collect_legal_actions(st).get("tunable") or {}):
            continue
        assert (
            note_mod.validate_next(
                {"next": {"kind": "feature", "name": already, "why": "on"}},
                note_mod.collect_legal_actions(st),
            )
            is None
        )
    print("already-enabled no-tunable feature is not MUST-able OK")

    crash_p = note_mod.build_revision_prompt(
        grid,
        legal,
        cause="crash",
        result="crash",
        description="HIDDEN_DIM boom",
        crash_digest="RuntimeError: BFloat16 overflow",
    )
    assert "run.log" not in crash_p.lower()
    assert "vetoes switching" in crash_p
    print("crash revision prompt vetoes layer-switch OK")

    def _crash_ok(_prompt: str) -> str:
        return json.dumps(
            {"next": {"kind": "feature", "name": "kmer", "why": "avoid HIDDEN_DIM"}}
        )

    grid["next"] = {"kind": "knob", "name": "HIDDEN_DIM", "why": "capacity"}
    after = note_mod.revise_after_round(
        grid,
        cause="crash",
        result="crash",
        description="HIDDEN_DIM boom",
        crash_digest="RuntimeError: BFloat16 overflow",
        state=st,
        completer=_crash_ok,
    )
    assert after["next"]["name"] == "kmer", after["next"]
    assert after["crash_events"], after.get("crash_events")
    print("crash revision writes different next OK")

    st_c1 = _state()
    st_c1.fail_knob = 3
    assert st_c1.next_do_token() == "", st_c1.next_do_token()
    must = note_mod.n1_must_action(fake, st_c1)
    assert must is not None and must["kind"] == "feature", must
    assert must["name"] == "struct_dotbracket", must
    with tempfile.TemporaryDirectory() as tmp_empty:
        old_path = note_mod.NOTE_PATH
        note_mod.NOTE_PATH = Path(tmp_empty) / "none.json"
        try:
            fw = orch.format_framework_block(st_c1)
            assert "MUST be [feature]" not in fw, fw
        finally:
            note_mod.NOTE_PATH = old_path
    print("fail[knob]>=3 does not lock layer; note MUST name wins OK")

    st_crash = _state()
    st_crash.last_non_sample_crash = True
    must_c = note_mod.n1_must_action(fake, st_crash)
    assert must_c is not None and must_c["kind"] == "crash", must_c
    block_c = orch.format_note_block(fake, state=st_crash)
    assert "[crash]" in block_c and "struct_dotbracket" in block_c
    fw_c = orch.format_framework_block(st_crash)
    assert "MUST repair" in fw_c, fw_c
    print("N1 crash veto over note MUST OK")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sample_note.json"
        samp = note_mod.empty_note()
        samp["data"] = data
        samp["next"] = {"kind": "knob", "name": "LR", "why": "before expand"}
        note_mod.save_note(samp, path)
        st_s = _state()
        st_s.sample_on = True
        st_s.k_train = 10
        st_s.k_eval = 10
        st_s.k_train0 = 10
        st_s.sample_exhausted = True
        out_s = note_mod.apply_sample_expand_to_note(st_s, status="discard", path=path)
        assert out_s["S"]["last_kept"] is False
        assert (out_s.get("S") or {}).get("evidence")
        assert out_s["next"] is not None and out_s["next"]["name"] == "LR", out_s.get("next")
        print("sample expand updates S cell without LLM OK")

        path_fw = Path(tmp) / "fw_note.json"
        note_mod.save_note(fake, path_fw)
        old_path = note_mod.NOTE_PATH
        note_mod.NOTE_PATH = path_fw
        try:
            fw_must = orch.format_framework_block(_state())
            assert "MUST be [feature] struct_dotbracket" in fw_must, fw_must
            print("framework block MUST note name OK")
        finally:
            note_mod.NOTE_PATH = old_path

    print("smoke_campaign_note: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
