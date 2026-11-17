#!/usr/bin/env python3
"""Walk through the campaign diagnostic role (N1 policy preview).

Does not change overnight search. Does not write workspace/campaign_note.json
unless --write.

Story:
  1. Opening (no Y) → MUST first shot
  2. Feature discard → revise next
  3. Crash → note next changes, but crash block still vetoes switching layers
  4. Fallback A: illegal / head / empty → keep last legal, or no MUST

Usage:
  python tools/demo_campaign_diagnose.py
  python tools/demo_campaign_diagnose.py --live
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

os.environ.setdefault(
    "RNAOMNI_COMPARE_ROOT",
    str(REPO / "dataset" / "forcompare" / "Class_88fam_200%noise"),
)


def _hr(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def _must_line(nxt: dict | None, *, crash_veto: bool = False) -> None:
    if crash_veto:
        print("Iter MUST this round: [crash] one repair of the short signature")
        print("                       crash block vetoes switching feature/head/layer")
        if nxt:
            print(
                f"Note next (after crash is cleared): {nxt.get('kind')} {nxt.get('name')}"
            )
        else:
            print("Note next (after crash is cleared): (none)")
        return
    if not nxt:
        print("Iter MUST this round: (none — agent may choose feature or knob)")
        return
    kind = nxt.get("kind")
    name = nxt.get("name")
    why = nxt.get("why") or ""
    if kind == "feature":
        tag = "[feature]"
    elif kind == "sample":
        tag = "[sample] expand (orchestrator, no LLM)"
    else:
        tag = "[model] stay"
    print(f"Iter MUST this round: {tag} {name}")
    if why:
        print(f"                      why: {why}")


def _canned_opening(_prompt: str) -> str:
    return json.dumps(
        {
            "prior": {
                "suspect": ["structure", "composition"],
                "reason": "Family labels often track motif/composition; structure still off.",
            },
            "next": {
                "kind": "feature",
                "name": "struct_dotbracket",
                "why": "MFE pairing not enabled; cheap probe of structure.",
            },
        }
    )


def _canned_after_discard(_prompt: str) -> str:
    return json.dumps(
        {
            "next": {
                "kind": "knob",
                "name": "HIDDEN_DIM",
                "why": "Structure probe did not lift Y; try capacity on current head.",
            }
        }
    )


def _canned_after_crash(_prompt: str) -> str:
    return json.dumps(
        {
            "next": {
                "kind": "feature",
                "name": "kmer",
                "why": "Avoid repeating the crashed HIDDEN_DIM change; try kmer.k instead.",
            }
        }
    )


def _canned_switch_head(_prompt: str) -> str:
    return json.dumps(
        {
            "next": {
                "kind": "head",
                "name": "tf_mean",
                "why": "skip the crash, just switch the skeleton",
            }
        }
    )


def _canned_illegal(_prompt: str) -> str:
    return json.dumps(
        {"next": {"kind": "feature", "name": "not_a_registered_feature", "why": "oops"}}
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Campaign diagnostic demo")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Call isolated OPENCODE_MODEL/CURSOR_MODEL (same as Iter). Slow.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Save the demo note to workspace/campaign_note.json",
    )
    args = parser.parse_args()

    import rnaomni.orch.note as note_mod
    from rnaomni.orch.search import SearchState, format_cause_block
    from rnaomni.orch.tsv import format_crash_prompt_block

    state = SearchState(
        decision_metric="val_accuracy",
        higher_is_better=True,
        min_delta=0.001,
        patience=15,
        min_iterations=10,
        max_iterations=100,
        max_agent_failures=3,
        eff_window=5,
        sample_expand_patience=3,
    )
    data = note_mod.collect_data_block(task_type="classification", state=state)
    legal = note_mod.collect_legal_actions(state)

    _hr("0. N1 policy this demo is showing")
    print("C1 inner MUST is removed; note drives F/θ.")
    print("First Iter MUST opening next (option A).")
    print("Revise after keep, discard, and crash. Sample-expand: orchestrator only.")
    print("Fallback A: illegal/failed LLM keeps last legal next; else no MUST.")
    print("Crash block still vetoes skipping the repair to switch layers.")

    _hr("1. Data (no Y, no TSV scores)")
    print(json.dumps(data, ensure_ascii=False, indent=2))

    opening_prompt = note_mod.build_opening_prompt(data, legal)
    leaks = note_mod.opening_prompt_leaks(opening_prompt)
    _hr("2. Opening prompt (isolated API, no tools)")
    print("chars:", len(opening_prompt), "score-leaks:", leaks or "none")
    print(opening_prompt[:720])
    if len(opening_prompt) > 720:
        print("...")

    note = note_mod.empty_note()
    note["data"] = data
    demo_path = None
    if args.live or args.write:
        demo_path = (
            note_mod.NOTE_PATH
            if args.write
            else Path(tempfile.mkdtemp(prefix="rnaomni_demo_")) / "campaign_note.json"
        )

    if args.live:
        _hr("3. Opening diagnosis (LIVE isolated API)")
        note = note_mod.ensure_opening_note(
            state,
            task_type="classification",
            call_llm=True,
            path=demo_path,
        )
        note["next"] = note_mod.n1_executable_next(note.get("next"), legal)
    else:
        _hr("3. Opening diagnosis (canned JSON, not live)")
        parsed = note_mod.extract_json_object(_canned_opening(""))
        note["prior"] = note_mod.parse_prior(parsed)
        note["next"] = note_mod.n1_executable_next(
            (parsed or {}).get("next"), legal
        )

    print("prior:", json.dumps(note.get("prior"), ensure_ascii=False))
    _must_line(note.get("next"))
    print(note_mod.format_note_block(note))

    _hr("4. After [feature] discard → revise API (keep/discard)")
    note = note_mod.revise_after_round(
        note,
        cause="feature",
        result="discard",
        delta=-0.002,
        description="[feature] cause=feature struct_dotbracket enable",
        state=state,
        completer=None if args.live else _canned_after_discard,
        call_llm=True,
    )
    print("F evidence:", json.dumps(note.get("F"), ensure_ascii=False))
    _must_line(note.get("next"))

    crashed_name = (note.get("next") or {}).get("name")
    _hr("5. After crash: note revises next, crash block still vetoes layer switch")
    state.last_non_sample_crash = True
    state.last_do = "crash"
    state.last_result = "crash"
    print(format_cause_block(state).rstrip())
    crash_rows = [
        {
            "commit": "abc1234",
            "status": "crash",
            "description": (
                f"[model] framework=baseline cause=knob {crashed_name} boom"
                " || crash: RuntimeError: BFloat16 overflow | File train.py, line 280"
            ),
        }
    ]
    print(format_crash_prompt_block(crash_rows).rstrip())

    note = note_mod.revise_after_round(
        note,
        cause="crash",
        result="crash",
        description=f"[model] framework=baseline cause=knob {crashed_name} boom",
        crash_digest="RuntimeError: BFloat16 overflow | File train.py, line 280",
        state=state,
        completer=None if args.live else _canned_after_crash,
        call_llm=True,
    )
    print("crash_events:", json.dumps(note.get("crash_events"), ensure_ascii=False))
    _must_line(note.get("next"), crash_veto=True)
    nxt = note.get("next") or {}
    if crashed_name and nxt.get("name") == crashed_name:
        print(f"WARN: note next still names the crashed action {crashed_name!r}")
    else:
        print(f"OK: note next {nxt.get('name')!r} != crashed {crashed_name!r}")
    print("OK: Iter this round is still crash-repair, not the note's next")

    _hr("6. Fallback A: illegal next keeps last legal")
    before = dict(note.get("next") or {})
    kept = note_mod.apply_revision_reply(
        note,
        note_mod.extract_json_object(_canned_illegal("")),
        legal,
    )
    print("LLM proposed: not_a_registered_feature")
    print("kept:", kept)
    assert (kept or {}).get("name") == before.get("name"), (before, kept)
    print("OK: previous legal next retained")

    _hr("7. Fallback A: next=head is ignored (N1 does not draw heads)")
    before = dict(note.get("next") or {})
    fake = dict(note)
    fake["next"] = {"kind": "knob", "name": "HIDDEN_DIM", "why": "pre-crash"}
    parsed_head = note_mod.extract_json_object(_canned_switch_head(""))
    kept_head = note_mod.apply_revision_reply(fake, parsed_head, legal)
    print("LLM proposed: head tf_mean (skip crash, switch skeleton)")
    print("kept:", kept_head)
    assert kept_head is not None and kept_head.get("kind") != "head"
    assert kept_head.get("name") == "HIDDEN_DIM", kept_head
    print("OK: head dropped; last legal knob retained")
    print("OK: crash block would still MUST repair this round (not switch layers)")
    note["next"] = before

    _hr("8. Fallback A: no legal next at all → no MUST")
    empty = note_mod.empty_note()
    empty["data"] = data
    none = note_mod.apply_revision_reply(
        empty,
        note_mod.extract_json_object(_canned_illegal("")),
        legal,
    )
    print("opening also failed / next empty; LLM illegal →", none)
    _must_line(none)
    assert none is None
    print("OK: no MUST")

    if args.write:
        note_mod.save_note(note)
        print("\nWrote", note_mod.NOTE_PATH)
    elif demo_path is not None and demo_path.is_file():
        print("\nLive note saved to", demo_path)

    _hr("Done")
    print("Default is canned. --live calls the Iter model in an empty cwd.")
    print("Overnight loop uses this MUST / veto order. --live calls the Iter model in an empty cwd.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
