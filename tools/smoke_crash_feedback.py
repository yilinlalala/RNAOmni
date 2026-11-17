#!/usr/bin/env python3
"""Smoke crash digest + capped Iter prompt block (no train.py)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


FAKE_LOG = """\
Epoch 0 step 3
Traceback (most recent call last):
  File "train.py", line 390, in <module>
    main()
  File "train.py", line 280, in main
    loss.backward()
RuntimeError: value cannot be converted to type at::BFloat16 without overflow
"""


def main() -> int:
    import orchestrator as orch

    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "run.log"
        log.write_text(FAKE_LOG, encoding="utf-8")
        digest = orch.summarize_run_log(log)
        assert "BFloat16" in digest, digest
        assert "Traceback" not in digest, digest
        assert "\n" not in digest, digest
        assert len(digest) <= orch.CRASH_DIGEST_MAX
        print("summarize_run_log OK:", digest)

        huge = Path(tmp) / "huge.log"
        huge.write_text("Traceback (most recent call last):\n" + ("x" * 5000) + "\nRuntimeError: boom\n")
        d2 = orch.summarize_run_log(huge)
        assert d2.startswith("RuntimeError")
        assert len(d2) <= orch.CRASH_DIGEST_MAX
        print("summarize cap OK:", len(d2))

    rows_keep = [
        {"commit": "aaa1111", "status": "crash", "description": "old || crash: old error"},
        {"commit": "bbb2222", "status": "keep", "description": "[model] recovered"},
    ]
    assert orch.format_crash_prompt_block(rows_keep) == ""

    rows_discard = [
        {"commit": "aaa1111", "status": "crash", "description": "x || crash: old"},
        {"commit": "ccc3333", "status": "discard", "description": "[model] noop"},
    ]
    assert orch.format_crash_prompt_block(rows_discard) == ""

    rows_crash = [
        {"commit": "keep001", "status": "keep", "description": "[model] ok"},
        {"commit": "388c860", "status": "crash", "description": "[model] xc pool || crash: RuntimeError: BFloat16 without overflow | File \"train.py\", line 280"},
    ]
    block = orch.format_crash_prompt_block(rows_crash)
    assert block.startswith("Last train crash")
    assert "388c860" in block
    assert "BFloat16" in block
    assert "do not read run.log" in block
    assert block.count("\n") <= 4
    assert len(block) < 500, len(block)
    print("prompt block OK:", repr(block))

    sample_then_crash = [
        {"commit": "s1", "status": "crash", "description": "[sample] x || crash: should-skip"},
        {"commit": "388c860", "status": "crash", "description": "[model] foo || crash: RuntimeError: boom"},
    ]
    block2 = orch.format_crash_prompt_block(sample_then_crash)
    assert "boom" in block2
    assert "should-skip" not in block2
    print("skip sample crash then use model crash OK")

    print("smoke_crash_feedback: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
