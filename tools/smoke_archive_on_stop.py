#!/usr/bin/env python3
"""Smoke: parallel env skips orchestrator archive; standalone still calls it."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> int:
    from rnaomni.orch.loop import _archive_on_stop

    saved = os.environ.pop("RNAOMNI_BATCH_ID", None)
    try:
        with patch("rnaomni.archive.archive_run", return_value=Path("/tmp/fake-run")) as mocked:
            _archive_on_stop("patience")
            mocked.assert_called_once_with(stop_reason="patience")
        print("standalone archive_run OK")

        os.environ["RNAOMNI_BATCH_ID"] = "   "
        with patch("rnaomni.archive.archive_run", return_value=Path("/tmp/fake-run")) as mocked:
            _archive_on_stop("patience")
            mocked.assert_called_once_with(stop_reason="patience")
        print("whitespace BATCH_ID still archives OK")

        os.environ["RNAOMNI_BATCH_ID"] = "20260815_005404_Reg_TE_PC3"
        with patch("rnaomni.archive.archive_run", return_value=Path("/tmp/fake-run")) as mocked:
            _archive_on_stop("patience")
            mocked.assert_not_called()
        print("parallel skip archive_run OK")
    finally:
        if saved is None:
            os.environ.pop("RNAOMNI_BATCH_ID", None)
        else:
            os.environ["RNAOMNI_BATCH_ID"] = saved

    print("smoke_archive_on_stop: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
