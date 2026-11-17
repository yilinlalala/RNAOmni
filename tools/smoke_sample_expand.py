#!/usr/bin/env python3
"""Unit-style smoke for sample-expand state machine (no train.py).

Usage:
  python tools/smoke_sample_expand.py
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
os.environ.pop("COMPARE_FULL", None)
os.environ.pop("COMPARE_NO_SAMPLE", None)
os.environ.pop("RNAOMNI_DATA_MODE", None)


def _state(**kwargs):
    import orchestrator as orch

    defaults = dict(
        decision_metric="val_accuracy",
        higher_is_better=True,
        min_delta=3e-4,
        patience=10,
        min_iterations=10,
        max_iterations=100,
        max_agent_failures=3,
        eff_window=5,
        sample_expand_patience=3,
    )
    defaults.update(kwargs)
    st = orch.SearchState(**defaults)
    st.configure_sampling_from_env()
    return st


def _cls_row(commit, val, test, f1, mem, mode, status, desc) -> str:
    return (
        f"{commit}\t{val:.6f}\t{test:.6f}\t{f1:.6f}\t{mem}\t{mode}\t{status}\t{desc}\n"
    )


def main() -> int:
    import orchestrator as orch
    from rnaomni.results_io import RESULTS_HEADER_CLASSIFICATION

    assert "\tdata_mode\tstatus\t" in RESULTS_HEADER_CLASSIFICATION

    desc = orch.format_sample_description(
        scale=2.0, old_k=10, trial_k=20, k_eval=10, kept=True
    )
    parsed = orch.parse_sample_description(desc)
    assert parsed is not None, desc
    assert parsed["old_k"] == 10 and parsed["trial_k"] == 20
    assert parsed["k_eval"] == 10 and parsed["kept"] is True
    print("format/parse OK:", desc)

    state = _state()
    assert state.sample_on is False
    assert state.data_mode == "full"
    assert state.can_grow_sample() is False
    assert state.should_expand_sample() is False
    state.switch_to_sample("classification")
    assert state.sample_on is True
    assert state.data_mode == "sample"
    assert state.best is None
    assert state.k_train == 10 and state.k_eval == 10 and state.k_train0 == 10
    assert state.sample_expand_patience == 3
    assert state.can_grow_sample() is True
    assert state.trial_k_train() == 20
    assert state.should_expand_sample() is False

    state.best = 0.1
    state.no_improve_streak = 2
    state.valid_rounds = 5
    assert state.should_expand_sample() is False
    state.no_improve_streak = 3
    assert state.should_expand_sample() is True
    assert state.should_stop(5) is None  # expand preferred over patience

    # Prior expand keep does not lock the next one.
    state.sample_expand_done = True
    state.sample_exhausted = False
    state.sample_plateau = False
    state.no_improve_streak = 3
    state.valid_rounds = 10
    assert state.should_expand_sample() is True
    assert state.should_stop(10) is None

    # Model-axis discard → freeze sample axis; search continues (ordinary patience).
    state.sample_exhausted = True
    state.sample_plateau = False
    assert state.should_expand_sample() is False
    stop = state.should_stop(10)
    assert stop is None, stop

    # Feature-axis: grow blocked until one wall-clock retry.
    state.sample_exhausted = False
    state.sample_grow_blocked = True
    state.sample_wallclock_retry = False
    assert state.can_grow_sample() is False
    state.sample_wallclock_retry = True
    assert state.can_grow_sample() is True
    assert state.should_expand_sample() is True

    # Patience stop when K cannot grow.
    state.sample_grow_blocked = False
    state.sample_wallclock_retry = False
    state.sample_exhausted = False
    state.sample_plateau = False
    state.k_train = state.k_train_cap
    state.no_improve_streak = 10
    state.valid_rounds = 10
    assert state.should_expand_sample() is False
    stop = state.should_stop(10)
    assert stop is not None and stop.startswith("patience"), stop
    state.k_train = 10
    print("should_expand / should_stop OK")

    assert orch.head_unchanged("abc", "abc") is True
    assert orch.head_unchanged("abc", "def") is False
    print("head_unchanged OK")

    with tempfile.TemporaryDirectory() as tmp:
        fake = Path(tmp) / "compare_fake.tsv"
        fake.write_text(
            RESULTS_HEADER_CLASSIFICATION
            + _cls_row("abc1234", 0.10, 0.09, 0.08, 0.1, "sample", "keep", "[feature] baseline")
            + _cls_row("abc1234", 0.10, 0.09, 0.08, 0.1, "sample", "discard", "[model] noop")
            + _cls_row("abc1234", 0.1001, 0.09, 0.08, 0.1, "sample", "discard", "[model] noop2")
            + _cls_row("abc1234", 0.1001, 0.09, 0.08, 0.1, "sample", "discard", "[model] noop3")
            + _cls_row(
                "abc1234",
                0.15,
                0.14,
                0.13,
                0.2,
                "sample",
                "keep",
                orch.format_sample_description(
                    scale=2.0, old_k=10, trial_k=20, k_eval=10, kept=True
                ),
            ),
            encoding="utf-8",
        )
        orig_results = orch.results_tsv_path
        orch.results_tsv_path = lambda: fake  # type: ignore[assignment]
        try:
            st = _state()
            st.switch_to_sample("classification")
            st.sync_from_tsv()
            assert st.k_train == 20
            assert st.sample_scale == 2.0
            assert abs(st.best - 0.15) < 1e-9
            assert st.no_improve_streak == 0
            assert st.sample_exhausted is False
            assert st.sample_plateau is False
            assert st.should_expand_sample() is False
            st.no_improve_streak = 3
            assert st.should_expand_sample() is True
            print(
                "tsv replay keep OK:",
                f"best={st.best} K_train={st.k_train} streak={st.no_improve_streak}",
            )
        finally:
            orch.results_tsv_path = orig_results  # type: ignore[assignment]

    with tempfile.TemporaryDirectory() as tmp:
        fake = Path(tmp) / "compare_fake.tsv"
        fake.write_text(
            RESULTS_HEADER_CLASSIFICATION
            + _cls_row("abc1234", 0.20, 0.19, 0.18, 0.1, "sample", "keep", "[feature] baseline")
            + _cls_row("abc1234", 0.19, 0.18, 0.17, 0.1, "sample", "discard", "[model] d1")
            + _cls_row("abc1234", 0.19, 0.18, 0.17, 0.1, "sample", "discard", "[model] d2")
            + _cls_row("abc1234", 0.19, 0.18, 0.17, 0.1, "sample", "discard", "[model] d3")
            + _cls_row(
                "abc1234",
                0.19,
                0.18,
                0.17,
                0.1,
                "sample",
                "discard",
                orch.format_sample_description(
                    scale=2.0, old_k=10, trial_k=20, k_eval=10, kept=False
                ),
            ),
            encoding="utf-8",
        )
        orig_results = orch.results_tsv_path
        orch.results_tsv_path = lambda: fake  # type: ignore[assignment]
        try:
            st = _state()
            st.switch_to_sample("classification")
            st.sync_from_tsv()
            assert st.k_train == 10
            assert abs(st.best - 0.2) < 1e-9
            assert st.sample_exhausted is True
            assert st.sample_plateau is False
            assert st.should_expand_sample() is False
            stop = st.should_stop(10)
            assert stop is None, stop
            print(
                "tsv replay discard → sample axis frozen OK:",
                f"best={st.best} K_train={st.k_train} streak={st.no_improve_streak}",
            )
        finally:
            orch.results_tsv_path = orig_results  # type: ignore[assignment]

    with tempfile.TemporaryDirectory() as tmp:
        fake = Path(tmp) / "compare_fake.tsv"
        fake.write_text(
            RESULTS_HEADER_CLASSIFICATION
            + _cls_row("abc1234", 0.50, 0.40, 0.30, 0.1, "full", "keep", "[feature] full baseline")
            + _cls_row("abc1234", 0.10, 0.09, 0.08, 0.1, "sample", "keep", "[feature] sample baseline"),
            encoding="utf-8",
        )
        orig_results = orch.results_tsv_path
        orch.results_tsv_path = lambda: fake  # type: ignore[assignment]
        try:
            st = _state()
            st.switch_to_sample("classification")
            st.sync_from_tsv()
            assert abs(st.best - 0.10) < 1e-9, st.best
            print("tsv replay filters by data_mode OK")
        finally:
            orch.results_tsv_path = orig_results  # type: ignore[assignment]

    print("smoke_sample_expand: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
