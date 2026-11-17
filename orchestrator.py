#!/usr/bin/env python3
"""RNAOmni overnight orchestrator (one agent round per prompt).

Transport: ``agent_backend`` (default OpenCode; ``AGENT_BACKEND=cursor``
keeps Cursor SDK). Skill loading is transport-agnostic via
skill-library/registry.yaml + tools/resolve_skills.py.

Each Iter agent session is edit+commit only. Feature encoding and
train.py run outside the agent so long jobs do not hold the transport
(Cursor bridge ~15min idle/token failure mode).

Search stop (orchestrator-owned): hard budget, patience/min_delta,
and consecutive agent-failure fuse. Keep threshold shares min_delta
with the stop rule so noise keeps cannot reset patience.

Process-restart safety: keep/discard progress is replayed from
workspace/<stem>.tsv; iteration (+ prior stop_reason) lives in
workspace/orchestrator_state.json so patience / max_iterations survive
orchestrator restarts. consecutive_agent_failures stays in-memory
only (resets on restart — intentional for transport-death recovery).
Transport death (Connection refused / bridge timeout / opencode down)
exits with code 75 without counting toward max_agent_failures, so an
external restart can continue the search uninterrupted.
"""

from __future__ import annotations

import sys
import types


class _OrchestratorModule(types.ModuleType):
    """Forward ``results_tsv_path`` patches into ``rnaomni.orch.tsv`` (smoke tests)."""

    def __setattr__(self, name: str, value):
        if name == "results_tsv_path":
            import rnaomni.orch.tsv as tsv

            tsv.results_tsv_path = value
        super().__setattr__(name, value)


def _install() -> None:
    import rnaomni.orch as pkg

    facade = _OrchestratorModule(__name__, __doc__)
    facade.__file__ = __file__
    facade.__package__ = __package__
    skip = {"__name__", "__doc__", "__package__", "__file__", "__loader__", "__spec__"}
    for name in dir(pkg):
        if name in skip or name.startswith("__"):
            continue
        object.__setattr__(facade, name, getattr(pkg, name))
    sys.modules[__name__] = facade


if __name__ == "__main__":
    from rnaomni.orch.loop import main

    raise SystemExit(main())
else:
    _install()
