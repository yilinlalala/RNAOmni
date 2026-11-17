"""Runtime path table.

Live campaign scratch (gitignored) lives under ``workspace/``
(same layout as RNAClaw ``results/``):
``run.log``, ``orchestrator.log``, ``orchestrator_state.json``,
``campaign_note.json``, ``deploy_feature.log``. Search logs are
``workspace/<stem>.tsv``; archives and finalize under ``workspace/runs/``.
Prep metadata (data cards, splits) and L4 smoke stay under ``artifacts/``.
MSA stays under ``artifacts/msa`` / ``artifacts/msa_dbs``.
Repo-root ``results.tsv`` is not written.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).absolute().parent.parent

WORKSPACE_DIR = REPO / "workspace"
RESULTS_DIR = WORKSPACE_DIR  # alias: same role as RNAClaw results/
RUN_LOG = RESULTS_DIR / "run.log"
ORCH_LOG = RESULTS_DIR / "orchestrator.log"
STATE_PATH = RESULTS_DIR / "orchestrator_state.json"
NOTE_PATH = RESULTS_DIR / "campaign_note.json"
DEPLOY_LOG = RESULTS_DIR / "deploy_feature.log"

RUNS_DIR = RESULTS_DIR / "runs"
INDEX_PATH = RUNS_DIR / "INDEX.tsv"
LEGACY_RESULTS_TSV = REPO / "results.tsv"

DATA_CARD_DIR = REPO / "artifacts" / "data_cards"
SPLIT_DIR = REPO / "artifacts" / "splits"

__all__ = [
    "DATA_CARD_DIR",
    "DEPLOY_LOG",
    "INDEX_PATH",
    "LEGACY_RESULTS_TSV",
    "NOTE_PATH",
    "ORCH_LOG",
    "REPO",
    "RESULTS_DIR",
    "RUN_LOG",
    "RUNS_DIR",
    "SPLIT_DIR",
    "STATE_PATH",
    "WORKSPACE_DIR",
]
