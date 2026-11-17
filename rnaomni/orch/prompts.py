"""Setup / Iter prompt templates and builders."""

from __future__ import annotations

import os
import sys

from rnaomni.paths import REPO
from tools.user_env import load_user_env

load_user_env(REPO)

sys.path.insert(0, str(REPO / "tools"))
from resolve_skills import format_prompt_block  # noqa: E402

from prepare import (
    ACTIVE_CSV,
    USE_COMPARE_SPLITS,
    clean_dataset,
    data_mode,
    infer_task_type,
    load_compare_splits_dataframe,
    load_dataset,
)
from rnaomni.results_io import results_tsv_rel
from rnaomni.orch.search import (  # noqa: E402
    DEFAULT_FRAMEWORK,
    DEFAULT_MIN_DELTA_CLASSIFICATION,
    DEFAULT_MIN_DELTA_REGRESSION,
    format_cause_block,
)
from rnaomni.orch.note import format_note_block, n1_must_action  # noqa: E402
from rnaomni.orch.tsv import format_crash_prompt_block

PROMPT_SETUP_TEMPLATE = """\
You are invoked by orchestrator.py in orchestrator mode (SDK is only the current transport).
Read program.md section "SDK orchestrator mode", Active skills, and complete Setup only.

{skill_block}
{encoder_pin_block}
Rules for this invocation:
- Full autonomy: never ask the human for confirmation.
- Do NOT start the experiment loop; do NOT run train.py.
- Do NOT run feature encoding / full prepare encode (that happens later outside the SDK).
- Use a run tag from the current timestamp; create branch RNAclaw/<tag> from master.
- Read README.md, program.md, prepare.py, train.py, features/active.yaml, features/registry.yaml.
- Respect program.md Dataset contract. {dataset_route_block}
- Verify features/registry.yaml and features/active.yaml exist and are readable.
- Run: python prepare.py --skip-encode  (splits/data-card only; NO feature encoding in this session)
- Results logging is orchestrator-managed: {results_file}
  (under workspace/; there is no repo-root results.tsv). Do NOT recreate or commit it.

When setup is complete, STOP. Baseline is the next Iter invocation.
"""

PROMPT_ITER_TEMPLATE = """\
You are invoked by orchestrator.py in orchestrator mode (SDK is only the current transport).
Read program.md ("SDK orchestrator mode" + Experimentation), Active skills / agent-ops, {results_file}, git log, features/active.yaml, and train.py.

{skill_block}
{encoder_pin_block}
Active task_type (from prepare inference): {task_type}
data_mode: {data_mode}  (full = all rows; sample = subsampled. Do not treat a mode switch as your commit's gain.)
Decision metric: {decision_metric} ({decision_direction})
Keep threshold (shared with orchestrator stop rule): improvement must be >= {min_delta}
Current best {decision_metric}: {best_display}
{framework_block}
{crash_feedback}{cause_block}{note_block}
IMPORTANT — this SDK session is SHORT. The orchestrator runs train.py (feature encode + training)
and keep/discard git reset OUTSIDE the SDK after you stop. Do NOT run train.py, prepare.py,
or any long encoding/training command in this session.

Run exactly ONE experiment iteration (edit+commit only), then STOP:
1. Inspect git state (branch, HEAD, recent commits) and {results_file}.
2. Declare round_type (feature|model). Prefer whichever hypothesis is clearest; either is fine.
   If next_framework is set (framework block above): this round MUST be model implementing that name only.
   If bottleneck next_do=crash (or crash block above): this round MUST repair that error; do not switch layers.
   If bottleneck next_do=feature: this round MUST be [feature]; write cause=feature.
   If campaign note MUST names a feature or knob and none of the above vetoes apply: execute that name this round. Do not default to twisting LR or concatenating another feature. If the note names a remaining head, the orchestrator injects next_framework — implement that name only; never pick a skeleton yourself.
   If note MUST is absent and next_do is empty: choose [feature] or [model] stay.
   - feature: edit features/active.yaml whitelist only (recipe_id; fusion=concat|add|mean; optional fusion_include=[enabled registered names]; features[].name/enabled; params only if in that feature's agent_tunable: kmer.k in 1..4, rna_fm.repr_layer in 0..12). L4 rna_fm/rinalmo/evo/evo2 may be enabled; missing local weights crash — no pip/download. evo/evo2 are 4096-d / ~12GB VRAM. Prefer one change per round (one enabled, one tunable, or fusion_include). add/mean need equal C among included channels. May adapt train.py for input_dim. Enable only registry-listed features.
   - model: edit train.py only (first successful run may be an unchanged baseline with a no-op or docs-only commit if needed). Opening active_framework is baseline (current Predictor / PooledMLP). Stay on it unless next_framework is set — then set Predictor to the matching class already in train.py (do not retune LR). In-framework knobs: HIDDEN_DIM / LR / BATCH_SIZE / dropout / loss / EMA; resnet1d_pool kernel 3|7 and residual blocks <=4. Named heads: skill-library/feature/frameworks.md. Description must include framework=<active_or_next> and cause=<head|feature|knob> (cause=crash if fixing a crash). Never write cause=head to pick a skeleton; only when next_framework is set.
3. git commit (whitelist files only) with subject prefixed by [feature] or [model]. Do not commit {results_file}.
4. STOP immediately after the commit. Do not run train.py. Do not grep run.log. Do not edit {results_file}.

Sample expand: a [sample] row already retrained the previous best HEAD at the new K_train.
Do NOT re-run that commit or any earlier keep/discard experiment. This round must be a NEW
edit+commit. If you cannot produce a new edit, STOP without a no-op / empty commit.

Then STOP. Do not ask the human anything. Do not start another iteration in this session.
"""


def detect_task_type() -> str:
    try:
        if USE_COMPARE_SPLITS:
            # Tiny stratified preview for label-cardinality only — quiet so it
            # does not look like the training COMPARE_PER_CLASS load.
            df = load_compare_splits_dataframe(per_class=1, quiet=True)
            return infer_task_type(df["predicted_target"].to_numpy())
        df = load_dataset(ACTIVE_CSV)
        _, _, task_type = clean_dataset(df)
        return task_type
    except Exception:
        return "regression"


def decision_spec(task_type: str) -> tuple[str, bool, str]:
    """Return (metric_name, higher_is_better, metric_grep)."""
    if task_type == "classification":
        return (
            "val_accuracy",
            True,
            r"^val_accuracy:\|^test_accuracy:\|^peak_vram_mb:",
        )
    return (
        "val_spearman",
        True,
        r"^val_spearman:\|^test_RMSE:\|^peak_vram_mb:",
    )


def default_min_delta(task_type: str) -> float:
    if task_type == "classification":
        return DEFAULT_MIN_DELTA_CLASSIFICATION
    return DEFAULT_MIN_DELTA_REGRESSION


def format_encoder_pin_block() -> str:
    """Campaign pin from RNAOMNI_PIN_L4 / RNAOMNI_FREEZE_L4 (empty = no pin)."""
    pin = (os.environ.get("RNAOMNI_PIN_L4") or "").strip()
    if not pin:
        return ""
    freeze_raw = (os.environ.get("RNAOMNI_FREEZE_L4") or "rna_fm,rinalmo,evo2").strip()
    frozen = [x.strip() for x in freeze_raw.split(",") if x.strip() and x.strip() != pin]
    frozen_s = ", ".join(frozen) if frozen else "(none)"
    return (
        f"ENCODER PIN (this campaign): {pin}.enabled MUST stay true. "
        f"Do not enable, disable, or retune: {frozen_s}. "
        "train.py knobs and named heads may still change.\n"
    )


def build_prompt(template: str, **extra) -> str:
    skill_block = format_prompt_block(overnight=True).rstrip()
    extra.setdefault("results_file", results_tsv_rel())
    extra.setdefault("encoder_pin_block", format_encoder_pin_block())
    return template.format(skill_block=skill_block, **extra)


def format_framework_block(state: object | None) -> str:
    """Inject active/next framework. Empty-safe when state is missing."""
    if state is None:
        return (
            f"Framework: active={DEFAULT_FRAMEWORK}  next=(none)\n"
            f"Stay on {DEFAULT_FRAMEWORK}. [model] description must include "
            f"framework={DEFAULT_FRAMEWORK}.\n"
        )
    active = getattr(state, "active_framework", None) or DEFAULT_FRAMEWORK
    streak = int(getattr(state, "framework_no_improve_streak", 0) or 0)
    patience = int(getattr(state, "framework_patience", 10) or 10)
    tried = getattr(state, "frameworks_tried", None) or [active]
    if isinstance(tried, str):
        tried_s = tried
    else:
        tried_s = ",".join(str(x) for x in tried)
    nxt = getattr(state, "next_framework", None) or ""
    nxt_s = nxt if nxt else "(none)"
    next_do = ""
    if hasattr(state, "next_do_token"):
        next_do = str(state.next_do_token() or "")
    lines = [
        f"Framework: active={active}  streak={streak}/{patience}  tried={tried_s}  next={nxt_s}",
        "Named heads: skill-library/feature/frameworks.md (WIDTH=HIDDEN_DIM; do not hardcode 4/69/266/532).",
    ]
    if nxt:
        lines.append(
            f"SWITCH ROUND: implement next_framework={nxt} only. round_type=model. "
            "Set Predictor to the matching class already in train.py. "
            "Inherit HIDDEN_DIM / LR / batch / loss / EMA. "
            "Do not edit features/active.yaml or CONV_CHANNELS. Do not retune LR this round. "
            f"Commit description: [model] framework={nxt} framework_base cause=head … "
            "Do not invent another name."
        )
    elif next_do == "crash":
        lines.append(
            f"Stay on {active}. This round MUST repair the last crash (see crash block). "
            "Do not switch feature/head/layer. Do not follow note next until the crash is cleared."
        )
    elif next_do == "feature":
        action = n1_must_action(state=state)
        name = (action or {}).get("name") or ""
        named = f" name={name}" if name else ""
        lines.append(
            f"Stay on {active}. This round MUST be [feature]{named} "
            "(bottleneck next_do=feature). Do not switch the named head."
        )
    else:
        action = n1_must_action(state=state)
        kind = (action or {}).get("kind") or ""
        name = (action or {}).get("name") or ""
        if kind == "feature":
            lines.append(
                f"Stay on {active}. This round MUST be [feature] {name}. "
                "Write cause=feature. Do not switch the named head."
            )
        elif kind == "knob":
            lines.append(
                f"Stay on {active}. This round MUST be [model] stay knob={name}. "
                f"[model] description must include framework={active} and cause=knob. "
                "Do not switch the named head this round."
            )
        else:
            lines.append(
                f"Stay on {active}. Choose [feature] or [model] stay "
                f"(cause=knob). [model] description must include framework={active}. "
                "Do not switch the named head this round."
            )
    return "\n".join(lines) + "\n"


def build_iter_prompt(
    *,
    min_delta: float,
    best: float | None,
    state: object | None = None,
) -> str:
    task_type = detect_task_type()
    decision_metric, higher_is_better, metric_grep = decision_spec(task_type)
    direction = "higher is better" if higher_is_better else "lower is better"
    best_display = f"{best:.6f}" if best is not None else "(none yet — first keep establishes best)"
    return build_prompt(
        PROMPT_ITER_TEMPLATE,
        task_type=task_type,
        decision_metric=decision_metric,
        decision_direction=direction,
        min_delta=f"{min_delta:.6g}",
        best_display=best_display,
        framework_block=format_framework_block(state),
        crash_feedback=format_crash_prompt_block(),
        cause_block=format_cause_block(state),
        note_block=format_note_block(state=state),
        data_mode=data_mode(),
    )
