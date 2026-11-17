#!/usr/bin/env python3
"""RNAOmni Feature Deploy launcher (S2b-Deploy) via agent_backend.

Human-triggered only. Does NOT use orchestrator.py overnight prompts.
SOP: skill-library/feature/deploy-ops.md

Default transport: OpenCode (``AGENT_BACKEND=cursor`` for Cursor SDK).

Assets-driven (not model-hardcoded). Examples:
  python -m rnaomni.deploy --assets skill-library/feature/examples/rinalmo.assets.yaml \\
      --code-root /abs/path/to/RiNALMo --dry-run
  python -m rnaomni.deploy --assets skill-library/feature/examples/rna_fm.assets.yaml --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from rnaomni.paths import DEPLOY_LOG, REPO
from tools.user_env import load_user_env

load_user_env(REPO)
sys.path.insert(0, str(REPO / "tools"))
import rnaomni.agent_backend as agent_backend
from rnaomni.agent_backend import prompt as agent_prompt, resolve_backend, resolve_model
from pretrained_assets import (  # noqa: E402
    expected_dim_for,
    infer_lm_config,
    package_marker_ok,
    parse_assets,
    resolve_code_root,
    resolve_local_weight,
    write_whitelist_for,
)
from resolve_skills import format_deploy_prompt_block  # noqa: E402

DEFAULT_ASSETS = REPO / "skill-library" / "feature" / "examples" / "rinalmo.assets.yaml"

PROMPT_DEPLOY_TEMPLATE = """\
You are invoked by python -m rnaomni.deploy (Feature Deploy / S2b-Deploy).
Agent transport is only a vehicle. This is NOT orchestrator overnight mode.

{skill_block}

## Preflight (already checked by the launcher — re-verify)
- feature_name: **{feature_name}**
- conda env: **{conda_env}**
- code_root: `{code_root}` (zero-install: PYTHONPATH={code_root_abs})
- resolved local weight: `{weight_path}`
- lm_config: `{lm_config}` (must match weight scale)
- expected_dim_doc (hint): {expected_dim}; registry.dim MUST come from smoke

## Mission
Follow deploy-ops.md end-to-end for feature **{feature_name}**:
1. Embedding check with **local** weight only (no download). Record real **dim** and device
   (short Python snippet under `PYTHONPATH={code_root_abs}`, or enable the feature and run `prepare.py` / a short `train.py`).
2. If ImportError / missing hard dep / check fails: **STOP and ask the human**.
   Cite `{code_root}/environment.yml` / setup files as suggestions only.
   Do **not** online `pip install` / `conda install` / hub / gdown without a local path.
3. If check OK: write builder + registry L4 entry + features/active.yaml placeholder with **enabled: false**.
   Lock `default_params.model_path` (absolute or repo-relative to the resolved file), `lm_config`, `device`.
   Set `registry.dim` to measured dim (update if user later switches micro/mega/giga).
4. Re-check via builder path; assert C == registry.dim.
5. Write short handoff under skill-library/feature/examples/{feature_name}.handoff.md; STOP.

## Write whitelist (only these)
{write_whitelist}

## Hard bans
- Do NOT edit train.py, prepare.py, orchestrator.py, PROMPT_*, folds, metrics, TIME_BUDGET.
- Do NOT set enabled: true or run train.py / results.tsv experiments.
- Do NOT follow agent-ops.md.
- Do NOT download weights.

When done or blocked on human input, STOP.
"""


def setup_logging(log_file: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )


def preflight(
    assets: dict,
    *,
    code_root_override: str | None,
    weight_override: Path | None,
) -> tuple[Path, Path, str, int]:
    code_abs = resolve_code_root(assets, repo=REPO, code_root_override=code_root_override)
    if not code_abs.is_dir():
        raise FileNotFoundError(f"code_root missing: {code_abs}")
    package_marker_ok(code_abs, assets)

    if weight_override is not None:
        weight = weight_override.expanduser().resolve()
        if not weight.is_file():
            raise FileNotFoundError(f"--weight not found: {weight}")
    else:
        candidates = [str(c) for c in (assets.get("weight_candidates") or [])]
        if not candidates:
            raise FileNotFoundError("assets has empty weight_candidates")
        weight = resolve_local_weight(candidates, code_root=code_abs, repo=REPO)

    lm_config = infer_lm_config(weight, assets)
    dim_hint = expected_dim_for(lm_config, assets)
    return code_abs, weight, lm_config, dim_hint


def build_prompt(
    assets: dict,
    code_abs: Path,
    weight: Path,
    lm_config: str,
    dim_hint: int,
) -> str:
    feature_name = str(assets.get("feature_name") or "pretrained")
    skill_block = format_deploy_prompt_block(repo=REPO).rstrip()
    whitelist = "\n".join(f"- `{p}`" for p in write_whitelist_for(feature_name))
    return PROMPT_DEPLOY_TEMPLATE.format(
        skill_block=skill_block,
        feature_name=feature_name,
        conda_env=assets.get("conda_env")
        or os.environ.get("RNAOMNI_CONDA_ENV", "rnaomni"),
        code_root=assets.get("code_root", str(code_abs)),
        code_root_abs=str(code_abs),
        weight_path=str(weight),
        lm_config=lm_config,
        expected_dim=dim_hint,
        write_whitelist=whitelist,
    )


def run_deploy(
    prompt: str,
    *,
    backend: str,
    model: str,
    api_key: str | None = None,
) -> int:
    logging.info(
        "Starting Deploy agent (backend=%s model=%s)",
        backend,
        model,
    )
    result = agent_prompt(
        prompt,
        cwd=REPO,
        model=model,
        backend=backend,
        api_key=api_key,
    )
    if result.transport_dead:
        logging.error(
            "Transport dead (backend=%s): %s — exit %s",
            backend,
            result.message,
            agent_backend.EXIT_TRANSPORT_DEAD,
        )
        return agent_backend.EXIT_TRANSPORT_DEAD

    logging.info("Deploy agent finished: status=%s backend=%s", result.status, backend)
    if result.result:
        preview = result.result.strip().replace("\n", " ")
        if len(preview) > 800:
            preview = preview[:800] + "..."
        logging.info("Result preview: %s", preview)

    if result.status == "error":
        if result.message:
            logging.error("Deploy error: %s", result.message[:500])
        return result.as_exit_code()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch Feature Deploy agent (S2b-Deploy) via agent_backend. "
            "Default transport: OpenCode. Assets + --code-root select which "
            "pretrained tree to deploy."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=["cursor", "opencode"],
        default=None,
        help=(
            "Agent transport (default: AGENT_BACKEND or "
            f"{agent_backend.DEFAULT_BACKEND})."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model id for the selected backend "
            "(default: OPENCODE_MODEL / CURSOR_MODEL by backend)."
        ),
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEPLOY_LOG,
        help="Append deploy launcher logs to this file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preflight + print prompt; do not call agent backend.",
    )
    parser.add_argument(
        "--assets",
        type=Path,
        default=DEFAULT_ASSETS,
        help="Path to *.assets.yaml (feature-specific contract).",
    )
    parser.add_argument(
        "--code-root",
        type=str,
        default=None,
        help="Absolute or repo-relative upstream code root (overrides assets.code_root).",
    )
    parser.add_argument(
        "--weight",
        type=Path,
        default=None,
        help="Override local weight path (must exist; no download).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_file)

    assets_path = args.assets
    if not assets_path.is_file():
        # allow repo-relative
        cand = (REPO / assets_path).resolve()
        if cand.is_file():
            assets_path = cand
        else:
            logging.error("Assets file missing: %s", args.assets)
            return 1

    try:
        assets = parse_assets(assets_path)
        code_abs, weight, lm_config, dim_hint = preflight(
            assets,
            code_root_override=args.code_root,
            weight_override=args.weight,
        )
        prompt = build_prompt(assets, code_abs, weight, lm_config, dim_hint)
    except FileNotFoundError as err:
        logging.error("Deploy preflight failed: %s", err)
        return 1

    logging.info(
        "Preflight OK: feature=%s conda_env=%s code_root=%s weight=%s lm=%s dim_hint=%s",
        assets.get("feature_name"),
        assets.get("conda_env"),
        code_abs,
        weight,
        lm_config,
        dim_hint,
    )

    if args.dry_run:
        print(prompt)
        logging.info("Dry-run complete; agent backend not invoked.")
        return 0

    try:
        backend = resolve_backend(args.backend)
    except ValueError as err:
        logging.error("%s", err)
        return 1
    model = resolve_model(backend, args.model)
    api_key = os.environ.get("CURSOR_API_KEY")
    if backend == "cursor" and not api_key:
        logging.error("CURSOR_API_KEY is not set (required for --backend cursor).")
        return 1
    logging.info("Agent transport: backend=%s model=%s", backend, model)

    return run_deploy(prompt, backend=backend, model=model, api_key=api_key)


if __name__ == "__main__":
    raise SystemExit(main())
