#!/usr/bin/env python3
"""Resolve skills from skill-library/registry.yaml (launcher-agnostic).

Used by orchestrator.py today; any future non-SDK launcher should call the same API.
Does not depend on .cursor/skills/ or the Cursor SDK.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = REPO_ROOT / "skill-library" / "registry.yaml"


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    return value


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in ("|", ">", "|-", ">-"):
        return ""
    value = _strip_quotes(value)
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in ("null", "~", ""):
        return None
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    return value


def _minimal_yaml_load(text: str) -> dict[str, Any]:
    """Load the restricted registry.yaml schema without PyYAML."""
    data: dict[str, Any] = {"skills": []}
    current: dict[str, Any] | None = None
    in_triggers = False
    in_folded_description = False
    folded_lines: list[str] = []

    def flush_folded() -> None:
        nonlocal in_folded_description, folded_lines, current
        if current is not None and in_folded_description:
            current["description"] = " ".join(folded_lines).strip()
        in_folded_description = False
        folded_lines = []

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()

        if in_folded_description:
            if indent >= 4 and not line.startswith("- "):
                folded_lines.append(line)
                continue
            flush_folded()

        if indent == 0 and line.startswith("version:"):
            data["version"] = _parse_scalar(line.split(":", 1)[1])
            current = None
            in_triggers = False
            continue

        if indent == 0 and line.startswith("skills:"):
            current = None
            in_triggers = False
            continue

        if indent == 2 and line.startswith("- "):
            flush_folded()
            current = {}
            data["skills"].append(current)
            in_triggers = False
            rest = line[2:].strip()
            if rest and ":" in rest:
                key, val = rest.split(":", 1)
                current[key.strip()] = _parse_scalar(val)
            continue

        if current is None:
            continue

        if in_triggers and indent >= 4 and line.startswith("- "):
            current.setdefault("triggers", []).append(_parse_scalar(line[2:]))
            continue

        if indent == 4 and ":" in line:
            in_triggers = False
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()
            if key == "triggers":
                current["triggers"] = []
                in_triggers = True
                continue
            if key == "description" and val in (">", ">-", "|", "|-"):
                in_folded_description = True
                folded_lines = []
                continue
            current[key] = _parse_scalar(val)

    flush_folded()
    return data


def load_registry(registry_path: Path | None = None) -> dict[str, Any]:
    path = registry_path or DEFAULT_REGISTRY
    if not path.is_file():
        raise FileNotFoundError(f"Skill registry not found: {path}")
    raw = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(raw)
    except ImportError:
        data = _minimal_yaml_load(raw)
    if not isinstance(data, dict) or "skills" not in data:
        raise ValueError(f"Invalid skill registry schema: {path}")
    return data


def resolve_skills(
    *,
    overnight: bool | None = None,
    enabled_only: bool = True,
    repo: Path | None = None,
    registry_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return skill entries from the registry, optionally filtered."""
    root = repo or REPO_ROOT
    data = load_registry(registry_path)
    out: list[dict[str, Any]] = []
    for entry in data.get("skills") or []:
        if not isinstance(entry, dict):
            continue
        if enabled_only and not entry.get("enabled", True):
            continue
        if overnight is not None and bool(entry.get("overnight", False)) != overnight:
            continue
        path = entry.get("path")
        if not path:
            raise ValueError(f"Skill entry missing path: {entry!r}")
        resolved = (root / path).resolve()
        item = dict(entry)
        item["resolved_path"] = resolved
        item["exists"] = resolved.is_file()
        out.append(item)
    return out


def format_prompt_block(
    skills: list[dict[str, Any]] | None = None,
    *,
    overnight: bool = True,
    require_exists: bool = True,
) -> str:
    """Build a prompt section that forces agents to read registered skills."""
    if skills is None:
        skills = resolve_skills(overnight=overnight)
    if require_exists:
        missing = [s["path"] for s in skills if not s.get("exists")]
        if missing:
            raise FileNotFoundError(
                "Registered skill files missing:\n  - " + "\n  - ".join(missing)
            )
    if not skills:
        return (
            "## Active skills (from skill-library/registry.yaml)\n"
            "No skills matched the current filter. Follow program.md only.\n"
        )
    lines = [
        "## Active skills (from skill-library/registry.yaml)",
        "Skill registration is repo-native. Do NOT rely on .cursor/skills/ discovery.",
        "Read each SKILL.md below before acting; follow its red lines.",
        "",
    ]
    for skill in skills:
        sid = skill.get("id", "?")
        path = skill.get("path", "?")
        desc = (skill.get("description") or "").strip()
        lines.append(f"- `{sid}`: read `{path}`")
        if desc:
            lines.append(f"  ({desc})")
    lines.append("")
    return "\n".join(lines)


def format_deploy_prompt_block(*, repo: Path | None = None) -> str:
    """Prompt block for deploy_feature.py (not overnight orchestrator)."""
    root = repo or REPO_ROOT
    paths = [
        "skill-library/feature/SKILL.md",
        "skill-library/feature/deploy-ops.md",
    ]
    missing = [p for p in paths if not (root / p).is_file()]
    if missing:
        raise FileNotFoundError(
            "Deploy skill files missing:\n  - " + "\n  - ".join(missing)
        )
    lines = [
        "## Deploy skills (human-triggered; not overnight)",
        "Read each file below before acting; follow deploy-ops red lines.",
        "",
    ]
    for path in paths:
        lines.append(f"- read `{path}`")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve RNAOmni skills from registry.yaml")
    parser.add_argument(
        "--overnight",
        action="store_true",
        help="Only skills with overnight: true",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="All enabled skills (ignore overnight filter)",
    )
    parser.add_argument(
        "--prompt",
        action="store_true",
        help="Print the prompt block for injection",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Override registry path",
    )
    args = parser.parse_args(argv)

    overnight: bool | None
    if args.all:
        overnight = None
    elif args.overnight:
        overnight = True
    else:
        overnight = True

    skills = resolve_skills(overnight=overnight, registry_path=args.registry)
    if args.prompt:
        print(format_prompt_block(skills, overnight=bool(overnight)))
        return 0

    for skill in skills:
        status = "ok" if skill["exists"] else "MISSING"
        print(f"{skill['id']}\t{skill['path']}\t{status}")
    return 0 if all(s["exists"] for s in skills) else 1


if __name__ == "__main__":
    raise SystemExit(main())
