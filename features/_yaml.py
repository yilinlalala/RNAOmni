"""Minimal YAML loader for features/registry.yaml and features/active.yaml."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    return value


def _parse_scalar(value: str) -> Any:
    value = value.strip()
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
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)
    return value


def load_yaml(path: Path | str) -> Any:
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(raw)
    except ImportError:
        return _minimal_yaml_load(raw)


def _minimal_yaml_load(text: str) -> dict[str, Any]:
    """Restricted loader for recipe/registry shapes used in this repo."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        container = stack[-1][1]

        if line.startswith("- "):
            item_body = line[2:].strip()
            if not isinstance(container, list):
                raise ValueError(f"List item without list context: {line}")
            if ":" in item_body:
                key, val = item_body.split(":", 1)
                item: dict[str, Any] = {key.strip(): _parse_scalar(val)}
                container.append(item)
                stack.append((indent, item))
            else:
                container.append(_parse_scalar(item_body))
            continue

        if ":" not in line:
            raise ValueError(f"Cannot parse YAML line: {line}")

        key, val = line.split(":", 1)
        key = key.strip()
        val = val.strip()

        if not isinstance(container, dict):
            raise ValueError(f"Key under non-mapping context: {line}")

        if val == "":
            if key in ("features", "skills", "triggers", "agent_tunable", "deps"):
                new_obj: Any = []
            else:
                new_obj = {}
            container[key] = new_obj
            stack.append((indent, new_obj))
            continue

        container[key] = _parse_scalar(val)

    return root
