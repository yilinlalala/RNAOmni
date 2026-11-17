"""Per-campaign data note (N0): deterministic data block + opening prior via isolated LLM."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Callable

from rnaomni.paths import DATA_CARD_DIR, NOTE_PATH, RESULTS_DIR
from rnaomni.orch.search import DEFAULT_FRAMEWORK, FRAMEWORK_MENU, SearchState

PRIOR_SUSPECTS: tuple[str, ...] = (
    "composition",
    "structure",
    "long_range",
    "sample_size",
)
_PRIOR_CLOSED = frozenset(PRIOR_SUSPECTS)
NEXT_KINDS: tuple[str, ...] = ("feature", "knob", "head", "sample", "crash")
_NEXT_CLOSED = frozenset(NEXT_KINDS)
KNOB_NAMES: tuple[str, ...] = (
    "HIDDEN_DIM",
    "LR",
    "BATCH_SIZE",
    "dropout",
    "loss",
    "EMA",
)
OPENING_PROMPT_FORBIDDEN: tuple[str, ...] = (
    "val_accuracy",
    "val_spearman",
    "best",
    "run.log",
    ".tsv",
)
NOTE_RENDER_MAX = 800
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


Completer = Callable[[str], Any]
HEAVY_VRAM_GB = 12.0


def _empty_recipe_snapshot() -> dict[str, Any]:
    return {
        "recipe_id": "",
        "fusion": "concat",
        "enabled": [],
        "registered": [],
        "tunable": {},
        "heavy": [],
        "light_l4": [],
    }


def feature_is_heavy(entry: dict[str, Any] | None) -> bool:
    """True when registry marks the feature too heavy to stack on a working L4."""
    if not isinstance(entry, dict):
        return False
    if bool(entry.get("heavy")):
        return True
    raw = entry.get("vram_gb")
    try:
        if raw is not None and float(raw) >= HEAVY_VRAM_GB:
            return True
    except (TypeError, ValueError):
        return False
    return False


def empty_note() -> dict[str, Any]:
    return {
        "data": {},
        "prior": {"suspect": [], "reason": ""},
        "F": {"status": "untried", "evidence": []},
        "H": {"shots": []},
        "knob": {"status": "untried", "evidence": []},
        "S": {"sample_short": None, "last_kept": None},
        "crash_events": [],
        "failed_features": [],
        "failed_knobs": [],
        "next": None,
    }


def load_note(path: Any | None = None) -> dict[str, Any]:
    target = path if path is not None else NOTE_PATH
    target = target if hasattr(target, "read_text") else NOTE_PATH
    if not target.is_file():
        return empty_note()
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        logging.warning("Ignoring corrupt campaign note: %s", err)
        return empty_note()
    if not isinstance(raw, dict):
        return empty_note()
    note = empty_note()
    note.update({k: raw[k] for k in note if k in raw})
    return note


def save_note(note: dict[str, Any], path: Any | None = None) -> None:
    target = path if path is not None else NOTE_PATH
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(note, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _recipe_snapshot() -> dict[str, Any]:
    try:
        from features.pipeline import load_active_recipe, load_feature_registry
    except Exception as err:
        logging.warning("Campaign note: recipe load failed: %s", err)
        return _empty_recipe_snapshot()
    try:
        recipe = load_active_recipe()
        registry = load_feature_registry()
    except Exception as err:
        logging.warning("Campaign note: recipe/registry unreadable: %s", err)
        return _empty_recipe_snapshot()
    enabled: list[str] = []
    for item in recipe.get("features") or []:
        if isinstance(item, dict) and item.get("enabled") and item.get("name"):
            enabled.append(str(item["name"]))
    registered: list[str] = []
    tunable: dict[str, list[str]] = {}
    heavy: list[str] = []
    light_l4: list[str] = []
    for entry in registry.get("features") or []:
        if not isinstance(entry, dict) or not entry.get("name"):
            continue
        name = str(entry["name"])
        registered.append(name)
        keys = [str(x) for x in (entry.get("agent_tunable") or [])]
        if keys:
            tunable[name] = keys
        if feature_is_heavy(entry):
            heavy.append(name)
        elif str(entry.get("layer") or "") == "L4":
            light_l4.append(name)
    return {
        "recipe_id": str(recipe.get("recipe_id") or ""),
        "fusion": str(recipe.get("fusion") or "concat"),
        "enabled": enabled,
        "registered": registered,
        "tunable": tunable,
        "heavy": heavy,
        "light_l4": light_l4,
    }


def _data_card_snapshot() -> dict[str, Any]:
    try:
        from rnaomni.results_io import active_results_stem
    except Exception:
        return {}
    path = DATA_CARD_DIR / f"{active_results_stem()}.json"
    if not path.is_file():
        return {}
    try:
        card = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(card, dict):
        return {}
    seq = card.get("seq_length") if isinstance(card.get("seq_length"), dict) else {}
    label = card.get("label") if isinstance(card.get("label"), dict) else {}
    out: dict[str, Any] = {
        "n_rows": card.get("n_rows"),
        "csv_name": card.get("csv_name"),
        "seq_len_min": seq.get("min"),
        "seq_len_max": seq.get("max"),
        "seq_len_median": seq.get("median"),
    }
    if label.get("num_classes") is not None:
        out["num_classes"] = label.get("num_classes")
    return {k: v for k, v in out.items() if v is not None}


def collect_data_block(
    *,
    task_type: str,
    state: SearchState | None = None,
) -> dict[str, Any]:
    """Deterministic snapshot. No decision-metric scores, no TSV rows."""
    from prepare import data_mode

    recipe = _recipe_snapshot()
    card = _data_card_snapshot()
    active = DEFAULT_FRAMEWORK
    tried: list[str] = [DEFAULT_FRAMEWORK]
    remaining_n = len(FRAMEWORK_MENU)
    k_train = None
    k_eval = None
    mode = data_mode()
    if state is not None:
        active = state.active_framework or DEFAULT_FRAMEWORK
        tried = list(state.frameworks_tried or [active])
        remaining_n = len(state.remaining_frameworks())
        k_train = state.k_train
        k_eval = state.k_eval
        mode = state.data_mode or mode
    block: dict[str, Any] = {
        "task_type": task_type,
        "data_mode": mode,
        "k_train": k_train,
        "k_eval": k_eval,
        "active_framework": active,
        "frameworks_tried": tried,
        "remaining_heads": remaining_n,
        "recipe_id": recipe["recipe_id"],
        "fusion": recipe["fusion"],
        "enabled": list(recipe["enabled"]),
        "registered": list(recipe["registered"]),
    }
    block.update(card)
    return block


def _normalize_exclude(raw: list[dict[str, str]] | None) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "").strip().lower()
        name = str(item.get("name") or "").strip()
        if kind not in _NEXT_CLOSED or kind == "crash" or not name:
            continue
        key = (kind, name)
        if key in seen:
            continue
        seen.add(key)
        out.append({"kind": kind, "name": name})
    return out


def _registered_names_in_text(text: str, registered: list[str]) -> list[str]:
    found: list[str] = []
    desc = re.sub(r"[\t\r\n]+", " ", text or "")
    for name in registered:
        token = str(name)
        if token and re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])", desc
        ):
            found.append(token)
    return found


def _shot_name_for_evidence(note: dict[str, Any], cause: str) -> str:
    nxt = note.get("next") if isinstance(note.get("next"), dict) else None
    if not nxt or not nxt.get("name"):
        return ""
    kind = str(nxt.get("kind") or "").strip().lower()
    name = str(nxt["name"]).strip()
    if cause == "crash":
        return name
    if kind == cause:
        return name
    return ""


def _shot_kind_for_evidence(note: dict[str, Any], cause: str) -> str:
    nxt = note.get("next") if isinstance(note.get("next"), dict) else None
    if cause == "crash" and nxt:
        kind = str(nxt.get("kind") or "").strip().lower()
        if kind in _NEXT_CLOSED and kind != "crash":
            return kind
    if cause in _NEXT_CLOSED and cause != "crash":
        return cause
    return ""


def _remember_failed_action(
    note: dict[str, Any],
    *,
    kind: str,
    name: str,
    framework: str = "",
) -> None:
    token = str(name or "").strip()
    if not token:
        return
    if kind == "feature":
        lst = [str(x) for x in (note.get("failed_features") or []) if str(x).strip()]
        if token not in lst:
            lst.append(token)
        note["failed_features"] = lst
        return
    if kind == "knob":
        items: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for item in note.get("failed_knobs") or []:
            if not isinstance(item, dict):
                continue
            n = str(item.get("name") or "").strip()
            fw = str(item.get("framework") or "").strip()
            if not n or (fw, n) in seen:
                continue
            seen.add((fw, n))
            items.append({"name": n, "framework": fw})
        key = (str(framework or "").strip(), token)
        if key not in seen:
            items.append({"name": token, "framework": key[0]})
        note["failed_knobs"] = items


def _forget_failed_action(
    note: dict[str, Any],
    *,
    kind: str,
    name: str,
    framework: str = "",
) -> None:
    token = str(name or "").strip()
    if not token:
        return
    if kind == "feature":
        note["failed_features"] = [
            str(x) for x in (note.get("failed_features") or []) if str(x).strip() != token
        ]
        return
    if kind == "knob":
        fw = str(framework or "").strip()
        kept: list[dict[str, str]] = []
        for item in note.get("failed_knobs") or []:
            if not isinstance(item, dict):
                continue
            n = str(item.get("name") or "").strip()
            item_fw = str(item.get("framework") or "").strip()
            if n == token and item_fw == fw:
                continue
            if n:
                kept.append({"name": n, "framework": item_fw})
        note["failed_knobs"] = kept


def _failed_feature_exclude(
    note: dict[str, Any] | None,
    *,
    enabled: set[str],
    registered: list[str],
) -> list[dict[str, str]]:
    """Discarded/crashed features that are not currently enabled stay illegal."""
    if not note:
        return []
    raw: list[dict[str, str]] = []
    for name in note.get("failed_features") or []:
        token = str(name).strip()
        if token and token not in enabled:
            raw.append({"kind": "feature", "name": token})
    rows: list[dict[str, Any]] = []
    cell = note.get("F") if isinstance(note.get("F"), dict) else {}
    for ev in cell.get("evidence") or []:
        if isinstance(ev, dict):
            rows.append(ev)
    for ev in note.get("crash_events") or []:
        if isinstance(ev, dict):
            rows.append(ev)
    for ev in rows:
        result = str(ev.get("result") or "").strip().lower()
        if result not in {"discard", "crash"}:
            continue
        name = str(ev.get("name") or "").strip()
        if not name:
            mentioned = _registered_names_in_text(
                str(ev.get("desc") or ""), registered
            )
            name = next((n for n in mentioned if n not in enabled), "")
            if not name and mentioned:
                name = mentioned[0]
        if name and name not in enabled:
            raw.append({"kind": "feature", "name": name})
    return _normalize_exclude(raw)


def _failed_knob_exclude(
    note: dict[str, Any] | None,
    *,
    active_framework: str,
) -> list[dict[str, str]]:
    """Knobs discarded/crashed on this head stay illegal; other heads may retry."""
    if not note:
        return []
    active = str(active_framework or "").strip()
    raw: list[dict[str, str]] = []
    for item in note.get("failed_knobs") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        fw = str(item.get("framework") or "").strip()
        if name and fw == active:
            raw.append({"kind": "knob", "name": name})
    cell = note.get("knob") if isinstance(note.get("knob"), dict) else {}
    for ev in cell.get("evidence") or []:
        if not isinstance(ev, dict):
            continue
        result = str(ev.get("result") or "").strip().lower()
        if result not in {"discard", "crash"}:
            continue
        name = str(ev.get("name") or "").strip()
        fw = str(ev.get("framework") or "").strip()
        if name and fw == active:
            raw.append({"kind": "knob", "name": name})
    return _normalize_exclude(raw)


def _heavy_feature_exclude(
    recipe: dict[str, Any],
    *,
    data_mode: str,
    enabled: set[str],
) -> list[dict[str, str]]:
    """Full + a light L4 already on → evo/evo2 (heavy) cannot be MUST."""
    if str(data_mode or "") != "full":
        return []
    light = [str(x) for x in (recipe.get("light_l4") or [])]
    if not any(name in enabled for name in light):
        return []
    raw: list[dict[str, str]] = []
    for name in recipe.get("heavy") or []:
        token = str(name).strip()
        if token and token not in enabled:
            raw.append({"kind": "feature", "name": token})
    return _normalize_exclude(raw)


def collect_legal_actions(
    state: SearchState | None = None,
    *,
    exclude: list[dict[str, str]] | None = None,
    note: dict[str, Any] | None = None,
) -> dict[str, Any]:
    recipe = _recipe_snapshot()
    remaining = list(FRAMEWORK_MENU)
    can_grow = False
    suppressed: list[str] = []
    valid_rounds = 0
    if state is not None:
        remaining = state.remaining_frameworks()
        can_grow = bool(state.can_grow_sample())
        suppressed = [str(x) for x in (getattr(state, "suppressed", None) or [])]
        valid_rounds = int(getattr(state, "valid_rounds", 0) or 0)
    head_allowed = bool(remaining) and "head" not in suppressed and valid_rounds > 0
    enabled = [str(x) for x in (recipe.get("enabled") or [])]
    enabled_set = set(enabled)
    tunable = dict(recipe.get("tunable") or {})
    registered = list(recipe.get("registered") or [])
    # Already-on features with no agent_tunable are a no-op if MUST again.
    actionable = [n for n in registered if n not in enabled_set or n in tunable]
    failed = _failed_feature_exclude(
        note, enabled=enabled_set, registered=registered
    )
    active_fw = DEFAULT_FRAMEWORK
    mode = "full"
    if state is not None:
        active_fw = str(getattr(state, "active_framework", None) or DEFAULT_FRAMEWORK)
        mode = str(getattr(state, "data_mode", None) or mode)
    else:
        try:
            from prepare import data_mode as _data_mode

            mode = str(_data_mode() or mode)
        except Exception:
            pass
    knobs_failed = _failed_knob_exclude(note, active_framework=active_fw)
    heavy = _heavy_feature_exclude(recipe, data_mode=mode, enabled=enabled_set)
    heavy_names = {str(item.get("name")) for item in heavy}
    if heavy_names:
        actionable = [n for n in actionable if n not in heavy_names]
    merged = _normalize_exclude(
        list(exclude or []) + failed + knobs_failed + heavy
    )
    return {
        "features": registered,
        "features_actionable": actionable,
        "enabled": enabled,
        "tunable": tunable,
        "knobs": list(KNOB_NAMES),
        "heads_remaining": remaining,
        "sample_expand": can_grow,
        "head_allowed": head_allowed,
        "exclude": merged,
    }


def build_opening_prompt(data: dict[str, Any], legal: dict[str, Any]) -> str:
    """Opening diagnosis: data + legal actions only. Must not leak scores."""
    payload = {"data": data, "legal": legal}
    body = json.dumps(payload, ensure_ascii=False, indent=2)
    return (
        "You diagnose one RNAOmni overnight campaign. Reply with JSON only. "
        "Do not use tools. Do not edit files. Do not ask questions.\n"
        "You see dataset traits and the legal action menu. You do not see "
        "training scores or previous experiment rows.\n"
        "prior.suspect: zero or more of composition, structure, long_range, sample_size.\n"
        "Opening next.kind: feature | knob | sample (not head — first do(H) waits "
        "until after a keep/discard). next.name must be on the legal list "
        "(feature=legal.features_actionable: off, or on with agent_tunable; "
        "knob=HIDDEN_DIM|LR|BATCH_SIZE|dropout|loss|EMA). "
        "Do not name an already-enabled feature that has no tunable. "
        "One concrete action, not 'try something else'.\n"
        "Schema:\n"
        '{"prior":{"suspect":["structure"],"reason":"one short sentence"},'
        '"next":{"kind":"feature","name":"struct_thermo","why":"ties to prior"}}\n'
        f"Input:\n{body}\n"
    )


def opening_prompt_leaks(prompt: str) -> list[str]:
    lower = prompt.lower()
    return [tok for tok in OPENING_PROMPT_FORBIDDEN if tok.lower() in lower]


def extract_json_object(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    fence = _JSON_FENCE_RE.search(raw)
    if fence:
        raw = fence.group(1)
    else:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return None
        raw = raw[start : end + 1]
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def parse_prior(blob: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {"suspect": [], "reason": ""}
    if not blob:
        return out
    prior = blob.get("prior") if isinstance(blob.get("prior"), dict) else blob
    if not isinstance(prior, dict):
        return out
    suspects: list[str] = []
    raw_s = prior.get("suspect") or prior.get("suspects") or []
    if isinstance(raw_s, str):
        raw_s = [raw_s]
    if isinstance(raw_s, list):
        for item in raw_s:
            token = str(item).strip().lower().replace("-", "_").replace(" ", "_")
            aliases = {
                "composition": "composition",
                "structure": "structure",
                "long_range": "long_range",
                "longrange": "long_range",
                "sample_size": "sample_size",
                "sample": "sample_size",
            }
            token = aliases.get(str(item).strip(), token)
            if token in _PRIOR_CLOSED and token not in suspects:
                suspects.append(token)
    out["suspect"] = suspects
    reason = str(prior.get("reason") or "").strip().replace("\n", " ")
    if len(reason) > 160:
        reason = reason[:159] + "…"
    out["reason"] = reason
    return out


def validate_next(
    blob: dict[str, Any] | None,
    legal: dict[str, Any],
) -> dict[str, Any] | None:
    if not blob:
        return None
    nxt = blob.get("next") if isinstance(blob.get("next"), dict) else None
    if not isinstance(nxt, dict):
        return None
    kind = str(nxt.get("kind") or "").strip().lower()
    if kind not in _NEXT_CLOSED or kind == "crash":
        return None
    name = str(nxt.get("name") or "").strip()
    why = str(nxt.get("why") or "").strip().replace("\n", " ")
    if len(why) > 160:
        why = why[:159] + "…"
    if "features_actionable" in legal:
        features = {str(x) for x in (legal.get("features_actionable") or [])}
    else:
        features = {str(x) for x in (legal.get("features") or [])}
    knobs = {str(x) for x in (legal.get("knobs") or [])}
    heads = {str(x) for x in (legal.get("heads_remaining") or [])}
    for item in legal.get("exclude") or []:
        if isinstance(item, dict) and str(item.get("kind")) == kind and str(item.get("name")) == name:
            return None
    if kind == "feature":
        if name not in features:
            return None
    elif kind == "knob":
        if name not in knobs:
            return None
    elif kind == "head":
        if name not in heads:
            return None
    elif kind == "sample":
        if not legal.get("sample_expand"):
            return None
        name = name or "expand"
    return {"kind": kind, "name": name, "why": why}


def n1_executable_next(
    nxt: dict[str, Any] | None,
    legal: dict[str, Any],
) -> dict[str, Any] | None:
    """N1: feature/knob/sample only. head is ignored (not a draw)."""
    if not nxt:
        return None
    if str(nxt.get("kind") or "") == "head":
        return None
    return validate_next({"next": nxt}, legal)


def n2_executable_next(
    nxt: dict[str, Any] | None,
    legal: dict[str, Any],
) -> dict[str, Any] | None:
    """N2: N1 actions plus head when legal.head_allowed (remaining, not suppressed, after first keep/discard)."""
    if not nxt:
        return None
    if str(nxt.get("kind") or "") == "head":
        if not legal.get("head_allowed"):
            return None
        return validate_next({"next": nxt}, legal)
    return n1_executable_next(nxt, legal)


def legal_note_head_name(
    state: SearchState | None,
    note: dict[str, Any] | None = None,
) -> str | None:
    """Name to inject as next_framework, or None (opening / suppressed / illegal)."""
    payload = note if note is not None else load_note()
    legal = collect_legal_actions(state, note=payload)
    nxt = payload.get("next") if isinstance(payload.get("next"), dict) else None
    got = n2_executable_next(nxt, legal)
    if got and got.get("kind") == "head" and got.get("name"):
        return str(got["name"])
    return None


def release_crash_repair(
    state: SearchState | None = None,
    *,
    path: Any | None = None,
) -> dict[str, Any]:
    """After an empty crash-repair commit: drop the crashed name, keep a still-legal next."""
    note = load_note(path)
    events = note.get("crash_events") or []
    last = events[-1] if events and isinstance(events[-1], dict) else {}
    crashed = str(last.get("name") or "").strip()
    crashed_kind = str(last.get("kind") or "").strip()
    if crashed_kind not in {"feature", "knob"}:
        crashed_kind = "knob" if crashed in KNOB_NAMES else "feature"
    if crashed:
        _remember_failed_action(
            note,
            kind=crashed_kind,
            name=crashed,
            framework=str(last.get("framework") or "")
            or (
                str(getattr(state, "active_framework", "") or "")
                if state is not None
                else ""
            ),
        )
    legal = collect_legal_actions(state, note=note)
    note["next"] = retain_legal_next(note.get("next"), legal, n1=True)
    try:
        save_note(note, path)
    except OSError as err:
        logging.warning("Could not write campaign note after crash release: %s", err)
    logging.info(
        "Crash veto released; next=%s failed_features=%s",
        note.get("next"),
        note.get("failed_features"),
    )
    return note


def retain_legal_next(
    previous: dict[str, Any] | None,
    legal: dict[str, Any],
    *,
    n1: bool = True,
) -> dict[str, Any] | None:
    """Fallback A: keep last legal next; else None (no MUST). Head only if N2-allowed."""
    if n1:
        return n2_executable_next(previous, legal)
    return validate_next({"next": previous} if previous else None, legal)


def n1_must_action(
    note: dict[str, Any] | None = None,
    state: SearchState | None = None,
) -> dict[str, Any] | None:
    """What Iter MUST this round. crash / sample / next_framework veto the note.

    Returns kind/name/why plus veto=crash|sample|head|bottleneck or empty.
    None means no MUST (agent may choose feature or knob).
    """
    payload = note if note is not None else load_note()
    legal = collect_legal_actions(state, note=payload)
    suggested = n1_executable_next(
        payload.get("next") if isinstance(payload.get("next"), dict) else None,
        legal,
    )
    if state is not None:
        token = str(state.next_do_token() or "")
        if token == "crash":
            out: dict[str, Any] = {
                "kind": "crash",
                "name": "",
                "why": "crash block; one repair, do not switch layers",
                "veto": "crash",
            }
            if suggested:
                out["after"] = suggested
            return out
        if token == "sample":
            return {
                "kind": "sample",
                "name": "expand",
                "why": "orchestrator sample expand (no agent)",
                "veto": "sample",
            }
        if token == "head":
            name = str(getattr(state, "next_framework", "") or "")
            return {
                "kind": "head",
                "name": name,
                "why": "next_framework",
                "veto": "head",
            }
        if token == "feature":
            out = {
                "kind": token,
                "name": "",
                "why": f"bottleneck next_do={token}",
                "veto": "bottleneck",
            }
            if suggested and suggested.get("kind") == token:
                out["name"] = str(suggested.get("name") or "")
                out["why"] = str(suggested.get("why") or out["why"])
            return out
    if suggested:
        return {**suggested, "veto": ""}
    return None


def format_must_line(action: dict[str, Any] | None) -> str:
    if not action:
        return "MUST this round: (none — choose [feature] or [model] stay / cause=knob)"
    kind = str(action.get("kind") or "")
    name = str(action.get("name") or "")
    why = str(action.get("why") or "")
    veto = str(action.get("veto") or "")
    if kind == "crash":
        after = action.get("after") if isinstance(action.get("after"), dict) else None
        after_s = ""
        if after:
            after_s = f" Note next after repair: {after.get('kind')} {after.get('name')}"
        return (
            "MUST this round: [crash] one repair of the short signature "
            "(do not switch feature/head/layer)"
            f"{after_s}"
        )
    if kind == "sample":
        return "MUST this round: [sample] expand (orchestrator; no agent)"
    if kind == "head":
        return f"MUST this round: [model] next_framework={name or '(set)'} framework_base"
    if kind == "feature":
        tag = f"[feature] {name}".strip() if name else "[feature]"
    elif kind == "knob":
        tag = f"[model] stay knob={name}" if name else "[model] stay (cause=knob)"
    else:
        tag = f"{kind} {name}".strip()
    extra = " (bottleneck)" if veto == "bottleneck" else ""
    line = f"MUST this round: {tag}{extra}"
    if why:
        line = f"{line} — {why}"
    return line


def append_round_evidence(
    note: dict[str, Any],
    *,
    cause: str,
    result: str,
    delta: float | None = None,
    description: str = "",
    crash_digest: str = "",
    framework: str = "",
) -> None:
    """Deterministic grid update. Does not rewrite other layers' evidence."""
    desc = re.sub(r"[\t\r\n]+", " ", description or "").strip()[:80]
    digest = re.sub(r"[\t\r\n]+", " ", crash_digest or "").strip()[:120]
    digest = digest.replace("run.log", "run_log")
    shot_kind = _shot_kind_for_evidence(note, cause)
    shot_name = _shot_name_for_evidence(note, cause)
    if not shot_name and cause in {"feature", "crash"}:
        recipe = _recipe_snapshot()
        mentioned = _registered_names_in_text(desc, list(recipe.get("registered") or []))
        shot_name = mentioned[0] if mentioned else ""
        if not shot_kind and mentioned:
            shot_kind = "feature"
    if not shot_name and cause == "knob":
        for knob in KNOB_NAMES:
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(knob)}(?![A-Za-z0-9_])", desc):
                shot_name = knob
                shot_kind = "knob"
                break
    active_fw = str(framework or "").strip()
    if not active_fw:
        data = note.get("data") if isinstance(note.get("data"), dict) else {}
        active_fw = str(data.get("active_framework") or "").strip()
    row: dict[str, Any] = {
        "cause": cause,
        "result": result,
        "delta": delta,
        "desc": desc,
    }
    if shot_name:
        row["name"] = shot_name
    if shot_kind:
        row["kind"] = shot_kind
    if active_fw and (cause == "knob" or shot_kind == "knob"):
        row["framework"] = active_fw
    if digest:
        row["crash"] = digest

    if result in {"discard", "crash"} and shot_name:
        _remember_failed_action(
            note,
            kind=shot_kind or ("feature" if cause == "crash" else cause),
            name=shot_name,
            framework=active_fw,
        )
    elif result == "keep" and shot_name:
        _forget_failed_action(
            note,
            kind=shot_kind or cause,
            name=shot_name,
            framework=active_fw,
        )

    def _push(cell: dict[str, Any], key: str = "evidence") -> None:
        hist = list(cell.get(key) or [])
        hist.append(row)
        cell[key] = hist[-5:]

    if cause == "feature":
        cell = note.setdefault("F", {"status": "untried", "evidence": []})
        _push(cell)
        if result == "keep":
            cell["status"] = "tried_lifted"
        elif result in {"discard", "flat"}:
            cell["status"] = "tried_no_lift"
        return
    if cause == "knob":
        cell = note.setdefault("knob", {"status": "untried", "evidence": []})
        _push(cell)
        if result == "keep":
            cell["status"] = "tried_lifted"
        elif result in {"discard", "flat"}:
            cell["status"] = "tried_no_lift"
        return
    if cause == "head":
        cell = note.setdefault("H", {"shots": []})
        _push(cell, key="shots")
        return
    if cause == "sample":
        cell = note.setdefault("S", {"sample_short": None, "last_kept": None, "evidence": []})
        _push(cell)
        cell["last_kept"] = result == "keep"
        return
    if cause == "crash":
        hist = list(note.get("crash_events") or [])
        hist.append(row)
        note["crash_events"] = hist[-5:]


def build_revision_prompt(
    note: dict[str, Any],
    legal: dict[str, Any],
    *,
    cause: str,
    result: str,
    delta: float | None = None,
    description: str = "",
    crash_digest: str = "",
) -> str:
    """After one keep/discard/crash. No run.log. May include this round's delta."""
    grid = {
        "data": note.get("data") or {},
        "prior": note.get("prior") or {},
        "F": note.get("F") or {},
        "H": note.get("H") or {},
        "knob": note.get("knob") or {},
        "S": note.get("S") or {},
        "crash_events": note.get("crash_events") or [],
        "current_next": note.get("next"),
    }
    event = {
        "cause": cause,
        "result": result,
        "delta": delta,
        "description": re.sub(r"[\t\r\n]+", " ", description or "").strip()[:120],
        "crash_digest": re.sub(r"[\t\r\n]+", " ", crash_digest or "").strip()[:120],
    }
    event["crash_digest"] = event["crash_digest"].replace("run.log", "run_log")
    payload = {"grid": grid, "this_round": event, "legal": legal}
    body = json.dumps(payload, ensure_ascii=False, indent=2)
    crash_rule = ""
    if cause == "crash":
        crash_rule = (
            " Last round crashed. The crash prompt block still vetoes switching "
            "layers this Iter: the agent MUST make one repair, not skip to a new "
            "feature/head. Your next is for AFTER that repair: a different "
            "concrete feature|knob|sample|head than the crashed name. Do not "
            "repeat the crashed name."
        )
    head_rule = (
        " next.kind may be feature|knob|sample|head. head.name must be in "
        "legal.heads_remaining. Do not pick a tried / off-menu name."
        if legal.get("head_allowed")
        else " next.kind must be feature|knob|sample (not head)."
    )
    return (
        "You update one RNAOmni campaign note after a single experiment. "
        "Reply with JSON only. Do not use tools. Do not edit files.\n"
        "Only propose next. Do not rewrite prior or other layers' evidence."
        f"{head_rule} "
        "next.name must be on legal.features_actionable (or knobs / "
        "heads_remaining / sample_expand). Do not repeat this_round's "
        "kind+name (listed in legal.exclude). Do not name a feature that is "
        "already in legal.enabled unless it still has agent_tunable. "
        "One concrete different action. If nothing new is legal, reply "
        '{"next":null}.'
        f"{crash_rule}\n"
        "Schema: "
        '{"next":{"kind":"feature","name":"rna_fm","why":"different from last shot"}}\n'
        f"Input:\n{body}\n"
    )


def apply_revision_reply(
    note: dict[str, Any],
    parsed: dict[str, Any] | None,
    legal: dict[str, Any],
) -> dict[str, Any] | None:
    """Set note['next'] from LLM; illegal → fallback A (keep last legal)."""
    previous = note.get("next") if isinstance(note.get("next"), dict) else None
    proposed = n2_executable_next(
        (parsed or {}).get("next") if isinstance(parsed, dict) else None,
        legal,
    )
    if proposed is None and parsed:
        proposed = n2_executable_next(parsed if "kind" in parsed else None, legal)
    if proposed is not None:
        note["next"] = proposed
        return proposed
    kept = retain_legal_next(previous, legal, n1=True)
    note["next"] = kept
    return kept


def _just_tried_exclude(
    note: dict[str, Any],
    *,
    cause: str,
    description: str = "",
) -> list[dict[str, str]]:
    """The shot that just finished must not be MUST again."""
    raw: list[dict[str, str]] = []
    nxt = note.get("next") if isinstance(note.get("next"), dict) else None
    if nxt and nxt.get("name"):
        kind = str(nxt.get("kind") or cause).strip().lower()
        raw.append({"kind": kind, "name": str(nxt["name"])})
    desc = re.sub(r"[\t\r\n]+", " ", description or "")
    recipe = _recipe_snapshot()
    for name in recipe.get("registered") or []:
        token = str(name)
        if token and re.search(rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])", desc):
            raw.append({"kind": "feature" if cause == "feature" else cause, "name": token})
    if cause == "knob":
        for knob in KNOB_NAMES:
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(knob)}(?![A-Za-z0-9_])", desc):
                raw.append({"kind": "knob", "name": knob})
    return _normalize_exclude(raw)


def _refresh_note_data(note: dict[str, Any], state: SearchState | None) -> None:
    data = note.get("data") if isinstance(note.get("data"), dict) else {}
    task_type = str(data.get("task_type") or "")
    if not task_type:
        return
    try:
        note["data"] = collect_data_block(task_type=task_type, state=state)
    except Exception as err:
        logging.warning("Campaign note data refresh failed: %s", err)


def revise_after_round(
    note: dict[str, Any],
    *,
    cause: str,
    result: str,
    delta: float | None = None,
    description: str = "",
    crash_digest: str = "",
    state: SearchState | None = None,
    completer: Completer | None = None,
    backend: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    call_llm: bool = True,
) -> dict[str, Any]:
    """Append evidence, then isolated LLM rewrite of next. Failure keeps last legal next."""
    exclude = _just_tried_exclude(note, cause=cause, description=description)
    append_round_evidence(
        note,
        cause=cause,
        result=result,
        delta=delta,
        description=description,
        crash_digest=crash_digest,
        framework=str(getattr(state, "active_framework", "") or "") if state is not None else "",
    )
    _refresh_note_data(note, state)
    legal = collect_legal_actions(state, exclude=exclude, note=note)
    if not call_llm:
        note["next"] = retain_legal_next(note.get("next"), legal, n1=True)
        return note
    prompt = build_revision_prompt(
        note,
        legal,
        cause=cause,
        result=result,
        delta=delta,
        description=description,
        crash_digest=crash_digest,
    )
    if "run.log" in prompt.lower():
        prompt = prompt.replace("run.log", "run_log").replace("RUN.LOG", "run_log")
    try:
        if completer is not None:
            raw = completer(prompt)
        else:
            raw = _default_completer(
                prompt, backend=backend, model=model, api_key=api_key
            )
        text = _result_text(raw)
        parsed = extract_json_object(text)
        apply_revision_reply(note, parsed, legal)
    except Exception as err:
        logging.warning("Revision note LLM failed (keep last legal next): %s", err)
        note["next"] = retain_legal_next(note.get("next"), legal, n1=True)
    return note


def _crash_digest_from_row(row: dict[str, Any] | None) -> str:
    if not row:
        return ""
    desc = str(row.get("description") or "")
    from rnaomni.orch.tsv import CRASH_MARK

    if CRASH_MARK in desc:
        return desc.split(CRASH_MARK, 1)[1].strip()
    return ""


def revise_campaign_after_status(
    state: SearchState | None,
    *,
    status: str,
    backend: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    completer: Completer | None = None,
    path: Any | None = None,
    call_llm: bool = True,
) -> dict[str, Any]:
    """Keep/discard/crash: append evidence + isolated LLM rewrite of next. Never raises."""
    note = load_note(path)
    if status not in {"keep", "discard", "crash"}:
        return note
    from rnaomni.orch.search import infer_cause, parse_round_kind
    from rnaomni.orch.tsv import latest_results_row

    row = latest_results_row()
    desc = str((row or {}).get("description") or "")
    parsed = parse_round_kind(desc)
    cause = str(getattr(state, "last_do", None) or "") if state is not None else ""
    if status == "crash":
        cause = "crash"
    elif cause not in {"head", "feature", "knob", "sample", "crash"}:
        cause = infer_cause(parsed, status)
    result = str(getattr(state, "last_result", None) or status) if state is not None else status
    if status == "crash":
        result = "crash"
    delta = None
    if status != "crash" and state is not None:
        delta = getattr(state, "last_delta", None)
    digest = _crash_digest_from_row(row) if status == "crash" else ""
    try:
        note = revise_after_round(
            note,
            cause=cause or "knob",
            result=result or status,
            delta=delta,
            description=desc,
            crash_digest=digest,
            state=state,
            completer=completer,
            backend=backend,
            model=model,
            api_key=api_key,
            call_llm=call_llm,
        )
        save_note(note, path)
    except Exception as err:
        logging.warning("Campaign note revision failed (search continues): %s", err)
        try:
            legal = collect_legal_actions(state, note=note)
            note["next"] = retain_legal_next(note.get("next"), legal, n1=True)
            save_note(note, path)
        except Exception:
            pass
    logging.info(
        "Campaign note revised: cause=%s result=%s next=%s",
        cause,
        result,
        note.get("next"),
    )
    return note


def apply_sample_expand_to_note(
    state: SearchState | None,
    *,
    status: str,
    path: Any | None = None,
) -> dict[str, Any]:
    """Orchestrator-only S-cell update. No LLM. Re-validates next (fallback A)."""
    note = load_note(path)
    from rnaomni.orch.tsv import latest_results_row

    row = latest_results_row()
    desc = str((row or {}).get("description") or "")
    result = status if status in {"keep", "discard", "crash"} else "discard"
    delta = getattr(state, "last_delta", None) if state is not None else None
    if result == "crash":
        delta = None
    append_round_evidence(
        note,
        cause="sample",
        result=result,
        delta=delta,
        description=desc,
        crash_digest=_crash_digest_from_row(row) if result == "crash" else "",
    )
    task_type = "classification"
    data = note.get("data") if isinstance(note.get("data"), dict) else {}
    if data.get("task_type"):
        task_type = str(data["task_type"])
    note["data"] = collect_data_block(task_type=task_type, state=state)
    legal = collect_legal_actions(state, note=note)
    note["next"] = retain_legal_next(note.get("next"), legal, n1=True)
    try:
        save_note(note, path)
    except OSError as err:
        logging.warning("Could not write campaign note after sample expand: %s", err)
    logging.info(
        "Campaign note S-cell updated (no LLM): result=%s next=%s",
        result,
        note.get("next"),
    )
    return note


def format_note_block(
    note: dict[str, Any] | None = None,
    *,
    state: SearchState | None = None,
) -> str:
    """Short Iter injection. N1: MUST the note next unless a veto wins. Never dumps run.log."""
    payload = note if note is not None else load_note()
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    prior = payload.get("prior") if isinstance(payload.get("prior"), dict) else {}
    nxt = payload.get("next") if isinstance(payload.get("next"), dict) else None
    enabled = data.get("enabled") or []
    if isinstance(enabled, list):
        enabled_s = ",".join(str(x) for x in enabled) or "-"
    else:
        enabled_s = str(enabled)
    suspects = prior.get("suspect") or []
    if isinstance(suspects, list):
        sus_s = ",".join(str(x) for x in suspects) or "-"
    else:
        sus_s = str(suspects) or "-"
    reason = str(prior.get("reason") or "").strip()
    seq = ""
    if data.get("seq_len_median") is not None:
        seq = (
            f" seq_len={data.get('seq_len_min')}-{data.get('seq_len_max')}"
            f" med={data.get('seq_len_median')}"
        )
    n_rows = data.get("n_rows")
    n_s = f" n={n_rows}" if n_rows is not None else ""
    k_s = ""
    if data.get("k_train") is not None:
        k_s = f" K_train={data.get('k_train')}"
    next_s = "(none)"
    if nxt:
        next_s = f"{nxt.get('kind')} {nxt.get('name') or ''}".strip()
        if nxt.get("why"):
            next_s = f"{next_s} — {nxt['why']}"
    must = n1_must_action(payload, state)
    lines = [
        "Campaign note (MUST unless crash / next_framework veto):",
        (
            f"data: task={data.get('task_type') or '-'} mode={data.get('data_mode') or '-'}"
            f"{n_s}{seq} fusion={data.get('fusion') or '-'} enabled={enabled_s} "
            f"head={data.get('active_framework') or '-'} "
            f"remaining_heads={data.get('remaining_heads') if data.get('remaining_heads') is not None else '-'}"
            f"{k_s}"
        ),
        f"prior: {sus_s}" + (f" — {reason}" if reason else ""),
        f"note next: {next_s}",
        format_must_line(must),
    ]
    text = "\n".join(lines) + "\n"
    if "run.log" in text:
        text = text.replace("run.log", "run_log")
    if len(text) > NOTE_RENDER_MAX:
        text = text[: NOTE_RENDER_MAX - 1] + "…\n"
    return text


def _default_completer(
    prompt: str,
    *,
    backend: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> Any:
    from rnaomni.agent_backend import complete_isolated

    timeout = float(os.environ.get("NOTE_API_TIMEOUT", "120") or 120)
    return complete_isolated(
        prompt,
        backend=backend,
        model=model,
        api_key=api_key,
        timeout=timeout,
    )


def _result_text(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    body = getattr(result, "result", None)
    if getattr(result, "status", "ok") != "ok":
        return str(body or getattr(result, "message", "") or "")
    return str(body or "")


def ensure_opening_note(
    state: SearchState | None,
    *,
    task_type: str,
    backend: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    completer: Completer | None = None,
    path: Any | None = None,
    call_llm: bool = True,
) -> dict[str, Any]:
    """Refresh data block; call isolated LLM once if prior is empty (true opening).

    API failure leaves the data block and does not raise.
    """
    note = load_note(path)
    note["data"] = collect_data_block(task_type=task_type, state=state)
    legal = collect_legal_actions(state, note=note)
    prior = note.get("prior") if isinstance(note.get("prior"), dict) else {}
    has_prior = bool(prior.get("suspect") or prior.get("reason"))
    already_started = bool(state is not None and int(getattr(state, "valid_rounds", 0) or 0) > 0)
    should_call = call_llm and not has_prior and not already_started
    if should_call:
        prompt = build_opening_prompt(note["data"], legal)
        leaks = opening_prompt_leaks(prompt)
        if leaks:
            logging.warning("Opening note prompt leaked %s; skip LLM", leaks)
        else:
            try:
                if completer is not None:
                    raw = completer(prompt)
                else:
                    raw = _default_completer(
                        prompt, backend=backend, model=model, api_key=api_key
                    )
                text = _result_text(raw)
                parsed = extract_json_object(text)
                note["prior"] = parse_prior(parsed)
                # N1: opening next is MUST-able feature/knob/sample only (ignore head).
                note["next"] = n1_executable_next(
                    (parsed or {}).get("next") if isinstance(parsed, dict) else None,
                    legal,
                )
            except Exception as err:
                logging.warning("Opening campaign note LLM failed (search continues): %s", err)
    try:
        save_note(note, path)
    except OSError as err:
        logging.warning("Could not write campaign note: %s", err)
    logging.info(
        "Campaign note ready: prior=%s next=%s enabled=%s",
        note.get("prior"),
        note.get("next"),
        ",".join(str(x) for x in (note.get("data") or {}).get("enabled") or []),
    )
    return note
