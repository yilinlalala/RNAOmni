"""Load feature registry + active recipe, encode tensors."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

import numpy as np

from features._yaml import load_yaml

PROJECT_DIR = Path(__file__).resolve().parent.parent
FEATURES_DIR = PROJECT_DIR / "features"
REGISTRY_PATH = FEATURES_DIR / "registry.yaml"
RECIPE_PATH = FEATURES_DIR / "active.yaml"

DEFAULT_ALPHABET = "ACGU"
KMER_K_MIN = 1
KMER_K_MAX = 4
DEFAULT_ENCODE_CHUNK = 4096
ALLOWED_FUSION = frozenset({"concat", "add", "mean"})
AGENT_RECIPE_WHITELIST = frozenset(
    {
        "recipe_id",
        "fusion",
        "fusion_include",  # optional subset of enabled+registered names
        "features",  # name, enabled, and params keys listed in agent_tunable
    }
)


def load_feature_registry(path: Path | None = None) -> dict[str, Any]:
    data = load_yaml(path or REGISTRY_PATH)
    if not isinstance(data, dict) or "features" not in data:
        raise ValueError(f"Invalid feature registry: {path or REGISTRY_PATH}")
    return data


def load_active_recipe(path: Path | None = None) -> dict[str, Any]:
    data = load_yaml(path or RECIPE_PATH)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid recipe: {path or RECIPE_PATH}")
    fusion = data.get("fusion", "concat")
    if fusion not in ALLOWED_FUSION:
        raise ValueError(f"Unsupported fusion {fusion!r}; allowed={sorted(ALLOWED_FUSION)}")
    if "features" not in data or not data["features"]:
        raise ValueError("Recipe must list at least one feature")
    include = data.get("fusion_include")
    if include is not None:
        if not isinstance(include, list) or not include:
            raise ValueError(
                "fusion_include must be a non-empty list of enabled registered feature names"
            )
    return data


def _registry_by_name(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out = {}
    for entry in registry.get("features") or []:
        name = entry["name"]
        out[name] = entry
    return out


def _merged_params(entry: dict[str, Any], recipe_item: dict[str, Any]) -> dict[str, Any]:
    params = dict(entry.get("default_params") or {})
    extra = recipe_item.get("params")
    if extra is None:
        return params
    if not isinstance(extra, dict):
        raise ValueError(
            f"Recipe params for {entry['name']!r} must be a mapping, got {type(extra).__name__}"
        )
    if not extra:
        return params
    tunable = list(entry.get("agent_tunable") or [])
    unknown = [key for key in extra if key not in tunable]
    if unknown:
        raise ValueError(
            f"Recipe params {unknown} not in agent_tunable={tunable} "
            f"for feature {entry['name']!r}"
        )
    params.update(extra)
    _validate_merged_params(entry["name"], params)
    return params


def _validate_merged_params(name: str, params: dict[str, Any]) -> None:
    if name == "rna_fm":
        from features.builders.rna_fm import parse_repr_layer

        params["repr_layer"] = parse_repr_layer(params)


def _fusion_include_names(
    recipe: dict[str, Any],
    enabled_names: list[str],
    by_name: dict[str, dict[str, Any]],
) -> list[str] | None:
    raw = recipe.get("fusion_include")
    if raw is None:
        return None
    if not isinstance(raw, list) or not raw:
        raise ValueError(
            "fusion_include must be a non-empty list of enabled registered feature names"
        )
    enabled_set = set(enabled_names)
    names: list[str] = []
    seen: set[str] = set()
    for item in raw:
        name = str(item)
        if name not in by_name:
            raise ValueError(f"fusion_include name {name!r} not in features/registry.yaml")
        if name not in enabled_set:
            raise ValueError(f"fusion_include name {name!r} is not enabled")
        if name in seen:
            raise ValueError(f"fusion_include duplicate name {name!r}")
        seen.add(name)
        names.append(name)
    return names


def _channel_dim(entry: dict[str, Any], params: dict[str, Any], *, alphabet: str) -> int:
    if entry["name"] == "kmer":
        k = int(params.get("k", 3))
        if k < KMER_K_MIN or k > KMER_K_MAX:
            raise ValueError(
                f"kmer k must be in {KMER_K_MIN}..{KMER_K_MAX} (C=|{alphabet}|^k), got {k}"
            )
        return int(len(alphabet) ** k)
    return int(entry["dim"])


def _enabled_specs(
    recipe: dict[str, Any],
    by_name: dict[str, dict[str, Any]],
    *,
    alphabet: str,
) -> list[dict[str, Any]]:
    specs = []
    for item in recipe["features"]:
        name = item["name"]
        if not item.get("enabled", True):
            continue
        if name not in by_name:
            raise KeyError(f"Recipe feature {name!r} not in features/registry.yaml")
        entry = by_name[name]
        params = _merged_params(entry, item)
        specs.append(
            {
                "name": name,
                "module": entry["module"],
                "params": params,
                "channel_dim": _channel_dim(entry, params, alphabet=alphabet),
            }
        )
    if not specs:
        raise ValueError("Recipe has no enabled features")
    include = _fusion_include_names(recipe, [s["name"] for s in specs], by_name)
    if include is None:
        return specs
    by_spec = {s["name"]: s for s in specs}
    return [by_spec[name] for name in include]


def _fused_input_dim(channel_dims: list[int], fusion: str) -> int:
    if fusion == "concat":
        return int(sum(channel_dims))
    if fusion in ("add", "mean"):
        if len(set(channel_dims)) != 1:
            raise ValueError(f"fusion={fusion} needs equal C, got {channel_dims}")
        return int(channel_dims[0])
    raise ValueError(f"Unsupported fusion {fusion!r}; allowed={sorted(ALLOWED_FUSION)}")


def _fuse_channels(channels: list[np.ndarray], fusion: str) -> np.ndarray:
    dims = [int(arr.shape[-1]) for arr in channels]
    _fused_input_dim(dims, fusion)
    if fusion == "concat":
        return np.concatenate(channels, axis=-1)
    stacked = np.stack(channels, axis=0)
    if fusion == "add":
        return stacked.sum(axis=0)
    return stacked.mean(axis=0).astype(np.float32, copy=False)


def recipe_cache_key(recipe: dict[str, Any]) -> tuple:
    """Hash-stable recipe identity for prepare.py in-process memo."""
    enabled = []
    for item in recipe.get("features") or []:
        if not item.get("enabled", True):
            continue
        params = item.get("params") or {}
        if not isinstance(params, dict):
            params = {}
        enabled.append((item["name"], tuple(sorted(params.items()))))
    include = recipe.get("fusion_include")
    include_key = tuple(str(x) for x in include) if include is not None else None
    return (
        recipe.get("recipe_id"),
        recipe.get("fusion", "concat"),
        include_key,
        tuple(enabled),
    )


def resolve_input_dim(
    recipe: dict[str, Any] | None = None,
    registry: dict[str, Any] | None = None,
    *,
    alphabet: str = DEFAULT_ALPHABET,
) -> int:
    recipe = recipe or load_active_recipe()
    registry = registry or load_feature_registry()
    by_name = _registry_by_name(registry)
    specs = _enabled_specs(recipe, by_name, alphabet=alphabet)
    fusion = recipe.get("fusion", "concat")
    if fusion not in ALLOWED_FUSION:
        raise ValueError(f"Unsupported fusion {fusion!r}; allowed={sorted(ALLOWED_FUSION)}")
    return _fused_input_dim([s["channel_dim"] for s in specs], fusion)


def _sequence_lengths(sequences: list[str], max_len: int) -> np.ndarray:
    return np.asarray([min(len(s), max_len) for s in sequences], dtype=np.int64)


def encode_chunk_size() -> int:
    """Unique-table chunk size (15.3). ``RNAOMNI_ENCODE_CHUNK``; default 4096."""
    raw = os.environ.get("RNAOMNI_ENCODE_CHUNK")
    if raw is None or str(raw).strip() == "":
        return DEFAULT_ENCODE_CHUNK
    n = int(str(raw).strip())
    if n <= 0:
        return DEFAULT_ENCODE_CHUNK
    return n


def store_len_for(lengths: np.ndarray, cap: int) -> int:
    """Resident L: min(data/chunk max, cap). Never pad a stored table to 1024 by default."""
    cap = int(cap)
    if lengths.size == 0:
        return 1
    return max(1, min(int(lengths.max()), cap))


def normalize_seq_id(seq: str) -> str:
    """Match prepare.splits._normalize_sequence / compare sample (upper, T→U)."""
    return str(seq).upper().replace("T", "U")


def unique_sequence_index(sequences) -> tuple[list[str], np.ndarray]:
    """First-seen unique seq_ids and row→uid map. Labels stay per-row elsewhere."""
    keys = [normalize_seq_id(s) for s in sequences]
    uid_of: dict[str, int] = {}
    unique: list[str] = []
    row_to_uid = np.empty(len(keys), dtype=np.int64)
    for i, key in enumerate(keys):
        uid = uid_of.get(key)
        if uid is None:
            uid = len(unique)
            uid_of[key] = uid
            unique.append(key)
        row_to_uid[i] = uid
    return unique, row_to_uid


def _encode_chunk(
    sequences: list[str],
    max_len: int,
    alphabet: str,
    specs: list[dict[str, Any]],
    fusion: str,
) -> np.ndarray:
    """Encode one list; one channel at a time. Peak ≈ widest channel + fused buffer."""
    n = len(sequences)
    dims = [int(s["channel_dim"]) for s in specs]
    fused_dim = _fused_input_dim(dims, fusion)
    out = np.zeros((n, max_len, fused_dim), dtype=np.float32)
    offset = 0
    for spec in specs:
        mod = importlib.import_module(spec["module"])
        arr = mod.build(
            sequences,
            max_len=max_len,
            alphabet=alphabet,
            params=spec["params"],
        )
        if arr.ndim != 3 or arr.shape[0] != n or arr.shape[1] != max_len:
            raise ValueError(
                f"Builder {spec['name']} returned shape {arr.shape}, "
                f"expected (N={n}, L={max_len}, C)"
            )
        if arr.shape[2] != int(spec["channel_dim"]):
            raise ValueError(
                f"Builder {spec['name']} dim={arr.shape[2]} != computed dim={spec['channel_dim']}"
            )
        arr = arr.astype(np.float32, copy=False)
        width = int(arr.shape[2])
        if fusion == "concat":
            out[:, :, offset : offset + width] = arr
            offset += width
        else:
            out += arr
        del arr
    if fusion == "mean" and specs:
        out /= float(len(specs))
    return out


def _encode_sequence_list(
    sequences: list[str],
    *,
    max_seq_len: int,
    alphabet: str,
    recipe: dict[str, Any],
    registry: dict[str, Any],
):
    by_name = _registry_by_name(registry)
    specs = _enabled_specs(recipe, by_name, alphabet=alphabet)
    fusion = recipe.get("fusion", "concat")
    if fusion not in ALLOWED_FUSION:
        raise ValueError(f"Unsupported fusion {fusion!r}; allowed={sorted(ALLOWED_FUSION)}")

    lengths = _sequence_lengths(sequences, max_seq_len)
    l_store = store_len_for(lengths, max_seq_len)
    fused_dim = _fused_input_dim([int(s["channel_dim"]) for s in specs], fusion)
    n = len(sequences)
    x = np.zeros((n, l_store, fused_dim), dtype=np.float32)
    if n == 0:
        return x, lengths, specs, fusion

    chunk = encode_chunk_size()
    done = 0
    while done < n:
        end = min(done + chunk, n)
        chunk_seqs = sequences[done:end]
        l_chunk = store_len_for(lengths[done:end], l_store)
        fused = _encode_chunk(chunk_seqs, l_chunk, alphabet, specs, fusion)
        x[done:end, :l_chunk, :] = fused
        del fused
        done = end
        print(f"Data: encoded unique {done}/{n} (store_len={l_store}, chunk={chunk})", flush=True)
    return x, lengths, specs, fusion


def encode_sequence_batch(
    sequences: list[str],
    *,
    max_seq_len: int,
    alphabet: str,
    recipe: dict[str, Any] | None = None,
    registry: dict[str, Any] | None = None,
):
    """Pad this list to its own max length (capped). For finalize online collate (15.2A)."""
    recipe = recipe or load_active_recipe()
    registry = registry or load_feature_registry()
    by_name = _registry_by_name(registry)
    specs = _enabled_specs(recipe, by_name, alphabet=alphabet)
    fusion = recipe.get("fusion", "concat")
    if fusion not in ALLOWED_FUSION:
        raise ValueError(f"Unsupported fusion {fusion!r}; allowed={sorted(ALLOWED_FUSION)}")
    lengths = _sequence_lengths(sequences, max_seq_len)
    l_batch = store_len_for(lengths, max_seq_len)
    x = _encode_chunk(sequences, l_batch, alphabet, specs, fusion)
    return x, lengths


def encode_dataframe_from_recipe(
    df,
    *,
    sequence_col: str,
    label_col: str,
    max_seq_len: int,
    alphabet: str,
    recipe: dict[str, Any] | None = None,
    registry: dict[str, Any] | None = None,
    label_dtype=np.float32,
):
    """Encode unique sequences; return x (U,L,C), lengths (U,), labels (N,).

    ``meta['row_to_uid']`` maps each row to a unique-feature row. Duplicate
    sequences share features; labels are not merged.
    """
    recipe = recipe or load_active_recipe()
    registry = registry or load_feature_registry()

    sequences = df[sequence_col].astype(str).tolist()
    unique_seqs, row_to_uid = unique_sequence_index(sequences)
    labels = df[label_col].to_numpy().astype(label_dtype, copy=False)
    print(
        f"Data: encoding {len(unique_seqs)} unique sequences "
        f"(from {len(sequences)} rows) (max_len={max_seq_len}, "
        f"chunk={encode_chunk_size()}, alphabet={alphabet})...",
        flush=True,
    )
    x, lengths, specs, fusion = _encode_sequence_list(
        unique_seqs,
        max_seq_len=max_seq_len,
        alphabet=alphabet,
        recipe=recipe,
        registry=registry,
    )
    return x, lengths, labels, {
        "recipe_id": recipe.get("recipe_id"),
        "fusion": fusion,
        "fusion_include": list(recipe["fusion_include"]) if recipe.get("fusion_include") else None,
        "enabled_features": [s["name"] for s in specs],
        "input_dim": int(x.shape[-1]),
        "channel_dims": [s["channel_dim"] for s in specs],
        "row_to_uid": row_to_uid,
        "n_rows": int(len(sequences)),
        "n_unique": int(len(unique_seqs)),
        "store_len": int(x.shape[1]),
    }
