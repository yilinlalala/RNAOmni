"""Human-controlled feature builders + recipe encoding (path H)."""

from features.pipeline import (
    RECIPE_PATH,
    encode_dataframe_from_recipe,
    encode_sequence_batch,
    load_active_recipe,
    load_feature_registry,
    recipe_cache_key,
    resolve_input_dim,
    unique_sequence_index,
)

__all__ = [
    "RECIPE_PATH",
    "encode_dataframe_from_recipe",
    "encode_sequence_batch",
    "load_active_recipe",
    "load_feature_registry",
    "recipe_cache_key",
    "resolve_input_dim",
    "unique_sequence_index",
]
