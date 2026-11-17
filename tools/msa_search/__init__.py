"""Human-controlled MSA search precompute (online RNAcentral; local deferred).

Agent must not run these tools. Training reads profiles via msa_profile builder.
"""

from tools.msa_search.contract import (
    CACHE_KEY,
    DIM,
    PROFILE_CHANNELS,
    dataset_id_from_csv,
    normalize_rna,
    resolve_dataset_asset_dir,
    rnacentral_md5,
    seq_hash,
)

__all__ = [
    "CACHE_KEY",
    "DIM",
    "PROFILE_CHANNELS",
    "dataset_id_from_csv",
    "normalize_rna",
    "resolve_dataset_asset_dir",
    "rnacentral_md5",
    "seq_hash",
]
