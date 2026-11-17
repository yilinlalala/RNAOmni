# Evo (encode-only vendor)

Local encode-only vendor copy of [Evo](https://github.com/evo-design/evo).
Only the pieces needed for L4 encoding are kept.

Used as an L4 sequence encoder (`features.builders.evo`), same contract as RNA-FM:
`(N, max_len, 4096)` token hidden states, position `j` aligned with residue `j`.

## What was copied

| Path | Why |
|------|-----|
| `model/` | HuggingFace StripedHyena code + `evo-1-131k-base` shards |
| `evo/models.py`, `evo/scoring.py`, `evo/configs/` | official load + batch tokenize |
| `LICENSE`, `requirements.txt` | license / dep hint |

Weight shards are **hardlinks** of the original `model/*.safetensors` (same inode, no extra 13 GB).

Not copied: generation scripts, examples, `EVO_results/`, `streamer.py`.

## Zero-install

```bash
# conda myenv3.11 already has torch / transformers / flash_attn
PYTHONPATH=tools/Evo python tools/smoke_evo.py
python tools/smoke_evo_builder.py
```

Do not `pip install evo-model`. Do not download from HuggingFace.
