# Evo2 (encode-only vendor)

Local encode-only vendor copy of [Evo2](https://github.com/ArcInstitute/evo2).
Only the pieces needed for L4 encoding are kept.

Used as an L4 sequence encoder (`features.builders.evo2`), same contract as Evo / RNA-FM:
`(N, max_len, 4096)` token hidden states, position `j` aligned with residue `j`.

Default checkpoint: local `evo2_7b` (`weights/evo2_7b.pt`).

## What was copied

| Path | Why |
|------|-----|
| `evo2/` | official load (`Evo2`) + configs + scoring tokenize |
| `vortex/` | StripedHyena 2 runtime (`import vortex.model`) |
| `weights/evo2_7b.pt` | hardlink of the original 7B checkpoint |
| `LICENSE`, `requirements.txt` | license / dep hint |

Not copied: generation scripts, examples, `EVO2_results*`, vortex tests / CUDA `csrc`.

## Zero-install

```bash
# conda rnaomni: torch / flash_attn. Transformer Engine not required for 7B
# (Evo2 disables FP8 when TE is missing).
```

Do not `pip install evo2`. Do not download from HuggingFace.
