# Named sequence-level frameworks (overnight)

> Orchestrator injects `active_framework` / `next_framework`. Agent **must** implement the injected name. Do not invent menu names. Token / pairwise heads are out of contract.

Menu (unordered): `pool_mlp` | `attn_pool_mlp` | `resnet1d_pool` | `cls_mlp` | `tf_mean` | `tf_cls` | `tf_attn` | `tf_conv` | `max_pool_mlp` | `gated_pool_mlp` | `gru_pool` | `mean_deep_mlp`

Opening campaign is **`baseline`** (current `train.py` / `Predictor = PooledMLP`). `baseline` is **not** a menu name. Random draw is after that opening head’s consecutive `[model]` discard cap, from the **full** menu — including `pool_mlp`.

## When `next_framework` is set

1. `round_type=model`. Do **not** edit `features/active.yaml`.
2. Start from current HEAD (orchestrator already reset to the **global best** commit).
3. Implement that name as `Predictor`. Named classes already exist in `train.py` — set the alias only (`tf_mean`→`TfMean`, `tf_cls`→`TfCls`, `tf_attn`→`TfAttn`, `tf_conv`→`TfConv`, `max_pool_mlp`→`MaxPoolMLP`, `gated_pool_mlp`→`GatedPoolMLP`, `gru_pool`→`GruPool`, `mean_deep_mlp`→`MeanDeepMLP`). Inherit `HIDDEN_DIM` / `LR` / `BATCH_SIZE` / loss / EMA / dropout. Do **not** copy `CONV_CHANNELS`. Do **not** retune LR/batch this round.
4. Commit description: `[model] framework=<name> framework_base cause=head …`
5. One hypothesis: swap the skeleton only. Do not also retune LR/batch this round.

When `next_framework` is empty: stay on `active_framework`. `[model]` descriptions must include `framework=<active>`.

## In-framework knobs (no skeleton swap)

Stay on the injected name. One hypothesis per `[model]` round. Do **not** edit `features/active.yaml` on a model round.

| Knob | Constraint |
|------|------------|
| `HIDDEN_DIM` / `LR` / `BATCH_SIZE` / dropout | Positive; inherit these on a framework **switch** first shot |
| Loss / EMA | e.g. `LABEL_SMOOTHING`, `EMA_DECAY` — keep or drop, do not invent a new loss family |
| `resnet1d_pool` only | kernel ∈ {3, 7}; residual blocks ≤ 4 |

A switch first shot inherits `WIDTH=HIDDEN_DIM` and the training knobs above. Do not retune them in the same commit as the named-head swap.

## Tensor path (every named head)

```
x (B, L, C)                 C = data["input_dim"]  — never hardcode 4/69/266/532
  → stem to WIDTH           WIDTH = HIDDEN_DIM; identity if C == WIDTH
  → blocks at WIDTH
  → masked pool             (B, pool_dim)
  → MLP                     Linear(pool_dim, WIDTH) → … → num_outputs
                             num_outputs = 1 or data["num_classes"]
```

`forward(self, x, lengths)` must return `(B,)` if `num_outputs==1` else `(B, num_outputs)`.

| name | pool | `pool_dim` | notes |
|------|------|------------|--------|
| `pool_mlp` | mean+max | `2*WIDTH` | Named rebuild: stem→WIDTH then conv residual at WIDTH. Opening `baseline` may still be Conv(C→ch→C)+residual-on-C. |
| `attn_pool_mlp` | attn+max | `2*WIDTH` | Attention over length, `-inf` mask pad; concat attn-sum and max. |
| `resnet1d_pool` | mean | `WIDTH` | 1D ResNet, channels=`WIDTH`, **≤4** residual blocks, kernel 3 or 7. |
| `cls_mlp` | learned vector | `WIDTH` | One learned summary (CLS-style) over length, then MLP. **No** Transformer encoder. |
| `tf_mean` | mean | `WIDTH` | 1–2 layer `TransformerEncoder`, `nhead=4`, `dim_ff=2×WIDTH`, `src_key_padding_mask` from `lengths`. |
| `tf_cls` | CLS token | `WIDTH` | Same encoder; learned CLS prepended, take CLS after encoder. |
| `tf_attn` | attn | `WIDTH` | Same encoder as `tf_mean`, then length attention-pool (not `attn_pool_mlp`: encoder first). |
| `tf_conv` | mean | `WIDTH` | k=7 conv residual at WIDTH, then **1** encoder layer, then mean. |
| `max_pool_mlp` | max | `WIDTH` | Stem only, no conv / encoder. |
| `gated_pool_mlp` | gated sum | `WIDTH` | Per-position scalar gate, softmax over length. |
| `gru_pool` | mean | `2*WIDTH` | 1-layer bidirectional GRU; mean-pool outputs. |
| `mean_deep_mlp` | mean | `WIDTH` | Stem + mean-pool + deeper MLP; no conv / encoder. |

`tf_*` WIDTH should stay divisible by 4 (`nhead=4`; code falls back to `nhead=1`). This is **not** L4 `rna_fm` (frozen 12-layer embeddings).

Head first layer: `Linear(pool_dim, WIDTH)` (or `Linear(WIDTH, num_outputs)` for a shallow `resnet1d_pool` / `cls_mlp` head). Rebuild that Linear when `pool_dim` changes. Do not leave a stale `Linear(2*C, …)` after a switch.

## `Predictor` alias

```python
class AttnPoolMLP(nn.Module):
    ...

Predictor = AttnPoolMLP   # overnight / finalize instantiate this name
```

Keep the old class around if you want, but training and finalize use `Predictor`. Constructor should accept `input_dim`, `hidden_dim`, `dropout`, `num_outputs`. Do not require `conv_channels` on non-`pool_mlp` heads. Menu name → class is in `train.py` (see switch-round list above).

## Red lines

- Do not invent menu names. Do not add Mamba / GNN. Menu `tf_*` and the other named heads in the table are allowed.
- Do not add per-token or pairwise (contact) heads.
- Do not `pip` / download weights.
- Do not hardcode `input_dim` or `num_classes`.
