# Evo weights (not shipped)

Expected: full Hugging Face model directory under `tools/Evo/model/`
(`config.json`, `model.safetensors.index.json`, shard files).

```bash
mkdir -p tools/Evo/model
huggingface-cli download togethercomputer/evo-1-131k-base --local-dir tools/Evo/model
```

https://huggingface.co/togethercomputer/evo-1-131k-base
