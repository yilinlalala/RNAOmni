# RNAOmni

An end-to-end autonomous modeling agent for RNA downstream prediction tasks.

Building strong RNA models still needs heavy manual design. AutoML mostly tunes hyperparameters in fixed spaces, and LLM agents often rely on human blueprints. **RNAOmni** starts from raw data and jointly searches data processing, features, backbones, and hyperparameters (*Direct* / *Thinking* modes). On **17** tasks, it gains **59.50%** over AutoML and **18.4%** over AI research agents on average.

- **Task:** RNA sequence → regression or classification label  
- **Metric:** `val_spearman` / `val_accuracy` (higher better)  
- **Budget:** `TIME_BUDGET` (default 300s in `rnaomni.env`; encode excluded)  
- **Files:** `prepare.py` (fixed) · `train.py` (agent) · `program.md` (you) · `rnaomni.env`

## Quick start

**Needs:** 1× NVIDIA GPU. Create the env from [`environment.yml`](environment.yml) (or [`requirements.txt`](requirements.txt)). Keep `RNAOMNI_CONDA_ENV` in `rnaomni.env` in sync (default `rnaomni`). Optional: ViennaRNA (`struct_*`), `mmseqs` (classification splits). Default recipe is **`onehot_only`** (no weights).

```bash
conda env create -f environment.yml
conda activate rnaomni
# CUDA PyTorch for your GPU if needed, e.g.:
# pip install torch --index-url https://download.pytorch.org/whl/cu128

python prepare.py                 # demo: dataset/HEK_te_prediction_task.csv
python train.py                   # or: TIME_BUDGET=60 python train.py
grep -E '^val_spearman:|^test_RMSE:' workspace/run.log

# custom CSV / pre-split
python prepare.py --csv your_file.csv
python prepare.py --compare-root /path/to/task

# overnight search (API keys in .env only)
python orchestrator.py
```

Outputs: `workspace/<stem>.tsv`, `workspace/run.log`, archives under `workspace/runs/`.

## Optional pretrained weights

Place under `tools/*/`, then enable in `features/active.yaml`. See each `WEIGHTS.md`.

| Feature | Path | Download |
| --- | --- | --- |
| Evo2 (~13GB) | `tools/Evo2/weights/evo2_7b.pt` | [HF arcinstitute/evo2_7b](https://huggingface.co/arcinstitute/evo2_7b) |
| Evo (~13GB) | `tools/Evo/model/` | [HF togethercomputer/evo-1-131k-base](https://huggingface.co/togethercomputer/evo-1-131k-base) |
| RNA-FM (~1.2GB) | `tools/RNA-FM/pretrained/RNA-FM_pretrained.pth` | [RNA-FM](https://github.com/ml4bio/RNA-FM) |
| RiNALMo micro (~128MB) | `tools/RiNALMo/weights/rinalmo_micro_pretrained.pt` | [Zenodo](https://zenodo.org/records/15043668) |

`evo` / `evo2` need `transformers` + `flash_attn`. No weights: `onehot`, `length_frac`, `kmer`.

## Citation

> Yilin Yan, Ruihao Xia, Honglin Li, Yang Tang, Feng Qian.  
> **RNAOmni: An Autonomous Modeling Agent for RNA Downstream Tasks**.

```bibtex
@article{yan2026rnaomni,
  title   = {RNAOmni: An Autonomous Modeling Agent for RNA Downstream Tasks},
  author  = {Yan, Yilin and Xia, Ruihao and Li, Honglin and Tang, Yang and Qian, Feng},
  year    = {2026}
}
```

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — see [`LICENSE`](LICENSE). Third-party code under `tools/` keeps upstream licenses.
