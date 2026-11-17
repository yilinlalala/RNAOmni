# RNAOmni

Autonomous research framework for **RNA sequence modeling**.

RNAOmni trains models that map an RNA sequence to a biological label — **regression** (continuous) or **classification** (discrete). An AI coding agent iterates overnight: it edits training code, runs a fixed-time experiment, keeps improvements, and discards the rest. You get an experiment log and (hopefully) a stronger model. This is a **research** loop; there is no inference CLI/API for new sequences yet.

Workflow adapted from [autoresearch](https://github.com/karpathy/autoresearch): you mainly edit `program.md` (agent instructions); the agent modifies `train.py`.

中文说明：[`使用说明.md`](使用说明.md)

## Task

| | |
| --- | --- |
| **Input** | RNA nucleotide sequence |
| **Output** | Continuous value or class label |
| **Decision metric** | `val_spearman` (regression) or `val_accuracy` (classification), higher is better |
| **Time budget** | `TIME_BUDGET` wall-clock training per run (default **300s** in `rnaomni.env`; encode/startup excluded) |

Demo CSV: `dataset/HEK_te_prediction_task.csv`.

## How it works

| File | Role |
| --- | --- |
| `prepare.py` | Data load, encoding, evaluation — **do not modify** |
| `train.py` | Model + training loop — **agent edits this** |
| `program.md` | Agent protocol — **you edit this** |
| `rnaomni.env` | Default config table — agent must not edit; optional `.env` overlay |

## Quick start

**Requirements:** one NVIDIA GPU; Python 3.11; PyTorch (CUDA); NumPy; PyYAML. Conda env name is `RNAOMNI_CONDA_ENV` in `rnaomni.env` (default `myenv3.11`). Structure features need ViennaRNA; classification splits may need `mmseqs`.

Default recipe (`features/active.yaml`) is **`onehot_only`** — no pretrained weights required.

```bash
conda activate myenv3.11   # must match RNAOMNI_CONDA_ENV

python prepare.py          # one-time data prep (demo CSV included)
python train.py            # one experiment (default TIME_BUDGET=300s)

# shorter smoke run
TIME_BUDGET=60 python train.py
```

Check metrics at the end of the run, or:

```bash
grep -E '^val_spearman:|^test_RMSE:' workspace/run.log
```

Example summary:

```
---
val_spearman:     0.229061
test_RMSE:        1.000536
test_R2:          0.108261
test_spearman:    0.340659
test_pearson:     0.361309
training_seconds: 56.6
peak_vram_mb:     101.0
```

### Custom data

```bash
# CSV under dataset/ (columns: input_seq, label/target)
python prepare.py --csv your_file.csv
python train.py

# External train/val/test CSVs
python prepare.py --compare-root /path/to/task
python train.py
```

### Optional pretrained features (Evo2, RNA-FM, …)

Place weights under `tools/*/`, then enable them in `features/active.yaml`. Details: [`使用说明.md`](使用说明.md) and each `WEIGHTS.md`.

### Overnight agent search

```bash
python orchestrator.py
```

Backend: `AGENT_BACKEND` in `rnaomni.env`. Put API keys in `.env` only (gitignored).

Point a coding agent at the repo:

```
Hi, have a look at program.md and let's kick off a new experiment! Let's do the setup first.
```

## Outputs

Runtime outputs go under **`workspace/`** (gitignored):

| Path | Contents |
| --- | --- |
| `workspace/<dataset_stem>.tsv` | Search / experiment log (tab-separated) |
| `workspace/run.log` | Latest training log |
| `workspace/runs/` | Archived campaigns (`results.tsv`, `active.yaml`, `meta.json`, …) |
| `artifacts/` | Prep caches (data cards, splits) |

## Project layout

```
dataset/          demo CSV
features/         active recipe + feature builders
rnaomni/          prepare helpers, orchestrator, parallel, finalize
tools/            optional encoders + smoke scripts
skill-library/    agent skill docs for features
workspace/        runtime outputs (gitignored)
artifacts/        prep caches (gitignored)
prepare.py
train.py
orchestrator.py
program.md
rnaomni.env
```

## Design choices

- **One editable training file** — agent only changes `train.py`.
- **Fixed time budget** — comparable runs across architectures on the same GPU.
- **Fixed evaluation** — keep/discard uses the metric from `prepare.py`.
- **Simplicity** — equal metric with simpler code wins (see `program.md`).

## Platform

Single NVIDIA GPU + CUDA. On smaller GPUs, reduce depth / sequence length / batch size in `train.py` (or ask the agent to tune them).

## Citation

If you use RNAOmni, please cite:

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

[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). See [`LICENSE`](LICENSE).

Third-party code under `tools/` keeps its upstream licenses (see each subdirectory).
