# RNAclaw

This is an experiment to have the Agent do its own research on single-sequence, single-label RNA-related prediction.

## Dataset

Values live in `rnaomni.env` (optional gitignored `.env` overlay). Agent must **not** edit either. This file is protocol only.

- **Route:** no named compare root → main CSV: clean + assign splits. Named `RNAOMNI_COMPARE_ROOT` / `--compare-root` → **external** `train.csv` / `val.csv` / `test.csv` as given (no reclean, no re-split). Flat: those three at the root. Fold: `fold{N}/` of those three (default `N=0`). Any other layout **aborts**. Force main: `--no-compare` or `--csv` without `--compare-root`. Files under `forcompare/` do not auto-select.
- **Full then sample (one campaign):** HEK and COMPARE, classification and regression, all start **full** `train/val/test`. Train is still `TIME_BUDGET=60s` (encode excluded). **Encode** wall clock only, cap **1 h** (`ENCODE_TIMEOUT_SEC=3600`). Encode timeout on a **first** full encode (dataset itself cannot encode): kill encode, **no TSV row**, switch to sample, reset search best, retry the same HEAD. Encode timeout **after** a full keep/discard already exists: **crash** that shot + `git reset` the commit, **stay full** (a heavy feature is not a reason to throw away the full baseline). Encode timeout after sampling: **crash** row (`data_mode=sample`) + `git reset` the feature. Classification sample = A+D (K=10, every class ≥1). Regression = row cap **1000** on train/val/test (expand train only, ×2). Same results stem (`compare_<root>` or HEK CSV stem); extra column `data_mode` from the first row. Keep/discard and TSV replay filter by the current mode. `--formal` is a no-op. Old `*_explore.tsv` files are history.
- **Auto expand:** sampling on + `SAMPLE_EXPAND_PATIENCE` consecutive non-significant rounds → orchestrator scales `K_train` (or the regression train row cap) on the current best HEAD. Keep if the decision metric improves by ≥ `min_delta` (then may expand again). **Model discard** rolls the cap back and **freezes the sample axis only**; ordinary patience continues (does not stop the search). **Encode timeout** on expand (same features, more rows) rolls the cap back and blocks further grows until a heavy feature is dropped; then **one** wall-clock retry. `[sample]` in the description is a K/row expand, not `data_mode`. Expand never `git reset`.
- **Columns:** `input_seq` + `predicted_target`. Integer labels with small cardinality → classification (`val_accuracy`); else regression (`val_spearman`).
- **Clean (main only):** T→U; drop illegal bases / non-finite labels; collapse duplicate sequences; write data card. Raw CSV not overwritten.
- **Splits (main):** regression test holdout + CV; classification MMseqs then cluster-aware train/val/test. Numbers in `rnaomni.env`.

## Multi-dataset parallel

Overnight default is still **one dataset in this worktree** (`python orchestrator.py`). To run another dataset at the same time, submit it as a **separate job** — one command per dataset. Default GPU is **0**. Override with `CUDA_VISIBLE_DEVICES` / `--cuda-visible-devices`. Pin with `--gpu`. Concurrent running jobs are capped by `RNAOMNI_MAX_RUNNING_LANES` (default **2**; `0` = unlimited). At cap the next submit **waits**; `--no-wait` refuses. There is **no** GPU occupancy check. `python -m rnaomni.finalize` is not in this pool — run it off-peak, or submit it through `rnaomni.parallel` if you want it counted.

```bash
# terminal 1 (GPU 0)
python -m rnaomni.parallel --compare-root dataset/forcompare/Class_88fam_200%noise

# terminal 2 (also GPU 0 unless CUDA_VISIBLE_DEVICES lists more cards)
python -m rnaomni.parallel --compare-root dataset/forcompare/Class_88fam_0%noise

# cycle 0,1,2,3
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m rnaomni.parallel --compare-root dataset/forcompare/Reg_MRL
```

Agent must not `git worktree add` or edit `rnaomni.env` / `.env`.

| Rule | Detail |
|------|--------|
| Submit | One `--compare-root` = one job. Default GPU **0**. `CUDA_VISIBLE_DEVICES=0,1` cycles those ids. `--gpu N` pins that card. |
| Cap | None. Same card / same stem may run more than once. OOM is the backstop. |
| Inject | That job's process env sets `RNAOMNI_COMPARE_ROOT` and beats this machine's `.env`. |
| Isolation | Each job is its own git worktree. One job failing does **not** kill the other. |
| Results | Each worktree has its own `workspace/<stem>.tsv`. Pool panel: `workspace/batches/pool/dashboard.tsv` — **not** a cross-dataset leaderboard. |
| Code | Do **not** auto-merge `train.py` across jobs. |
| Status | `python -m rnaomni.parallel --status` |

`--spec` can still start several named jobs in one go (they follow `CUDA_VISIBLE_DEVICES`, default all on 0). There is **no** waiting queue that auto-starts the next dataset.

Cleanup one job: `python -m rnaomni.parallel --cleanup-only --batch-id <job_id>`.

## Active skills

Skill **registration** is repo-native (not Cursor IDE discovery):

| Source | Path | Role |
|--------|------|------|
| Registry (SSOT) | `skill-library/registry.yaml` | Which skills exist; `overnight: true` = night min-set |
| Resolver | `tools/resolve_skills.py` | Any launcher injects the same skill paths into prompts |
| Content | `skill-library/feature/SKILL.md` | Feature skill (path H: human builders; agent recipe/`train.py` only) |

Overnight / orchestrator must load skills with `overnight: true` from the registry (currently `rnaomni-feature`). Always read the registry or the SKILL paths above before changing features.

### Feature recipe

| Path | Who edits |
|------|-----------|
| `features/registry.yaml`, `features/builders/*` | Human only |
| `features/active.yaml` | Human + agent (**whitelist**: `recipe_id`, `fusion ∈ {concat,add,mean}`, optional `fusion_include`, `features[].name/enabled`, `features[].params` only for that name's `agent_tunable`: kmer `k` 1–4, `rna_fm.repr_layer` 0–12) |
| `prepare.py`, `rnaomni/prepare/` | Human orchestration only — **agents must not modify** (facade or package) |
| `rnaomni.env` | Human only — **agents must not modify** (default config table) |

After changing the recipe, run `train.py` directly — features are encoded at call time from `features/active.yaml` (no disk feature cache). `train.py` must use `load_prepared_data()["input_dim"]` (do not hardcode 4). For classification also use `data["num_classes"]` for the output head (do not hardcode 88).

## SDK orchestrator mode

`orchestrator.py` uses `rnaomni.agent_backend` as transport (backend/model: `rnaomni.env`; `--backend cursor` keeps Cursor SDK). Behavior is defined here; prompts inject Active skills via `resolve_skills.py`.

| Rule | Detail |
|------|--------|
| Autonomy | Never ask the human for confirmation, approval, or "should I continue?". Proceed and STOP only when the current invocation's checklist is done. |
| Fresh agent | Each Setup or Iter is a new agent invocation. |
| Setup | Complete the Setup section only. Do **not** run `train.py`. Do **not** start the experiment loop. Use `python prepare.py --skip-encode` (no feature encoding in the SDK session). |
| Iter | **Edit+commit only**, then STOP. Do **not** run `train.py` or read `workspace/run.log`. If the previous train crashed, the orchestrator injects one length-capped digest (also `|| crash:` on the TSV row). |
| Skills | Follow overnight Active skills / `agent-ops.md`. |
| Decision metric | **Regression**: keep/discard by **val_spearman** (higher is better). **Classification**: keep/discard by **val_accuracy** (higher is better). |
| Keep threshold | Improve vs current best by ≥ **min_delta** (orchestrator defaults; `--min-delta` overrides). Smaller → discard + reset. Same threshold resets patience. Enforced after external `train.py`. |
| Search stop | Orchestrator: max iterations, consecutive agent failures, or patience (**15** consecutive non-significant rounds after `min_iterations`). Sampling on: expand `K_train` after `SAMPLE_EXPAND_PATIENCE` non-improves; expand **model discard** freezes the sample axis only (ordinary patience continues). Encode-timeout expand blocks grows until a heavy feature is dropped, then one retry. On stop, archive to `workspace/runs/<tag>_<csv_stem>/`. |

## Setup

**Environment:** `conda activate` the name in `rnaomni.env` (`RNAOMNI_CONDA_ENV`). Classification prep needs `mmseqs` on PATH (or `MMSEQS_BIN`).

Set up a new experiment autonomously (no human confirmation):

1. **Run tag**: choose a tag from the current timestamp. The branch `RNAclaw/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b RNAclaw/<tag>` from current master.
3. **Read in-scope files**:
  - `README.md` — repository context.
  - `program.md` — dataset contract and experiment protocol (this file).
  - `prepare.py` / `rnaomni/prepare/` — splits, dataloader, evaluation. Do not modify.
  - `train.py` — model / training loop (agent may edit).
  - `features/active.yaml`, `features/registry.yaml` — feature contract.
  - Active skills (`skill-library/feature/SKILL.md`, `agent-ops.md`).
4. **Verify data**: `dataset/` must contain the active CSV. If missing, STOP with a clear error (do not invent data).
5. **Run data prep**: `python prepare.py --skip-encode` when invoked via orchestrator (splits / data-card only). Feature encoding happens later when the orchestrator runs `train.py` outside the SDK. Manual interactive setup may still run full `python prepare.py` if desired.
6. **Results logging**: orchestrator/train own `workspace/<dataset_stem>.tsv`. Do **not** recreate, symlink as repo-root `results.tsv`, or commit results files.
7. **Metric reminder**:
  - Regression: mean val Spearman over folds (`evaluate_spearman`); test via 5-fold ensemble (`evaluate_test_metrics`).
  - Classification: val accuracy on fixed val split (`evaluate_accuracy`); test accuracy / macro-F1 via ensemble softmax average.

When setup is complete, STOP. Baseline training is the next Iter invocation.

## Experimentation

Each experiment runs on a single GPU. Train with `python train.py` for `TIME_BUDGET` seconds (prepare.py; encode/startup excluded).

Declare `round_type` each round: `feature` | `model` (avoid `joint` in the first version). Opening head is `baseline` (current `train.py`). Named prediction heads (menu in `skill-library/feature/frameworks.md`, including `tf_*` / `gru_pool` / pool variants) are orchestrator-injected via `next_framework`: after at least one keep/discard the campaign note may name a remaining head; if it does not and `FRAMEWORK_PATIENCE` consecutive `[model]` discards elapse, the orchestrator still draws uniformly from the remaining menu (including `pool_mlp`). Implement that injected name only by setting the `Predictor` alias. Do not invent architecture names. If bottleneck `next_do=feature` (head suppressed at plateau), MUST `[feature]` that round. A wrong prefix does not change keep/discard.

**What you CAN do:**

- Modify `train.py` — stay on `baseline` (or the current named head) unless `next_framework` is set. In-framework knobs: `HIDDEN_DIM` / `LR` / `BATCH_SIZE` / dropout / loss / EMA; `resnet1d_pool` kernel ∈ {3, 7} and residual blocks ≤ 4 (`skill-library/feature/frameworks.md`). Must use `data["input_dim"]`. Classification must use `data["num_classes"]` for the head.
- Modify `features/active.yaml` whitelist keys only: `recipe_id`, `fusion` (`concat` default, or `add` / `mean` when included channels have the same C), optional `fusion_include` (non-empty list of **enabled + registered** names; omit = all enabled), `features[].name`, `features[].enabled`, and `features[].params` listed in that feature's `agent_tunable` (kmer `k` in 1–4; `rna_fm` `repr_layer` in 0–12). L4 `rna_fm` / `rinalmo` / `evo` / `evo2` may be toggled; missing local weights crash — no pip / download. `evo` / `evo2` are 4096-d (~12 GB VRAM). Only enable features already listed in `features/registry.yaml`. Prefer one main hypothesis per `[feature]` round.

**What you CANNOT do:**

- Modify `prepare.py`, `rnaomni/prepare/`, `features/builders/*`, `features/registry.yaml`, `rnaomni.env`, or the evaluation harness / split protocol.
- Install packages, add dependencies, or download new pretrained weights.
- Hardcode input dimension to `4` (or any fixed dim that ignores the recipe).
- Hardcode `num_classes` for classification.

**The goal**:
- Regression: highest val Spearman.
- Classification: highest val accuracy.

Time budget is always `TIME_BUDGET` of training (encode time is not counted).

**VRAM** is a soft constraint. Some increase is acceptable for meaningful metric gains; do not blow it up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A tiny metric gain that adds ugly complexity is not worth keeping. Equal-or-better results from deleting code are a win.

**Description prefix**: prefix `workspace/<stem>.tsv` descriptions with `[feature]` or `[model]` to match `round_type`. `[model]` must include `framework=<name>` (`baseline` until the first named switch); a framework switch first shot also adds `framework_base`. Also include `cause=<head|feature|knob>` (or `cause=crash` when fixing a crash). Never write `cause=head` to pick a skeleton; only when `next_framework` is set: `[model] framework=<name> framework_base cause=head …`.

**The first run**: establish baseline — run training with the current recipe/`train.py` as-is (or minimal no-op commit if a commit is required).

## Output format

### Regression

```
---
val_spearman:     0.790000
test_RMSE:        0.480000
test_R2:          0.620000
test_spearman:    0.790000
test_pearson:     0.810000
training_seconds: 60.1
test_eval_seconds:2.3
total_seconds:    328.2
peak_vram_mb:     450.2
...
```

```
grep "^val_spearman:\|^test_RMSE:\|^peak_vram_mb:" workspace/run.log
```

### Classification

```
---
val_accuracy:     0.850000
test_accuracy:    0.840000
test_macro_f1:    0.830000
training_seconds: 60.1
test_eval_seconds:2.3
total_seconds:    328.2
peak_vram_mb:     450.2
...
```

```
grep "^val_accuracy:\|^test_accuracy:\|^peak_vram_mb:" workspace/run.log
```

## Logging results

When an experiment is done, log it to the active results file (tab-separated, NOT comma-separated — commas break in descriptions).

**Per-dataset results:** logs live at `workspace/<dataset_stem>.tsv` (main CSV stem, or `compare_<root>` on the external route). There is **no** repo-root `results.tsv`. Switching `COMPARE_ROOT` / `ACTIVE_CSV` switches files automatically; switching `data_mode` or changing `K_train`/`K_eval` within a campaign does **not** switch files. Orchestrator ensures the correct task header on start and each iteration. First full-mode encode timeout (no prior full keep/discard) writes **no** row and switches to sample. Later full-mode encode timeout writes **crash** and stays full. Sample-mode encode timeout writes **crash**.

**Live campaign scratch** (gitignored, not at repo root): `workspace/run.log`, `workspace/orchestrator.log`, `workspace/orchestrator_state.json`. A new campaign truncates the orchestrator log; resume appends. Agent must not read `workspace/run.log`.

**Run archive (output side):** before switching datasets, preserve the search log. Orchestrator auto-archives on search stop; you can also run `python -m rnaomni.archive` manually. Snapshots go to `workspace/runs/<timestamp_or_branch_tag>_<csv_stem>/` (`results.tsv`, `active.yaml`, `orchestrator.log`, `train.py.snapshot`, `meta.json`) and append `workspace/runs/INDEX.tsv`. Working-tree results files are copied, not deleted. Finalize separately if needed: `python -m rnaomni.finalize --commit auto --output-dir workspace/runs/<tag>/finalize`.

### Regression header (10 columns)

```
commit	val_spearman	test_RMSE	test_R2	test_spearman	test_pearson	memory_gb	data_mode	status	description
```

Keep/discard by **val_spearman** only (higher is better). Keep requires improvement ≥ **min_delta** vs current best in the **current `data_mode`**.

### Classification header (8 columns)

```
commit	val_accuracy	test_accuracy	test_macro_f1	memory_gb	data_mode	status	description
```

Keep/discard by **val_accuracy** only (higher is better). Keep requires improvement ≥ **min_delta** vs current best in the **current `data_mode`**.

1. git commit hash (short, 7 chars)
2. decision metric (val_spearman or val_accuracy) — use 0.000000 for crashes
3–N. test metrics (task-specific) — use 0.000000 for crashes
4. peak memory in GB, round to .1f — use 0.0 for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description (prefix `[feature]` / `[model]`)

## The experiment loop

Runs on a dedicated `RNAclaw/<tag>` branch.

**Orchestrator / agent**: one Iter invocation = **edit+commit only**, then STOP. The orchestrator runs `python train.py` (encode+train) and enforces keep/discard `git reset` **outside** the agent so long jobs do not hold the transport (Cursor bridge or OpenCode session).

**Long-lived interactive session** (if not under orchestrator): repeat the full loop (including train) without asking the human; continue until manually interrupted.

Each round (interactive full loop; under orchestrator, steps 4–8 are done by `orchestrator.py`):

1. Inspect git state (branch, HEAD, recent commits) and the active `workspace/<stem>.tsv`.
2. Declare `round_type` (`feature` | `model`) and one main hypothesis. Follow Active skills / agent-ops.
3. Edit whitelist files only → `git commit` with `[feature]` / `[model]` subject prefix (do not commit results files).
4. Run: `python train.py > workspace/run.log 2>&1` (redirect everything — do NOT use tee). Features encode at train time from the active recipe.
5. Read metrics with the task-appropriate grep (see Output format).
6. If grep is empty, `tail -n 50 workspace/run.log`, fix easy bugs or log crash and revert.
7. Ensure `workspace/<stem>.tsv` has the row (`train.py` may append; fix status/description if needed).
8. Keep the commit only if the decision metric improves vs current best by ≥ **min_delta** (lower RMSE / higher accuracy); else `git reset` to the pre-experiment HEAD (recipe and `train.py` must match HEAD after discard). Tiny gains below min_delta are discard.

**Timeout**: Training should stay near `TIME_BUDGET` (+ encode/startup). Far over with no progress → kill, discard, revert. First-time MMseqs clustering is prep-only and may exceed this during Setup.

**Crashes**: Easy fixes (typo, missing import) → fix and re-run. Fundamentally broken idea → log `crash`, revert if needed, move on.

**Autonomy**: Do NOT pause to ask the human whether to continue. If out of ideas, read in-scope files and skills harder and try a new hypothesis.
