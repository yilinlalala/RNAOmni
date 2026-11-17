# Feature agent-ops (one overnight round)

> May edit `features/active.yaml` whitelist keys.  
> `prepare.py` / `rnaomni/prepare/` / `features/pipeline.py` / builders / registry / `rnaomni.env` must **not** be modified.  
> **No on-disk feature cache:** after changing the recipe, the next `train.py` start encodes with the current recipe (who runs train: see below).  
> **Unattended:** do not ask a human for confirmation. Orchestrator overnight: after the agent finishes 1–3, STOP immediately; 4–6 are done by the orchestrator. Interactive session: the agent does the full 1–6.  
> Stage status: [`../../devlog/20260719_000000_feature_skill_PLAN_PARTIAL.md`](../../devlog/20260719_000000_feature_skill_PLAN_PARTIAL.md).

## Recipe whitelist (minimal set)

Only: `recipe_id`, `fusion` (`concat` | `add` | `mean`), optional `fusion_include` (non-empty list; only names that are **enabled and registered**; omit = all enabled), `features[].name`, `features[].enabled`, and that name's `agent_tunable` keys (kmer `params.k` integer 1–4; `rna_fm` `params.repr_layer` integer 0–12).  
`name` must already be in `features/registry.yaml`. `add` / `mean` require the **included** channels to have the same C (e.g. onehot(4)+kmer(64) cannot add). Prefer changing one knob per round: one `enabled`, one tunable, or `fusion` / `fusion_include`.

## Round types

| `round_type` | Changes | Extra prepare |
|--------------|------|----------------|
| `model` | `train.py` only | No |
| `feature` | `features/active.yaml` ± `train.py` | No (features computed at train time) |
| `joint` | Major changes to both | No (avoid in v1) |

Suggested quota: about 70% model / 25% feature, **only when campaign notes have no MUST and the bottleneck card `next_do` is empty**. When Iter-injected campaign notes `MUST this round` name a concrete registered name / explicit knob, this round must execute that change; do not default to retuning LR or concatenating another feature without note support. A wrong `round_type` does not change keep/discard.  
`rna_fm` / `rinalmo` / `evo` / `evo2` (L4) are valid `[feature]` actions (on/off; `rna_fm` may also tune `repr_layer`). Missing local weights → crash; pip / downloads forbidden. `evo` / `evo2` are 4096-d / ~12 GB VRAM (registry `heavy: true`); when **full and a light L4 is already present** (`rna_fm` / `rinalmo`), the orchestrator removes them from valid MUST. Features that were discarded / crashed are not MUST again for the rest of the campaign; knobs are tracked as `(framework, knob)` and may be retried after a head change. If a crash fix has no new commit: clear the veto, do not count toward `max_agent_failures`, and the next round follows still-valid next items in the notes.  
After 10 consecutive `[model]` discards on the current framework, if notes do not name a valid remaining head, the orchestrator still samples one remaining menu item uniformly and injects `next_framework`. After at least one keep/discard, notes may name a remaining name and the orchestrator adopts it (sticky) without waiting for 10. Ignore `next=head` at the start. See [frameworks.md](frameworks.md). Start with `baseline`. Do not pick another name yourself. Veto order: crash block (must fix the short signature; do not swap layers) > expand K > `next_framework` > notes MUST. Do not use `cause=head` to self-pick a backbone name.  
When `[model]` does not change the backbone, only tune in-framework menu items (`HIDDEN_DIM` / `LR` / `BATCH_SIZE` / dropout / loss / EMA; for `resnet1d_pool`, kernel and residual blocks); see [frameworks.md](frameworks.md). In-framework keep/discard is relative to **that head's** `framework_best` (a global discard can become a keep). The first shot after changing the backbone still leaves `framework_base` in the TSV as that head's baseline; if global best was not raised, the orchestrator `git reset`s to the global-best HEAD and does not treat a worse head as the working base for later F/θ.

## Per-round steps

The step list and keep/discard rules are the same in both modes. **Who runs 4–6 differs:**

| Step | orchestrator overnight | Interactive, without orchestrator |
|------|-------------------|-------------------------|
| 1–3 Read state, declare hypothesis, whitelist commit | agent; STOP immediately after commit. Do not run `train.py`, read `workspace/run.log`, or edit the TSV. A one-line summary of the previous overnight crash is written into the Iter prompt by the orchestrator | agent |
| 4–6 `train.py`, write TSV, keep/discard reset | **orchestrator** | agent does it |

1. Read `workspace/<stem>.tsv`, `features/active.yaml`, git HEAD, this skill / handoff. If Iter injected campaign notes, first read `MUST this round`: with no veto, execute that name/knob. Crash block / `next_framework` still take priority. Do not self-pick a backbone name; when changing head, only implement the injected `next_framework`.  
2. Declare `round_type` + one main hypothesis (do not repeat already-discarded descriptions). Expand-sampling `[sample]` rows have already retrained the current best with the new `K_train`: do not checkout / rerun historical commits; this round must be a new hypothesis.  
3. Edit only whitelist files → `git commit` (do not commit tsv under `workspace/`). If there are no new changes, STOP; do not empty-commit (orchestrator will not retrain an unchanged HEAD). Overnight STOP here.  
4. `python train.py > workspace/run.log 2>&1` (internally `load_prepared_data` encodes per recipe).  
5. Log `workspace/<stem>.tsv` (prefix `[feature]` / `[model]`). `[model]` includes `framework=<name>` (start with `framework=baseline`); first shot after changing backbone adds `framework_base`. Closed-set `cause=<head|feature|knob|crash>` (`sample` is written by the orchestrator). Do not write `cause=head` when there is no `next_framework` (book as knob tuning, not a head change). First shot after changing backbone: `[model] framework=<name> framework_base cause=head …`.  
6. Decision metric improves vs **current best** by ≥ **min_delta** → keep; smaller improvement or worse → `git reset` to the pre-experiment HEAD.  
   - Regression: **val_spearman** higher is better  
   - Classification: **val_accuracy** higher is better  
   - `min_delta` value: see `rnaomni.env`; injected by orchestrator Iter prompt; shared with stop-search patience

## Registered features (on/off)

Authoritative source: `features/registry.yaml`. Currently includes:

| Layer | name | Notes |
|----|------|------|
| L0–L2 | `onehot` / `length_frac` / `kmer` | Low cost; often used as baseline |
| L3 | `struct_thermo` / `struct_dotbracket` / `struct_bpp` | ViennaRNA; slower than L0–L2; shared `_vienna` LRU |
| L4 | `rna_fm` / `rinalmo` / `evo` / `evo2` | High cost; see matching `examples/*.handoff.md` (C4 already allows on/off) |

May only `enabled` names already in `features/registry.yaml`. Unregistered names fail to load. MSA is ignored in this version.

Open criteria: **registered + allowed by handoff/procedure**. `agent_tunable`: kmer `[k]`, `rna_fm` `[repr_layer]`; `rinalmo` and most others are `[]` (on/off `enabled` only). There is no separate `agent_open` field.

## L4 notes

- Full encode wall-clock is **not counted** toward `TIME_BUDGET=60s` (exploration), but still consumes startup time; each round is a new process with no cross-round memo.
- Large `input_dim` requires adapting `train.py` (read `data["input_dim"]`); may tune batch / structure to avoid OOM.  
- Missing local weights → crash; **forbidden** to pip / download.  
- Details and re-check commands: `examples/rna_fm.handoff.md`, `examples/rinalmo.handoff.md`, `examples/evo.handoff.md`, `examples/evo2.handoff.md`.

## Red lines

- Do not change `features/builders/`, `features/registry.yaml`, `prepare.py`, `rnaomni/prepare/`, `rnaomni.env`, `.env`, folds / metric / `TIME_BUDGET`.  
- Do not install packages, download new weights, or run MSA precompute.  
- Decide only on the primary metric: regression **val_spearman**; classification **val_accuracy**; keep requires ≥ min_delta. Do not change splits / evaluation / `TIME_BUDGET`.  
- `train.py` must use `data["input_dim"]`; do not hardcode `4`; classification must use `data["num_classes"]`; do not hardcode the class count.  
- Changing framework only implements the Iter-injected `next_framework` (contract [frameworks.md](frameworks.md)). Do not invent class names outside the menu.  
- Deploy (`python -m rnaomni.deploy`) is not an overnight responsibility.
