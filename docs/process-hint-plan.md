# 过程提示搜索（落地步骤）

> 时间戳：`2026-08-22`  
> 类型：`PLAN`  
> 状态：`TODO`（C0/N0/N1/**N2** 已落地；C1 强制换层已删除）  
> 思路：[`process-hint.md`](process-hint.md)  
> 范围：按 **这一组数据的特点** 出下一刀实验。C0 的 `cause=` / `framework_base` 仍记账和否决；C1 计数锁层已删除。不改评测合同、keep/discard、`min_delta`、`TIME_BUDGET`。  
> 相关：`rnaomni/orch/search.py`、`rnaomni/orch/prompts.py`、`rnaomni/orch/loop.py`、`skill-library/feature/agent-ops.md`、`program.md`、`tools/smoke_bottleneck_card.py`  
> 对照：[`relax-model-feature-constraints.md`](relax-model-feature-constraints.md)（合法动作仍是菜单 + registry；本文改 **怎么选**，不发明积木）

过夜停搜数字不动：全局 `patience=15`、`SAMPLE_EXPAND_PATIENCE=3`、`min_delta`。  
`FRAMEWORK_PATIENCE=10`：笔记可在一次 keep/discard 之后点 remaining 头；没点名且满 10 次则仍均匀抽（护栏）。不上未试过概率地板。C2（计数满了提前均匀抽头）**取消**，并入 N2。

---

## 一句话

缺的不是再精细的「拧参 / 组特征」计数，而是 **针对这份数据记下来、并据此提出具体下一枪**。  
外置 API 角色：开局只看 X 出先验；每轮用这次 `do` 的结果修订信念表（不是追加长文）。Iter 执行表上的那一刀（哪个已登记特征、哪套剩余菜单头、还是扩 K）。

C0 记账和否决还在；C1 强制换层已从运行时删除。

---

## 和 C0/C1 的关系

| | C0（记账/否决） | N0–N2（选下一刀） |
|--|-----------------|------------------------|
| 问的问题 | 平台期该动哪一层 | 这份数据下一步具体做什么 |
| 主序 | 否决 crash / 扩 K / 压制 `head` | 笔记出具体实验 |
| 换头 | 编排器均匀随机；agent 不点名 | 笔记在 **剩余菜单** 里点名，编排器仍注入 `next_framework` |
| 保留 | `cause=` 记账、`framework_base` 直接效应、crash 短签名 | 上述作为笔记输入，不倒 TSV、不读 `run.log` |

一次 discard 仍不能写成生物学结论。修订笔记时：没做 `do(H)` 不准改 H 格；拧参失败不准改成「这类 RNA 不适合该头」。

---

## 本批范围

| 批次 | 做什么 | 行为变化 |
|------|--------|----------|
| **C0** | `cause=`、`framework_base`、瓶颈卡、压制 `head` | **DONE** |
| **C1** | `fail[knob/feature]>=3` 强制换层 | **REMOVED** 运行时已删；`fail[*]` 仍记在瓶颈卡上 |
| **C2** | 计数满了提前均匀抽头 | **取消**（改由笔记选头） |
| **N0** | 战役信念表 + 开局只看数据的先验；注入 Iter | **DONE** 每场有一份短笔记；规则主序暂不变 |
| **N1** | 每轮 API 修订表；空 `next_do` 时执行表上的具体 F/θ/S | **DONE** 不再默认「先宁参再组特征」；输出必须点到合法动作 |
| **N2** | 表可在剩余菜单里点名 `next_framework` | **DONE** 换头看这份数据；满 10 次仍可均匀抽护栏；非法头回退 A |

**不做（N 全程）：** 发明菜单外类名 / 未登记特征；改 keep/discard / folds / `TIME_BUDGET`；agent 读 `run.log`；用旧战役 TSV raw best 排头；一篇越写越长的报告当政策；最终 val 当 RL；复活 S5。合法动作仍只有菜单头 + `features/active.yaml` 白名单 + 框架内旋钮 + 编排器扩 K。策略再聪明也变不出没上架的积木。

---

## 已拍板（N 实现时当约束）

### 角色

外置 **诊断角色**（独立短 API，prompt → 结构化回答；可用现有 `agent_backend`，不必再开 overnight Iter）。  
**没有执行权：** 不改 git、不跑 `train.py`、不改 keep/discard。只读写信念表。实现仍由 Iter agent 按注入做。

### 信念表（分格，不准长文）

每场一份，建议 `workspace/campaign_note.json`（或 checkpoint 旁）。整份注入 Iter 时 **<800 字** 渲染。字段闭集：

```
data     任务类型、类数/回归、长度摘要、data_mode、K、enabled[]、active 头
         （能算的先算，模型只解释，不编造统计）
prior    开局、不看 Y：更像缺组成 / 结构 / 长程 / 样本量（可多选，带一句理由）
F        未试 | 试过抬了 | 试过没抬     证据=哪次 do(F)
H        仅记录各已抽头的 framework_base Δ（相对换头前全局 best）
θ        剩余合法旋钮；幅度加权的连续失败（可选）
S        K 是否还是短板；上次扩 K 的 kept
next     具体一枪（见下）+ 对应 data/prior 的哪一条
```

每轮 API **只改被这次 `cause=` 对上的格子** + 重写 `next`。禁止改写历史格子里的证据行。

开局：只填 `data` + `prior`，不准看 Y / TSV 分数。

### `next` 闭集（必须具体）

`next` 必须是下面之一，禁止「再试试别的」：

| kind | 内容 | 谁执行 |
|------|------|--------|
| `feature` | 一个已注册 name 的开关或该 name 的 `agent_tunable` | Iter `[feature]` |
| `knob` | 一个框架内明旋钮（`HIDDEN_DIM`/`LR`/…） | Iter `[model]` 停留 |
| `head` | **剩余菜单** 里的一个名字 | 编排器注入 `next_framework`；Iter 只实现它 |
| `sample` | 扩 K | 现有 sample 轮（无 agent） |
| `crash` | 修短签名 | 现有 crash 块 |

点了不在剩余菜单 / 未注册 / 不可扩 K 的动作 → 丢弃。N0 期间即丢掉建议；N1 起走回退 A（上一份合法 `next`；没有则不 MUST）。

### 否决（硬，笔记不能推翻）

1. 最近非 sample 行是 crash → 本轮 `crash`  
2. `should_expand_sample()` 已触发 → 本轮走现有扩 K（可与笔记 `next=sample` 对齐）  
3. 非法 `next`（上表）→ 回退  
4. ~~C1 `fail>=3` MUST 换层~~ **已删除**  

C0 的 `head` 压制：**N2 前仍生效**（无效 `framework_base` 后不准只换头）。N2 若笔记点头，须 `head` 未被压且名字仍在 remaining；被压时笔记只能出 F/θ/S。

### API 失败

超时 / 非 JSON / 缺字段 / `next=head`（N1）→ 本轮当笔记未更新：**沿用上一份合法 `next`**；一份都没有则不 MUST。战役不红。crash 块仍否决换层。

---

## N0 — 表 + 开局先验（先做，可不调选层）

**状态：DONE**  
诊断调用：`complete_isolated`（空临时目录，无仓库工具）；模型与 Iter 相同（`OPENCODE_MODEL` / `CURSOR_MODEL`）。笔记路径：`workspace/campaign_note.json`。  
**触及：** `rnaomni/orch/note.py`、`agent_backend.py` `complete_isolated`、`prompts.py`、`loop.py` 开局一次、agent-ops、`tools/smoke_campaign_note.py`

### 步骤

1. 确定性 `data` 块：task_type、metric、`active.yaml` enabled、fusion、`data_mode`/K、`active_framework`、remaining 个数。不读 `run.log`。  
2. 开局 API：只给 `data` + 合法动作清单（registry 名、菜单剩余、明旋钮），**不给 Y**。填 `prior` + 建议 `next`（N0 只注入给 Iter 看，**不强制**）。  
3. Iter 在瓶颈卡下渲染笔记（<800 字；无 `run.log`）。C1 MUST 与换头日程暂不变。  
4. 不改 TSV 表头；不改 keep/discard。

### smoke

| 例 | 期望 |
|----|------|
| 无网络 / API 失败 | 仍有确定性 `data` 块；战役不红 |
| 开局 prompt | 不含 val / best / TSV 行 |
| 渲染 | <800 字；含 enabled 列表；无 `run.log` |
| 旧 C0/C1 smoke | 仍绿 |

---

## N1 — 每轮修订；笔记驱动 F/θ/S

**状态：DONE**  
C1 强制换层已删除。开局 `next` 第一枪 MUST；keep/discard/crash 打修订 API；扩 K 只改 S 格；回退 A；crash 块否决换层。  
**触及：** `rnaomni/orch/note.py`、`prompts.py`、`loop.py`、`agent-ops.md`、`tools/demo_campaign_diagnose.py`、`tools/smoke_campaign_note.py`  
N0 之后。不在本批让笔记点头。  
**诊断 demo：** `python tools/demo_campaign_diagnose.py`（`--live` 才打真实 API）。

### 已拍板

1. **C1 内层 MUST 已删除**（`fail[*]` 只记账，不锁层）。  
2. **第一枪就 MUST 开局 `next`**（选项 A）。开局 API 失败且没有合法 `next` → 本轮不 MUST。  
3. **修订 API：keep / discard / crash 都打。** 扩 K 由编排器改 S 格，不打 LLM。  
4. **回退 A：** 非法 JSON / 未注册 name / `next=head` / API 失败 → 沿用上一份仍合法的 `next`；一份都没有 → **不 MUST**（agent 自选 F/θ）。刚试过的 `kind+name`、以及已经 `enabled` 且无 `agent_tunable` 的 feature，都算非法（防止 keep 后再 MUST 同一刀、空 commit 三次出局）。不按「剩余动作最多的一层」硬锁层。  
5. **crash 块继续否决「不修直接换层」。** `next_do=crash` 盖过笔记：本轮必须按短签名修一枪。crash 修订写出的 `next` 是修好之后的下一刀，且必须避开刚炸的 name。

### 步骤

1. 每轮 train+reconcile 之后：把这次 `cause=`、`result`、Δ（相对所用 baseline）、description 截断，交给 API，只许改对应格 + `next`。crash 也打，避免下一枪重复同一失败。  
2. 下一轮 Iter：若否决条未命中且 `next` 为 `feature`/`knob`/`sample` → MUST 该具体动作（name 或旋钮写进卡）。开局那条从第 1 枪起同样 MUST。  
3. `next=head` 在 N1 **忽略**（尚未授权选头），走回退 A。  
4. 错写 `round_type` 仍不改 keep/discard。  
5. agent-ops：禁止无笔记依据的「默认再拧 LR / 再 concat 一个」。必须执行卡上的 name/旋钮，或 STOP 不空 commit。

### smoke / demo

| 例 | 期望 |
|----|------|
| API 返回 `feature` + 已注册 name | 卡上 MUST 该 name |
| API 返回未注册名或 `head` | 丢掉，沿用上一份合法 `next`；没有则不 MUST |
| 连续两次修订 | `F` 格证据行追加，不删旧证据 |
| crash 修订 | 笔记 `next` 换名；本轮 Iter 仍是 crash 块，不可换层 |
| `fail[knob]>=3` | **不** MUST feature，除非笔记如此 |
| `python tools/demo_campaign_diagnose.py` | 走查开局 MUST + 修订 + 回退 A + crash 否决 |
| `python tools/smoke_framework_search.py` | 仍绿 |

---

## N2 — 笔记在剩余菜单里点头

**状态：DONE**  
**已拍板：** (1B) 当前头至少一次 keep/discard 之后才允许点头；开局 `next=head` 忽略。(2B) 笔记点了合法 remaining 名 → 采用；没点且 streak≥10 → 仍均匀抽。(3A) 非法 / 已 tried 头走回退 A；不上概率地板。  
**触及：** `search.py` `ensure_next_framework(..., note_head=)`、`note.py` `n2_executable_next` / `legal_note_head_name`、`loop.py`、`prompts.py`、`agent-ops.md`

### 步骤

1. `next.kind=head` 且 name∈remaining 且 `head` 未被压且 `valid_rounds>0` → `ensure_next_framework` **采用该 name**（不再等满 10 次才抽）。sticky：抽出后未实现前保持。  
2. 笔记没点合法头且 `FRAMEWORK_PATIENCE` 已到 → 仍从 remaining 均匀随机（护栏）。  
3. 第一枪仍 `framework_base`、仍从全局 best 检出；Δ 写回 H 格。无效 → C0 压制，笔记下一枪不能再只点头。  
4. 排序依据只有：**本场** `framework_base` Δ + `prior`/`data`。禁止旧战役 TSV raw best。  
5. 全局 patience 仍停搜。非法头不改抽、不上未试过地板。

### smoke

| 例 | 期望 |
|----|------|
| 笔记点 remaining 中的 `tf_mean` 且已有 keep/discard | `next_framework=tf_mean`，非随机 |
| 开局 / `valid_rounds=0` 点头 | 忽略 |
| 点已 tried 或不在菜单 | 丢弃，回退 A |
| `head` 已压制 | 即使点头也不抽，卡上须 F/θ/S |
| 满 10 次且笔记未点头 | 仍均匀抽 remaining（旧框架 smoke） |
| `python tools/smoke_framework_search.py` | 仍绿 |

---

## 以后再做（本文件不排期）

- 人上架新头 / 新 builder（动作集变大，不是本策略能变出来的）。  
- 特征 keep 后的适配轮注入。  
- 层上 Thompson/UCB（若笔记主序不够，再把 `next` 的 F/θ 选择换成信念抽样）。  
- 受控 `joint`。

---

## 红线

- 不改评测 / folds / `TIME_BUDGET` / `min_delta` / keep/discard。`framework_base` 仍留下当该头基线。  
- 不发明菜单外类名、不 `enabled` 未登记特征、不读 `run.log`。  
- 笔记不能否决 crash / 非法动作 /（N2 前）`head` 压制。  
- 不用旧战役 raw best 排头。  
- 信念表分格修订，不当成长篇战役日记。  
- N0 / N1 / N2 不要捆同一 commit。C1 强制换层已从运行时删除。

---

## 完成定义

**N0：** 开局有 `data`+`prior`；Iter 能看见；API 挂了不红；C0/C1 smoke 仍绿。  
**N1：** 每轮表会按 `cause=` 修订；`next` 的 feature/knob 能 MUST 到具体 name/旋钮；C1 锁层已删。  
**N2：** 笔记可在 remaining 里点头并注入（须已有 keep/discard）；无效 `framework_base` 仍压制；满 10 次无点名仍均匀抽；非法头回退 A。

---

## 已落地存档（C0）

实现细节以代码为准。C1 计数锁层已删除；`fail[*]` 仍写在瓶颈卡上供查看。

### 词表 `cause=`

闭集：`head` / `feature` / `knob` / `sample` / `crash`。缺省按 kind 推断。无 `next_framework` 却写 `cause=head` → 记 `knob`。开局 `baseline` 不是 `do(H)`。这是过程层标记，不是因果估计。

`do(H)` 有效 = `framework_base` 相对换头前全局 best ≥ `min_delta`，否则 `result=flat` 并压制 `head`。`[feature]` keep 或样本 keep → 清压制与 `fail[*]`。

### C0 瓶颈卡

字段：`last_do` / `result` / `fail[head|feature|knob|sample]` / `suppressed` / `next_do`。`<400` 字。smoke：`tools/smoke_bottleneck_card.py`。

### C1

**已删除。** 不再根据 `fail[knob/feature]>=3` MUST 换层。
