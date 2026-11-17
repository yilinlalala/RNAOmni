# finalize 排障备忘：`**kwargs` 推断 bug 与启动环境

> 时间戳：`2026-08-29`
> 类型：`NOTE`（排障经验，勿删）
> 范围：`rnaomni.finalize` 对 Predictor 的 `_model_kwargs` 推断 + 各 worktree 的启动方式
> 触发案例：`worktrees/20260828_020937_Reg_EL_PC3` 的 finalize 多次失败

---

## 一、`**kwargs` 推断 bug（会波及所有带 `**kwargs` 的 Predictor）

### 现象
```
Cannot infer Predictor argument 'kwargs' from train.py snapshot.
EXIT=1
```

### 触发条件
某个 best commit 的 `train.py` 里，被选为 `Predictor` 的类 `__init__` 带 `**kwargs`。
例：`d577755` 中 `Predictor = AttnPoolMLP`，而
`AttnPoolMLP.__init__(self, input_dim, hidden_dim=256, dropout=0.1, num_outputs=1, **kwargs)`。

### 根因
`rnaomni/finalize.py` 的 `_model_kwargs()` 遍历 `__init__` 的每个参数来推断模型构造参数；
遇到 `**kwargs`（`VAR_KEYWORD`，无默认值）时走了 `else` 分支直接 `SystemExit`。

### 修复（`_model_kwargs` 循环里跳过可变参数）
```python
for pname, param in sig.parameters.items():
    if pname == "self":
        continue
    if param.kind in (
        inspect.Parameter.VAR_POSITIONAL,
        inspect.Parameter.VAR_KEYWORD,
    ):
        continue
    # ...原有 attr_map / default / else 逻辑
```

### 重要提示
> **凡是 Predictor 类带 `**kwargs` 的 worktree，只要 best commit 的 Predictor 是这种签名，
> finalize 就会在这里崩。**
> 目前（`013253 / 022350 / 022406 / 171221 / 171233`）能跑，是因为这些 worktree 当前
> commit 的 Predictor **恰好不带 `**kwargs`**，不代表以后不会。
> **每个 worktree 是独立的一份 `rnaomni/finalize.py`（相互不共享）**，所以：
> - 该修复应**分别同步到所有在跑 / 待跑的 worktree** 的 `rnaomni/finalize.py`；
> - 只改当前 worktree 不影响其它分支 / worktree（各 worktree 多为 detached HEAD 或独立分支）。

---

## 二、启动环境（比代码更常见的坑）

同一 finalize 进程反复崩，很多并不是代码问题，而是**启动时用了错的环境**：

### 排障结论
- 正确运行时用 **`RNAOMNI_CONDA_ENV`**（默认 `myenv3.11`）里的 Python 3.11，例如 `$(conda info --base)/envs/$RNAOMNI_CONDA_ENV/bin/python`
- 错误代价最小的命令 `which python` 指向的是 **base `anaconda3` 的 Python 3.8.8** → 连环报错：
  1. old torch（3.8）在 `set_num_interop_threads` 时 `std::terminate` 硬崩
     （`c10::Error: cannot set number of interop threads after parallel work has started`）
  2. base 3.8 没有 ViennaRNA → `import RNA` 抛 `ModuleNotFoundError`
  3. 还可能有 `GLIBCXX_3.4.31 not found`（libstdc++ 版本不够）

### 标准启动模板（每个要重启 finalize 的 worktree 都照这个来）
```bash
WTR=<worktree 根目录>
ENV_NAME=${RNAOMNI_CONDA_ENV:-myenv3.11}
PYBIN="$(conda info --base)/envs/${ENV_NAME}/bin/python"
CONDA_LIB="$(conda info --base)/envs/${ENV_NAME}/lib"

export CUDA_VISIBLE_DEVICES=<空闲GPU> ENCODE_TIMEOUT_SEC=0 RNAOMNI_ENCODE_MODE=online RNAOMNI_CPU_THREADS=4
export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$WTR${PYTHONPATH:+:$PYTHONPATH}"
$PYBIN -u -m rnaomni.finalize --commit <best commit> --output-dir 'workspace/runs/<tag>/finalize'
```

要点：
- **用 `myenv3.11` 的 python，不是 base**
- **把 conda lib 放到 `LD_LIBRARY_PATH` 最前**（避免/系统 libstdc++ 版本不够）
- `PYTHONPATH` 指向当前 worktree 根

---

## 三、顺手改的健壮性（可选，已在 020937 提交）
- `finalize.py`：跳过 `VAR_KEYWORD / VAR_POSITIONAL`（见第一节）。
- `threads.py`：去掉 `set_num_interop_threads` 调用。某些 torch 构建在 interop 池已启动后会
  `std::terminate`（不能靠 `except RuntimeError` 捕获）。interop 线程本 run 用不到，
  保留 `set_num_threads` 即可。参见 `tools/smoke_tf_heads.py` 里同样的规避注释。

### 排查清单（finalize 崩了先看）
1. log 末尾 `END finalize exit=` 前最后的 traceback 用的是哪个 python 路径？
   （`runpy.py` / `site-packages` 路径可判断是 3.8 还是 3.11）
2. `/_vienna.py` 报 `import RNA` 失败 → 几乎肯定是 base 3.8（myenv3.11 有 RNA）。
3. `c10::Error ... interop threads` / `GLIBCXX_3.4.31` → 环境 / libstdc++ / 线程问题。
4. `Cannot infer Predictor argument 'xxx'` → 代码推断问题，走第一节的修复。
