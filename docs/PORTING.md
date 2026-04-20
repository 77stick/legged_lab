# 把 legged_lab 移植到新机器

这份文档描述如何在**新机器**上从打包文件恢复本项目，并能跑通训练/回放。

打包过程见 `scripts/tools/pack_project.sh --help`。

---

## 0. 前置条件（新机器）

- Linux（推荐 Ubuntu 22.04 / 24.04）
- NVIDIA GPU + 匹配的驱动（Isaac Sim 要求 NVIDIA driver ≥ 535，RTX 卡）
- 可上网（拉 Isaac Lab、pip 包）
- 已装 `conda`（Miniconda 或 Anaconda）

确认命令能跑：

```bash
nvidia-smi
conda --version
```

---

## 1. 解压

```bash
tar -xf legged_lab-<stamp>.tar.zst      # 或 .tar.gz
cd legged_lab
```

> 如果包里带了 `.git/`，`git status` 应该能看到历史。如果没带，你在这台机器上就是一个干净的快照。

---

## 2. 装 Isaac Lab

`legged_lab` 本身**不打包 Isaac Lab**——Isaac Lab 自带 Isaac Sim 二进制、体积巨大，且强依赖机器的驱动版本，应该在每台机器独立安装。

1. 按照 [Isaac Lab 官方安装文档](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) 选 **pip 安装** 或 **binary 安装**。本项目测试过的 Isaac Lab 版本 ≈ `0.47.x`、Isaac Sim `4.5.0`。
2. 把 Isaac Lab 放到**绝对路径**，例如：`/home/<user>/IsaacLab`。
3. 按 Isaac Lab 文档执行过一次 `./isaaclab.sh --install` 后，记住它用的那个 conda env 名字（常见是 `env_isaaclab` 或 `isaaclab`）——下一步要用。

---

## 3. 创建 `.codex/env.local.sh`

这是告诉本项目的包装脚本去激活哪个 conda env、Isaac Lab 在哪：

```bash
cp .codex/env.local.sh.example .codex/env.local.sh
vim .codex/env.local.sh
```

改成你本机的真实值：

```bash
export CODEX_CONDA_ENV="env_isaaclab"                    # 上一步装好的 env 名
export ISAACLAB_PATH="/home/<you>/IsaacLab"              # 绝对路径
# 可选：如果 conda 不在常见位置：
# export CODEX_CONDA_SH="/home/<you>/miniconda3/etc/profile.d/conda.sh"
```

> `.codex/env.local.sh` 被 `.gitignore` 了（每台机器独立配），打包时也主动排除。

---

## 4. 装项目自身的 Python 包

**所有 Python 命令必须通过 `bash .codex/run-in-env.sh ...`**，这条脚本会先激活 `CODEX_CONDA_ENV`，然后把 `ISAACLAB_PATH` 注入 `PYTHONPATH`。

### 4.1 装本仓的 `rsl_rl` 分叉（必须，不是 PyPI 的 `rsl-rl-lib`）

```bash
bash .codex/run-in-env.sh python -m pip install -e ./rsl_rl
```

> ⚠️ 本项目用的是**带 AMP 支持**的定制版 `rsl_rl`，位置在 `./rsl_rl/`。PyPI 上的 `rsl-rl-lib` **没有** `AMPRunner`，装错会报 `ImportError: cannot import name 'AMPRunner'`。

### 4.2 装 legged_lab 主包（editable）

```bash
bash .codex/run-in-env.sh python -m pip install -e ./source/legged_lab
```

### 4.3 可选依赖

```bash
bash .codex/run-in-env.sh python -m pip install tensorboard wandb
```

---

## 5. 验证

列出所有已注册的 Gym 任务：

```bash
bash .codex/run-in-env.sh python scripts/list_envs.py
```

应该能看到像 `LeggedLab-V1-AMP-v0`、`LeggedLab-G1-AMP-v0` 这样的条目。

快速回放 V1 动画数据（需要 GUI；远程服务器用 `--headless`）：

```bash
bash .codex/run-in-env.sh python scripts/play_anim.py --robot v1 --num_envs 4
```

开始训练：

```bash
bash .codex/run-in-env.sh python scripts/rsl_rl/train.py \
  --task LeggedLab-V1-AMP-v0 \
  --headless
```

---

## 6. Git LFS（如果你的打包带了 `.git/`）

本项目部分资产（`*.usd`、`*.dae`、`*.obj`、`*.png` 等）在原仓库里是 **git-lfs** 跟踪的。打包脚本会把 `.git/lfs/` 一起带过来，**不需要再 `git lfs pull`**，工作区文件是完整的。

如果你是从 GitHub 上重新 `git clone` 而不是解压包，则需要：

```bash
sudo apt install git-lfs
git lfs install
git lfs pull
```

---

## 7. 常见坑位

| 症状 | 原因 | 修复 |
|---|---|---|
| `ImportError: cannot import name 'AMPRunner'` | 装了 PyPI 的 `rsl-rl-lib`，不是本仓 `./rsl_rl` | `pip uninstall rsl-rl-lib && pip install -e ./rsl_rl` |
| `.codex/env.local.sh is missing` | 没从 example 复制 | 执行第 3 步 |
| `CODEX_CONDA_ENV ... cannot be activated` | 名字写错 / env 不存在 | `conda env list` 确认名称 |
| `ISAACLAB_PATH missing or invalid` | Isaac Lab 没装 / 路径写错 | 走完第 2 步 |
| `PhysX error: Static friction ...` | 非致命，可忽略 | — |
| `python` 直接跑报 `ModuleNotFoundError` | 没走包装脚本 | 所有 Python 入口都要 `bash .codex/run-in-env.sh ...` |

---

## 8. checkpoint 迁移

训练日志（`logs/`）默认**不打包**（通常几 GB）。如果新机器要继续某次训练，打包时显式带上具体的 checkpoint：

```bash
bash scripts/tools/pack_project.sh \
  --with-checkpoint logs/rsl_rl/v1_amp/2026-04-20_14-37-03/model_3000.pt
```

在新机器上通过 `--resume` 恢复即可：

```bash
bash .codex/run-in-env.sh python scripts/rsl_rl/train.py \
  --task LeggedLab-V1-AMP-v0 \
  --resume \
  --load_run 2026-04-20_14-37-03 \
  --checkpoint model_3000.pt \
  --headless
```
