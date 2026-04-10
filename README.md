# hello-cute

## 简介

众所周知，CuTe 的学习曲线比 vim 还陡峭，文档又比较简略，尤其是 layout 的各种变换，对新手并不友好。这个仓库打算按照 [CuTe 官方文档](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/index.html) 的顺序写一套教程，重点放在 layout 的理解和运算上。

每个章节会逐步补充配套练习。读者可以把仓库 clone 到本地，直接运行测试脚本做题并查看结果。

## 目录

- [hello-cute01：cute概述及layout入门.md](hello-cute01：cute概述及layout入门.md)
- [hello-cute02：layout的代数运算.md](hello-cute02：layout的代数运算.md)

## 环境部署

下面的命令默认在 Linux/macOS shell 下执行；如果你的环境里 `python` 指向的不是 Python 3，请把命令中的 `python` 改成 `python3`。

### 1. 拉取仓库和 submodule

首次 clone 时，推荐直接带上 `submodule`：

```bash
git clone --recursive <repo-url> hello-cute
cd hello-cute
```

如果你已经 clone 过仓库，但还没有拉取 submodule，请执行：

```bash
git submodule update --init --recursive
```

当前仓库依赖的 submodule 是：

- `third_party/cutlass`

说明：

- 当前测试脚本会在需要时回退到仓库内置的 `pycute` 实现，而 `pycute` 位于 `third_party/cutlass/python` 下。
- 因此如果没有初始化 submodule，测试脚本无法正常工作。

### 2. 创建 Python 虚拟环境

推荐使用独立虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3. 安装可选依赖

如果你只是运行当前仓库里的交互测试，完成 submodule 初始化后通常就够用了，不强制要求安装额外 Python 包。

如果你希望同时安装官方的 CuTe Python DSL，可以按 CUDA 版本选择：

CUDA 12 / 默认环境：

```bash
pip install -r requirements.txt
```

CUDA 13：

```bash
pip install -r requirements-cu13.txt
```

说明：

- `requirements.txt` 会继续引用 `third_party/cutlass/python/CuTeDSL/requirements.txt`
- `requirements-cu13.txt` 会继续引用 `third_party/cutlass/python/CuTeDSL/requirements-cu13.txt`
- 当前版本的 `cutlass.cute` 更偏向在 `@cute.jit` 场景里使用；对于本仓库这种命令行交互题，测试脚本会在必要时自动回退到等价的 `pycute` 判分实现

## 测试方式

当前已提供的测试名如下：

- 第一章：`layout_base`
- 第二章：`layout_algebra`

### 1. 交互式测试

```bash
python tests/run.py layout_base
python tests/run.py layout_algebra
```

- `layout_base` 会生成 5 道基础 layout / `crd2idx` 题。
- `layout_algebra` 会生成 2 道 layout 代数题，分别考 `logical_divide` 和 `logical_product`，需要直接写出结果 layout。

### 2. 固定随机种子，复现同一套题

```bash
python tests/run.py layout_base --seed 20260409
python tests/run.py layout_algebra --seed 20260410
```

### 3. 查看标准答案

```bash
python tests/run.py layout_base --show-answers
python tests/run.py layout_base --seed 20260409 --show-answers
python tests/run.py layout_algebra --show-answers
python tests/run.py layout_algebra --seed 20260410 --show-answers
```

### 4. 非交互判分

适合自测或写脚本时使用：

```bash
python tests/run.py layout_base --seed 20260409 --answers 5 9 11 14 7
python tests/run.py layout_algebra --seed 20260410 --answers "(4,3):(1,4)" "((3,4),(2,2)):((1,3),(12,24))"
```

### 5. 测试脚本位置

- `tests/run.py`
- `tests/layout_base/quiz.py`
- `tests/layout_algebra/quiz.py`

## 参考

- [CuTe 官方文档](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/index.html)
