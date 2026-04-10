# CuTe 入门（Getting Started With CuTe）

原文：<https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/00_quickstart.html>

CuTe 是一组基于 C++ CUDA 模板（template）的抽象，用来定义并操作线程（threads）与数据（data）的分层多维布局（hierarchically multidimensional layouts）。CuTe 提供了 `Layout` 和 `Tensor` 两类核心对象，用比较紧凑的方式封装数据的类型（type）、形状（shape）、内存空间（memory space）和布局（layout），同时替用户处理复杂的索引计算。这样，程序员可以把注意力集中在算法的逻辑描述上，而把机械性的 bookkeeping 交给 CuTe。借助这些工具，我们可以更快地设计、实现并修改各种稠密线性代数（dense linear algebra）运算。

CuTe 的核心抽象是“分层多维布局”。布局可以和数据数组组合，表示张量（tensor）。这种布局表达能力非常强，足以覆盖高效实现稠密线性代数时需要的大多数场景。布局还可以通过函数式组合（functional composition）做进一步运算，而很多常见操作，例如分块（tiling）和分区（partitioning），正是建立在这种组合能力之上的。

## 系统要求（System Requirements）

CuTe 与 CUTLASS 3.x 共享软件要求，包括 `NVCC` 和支持 C++17 的宿主编译器（host compiler）。

## 知识前置（Knowledge Prerequisites）

CuTe 是一个只含头文件（header-only）的 CUDA C++ 库，要求使用 C++17。

本文默认读者具备中等水平的 C++ 经验。例如，你需要能读写模板函数和模板类，并理解 `auto` 关键字如何让编译器推导返回类型。文中会尽量温和地介绍 C++ 细节，但不会从零开始。

同时，本文也默认读者具备中等水平的 CUDA 使用经验。例如，你需要知道 host code 和 device code 的区别，并知道如何启动 kernel。

## 构建测试与示例（Building Tests and Examples）

CuTe 的测试和示例会随着 CUTLASS 的常规构建流程一起编译和运行。

- 单元测试（unit tests）位于 [`test/unit/cute`](https://github.com/NVIDIA/cutlass/tree/main/test/unit/cute)。
- 示例（examples）位于 [`examples/cute`](https://github.com/NVIDIA/cutlass/tree/main/examples/cute)。

## 库的组织方式（Library Organization）

CuTe 是一个 header-only C++ 库，因此没有需要单独编译的源文件。库头文件位于顶层目录 [`include/cute`](https://github.com/NVIDIA/cutlass/tree/main/include/cute)，不同语义的组件分别放在不同子目录下。

| 目录 | 内容 |
| --- | --- |
| [`include/cute`](https://github.com/NVIDIA/cutlass/tree/main/include/cute) | 顶层头文件基本都对应 CuTe 的一个基础构件，例如 [`Layout`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/layout.hpp) 和 [`Tensor`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/tensor.hpp)。 |
| [`include/cute/container`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/container) | STL 风格对象的实现，例如 `tuple`、`array`、`aligned array`。 |
| [`include/cute/numeric`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/numeric) | 基础数值类型，包括非标准浮点类型、非标准整数类型、复数和整数序列（integer sequence）。 |
| [`include/cute/algorithm`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm) | 各种通用算法实现，例如 `copy`、`fill`、`clear`；在硬件支持时会自动利用架构特定能力。 |
| [`include/cute/arch`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/arch) | 对架构相关的矩阵乘法和拷贝指令的封装。 |
| [`include/cute/atom`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/atom) | `arch` 指令的元信息（meta-information），以及分区（partitioning）、分块（tiling）等工具。 |

## 教程导航（Tutorial）

这一组文档本身就是 CuTe 的教程。建议先从 [`0x_gemm_tutorial.zh-CN.md`](./0x_gemm_tutorial.zh-CN.md) 开始，它讲的是如何只依靠 CuTe 组件从零实现一个 GEMM，能帮助你建立整体认识。

目录中的其他文章分别聚焦 CuTe 的不同部分：

- [`01_layout.zh-CN.md`](./01_layout.zh-CN.md)：介绍 `Layout`，这是 CuTe 最核心的抽象之一。
- [`02_layout_algebra.zh-CN.md`](./02_layout_algebra.zh-CN.md)：介绍更高级的 `Layout` 操作，以及 CuTe 的布局代数（layout algebra）。
- [`03_tensor.zh-CN.md`](./03_tensor.zh-CN.md)：介绍 `Tensor`，它把 `Layout` 与数据数组组合成多维数组抽象。
- [`04_algorithms.zh-CN.md`](./04_algorithms.zh-CN.md)：概述作用在 `Tensor` 上的通用算法。
- [`0t_mma_atom.zh-CN.md`](./0t_mma_atom.zh-CN.md)：介绍与 GPU 架构相关的矩阵乘加指令（MMA, Matrix Multiply-Accumulate）在 CuTe 中的元信息和接口。
- [`0x_gemm_tutorial.zh-CN.md`](./0x_gemm_tutorial.zh-CN.md)：演示如何使用 CuTe 从零搭建 GEMM。
- [`0y_predication.zh-CN.md`](./0y_predication.zh-CN.md)：说明当分块不能整除矩阵时，应该如何处理。
- [`0z_tma_tensors.zh-CN.md`](./0z_tma_tensors.zh-CN.md)：介绍一种用于支持 TMA load/store 的高级 `Tensor` 形式。

## 快速提示（Quick Tips）

### 如何在 host 或 device 上打印 CuTe 对象？

`cute::print` 对几乎所有 CuTe 类型都提供了重载，包括指针（Pointers）、整数（Integers）、步长（Strides）、形状（Shapes）、布局（Layouts）和张量（Tensors）。如果你不确定某个对象该怎么观察，先试着对它调用 `print`。

CuTe 的打印函数既能在 host 上工作，也能在 device 上工作。但要注意，在 device 上打印非常昂贵。即使只是把打印代码留在 device 路径里，而运行时并没有真的执行到它，例如在一个永远不满足的 `if` 分支中，也可能让编译器生成更慢的代码。因此，调试结束后应尽量删掉 device 侧打印代码。

很多时候，你只想在每个 threadblock 的线程 0，或者整个 grid 的 threadblock 0 上打印。`thread0()` 只会在“全局线程 0”时返回 `true`，也就是 threadblock 0 中的 thread 0。下面是只在全局线程 0 上打印 CuTe 对象的常见写法：

```c++
if (thread0()) {
  print(some_cute_object);
}
```

有些算法依赖的并不是 0 号线程或 0 号 threadblock，这时你可能需要在其他线程上打印。头文件 [`cute/util/debug.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/util/debug.hpp) 除了其他调试工具外，还提供了 `bool thread(int tid, int bid)`，当当前执行上下文正好是线程 `tid`、threadblock `bid` 时返回 `true`。

#### 其他输出格式（Other Output Formats）

某些 CuTe 类型还有专门的打印函数，输出格式会更适合阅读。

- `cute::print_layout` 可以把任意 rank-2 layout 打印成纯文本表格，非常适合可视化“坐标到索引”的映射关系。
- `cute::print_tensor` 可以把 rank-1、rank-2、rank-3 和 rank-4 tensor 打印成纯文本多维表格；它会把 tensor 中的值也打印出来，方便你验证一次 copy 之后，tile 的数据是不是你预期的那块。
- `cute::print_latex` 会输出一组 LaTeX 命令，你可以用 `pdflatex` 生成排版更漂亮、带颜色的表格。它支持 `Layout`、`TiledCopy` 和 `TiledMMA`，对理解 CuTe 中的布局模式（layout patterns）和分区模式（partitioning patterns）很有帮助。

## 版权（Copyright）

以下 BSD-3-Clause 许可证文本保留原文：

Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. SPDX-License-Identifier: BSD-3-Clause

```console
  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

  1. Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
