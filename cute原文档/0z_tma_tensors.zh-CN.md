# CuTe TMA 张量（CuTe TMA Tensors）

原文：<https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0z_tma_tensors.html>

在阅读 CuTe 代码时，你可能会见到一些“长得很怪”的 Tensor，打印出来像这样：

```console
ArithTuple(0,_0,_0,_0) o ((_128,_64),2,3,1):((_1@0,_1@1),_64@1,_1@2,_1@3)
```

这里的 `ArithTuple` 是什么？这些看起来像 stride 的东西到底表示什么？这种 Tensor 又是拿来做什么的？

本文的目标就是回答这些问题，并介绍 CuTe 里相对高级的一组能力。

## TMA 指令简介（Introduction to TMA Instructions）

Tensor Memory Accelerator（TMA）是一组用于在全局内存（GMEM）和共享内存（SMEM）之间搬运多维数组的指令。TMA 在 Hopper 架构中引入。单条 TMA 指令可以一次性复制整个数据 tile，因此硬件不再需要为 tile 中的每个元素分别计算地址、逐个发射 copy 指令。

为了做到这一点，TMA 需要一个 *TMA descriptor*。它是对全局内存中 1 到 5 维张量的一种打包描述，通常包含：

- 张量基地址（base pointer）
- 元素数据类型，例如 `int`、`float`、`double`、`half`
- 每个维度的大小（size）
- 每个维度中的步长（stride）
- 其他控制信息，例如 SMEM box 大小、SMEM swizzle 模式和越界行为

这个 descriptor 需要在 host 侧、kernel 启动之前构造完成。所有会发出 TMA 指令的 thread block 共享同一个 descriptor。进入 kernel 后，一次 TMA 操作通常需要下面这些参数：

- 指向 TMA descriptor 的指针
- 指向 SMEM 的指针
- 指向 descriptor 所描述 GMEM 张量中的“坐标（coordinates）”

例如，一个接收 3 维坐标的 TMA store 接口可以长这样：

```cpp
struct SM90_TMA_STORE_3D {
  CUTE_DEVICE static void
  copy(void const* const desc_ptr,
       void const* const smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2) {
    // ... invoke CUDA PTX instruction ...
  }
};
```

这里最关键的一点是：TMA 指令并不直接接收全局内存指针。全局内存的基地址已经封装在 descriptor 里，而且被视为常量，不会作为独立参数再次传入。TMA 指令真正消费的是“在 descriptor 所定义视图中的 TMA 坐标”。

这意味着，一个普通 CuTe Tensor 如果只是保存 GMEM 指针，并通过 layout 计算偏移得到新指针，那么它对 TMA 来说并没有直接用处。

那该怎么办？

## 构造一个 TMA Tensor（Building a TMA Tensor）

### 隐式 CuTe Tensor（Implicit CuTe Tensors）

所有 CuTe Tensor 本质上都是 `Layout` 与 `Iterator` 的组合。普通全局内存 Tensor 的 iterator 就是它的全局内存指针。但 CuTe 的 iterator 不一定非得是指针，只要它满足随机访问（random-access）语义即可。

一个典型例子是 counting iterator。它表示一个“从某个初值开始的整数序列”，这个序列并不会真的存储在内存里，因此这些整数是“隐式的（implicit）”。iterator 本身只保存当前位置。

例如，可以用 counting iterator 构造一个“隐式整数 Tensor”：

```cpp
Tensor A = make_tensor(counting_iterator<int>(42), make_shape(4,5));
print_tensor(A);
```

输出是：

```console
counting_iter(42) o (4,5):(_1,4):
   42   46   50   54   58
   43   47   51   55   59
   44   48   52   56   60
   45   49   53   57   61
```

这个 Tensor 会把逻辑坐标映射成“按需计算”的整数。因为它仍然是个标准的 CuTe Tensor，所以你仍然可以像操作普通 Tensor 那样对它做 tiling、partitioning 和 slicing，本质上只是把整数偏移累计到 iterator 上。

但 TMA 需要的不是普通整数，也不是指针，而是“坐标”。那么，我们能不能构造一个“隐式 TMA 坐标 Tensor”，让 TMA 指令直接消费这些坐标？如果可以，那么这个坐标 Tensor 也就能被同样地切块、切片和分区，从而在任意位置都得到正确的 TMA 坐标。

### ArithTupleIterator 与 ArithTuple

第一步，是为 TMA 坐标构造一个类似 counting iterator 的东西。它至少需要支持两件事：

- 解引用（dereference）得到一个 TMA 坐标
- 用另一个 TMA 坐标做偏移

CuTe 把这种对象叫做 `ArithmeticTupleIterator`。它内部保存的是一个坐标，这个坐标本身表示为 `ArithmeticTuple`。

`ArithmeticTuple` 可以看成是 `cute::tuple` 的一个公开子类（public subclass），并且额外重载了 `operator+`，使得它能和另一个 tuple 做逐元素相加。换句话说，两个 tuple 的和，仍然是一个 tuple，其各个位置分别是对应元素之和。

于是，类似于 `counting_iterator(42)`，我们就能构造一个“能解引用、也能被 tuple 偏移”的隐式迭代器：

```cpp
ArithmeticTupleIterator citer_1 = make_inttuple_iter(42, Int<2>{}, Int<7>{});
ArithmeticTupleIterator citer_2 = citer_1 + make_tuple(Int<0>{}, 5, Int<2>{});
print(*citer_2);
```

输出为：

```console
(42,7,_9)
```

TMA Tensor 就可以利用这样的 iterator 来保存当前 TMA 坐标“偏移”。这里之所以给“偏移”加引号，是因为它显然不是传统的一维地址偏移，也不是指针加法的那种 offset。

可以这样总结：

- 你先为 *整个全局内存 Tensor* 构造一个 TMA descriptor。
- descriptor 定义了 TMA 对这个全局张量的一个视图。
- TMA 指令接收的是这个视图上的坐标。
- 为了生成和追踪这些坐标，我们再构造一个“隐式的 CuTe 坐标 Tensor”。
- 这个坐标 Tensor 可以像普通 Tensor 一样被 tile、slice、partition。

现在我们已经能用 iterator 去保存和偏移 TMA 坐标了，但还有一个问题：CuTe 的 `Layout` 默认是产生整数偏移的，怎样让它生成“非整数”的偏移呢？

### Stride 不一定非得是整数

普通 Tensor 的 layout 会把逻辑坐标 `(i,j)` 映射成一维线性索引 `k`。这个映射本质上就是“坐标”和“stride”的内积（inner product）。

但 TMA Tensor 的 iterator 保存的是 TMA 坐标，因此它的 layout 也必须把逻辑坐标映射成“TMA 坐标”，而不是一维整数索引。

要做到这一点，我们就得把“stride”的概念推广一下。stride 不一定非得是整数，它可以是任意一种能和整数做内积的代数对象。一个自然的选择，就是前面用过的 `ArithmeticTuple`。只要再给它补上 `operator*`，使它能被整数缩放，那么它就能充当一种更一般的 stride。

#### 题外话：整数模（Integer-Module）作为 Stride

如果一组对象支持“元素之间相加”和“整数与元素相乘”，那么在代数上，这种结构叫做整数模（integer-module）。

形式化地说，一个整数模是一个阿贝尔群 `(M,+)`，并配备一个从 `Z * M -> M` 的乘法，其中 `Z` 是整数集合。也就是说，整数模中的元素能够与整数做内积或缩放运算。

整数本身当然是整数模。由整数构成的 R 维 tuple 也是整数模。

从这个角度看，CuTe 中的 layout stride 原则上可以是任何整数模。

#### 基元素（Basis Elements）

CuTe 的基元素定义在 `cute/numeric/arithmetic_tuple.hpp` 中。为了便于创建可作为 stride 的 `ArithmeticTuple`，CuTe 提供了一个类型别名 `E`，表示“归一化（normalized）的基元素”。所谓归一化，就是它的缩放系数在编译期固定为 1。

| C++ 对象 | 描述 | 打印形式 |
| --- | --- | --- |
| `E<>{}` | `1` | `1` |
| `E<0>{}` | `(1,0,...)` | `1@0` |
| `E<1>{}` | `(0,1,0,...)` | `1@1` |
| `E<0,0>{}` | `((1,0,...),0,...)` | `1@0@0` |
| `E<0,1>{}` | `((0,1,0,...),0,...)` | `1@1@0` |
| `E<1,0>{}` | `(0,(1,0,...),0,...)` | `1@0@1` |
| `E<1,1>{}` | `(0,(0,1,0,...),0,...)` | `1@1@1` |

表中的“描述”一栏把每个基元素看成一个无限长整数 tuple，其中未显式指定的位置全部补 0。位置索引从左到右、从 0 开始计数。例如：

- `E<1>{}` 在位置 1 上是 1，因此表示 `(0,1,0,...)`
- `E<3>{}` 在位置 3 上是 1，因此表示 `(0,0,0,1,0,...)`

基元素还可以嵌套（nested）。例如 `E<0,1>{}` 表示“位置 0 上放一个 `E<1>{}`”，所以得到 `((0,1,0,...),0,...)`。从打印结果 `1@1@0` 也能看出来：先把 `1` 提升到位置 1，得到 `1@1`，再把整个结果提升到位置 0。

基元素还能被整数缩放（scaled）。例如 `5*E<1>{}` 的缩放因子是 5，它会打印成 `5@1`，代表 `(0,5,0,...)`。这个缩放因子在嵌套结构里同样成立，比如 `5*E<0,1>{}` 会打印成 `5@1@0`，意思是 `((0,5,0,...),0,...)`。

基元素之间也可以相加，只要它们的层级结构兼容（compatible）。例如：

`3*E<0>{} + 4*E<1>{}`

会得到 `(3,4,0,...)`。

#### Stride 的线性组合（Linear Combinations of Strides）

Layout 的工作方式，本来就是把“自然坐标（natural coordinate）”和“stride”做内积。对于普通整数 stride，例如 `(1,100)`，输入坐标 `(i,j)` 和 stride 的内积就是 `i + 100j`。然后把这个线性索引加到普通 Tensor 的基指针上，就能拿到 `(i,j)` 对应的元素地址。

如果 stride 变成基元素，例如 `(1@0,1@1)`，那么 `(i,j)` 与 stride 的内积就是：

`i@0 + j@1 = (i,j)`

也就是说，layout 的结果不再是一维索引，而是一个二元 TMA 坐标 `(i,j)`。如果想把两个坐标位置对调，那么 stride 可以写成 `(1@1,1@0)`，于是：

`i@1 + j@0 = (j,i)`

因此，基元素的线性组合可以自然地表示多维、甚至分层的坐标。例如：

`2*2@1@0 + 3*1@1 + 4*5@1 + 7*1@0@0`

可以化成：

`((0,4,...),0,...) + (0,3,0,...) + (0,20,0,...) + ((7,...),...) = ((7,4,...),23,...)`

也就是说，它可以被理解成坐标 `((7,4),23)`。

这就是为什么“stride 的线性组合”足以生成 TMA 坐标；而一旦这些坐标能被 layout 生成，它们就可以继续去驱动 TMA 坐标 iterator。

### 应用于 TMA Tensor（Application to TMA Tensors）

现在，我们终于可以构造出类似文档开头那样的 CuTe Tensor 了：

```cpp
Tensor a = make_tensor(make_inttuple_iter(0,0),
                       make_shape (     4,      5),
                       make_stride(E<0>{}, E<1>{}));
print_tensor(a);

Tensor b = make_tensor(make_inttuple_iter(0,0),
                       make_shape (     4,      5),
                       make_stride(E<1>{}, E<0>{}));
print_tensor(b);
```

输出如下：

```console
ArithTuple(0,0) o (4,5):(_1@0,_1@1):
  (0,0)  (0,1)  (0,2)  (0,3)  (0,4)
  (1,0)  (1,1)  (1,2)  (1,3)  (1,4)
  (2,0)  (2,1)  (2,2)  (2,3)  (2,4)
  (3,0)  (3,1)  (3,2)  (3,3)  (3,4)

ArithTuple(0,0) o (4,5):(_1@1,_1@0):
  (0,0)  (1,0)  (2,0)  (3,0)  (4,0)
  (0,1)  (1,1)  (2,1)  (3,1)  (4,1)
  (0,2)  (1,2)  (2,2)  (3,2)  (4,2)
  (0,3)  (1,3)  (2,3)  (3,3)  (4,3)
```

第一个 tensor 用 `(E<0>{}, E<1>{})` 作为 stride，因此逻辑坐标 `(i,j)` 会被映射成坐标 `(i,j)`。第二个 tensor 则用 `(E<1>{}, E<0>{})` 作为 stride，因此会把两个坐标轴交换，变成 `(j,i)`。

这就是 CuTe 里 TMA Tensor 的核心思想：让 layout 不再只产生一维地址偏移，而是直接产生 TMA 所需的多维坐标。

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
