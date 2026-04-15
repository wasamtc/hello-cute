# CuTe 张量算法（CuTe Tensor Algorithms）
这一节也没什么好说的，还是直接用官方文档。

原文：<https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/04_algorithms.html>

本文概述 CuTe 中作用在 `Tensor` 上的一些常见数值算法（numerical algorithms）及其接口风格。

这些算法的实现主要位于 [`include/cute/algorithm/`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm/) 目录。

## `copy`

CuTe 的 `copy` 算法把源张量（source `Tensor`）的元素复制到目标张量（destination `Tensor`）中。各种重载定义在 [`include/cute/algorithm/copy.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm/copy.hpp)。

### 接口与特化机会（Interface and Specialization Opportunities）

`Tensor` 会把数据类型、数据所在位置，以及有时还能把 shape 和 stride 一并编码到类型里。因此，`copy` 可以根据参数类型自动分派（dispatch）到不同实现，包括同步（synchronous）和异步（asynchronous）的硬件拷贝指令。

`copy` 有两个主要重载。第一种只接收源张量和目标张量：

```c++
template <class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(Tensor<SrcEngine, SrcLayout> const& src,
     Tensor<DstEngine, DstLayout>      & dst);
```

第二种在此基础上额外接收一个 `Copy_Atom`：

```c++
template <class... CopyArgs,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(Copy_Atom<CopyArgs...>       const& copy_atom,
     Tensor<SrcEngine, SrcLayout> const& src,
     Tensor<DstEngine, DstLayout>      & dst);
```

双参数版本只依据两个 `Tensor` 的类型来选择默认实现。带 `Copy_Atom` 的版本则允许调用者显式覆盖默认实现，指定一个非默认的拷贝路径。

### 并行性与同步语义取决于参数类型

无论是默认实现，还是通过 `Copy_Atom` 选中的实现，都可能只使用极少并行性，也可能充分利用所有可用并行性；同步语义（synchronization semantics）也可能不同。最终行为取决于 `copy` 的参数类型。使用者需要结合自己运行的 GPU 架构来理解这一点，现实中很多高性能 kernel 也确实会针对不同架构手写优化版本。

`copy` 可能是“每个线程顺序执行”的，也可能在某个线程集合上并行执行，例如整个 thread block，甚至 cluster。

如果 `copy` 是并行的，那么在参与拷贝的线程集合中，往往需要显式同步之后，线程才能安全地假定拷贝已经完成。举例来说，如果参与者是一个线程块，那么在使用 `copy` 结果之前，通常需要调用 `__syncthreads()` 或 Cooperative Groups 中的等价同步原语。

`copy` 还可能使用异步拷贝指令，例如 `cp.async`，或者它的 C++ 接口 `memcpy_async`。这时，在真正使用拷贝结果之前，还需要执行与该底层机制相匹配的额外同步。CuTe 的 GEMM 教程里就展示了一种做法；而更高性能的 GEMM 往往会进一步用流水线（pipelining）把异步 `copy` 与其他计算重叠起来。

### 一个通用 `copy` 实现

对任意两个 `Tensor`，最朴素的通用实现大概如下：

```c++
template <class TA, class ALayout,
          class TB, class BLayout>
CUTE_HOST_DEVICE
void
copy(Tensor<TA, ALayout> const& src,  // Any logical shape
     Tensor<TB, BLayout>      & dst)  // Any logical shape
{
  for (int i = 0; i < size(dst); ++i) {
    dst(i) = src(i);
  }
}
```

这个版本使用一维逻辑坐标去访问两个 `Tensor`，因此会按逻辑上的列主序（logical column-major order）遍历。

如果从“与架构无关但更合理”的角度继续优化，大致可以做下面这些事情：

1. 如果两个 `Tensor` 的内存空间支持某些专门指令，例如 `cp.async`，那么就分派到对应的硬件实现。
2. 如果两个 `Tensor` 的 layout 是静态的，并且能够静态证明向量化是合法的，例如可以把四个 `ld.global.b32` 合成一个 `ld.global.b128`，那么就对源和目标张量做向量化访问。
3. 如果可能，要验证当前选择的拷贝指令是否真的适用于这两个张量。

CuTe 的优化版 `copy` 实现会做这些事情。

## `copy_if`

CuTe 的 `copy_if` 也定义在 [`include/cute/algorithm/copy.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm/copy.hpp) 中。

它和 `copy` 一样接收源 `Tensor` 与目标 `Tensor`，但还额外接收一个形状相同的“谓词张量（predication / predicate tensor）”。只有当对应位置的谓词值非零时，源张量元素才会被复制到目标张量。

为什么要这样做，以及应该怎样使用 `copy_if`，可以参考谓词化教程 [`0y_predication.zh-CN.md`](./0y_predication.zh-CN.md)。

## `gemm`

### `gemm` 到底计算什么（What `gemm` Computes）

`gemm` 接收三个 `Tensor`：A、B 和 C。它的具体行为取决于这三个张量各自拥有多少模式（modes）。文档用以下字母来表示这些模式：

- `V` 表示“向量维（vector mode）”，即彼此独立的一组元素。
- `M` 和 `N` 分别表示 BLAS GEMM 中结果矩阵 C 的行数与列数。
- `K` 表示 GEMM 的归约维（reduction mode），也就是沿着这个维度做求和。详细背景可参考 GEMM 教程 [`0x_gemm_tutorial.zh-CN.md`](./0x_gemm_tutorial.zh-CN.md)。

文档用 `(...) x (...) => (...)` 的形式列出 A、B、C 的模式：

1. `(V) x (V) => (V)`：向量逐元素乘法，`Cv += Av * Bv`。会分派到 FMA 或 MMA。
2. `(M) x (N) => (M,N)`：向量外积（outer product），`Cmn += Am * Bn`。可视为情形 4 在 `V=1` 时的特例。
3. `(M,K) x (N,K) => (M,N)`：矩阵乘法，`Cmn += Amk * Bnk`。可以看作对每个 `K` 重复情形 2。
4. `(V,M) x (V,N) => (V,M,N)`：批量向量外积（batched outer product），`Cvmn += Avm * Bvn`。会为寄存器复用（register reuse）做优化，并对每个 `M,N` 调用情形 1。
5. `(V,M,K) x (V,N,K) => (V,M,N)`：批量矩阵乘法，`Cvmn += Avmk * Bvnk`。可以看作对每个 `K` 重复情形 4。

关于各个模式在 CuTe 中的排列顺序，建议直接看 GEMM 教程。一个重要约定是：如果出现 `K`，它总是最右边；如果出现 `V`，它总是最左边。

### 分派到优化实现（Dispatch to Optimized Implementations）

和 `copy` 一样，CuTe 的 `gemm` 也会根据 `Tensor` 参数的类型自动分派到适当的优化实现。

同样地，`gemm` 还支持一个可选的 `MMA_Atom` 参数，用来覆盖 CuTe 默认选出的 `FMA` 指令或实现策略。

如果想进一步了解 `MMA_Atom`，以及 `gemm` 如何针对不同架构做特化，建议继续看 MMA 教程 [`0t_mma_atom.zh-CN.md`](./0t_mma_atom.zh-CN.md)。

## `axpby`

`axpby` 定义在 [`include/cute/algorithm/axpby.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm/axpby.hpp) 中。

它执行的是

`y = alpha * x + beta * y`

其中 `alpha` 和 `beta` 是标量（scalars），`x` 和 `y` 是 `Tensor`。名字 `axpby` 就是 “Alpha times X Plus Beta times Y” 的缩写，可以看作 BLAS 里 `AXPY`（Alpha times X Plus Y）的推广。

## `fill`

`fill` 定义在 [`include/cute/algorithm/fill.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm/fill.hpp) 中。它会把输出 `Tensor` 的所有元素覆盖为给定的标量值。

## `clear`

`clear` 定义在 [`include/cute/algorithm/clear.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm/clear.hpp) 中。它会把输出 `Tensor` 的所有元素清零。

## 其他算法（Other Algorithms）

CuTe 还提供了其他算法，对应头文件都可以在 [`include/cute/algorithm`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm) 目录中找到。
