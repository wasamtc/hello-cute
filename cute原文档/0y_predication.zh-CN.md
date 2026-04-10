# 谓词化（Predication）：当分块无法整除时怎么办

原文：<https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0y_predication.html>

[GEMM 教程](./0x_gemm_tutorial.zh-CN.md) 说明了如何通过遍历输入矩阵和输出矩阵的各个 tile 来完成矩阵乘法。那里的示例默认 tile 可以整齐地铺满矩阵，没有余数（remainder）。但现实里经常不是这样。比如，我们可能想把一个 `41 x 55` 的矩阵切成 `4 x 8` 的 tile，可是 `41 / 4 = 10` 余 `1`，`55 / 8 = 6` 余 `7`。那么那些“剩下来的”区域应该怎么处理？

首先要注意，`logical_divide` 是 CuTe 里对 layout 做分块（tiling）的方式，它的行为相当于“向上取整（round up）”。例如，如果 `N` 是 `1000:1` 这个 layout，而 `B` 是 `128:1` 这个 layout，那么 `logical_divide(N, B)` 得到的是 `(128, 8):(1, 128)`。换句话说，它会把原始 shape `N = 1000` 视作一个 `128 x 8` 的矩阵来处理，好像长度是 `1024` 一样。那最后多出来的 24 个元素怎么办？最后一个 tile 要怎么处理，才能避免越界访问（out-of-bounds）？

CuTe 采用的思路和很多 CUDA 入门教材一致，即“谓词化（predication）”。它不去显式表示“7 个大小为 128 的 tile，再加 1 个大小为 104 的 tile”，而是统一把问题向上补齐成“8 个大小为 128 的 tile”，然后再构造谓词（predicates），保证 kernel 只会访问每个 tile 中那些在原始矩阵边界之内的元素。

这种做法和 GPU 的优化方式也很契合：只要不引入 warp divergence，分支通常是可以接受的。它本质上也和经典 CUDA 写法一致，例如把一维 `N` 个工作项映射到若干 block 时，先算出“我这个线程”对应的下标，再判断它是否越界。

考虑一个更一般的例子：把长度为 `1000` 的向量切成大小为 `128` 的块。可以这样构造一个 predication tensor：

```c++
Tensor gmem = ...     // e.g. size 1000
Tensor smem = ...     // e.g. size 128

// Tile the gmem for smem
Tensor gmem_tiled = logical_divide(gmem, size(smem));      // e.g. (128,8)

// Create an identity layout for gmem and tile it similarly
Layout id_layout = make_layout(shape(gmem));               // e.g. 1000:1, explicitly constructed as identity function
Layout id_tiled  = logical_divide(id_layout, size(smem));  // e.g. (128,8):(1,128), but many elements aren't "valid"

// Create a predicate tensor
Tensor pred = make_tensor<bool>(shape(id_tiled));          // e.g. (128,8)
for (int i = 0; i < size(pred); ++i) {
  pred(i) = id_tiled(i) < size(id_layout);  // Predicate: Is the offset within the original shape?
}

// ... intervening code ...

// Note that gmem_tiled, id_tiled, and pred tensors are all congruent
// For tile tile_i, determine if element value_j is in-bounds and copy to smem
if (pred(value_j,tile_i)) { smem(value_j) = gmem_tiled(value_j,tile_i); }
```

它的通用流程可以总结成四步：

1. 创建一个和原始数据同形状的“恒等（identity）”layout，例如上面的 `id_layout = make_layout(shape(gmem))`。
2. 对这个 identity layout 执行与原始数据完全相同的 tiling / partitioning / slicing 流程，即使这个过程会向上补齐。
3. 通过把该参考 layout 的坐标与原始 layout 的边界做比较，构造一个谓词张量（predicate tensor）。
4. 用这个谓词张量把所有越界访问屏蔽掉。

下面看一个稍微具体一些的例子：对 GEMM 的 epilogue 做 predication。假设我们已经把 `mC` 先切成 CTA tile，再按 MMA 的线程映射做了线程级分区：

```cpp
// CTA partitioning
auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

// Thread partitioning
auto thr_mma = mma.get_slice(threadIdx.x);
Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)
Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)

// ... Compute gemms and accumulate into tCrC ...

// axpby epilogue
for (int i = 0; i < size(tCgC); ++i) {
  tCgC(i) = alpha * tCrC(i) + beta * tCgC(i);
}
```

按照前面的流程，谓词化非常直接：

```cpp
// A coordinate tensor the same shape as mC: (m,n) -> (m,n)
Tensor cC     = make_identity_tensor(shape(mC));

// Repeat partitioning steps applied to mC to our coordinate tensor cC
// CTA partitioning
Tensor cta_cC = local_tile(cC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N) -> (m,n)
// Thread partitioning
Tensor tCcC   = thr_mma.partition_C(cta_cC);                             // (MMA,MMA_M,MMA_N) -> (m,n)

// Predicated axpby epilogue
for (int i = 0; i < size(tCgC); ++i) {
  if (elem_less(tCcC(i), shape(mC))) {  // if coord is in-bounds
    tCgC(i) = alpha * tCrC(i) + beta * tCgC(i);
  }
}
```

这里，CTA 负责对 `mC` 做第一层 tiling / partitioning，MMA 负责对 `gC` 做第二层 tiling / partitioning，所以这两步也必须原样作用在 identity tensor 上。得到的坐标张量 `tCcC` 与寄存器片段 `tCrC` 以及全局内存分区张量 `tCgC` 彼此同构（congruent），表示的都是当前线程要处理的那部分子张量；不同之处在于，`tCcC` 的值域（codomain）仍然是原始张量 `mC` 中的全局坐标。因此，只需要把 `tCcC(i)` 与 `shape(mC)` 比较，就能判断当前位置是否合法。

这种“参考 identity tensor”或“坐标张量（coordinate tensor）”方法有几个明显优点：

1. 它不依赖被谓词化 tensor 的具体 layout 或 strides，只依赖逻辑边界。
2. partitioning 阶段可以是任意形式。CTA tiling、线程分区、`TiledMMA`、`TiledCopy` 都可以作用在坐标张量上。
3. 它能自然扩展到任意维数。
4. 它本质上就是普通 CUDA 一维并行访问模式的高维推广。

普通 CUDA 一维写法如下：

```cpp
int idx = blockDim.x * blockIdx.x + threadIdx.x;
if (idx < N)  // idx is a "coord" into gmem and N is the "bound"
  gmem_ptr[idx] = ...;
```

在 SIMT 编程模型里，不推荐通过修改 tensor extent 来避免循环越界。更通用的方法是保留原始循环结构，再用 predication 去查询“原始坐标是否越界”。这样可以避免动态循环边界，转而使用指令级 predication，同时保持线程一致性（thread coherence）和负载均衡（load balance）。而且它足够通用，适用于所有 rank、所有线程与数据布局，以及所有 tiling / partitioning 模式。对于特殊场景，也可以把额外假设编码到坐标张量或 predicate tensor 中。

再看一个更复杂一点的例子：在 GEMM 中对 A、B 的加载做 `m` / `n` 方向的 predication。假设我们已经这样把 A 和 B 的 tile 分配到 CTA 和线程上：

```c++
// CTA partitioning
auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)

Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);            // (BLK_M,BLK_K)
Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);            // (BLK_N,BLK_K)

// Thread partitioning
Tensor tAgA = local_partition(gA, tA, thread_idx);                   // (THR_M,THR_K,k)
Tensor tAsA = local_partition(sA, tA, thread_idx);                   // (THR_M,THR_K)

Tensor tBgB = local_partition(gB, tB, thread_idx);                   // (THR_N,THR_K,k)
Tensor tBsB = local_partition(sB, tB, thread_idx);                   // (THR_N,THR_K)
```

这里，`gA` 和 `gB` 分别是 `mA`、`mB` 依照 `cta_tiler` 和 `cta_coord` 切出的 CTA 级 tile；`tAgA` 和 `tBgB` 则是在此基础上，按线程布局 `tA`、`tB` 和 `thread_idx` 得到的线程级分区。

接着构造两个 identity tensor，分别表示 `(m,k) -> (m,k)` 和 `(n,k) -> (n,k)`：

```c++
// Coordinate tensors
Tensor cA = make_identity_tensor(shape(mA));   // (m,k) -> (m,k)
Tensor cB = make_identity_tensor(shape(mB));   // (n,k) -> (n,k)
```

然后，把它们按照和 `mA`、`mB` 一样的方式做 tiling 和 partitioning：

```c++
// CTA partitioning
Tensor cta_cA = local_tile(cA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k) -> (m,k)
Tensor cta_cB = local_tile(cB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k) -> (n,k)

// Thread partitioning
Tensor tAcA = local_partition(cta_cA, tA, thread_idx);                   // (THR_M,THR_K,k) -> (m,k)
Tensor tBcB = local_partition(cta_cB, tB, thread_idx);                   // (THR_N,THR_K,k) -> (m,k)
```

下面再构造与 `tAgA`、`tBgB` 相对应的谓词张量。它们会在 prologue 阶段只计算一次，之后用于屏蔽 inner loop 中的越界访问：

```c++
Tensor tApA = make_tensor<bool>(make_shape (size<0>(tAcA), size<1>(tAcA)),
                                make_stride(     Int<1>{},      Int<0>{}));
Tensor tBpB = make_tensor<bool>(make_shape (size<0>(tBcB), size<1>(tBcB)),
                                make_stride(     Int<1>{},      Int<0>{}));
```

这里隐含了几个前提：

- 我们一次只关心一个数据 tile 的谓词。
- 我们只关心 `m` 和 `n` 两个模式上的谓词，`k` 模式的处理交给别的逻辑。
- `m` / `n` 方向的谓词在每个 `k` tile 上都是不变的，因此可以在 mainloop 的每次迭代中重复复用。

所以这里只存储 `m` 和 `n` 方向的谓词，并通过 stride-0 的广播方式把它们应用到所有 `k` 位置上。

填充这些张量时，同样沿用这个假设：

```c++
// Populate the m- and n-predicates
CUTE_UNROLL
for (int m = 0; m < size<0>(tApA); ++m) {
  tApA(m,0) = elem_less(get<0>(tAcA(m,0,0)), shape<0>(mA));  // Compare the m-coordinate
}
CUTE_UNROLL
for (int n = 0; n < size<0>(tBpB); ++n) {
  tBpB(n,0) = elem_less(get<0>(tBcB(n,0,0)), shape<0>(mB));  // Compare the n-coordinate
}
```

也就是说，我们只看第 0 个 `k`-tile、第 0 个 `k`-block 上的 `m` / `n` 坐标，然后利用 stride-0 广播，让这组数据自动充当整块 tile 的 predicate tensor。

最后，在执行 `copy_if` 时，就可以只搬运谓词为 `true` 的元素：

```c++
// Copy a k_tile from global memory to shared memory
copy_if(tApA, tAgA(_,_,k_tile), tAsA);
copy_if(tBpB, tBgB(_,_,k_tile), tBsB);
```
