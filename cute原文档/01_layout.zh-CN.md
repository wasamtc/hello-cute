# CuTe 布局（CuTe Layouts）

原文：<https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html>

本文介绍 `Layout`，这是 CuTe 中最核心的抽象之一。从语义上说，`Layout` 是“从坐标空间（coordinate space）到索引空间（index space）的映射”。

`Layout` 为多维数组访问提供了统一接口，把“数组元素在内存里如何组织”这件事抽象了出来。这样一来，你写算法时可以只针对“逻辑上的多维数组”编程，而不必把代码绑死在某种特定内存布局上。例如，一个 `M x N` 的 row-major 布局和一个 `M x N` 的 column-major 布局，在使用者代码看来可以是同一种东西。

CuTe 还提供了一整套“布局代数（algebra of `Layout`s）”。你可以把多个 `Layout` 组合起来、改写它们，甚至把一个布局按另一个布局去分块（tile）。这对“把数据布局映射到线程布局上”这类任务尤其有用。

## 基础类型与概念（Fundamental Types and Concepts）

### 整数（Integers）

CuTe 同时大量使用动态整数（dynamic integers）和静态整数（static integers）。

- 动态整数就是运行期才知道值的普通整数类型，例如 `int`、`size_t`、`uint16_t` 等。凡是 `std::is_integral` 接受的类型，在 CuTe 里都可以视为动态整数。
- 静态整数则是像 `std::integral_constant` 这样的类型实例，它们把值编码在类型里，通过 `static constexpr` 成员暴露。CuTe 为 CUDA 提供了自己的一套静态整数类型 `cute::C`，并且重载了运算符，使静态整数之间的运算结果仍然尽量保持为静态整数。

CuTe 还提供了一些常用别名，例如：

- `Int<1>`、`Int<2>`、`Int<3>`
- `_1`、`_2`、`_3`

文档中的大量例子都会使用这些简写。

CuTe 尽量把静态整数和动态整数统一对待。很多示例里，动态整数和静态整数都可以互换。当 CuTe 文档里说“integer”时，绝大多数情况下都表示“动态或静态整数都可以”。

与整数相关的常见 traits 包括：

- `cute::is_integral`：检查 `T` 是否是静态或动态整数。
- `cute::is_std_integral`：检查 `T` 是否是动态整数，本质上等价于 `std::is_integral`。
- `cute::is_static`：检查 `T` 是否为空类型（empty type），即它的实例不依赖运行时信息，本质上等价于 `std::is_empty`。
- `cute::is_constant`：检查 `T` 是否为静态整数，且其值是否等于给定的 `N`。

更细节的实现可以参考 [`integral_constant` 相关代码](https://github.com/NVIDIA/cutlass/tree/main/include/cute/numeric/integral_constant.hpp)。

### Tuple

tuple 是一个“有序、有限”的元素列表，可以有零个或多个元素。CuTe 的 [`cute::tuple`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/container/tuple.hpp) 行为上类似 `std::tuple`，但同时支持 host 和 device，并对模板参数和实现做了裁剪，以换取更好的简洁性与性能。

### IntTuple

CuTe 中的 `IntTuple` 概念定义为：

- 一个整数，或者
- 一个由 `IntTuple` 组成的 tuple

注意这是一个递归定义。

一些 `IntTuple` 例子：

- `int{2}`：动态整数 2
- `Int<3>{}`：静态整数 3
- `make_tuple(int{2}, Int<3>{})`
- `make_tuple(uint16_t{42}, make_tuple(Int<1>{}, int32_t{3}), Int<17>{})`

CuTe 用 `IntTuple` 来承载很多概念，例如 `Shape`、`Stride`、`Step` 和 `Coord`。

常见操作包括：

- `rank(IntTuple)`：元素个数。单个整数的 rank 为 1；tuple 的 rank 为 `tuple_size`。
- `get<I>(IntTuple)`：取第 `I` 个元素。对于单个整数，`get<0>` 就是它本身。
- `depth(IntTuple)`：层级深度。单个整数深度为 0；由整数构成的 tuple 深度为 1；包含 tuple 的 tuple 深度为 2；以此类推。
- `size(IntTuple)`：所有元素的乘积。

文档中会用括号表示 `IntTuple` 的层次，例如：`6`、`(2)`、`(4,3)`、`(3,(6,2),8)` 都是合法的 `IntTuple`。

### Shape 与 Stride

`Shape` 和 `Stride` 本质上都属于 `IntTuple` 概念。

### Layout

`Layout` 可以理解为一对 `(Shape, Stride)`。语义上，它表示：

- 输入：`Shape` 内的一个合法坐标
- 输出：根据 `Stride` 计算出的一个索引

### Tensor

`Layout` 可以再和数据（例如指针或数组）组合成 `Tensor`。`Layout` 算出来的索引会被用来索引 iterator，从而取到真正的数据。`Tensor` 细节见 [`03_tensor.zh-CN.md`](./03_tensor.zh-CN.md)。

## 创建与使用 Layout（Layout Creation and Use）

一个 `Layout` 就是一对 `IntTuple`：`Shape` 和 `Stride`。`Shape` 定义逻辑形状，`Stride` 定义从该形状内部坐标到索引空间的映射方式。

和 `IntTuple` 类似，`Layout` 也有一批配套操作：

- `rank(Layout)`：模式（mode）数量
- `get<I>(Layout)`：第 `I` 个子布局
- `depth(Layout)`：shape 的层级深度
- `shape(Layout)`：取出 shape
- `stride(Layout)`：取出 stride
- `size(Layout)`：定义域大小，等价于 `size(shape(Layout))`
- `cosize(Layout)`：余域大小（严格来说不一定等于值域大小），等价于 `A(size(A) - 1) + 1`

### 层级访问函数（Hierarchical Access Functions）

由于 `IntTuple` 和 `Layout` 都可以任意嵌套，CuTe 提供了一组“可接受多个模板整数参数”的访问函数，方便深入嵌套结构内部：

- `get<I...>(x)`：相当于一层层 `get` 下去
- `rank<I...>(x)`：先 `get<I...>(x)`，再求 rank
- `depth<I...>(x)`：先取子元素，再求 depth
- `shape<I...>(x)`：取某个子元素的 shape
- `size<I...>(x)`：取某个子元素的 size

所以你会经常看到诸如 `size<0>(layout)`、`size<1>(layout)` 这样的写法。

### 构造 Layout（Constructing a Layout）

`Layout` 的构造方式很多，静态整数和动态整数可以混用：

```c++
Layout s8 = make_layout(Int<8>{});
Layout d8 = make_layout(8);

Layout s2xs4 = make_layout(make_shape(Int<2>{},Int<4>{}));
Layout s2xd4 = make_layout(make_shape(Int<2>{},4));

Layout s2xd4_a = make_layout(make_shape (Int< 2>{},4),
                             make_stride(Int<12>{},Int<1>{}));
Layout s2xd4_col = make_layout(make_shape(Int<2>{},4),
                               LayoutLeft{});
Layout s2xd4_row = make_layout(make_shape(Int<2>{},4),
                               LayoutRight{});

Layout s2xh4 = make_layout(make_shape (2,make_shape (2,2)),
                           make_stride(4,make_stride(2,1)));
Layout s2xh4_col = make_layout(shape(s2xh4),
                               LayoutLeft{});
```

`make_layout` 会根据实参类型自动推导并返回合适的 `Layout`。类似地，`make_shape` 和 `make_stride` 也分别返回 `Shape` 与 `Stride`。CuTe 很喜欢使用这类 `make_*` 工厂函数，一方面是为了绕开 CTAD 限制，另一方面也可以避免在代码中反复写出静态/动态整数的完整类型。

如果省略 `Stride`，CuTe 默认使用 `LayoutLeft` 根据 `Shape` 自动生成 stride。`LayoutLeft` 会把 shape 从左到右做 exclusive prefix product，可以看作一种“广义列主序（generalized column-major）”生成方式。`LayoutRight` 则从右到左生成 stride，对于 depth 为 1 的 shape，可以把它近似理解成 row-major；但对分层 shape 来说，结果可能没有那么直观。

对上面的 layout 调用 `print`，会得到：

```console
s8        :  _8:_1
d8        :  8:_1
s2xs4     :  (_2,_4):(_1,_2)
s2xd4     :  (_2,4):(_1,_2)
s2xd4_a   :  (_2,4):(_12,_1)
s2xd4_col :  (_2,4):(_1,_2)
s2xd4_row :  (_2,4):(4,_1)
s2xh4     :  (2,(2,2)):(4,(2,1))
s2xh4_col :  (2,(2,2)):(_1,(2,4))
```

这里常见记法是 `Shape:Stride`。其中 `_N` 表示静态整数，其他写法则通常表示动态整数。可以看到 `Shape` 和 `Stride` 都允许静态、动态整数混合出现。

同时，`Shape` 和 `Stride` 必须是 **congruent** 的，也就是说它们的 tuple 结构要匹配：`Shape` 里每个整数位置，都要在 `Stride` 里有一个对应整数位置。可以这样检查：

```cpp
static_assert(congruent(my_shape, my_stride));
```

### 使用 Layout（Using a Layout）

`Layout` 的根本用途，是把 `Shape` 所定义的坐标映射到由 `Stride` 定义的索引上。

例如，如果想把一个任意 rank-2 layout 以二维表格形式打印出来，可以这样写：

```c++
template <class Shape, class Stride>
void print2D(Layout<Shape,Stride> const& layout)
{
  for (int m = 0; m < size<0>(layout); ++m) {
    for (int n = 0; n < size<1>(layout); ++n) {
      printf("%3d  ", layout(m,n));
    }
    printf("\n");
  }
}
```

对前面的示例调用它，会得到：

```console
> print2D(s2xs4)
  0    2    4    6
  1    3    5    7
> print2D(s2xd4_a)
  0    1    2    3
 12   13   14   15
> print2D(s2xh4_col)
  0    2    4    6
  1    3    5    7
> print2D(s2xh4)
  0    2    1    3
  4    6    5    7
```

这说明 `layout(m,n)` 返回的是逻辑二维坐标 `(m,n)` 对应的一维索引。

值得注意的是，`s2xh4` 既不是普通 row-major，也不是普通 column-major。更有意思的是，它虽然内部有三层 mode，却仍然可以被当成 rank-2 来使用，并接受二维坐标。这是因为它的第二个 mode 本身是一个“多模式（multi-mode）”，但这个多模式仍然可以用一个一维坐标来访问。

对 `s2xh4 = (2,(2,2)):(4,(2,1))`，可以把它看成：

- 外层第一个坐标 `m` 决定行，贡献 `m * 4`
- 第二个坐标 `n` 会先被映射成内部 `shape=(2,2)` 的坐标，再按 `stride=(2,1)` 计算偏移

因此内部 1D 坐标 `n=0,1,2,3` 会依次对应到偏移 `0,2,1,3`，所以 `print2D(s2xh4)` 的两行就是：

- `m=0` 时得到 `0,2,1,3`
- `m=1` 时再整体加上 `4`，得到 `4,6,5,7`

继续推广一下，如果把整个 layout 都看成一个单独的 multi-mode，那么它也可以接受一维坐标。下面这个 `print1D`：

```c++
template <class Shape, class Stride>
void print1D(Layout<Shape,Stride> const& layout)
{
  for (int i = 0; i < size(layout); ++i) {
    printf("%3d  ", layout(i));
  }
}
```

对上面的布局会得到：

```console
> print1D(s2xs4)
  0    1    2    3    4    5    6    7
> print1D(s2xd4_a)
  0   12    1   13    2   14    3   15
> print1D(s2xh4_col)
  0    1    2    3    4    5    6    7
> print1D(s2xh4)
  0    4    2    6    1    5    3    7
```

也就是说，layout 的任意 multi-mode，乃至整个 layout 本身，都可以接受一个一维坐标。

CuTe 还提供了更强的可视化工具：

- `print_layout`：输出格式化二维表格
- `print_latex`：输出 LaTeX，可进一步生成彩色矢量图

例如：

```text
> print_layout(s2xh4)
(2,(2,2)):(4,(2,1))
      0   1   2   3
    +---+---+---+---+
 0  | 0 | 2 | 1 | 3 |
    +---+---+---+---+
 1  | 4 | 6 | 5 | 7 |
    +---+---+---+---+
```

### 向量布局（Vector Layouts）

CuTe 里，凡是 `rank == 1` 的 `Layout`，都可以看作向量（vector）。

例如，`8:1` 可以理解成长度为 8 的连续向量：

```console
Layout:  8:1
Coord :  0  1  2  3  4  5  6  7
Index :  0  1  2  3  4  5  6  7
```

而 `8:2` 则表示 8 个元素，索引步长是 2：

```console
Layout:  8:2
Coord :  0  1  2  3  4  5  6  7
Index :  0  2  4  6  8 10 12 14
```

进一步地，`((4,2)):((2,1))` 虽然内部 shape 看起来像 `4x2` 的 row-major 矩阵，但由于最外层 rank 仍然是 1，所以它依旧会被当成向量：

```console
Layout:  ((4,2)):((2,1))
Coord :  0  1  2  3  4  5  6  7
Index :  0  2  4  6  1  3  5  7
```

这意味着：前 4 个元素步长为 2，然后再复制出一组新的 4 元素块，并给它们整体加一个 stride-1 偏移。

再看 `((4,2)):((1,4))`：

```console
Layout:  ((4,2)):((1,4))
Coord :  0  1  2  3  4  5  6  7
Index :  0  1  2  3  4  5  6  7
```

作为“整数到整数的函数”，它与 `8:1` 完全相同，都是恒等映射。

### 矩阵示例（Matrix Examples）

同理，凡是 `rank == 2` 的 `Layout`，都可以看成矩阵。

例如：

```console
Shape :  (4,2)
Stride:  (1,4)
  0   4
  1   5
  2   6
  3   7
```

这是一个 `4x2` 的 column-major 布局；而：

```console
Shape :  (4,2)
Stride:  (2,1)
  0   1
  2   3
  4   5
  6   7
```

则是一个 `4x2` 的 row-major 布局。所谓“major”，本质上只是看哪一维的 stride 是 1。

和向量一样，矩阵的每个 mode 也可以继续拆成 multi-mode。例如：

```console
Shape:  ((2,2),2)
Stride: ((4,1),2)
  0   2
  4   6
  1   3
  5   7
```

逻辑上它仍然是一个 `4x2` 矩阵，只不过列方向本身变成了多层 stride。由于它在逻辑上依然是 `4x2`，所以你仍可以用普通二维坐标去访问它。

## Layout 概念（Layout Concepts）

这一节介绍 `Layout` 能接受哪些坐标，以及坐标映射（coordinate mapping）和索引映射（index mapping）是如何计算的。

### Layout 兼容性（Layout Compatibility）

如果布局 A 的 shape 与布局 B 的 shape 兼容，我们就说 A 与 B 兼容。

shape A 与 shape B 兼容，当且仅当：

- A 的 `size` 等于 B 的 `size`
- A 中所有合法坐标，也都是 B 中的合法坐标

例如：

- `24` 与 `32` 不兼容
- `24` 与 `(4,6)` 兼容
- `(4,6)` 与 `((2,2),6)` 兼容
- `((2,2),6)` 与 `((2,2),(3,2))` 兼容
- `24` 与 `((2,2),(3,2))` 兼容
- `24` 与 `((2,3),4)` 兼容
- `((2,3),4)` 与 `((2,2),(3,2))` 不兼容
- `((2,2),(3,2))` 与 `((2,3),4)` 不兼容
- `24` 与 `(24)` 兼容
- `(24)` 与 `24` 不兼容
- `(24)` 与 `(4,6)` 不兼容

这里要注意，“兼容”是有方向的：`A` 与 `B` 兼容，表示 A 的所有合法坐标也都是 B 的合法坐标。

- `24` 只接受标量坐标 `0..23`
- `(24)` 除了等价的一维位置外，还接受一元 tuple 坐标 `(0)..(23)`

因此 `24 -> (24)` 可以成立，但 `(24) -> 24` 不成立；同理，`(4,6)` 也不接受 `(7)` 这种一元 tuple 坐标，所以 `(24) -> (4,6)` 也不成立。

因此，“compatible” 在 shape 上构成了一个弱偏序关系（weak partial order）：它满足自反、反对称和传递。

### Layout 坐标（Layout Coordinates）

基于上面的兼容性概念，需要强调一点：

> 一个 `Layout` 往往可以接受多种不同形式的坐标。

只要某种 shape 与该 layout 的 shape 兼容，那么该 shape 中的坐标通常都能拿来索引这个 layout。CuTe 通过一种 **colexicographical order**（从右往左读的字典序）在这些坐标集合之间建立对应关系。

因此，每个 `Layout` 都有两个核心映射：

1. 从输入坐标到自然坐标（natural coordinate）的映射，由 `Shape` 决定
2. 从自然坐标到线性索引的映射，由 `Stride` 决定

#### 坐标映射（Coordinate Mapping）

输入坐标到自然坐标的映射，本质上就是在 `Shape` 上应用一种 colexicographical 展开规则。

例如，对 shape `(3,(2,3))`，它同时拥有：

- 一维坐标集
- 二维坐标集
- 自然坐标（h-D coordinate）集

这三类坐标是等价的，只是表示方式不同。所有等价坐标最终都会映射到同一个自然坐标。

换句话说，一个 shape 为 `(3,(2,3))` 的 layout：

- 可以像 18 元素一维数组那样，用 `0..17` 去索引
- 也可以像 `3x6` 二维矩阵那样，用二维坐标索引
- 还可以像 `3 x (2x3)` 这样的分层张量，用自然坐标索引

前文的 `print1D` 正说明了这一点：当你从 `0` 迭代到 `size(layout)`，再用这个单整数去索引 layout，本质上就是在按广义列主序枚举其更高维坐标。

坐标映射由 `cute::idx2crd(idx, shape)` 完成：

```cpp
auto shape = Shape<_3,Shape<_2,_3>>{};
print(idx2crd(   16, shape));                                // (1,(1,2))
print(idx2crd(_16{}, shape));                                // (_1,(_1,_2))
print(idx2crd(make_coord(   1,5), shape));                   // (1,(1,2))
print(idx2crd(make_coord(_1{},5), shape));                   // (_1,(1,2))
print(idx2crd(make_coord(   1,make_coord(1,   2)), shape));  // (1,(1,2))
print(idx2crd(make_coord(_1{},make_coord(1,_2{})), shape));  // (_1,(1,_2))
```

#### 索引映射（Index Mapping）

自然坐标到索引的映射则更直接：把自然坐标与 `Stride` 做内积（inner product）。

例如，对于 layout：

`(3,(2,3)):(3,(12,1))`

自然坐标 `(i,(j,k))` 会被映射成：

`i*3 + j*12 + k*1`

也就是表格里这些值：

```console
       0     1     2     3     4     5     <== 1-D col coord
     (0,0) (1,0) (0,1) (1,1) (0,2) (1,2)   <== 2-D col coord (j,k)
    +-----+-----+-----+-----+-----+-----+
 0  |  0  |  12 |  1  |  13 |  2  |  14 |
    +-----+-----+-----+-----+-----+-----+
 1  |  3  |  15 |  4  |  16 |  5  |  17 |
    +-----+-----+-----+-----+-----+-----+
 2  |  6  |  18 |  7  |  19 |  8  |  20 |
    +-----+-----+-----+-----+-----+-----+
```

索引映射由 `cute::crd2idx(c, shape, stride)` 完成。它会先把输入坐标转成该 shape 对应的自然坐标，再与 stride 做内积：

```cpp
auto shape  = Shape <_3,Shape<  _2,_3>>{};
auto stride = Stride<_3,Stride<_12,_1>>{};
print(crd2idx(   16, shape, stride));       // 17
print(crd2idx(_16{}, shape, stride));       // _17
print(crd2idx(make_coord(   1,   5), shape, stride));  // 17
print(crd2idx(make_coord(_1{},   5), shape, stride));  // 17
print(crd2idx(make_coord(_1{},_5{}), shape, stride));  // _17
print(crd2idx(make_coord(   1,make_coord(   1,   2)), shape, stride));  // 17
print(crd2idx(make_coord(_1{},make_coord(_1{},_2{})), shape, stride));  // _17
```

## Layout 变换（Layout Manipulation）

### 取子布局（Sublayouts）

可以用 `layout`：

```cpp
Layout a   = Layout<Shape<_4,Shape<_3,_6>>>{}; // (4,(3,6)):(1,(4,12))
Layout a0  = layout<0>(a);                     // 4:1
Layout a1  = layout<1>(a);                     // (3,6):(4,12)
Layout a10 = layout<1,0>(a);                   // 3:4
Layout a11 = layout<1,1>(a);                   // 6:12
```

也可以用 `select`：

```cpp
Layout a   = Layout<Shape<_2,_3,_5,_7>>{};     // (2,3,5,7):(1,2,6,30)
Layout a13 = select<1,3>(a);                   // (3,7):(2,30)
Layout a01 = select<0,1,3>(a);                 // (2,3,7):(1,2,30)
Layout a2  = select<2>(a);                     // (5):(6)
```

或者 `take`：

```cpp
Layout a   = Layout<Shape<_2,_3,_5,_7>>{};     // (2,3,5,7):(1,2,6,30)
Layout a13 = take<1,3>(a);                     // (3,5):(2,6)
Layout a14 = take<1,4>(a);                     // (3,5,7):(2,6,30)
// take<1,1> not allowed. Empty layouts not allowed.
```

### 拼接（Concatenation）

可以直接用 `make_layout` 把多个 `Layout` 包起来并拼接：

```cpp
Layout a = Layout<_3,_1>{};                     // 3:1
Layout b = Layout<_4,_3>{};                     // 4:3
Layout row = make_layout(a, b);                 // (3,4):(1,3)
Layout col = make_layout(b, a);                 // (4,3):(3,1)
Layout q   = make_layout(row, col);             // ((3,4),(4,3)):((1,3),(3,1))
Layout aa  = make_layout(a);                    // (3):(1)
Layout aaa = make_layout(aa);                   // ((3)):((1))
Layout d   = make_layout(a, make_layout(a), a); // (3,(3),3):(1,(1),1)
```

这里的 `make_layout(L1, L2, ...)` 可以理解为：把每个输入 `Layout` 当成一个 mode，按顺序打包成更高层的 `Layout`。它会保留原有层次结构，不会自动展平；因此 `make_layout(row, col)` 得到的是 `((3,4),(4,3))`，而不是 `(3,4,4,3)`。

也可以用 `append`、`prepend` 和 `replace`：

```cpp
Layout a = Layout<_3,_1>{};                     // 3:1
Layout b = Layout<_4,_3>{};                     // 4:3
Layout ab = append(a, b);                       // (3,4):(1,3)
Layout ba = prepend(a, b);                      // (4,3):(3,1)
Layout c  = append(ab, ab);                     // (3,4,(3,4)):(1,3,(1,3))
Layout d  = replace<2>(c, b);                   // (3,4,4):(1,3,3)
```

### 分组与展平（Grouping and Flattening）

可以用 `group` 把多个 mode 组合成一个更高层 mode，用 `flatten` 把它们再铺平：

```cpp
Layout a = Layout<Shape<_2,_3,_5,_7>>{};  // (_2,_3,_5,_7):(_1,_2,_6,_30)
Layout b = group<0,2>(a);                 // ((_2,_3),_5,_7):((_1,_2),_6,_30)
Layout c = group<1,3>(b);                 // ((_2,_3),(_5,_7)):((_1,_2),(_6,_30))
Layout f = flatten(b);                    // (_2,_3,_5,_7):(_1,_2,_6,_30)
Layout e = flatten(c);                    // (_2,_3,_5,_7):(_1,_2,_6,_30)
```

`group<i,j>` 的意思是把第 `i` 到第 `j-1` 个 mode 打包成一个更高层 mode；`flatten` 则把这些分层 mode 再展开回一层。换句话说，`group` 更像“加括号”，`flatten` 更像“去括号”。它们主要改变的是坐标的层次结构，而不是数据总数。

分组、展平和重排 mode，可以让你原地重解释 tensor，例如把矩阵看成向量，把向量看成矩阵，或者把某些分层结构折叠成更简单的视图。

### 切片（Slicing）

`Layout` 也支持 slicing，但这类操作通常更适合在 `Tensor` 上进行。具体可参考 [`03_tensor.zh-CN.md`](./03_tensor.zh-CN.md)。

## 小结（Summary）

- `Layout` 的 `Shape` 定义了它可接受的坐标空间。
- 每个 `Layout` 都有一个一维坐标空间，因此总可以被视为“按某种顺序枚举的线性对象”。
- 每个 `Layout` 也有一个 rank 为 `R` 的 `R` 维坐标空间。
- 每个 `Layout` 还有一个分层的自然坐标空间（h-D / hierarchical coordinate space）。
- `Stride` 负责把自然坐标映射成线性索引，本质上就是做内积。

因此，对任意 `Layout`，都存在一个与它兼容的整数 shape，即 `size(layout)`。从这个角度可以得到一个很重要的结论：

> Layout 本质上是“从整数到整数的函数”。

如果你熟悉 C++23 的 `mdspan`，这里有个关键区别：CuTe 的 `Layout` 是一等公民（first-class citizen），原生支持层级结构，因此很自然地就能表示超越 row-major / column-major 的复杂映射；它也原生支持用分层坐标去索引。`mdspan` 虽然也能表达复杂布局，但通常需要用户显式定义自定义 layout mapping，而且一个多维 `mdspan` 不能直接接受一维坐标。

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
