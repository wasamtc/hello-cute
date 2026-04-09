# cute概述

cute是nvidia开发的一个用于高性能计算的cuda模板库，在cutlass3.x引入，相比与cutlass，cute更加灵活，可以使用其定义的layout和tensor开发高性能的算子。

# cute中的layout定义

layout是cute中最核心的抽象，其主要表示“数据元素在内存里如何组织”，这样我们写算子的时候可以针对cute表示的逻辑上的多维数组编程。
另外cute提供一整套layout的代数操作，可以方便的组合layout，从而实现更加复杂的数据布局。

# cute中的打印

在本教程中会用到大量cute的打印函数，cute的打印函数既能在host上工作，也能在device上工作。但要注意，在device上打印非常昂贵。即使只是把打印代码留在device路径里，而运行时并没有真的执行到它，例如在一个永远不满足的if分支中，也可能让编译器生成更慢的代码。因此，**调试结束后应尽量删掉device侧打印代码（非常重要！！！）**。

打印函数相关的函数主要有：

- cute::print：对几乎所有cute类型都提供了重载，包括指针、整数、步长、形状、布局、张量。如果你不确定某个对象该怎么观察，先试着对它调用print。
- cute::print_layout：可以把任意rank-2 layout打印成纯文本表格，非常适合可视化“坐标到索引”的映射关系。
- cute::print_tensor：可以把rank-1、rank-2、rank-3和rank-4 tensor打印成纯文本多维表格；它会把tensor中的值也打印出来，方便你验证一次copy之后，tile的数据是不是你预期的那块。
- cute::print_latex：会输出一组LaTeX命令，你可以用pdflatex生成排版更漂亮、带颜色的表格。它支持Layout、TiledCopy和TiledMMA，对理解cute中的布局模式和分区模式很有帮助。
- bool thread0()：只会在“全局线程0”时返回true，也就是threadblock 0中的thread 0。
- bool thread(int tid, int bid)：当当前执行上下文正好是线程tid、threadblock bid时返回true。

# 一些基础概念

## 整数

cute中大量使用动态整数和静态整数，动态整数就是运行时才知道值的普通整数类型，例如int、size_t、uint16_t等，凡是std::is_integral接受的类型，在cute里都可以视为动态整数。静态整数则是像std::integral_constant这样的类型实例，它们把值编码在类型里，通过static constexpr成员暴露。cute为cuda提供了一套自己的静态整数类型cute::C，并且重载了运算符，使静态整数之间的运算结果仍然尽量保持为静态整数。

cute还提供一些常用别名，例如Int<1>、Int<2>、Int<3>，_1、_2、_3。
在编写算子的时候尽量使用静态整数，因为静态整数在编译期就知道值，可以避免运行时计算且给编译期提供优化信息。

## tuple

tuple是一个有序且有限的元素列表，可以有零个或多个元素，行为上类似std::tuple。

## inttuple

inttuple是一个整数或者由inttuple组成的tuple（注意这是一个递归定义）。一些inttuple的例子：

- int{2}：动态整数 2
- Int<3>{}：静态整数 3
- make_tuple(int{2}, Int<3>{})
- make_tuple(uint16_t{42}, make_tuple(Int<1>{}, int32_t{3}), Int<17>{})

cute用inttuple来承载很多概念，**shape、stride、step和coord都是inttuple**。

常见操作包括：

- rank(inttuple)：元素个数。单个整数的rank为1；tuple的rank为tuple_size。例如rank(1, (1,2))=2。
- get\<I\>(inttuple)：取第I个元素。对于单个整数，get<0>就是它本身。
- depth(inttuple)：层级深度。单个整数深度为0；由整数构成的tuple深度为1；包含tuple的tuple深度为2；以此类推。
- size(inttuple)：所有元素的乘积。例如size(1, (1,2))=2。

## shape和stride

shape和stride都是inttuple，表示一个多维数组的维度。
shape表示一个多维数组的逻辑形状，例如(2,3)表示一个2x3的矩阵。
stride表示一个多维数组在内存中的步长，例如(2,3)表示dim0的步长为2，dim1的步长为3。

## layout

layout是一个pair，由shape和stride组成，表示一个多维数组的形状和步长。例如(2,3):(1,3)表示一个2x3的矩阵，其在行维度stride为1，在列维度stride为3。
其实从(2,3):(1,3)可以看出cute的强大，因为其内存布局为

| 物理地址 (1D) | 逻辑坐标 (2D) | 行号 (r) <br> `stride=1` | 列号 (c) <br> `stride=3` | 状态说明 |
| :---: | :---: | :---: | :---: | :--- |
| **0** | `(0, 0)` | 0 | 0 | 📍 第 0 列起点 |
| **1** | `(1, 0)` | 1 | 0 | ⬇️ 行号 +1，地址 +1（内存连续） |
| **2** | `N/A` | - | - | 🕳️ **空洞 (Padding)** |
| **3** | `(0, 1)` | 0 | 1 | 📍 第 1 列起点（列号 +1，地址跨越 3） |
| **4** | `(1, 1)` | 1 | 1 | ⬇️ 行号 +1，地址 +1（内存连续） |
| **5** | `N/A` | - | - | 🕳️ **空洞 (Padding)** |
| **6** | `(0, 2)` | 0 | 2 | 📍 第 2 列起点（列号 +1，地址跨越 3） |
| **7** | `(1, 2)` | 1 | 2 | ⬇️ 行号 +1，地址 +1（内存连续） |

如果不使用cute的layout而想要实现这种布局是相当困难的，cute能够方便地实现且提供细粒度的控制。

## tensor

layout与数据（例如数组或指针）组合构成tensor。

# 创建与使用layout

## 构造layout

layout的构造方式很多，静态整数和动态整数可以混用：

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

类似地，make_shape 和 make_stride 也分别返回 Shape 与 Stride。

如果省略 Stride，CuTe 默认使用 LayoutLeft 根据 Shape 自动生成 stride。LayoutLeft 会把 shape 从左到右做 exclusive prefix product，可以看作一种“广义列主序（generalized column-major）”生成方式。LayoutRight 则从右到左生成 stride，对于 depth 为 1 的 shape，可以把它近似理解成 row-major；但对分层 shape 来说，结果可能没有那么直观。

对上面的 layout 调用 print，会得到：

```
s8        :  _8:_1
d8        :  8:_1
s2xs4     :  (_2,_4):(_1,_2)
s2xd4     :  (_2,4):(_1,_2)
s2xd4_a   :  (_2,4):(_12,_1)
s2xd4_col :  (_2,4):(_1,_2)
s2xd4_row :  (_2,4):(4,_1)
s2xh4     :  (2,(2,2)):(4,(2,1))
s2xh4_col :  (2,(2,2)):(4,(2,1))
```

这里常见记法是 Shape:Stride。其中 _N 表示静态整数，其他写法则通常表示动态整数。可以看到 Shape 和 Stride 都允许静态、动态整数混合出现。

## layout的核心映射

layout的根本用途是把坐标通过shape和stride映射到线性索引（物理地址）。例子可见[layout](#layout)。
要完成这种转换需要两个核心映射：

- idx2crd(idx, shape)：从输入坐标到逻辑坐标的映射，由 `Shape` 决定。
- crd2idx(crd, shape, stride)：从逻辑坐标crd到线性索引idx的映射，，由 `Stride` 决定。

### 从输入坐标到逻辑坐标
这部分映射主要由`shape`决定。
从输入坐标到逻辑坐标的映射由idx2crd完成。首先对于一个layout，其有多种坐标集表示方式，例如对于shape `(3,(2,3))`，它同时拥有：

- 一维坐标集
- 二维坐标集
- 自然坐标（h-D coordinate）集

这三类坐标是等价的，只是表示方式不同。所有等价坐标最终都会映射到同一个自然坐标。

换句话说，一个 shape 为 `(3,(2,3))` 的 layout：

- 可以像 18 元素一维数组那样，用 `0..17` 去索引
- 也可以像 `3x6` 二维矩阵那样，用二维坐标索引
- 还可以像 `3 x (2x3)` 这样的分层张量，用自然坐标索引

官方的例子：

```c++
auto shape = Shape<_3,Shape<_2,_3>>{};
print(idx2crd(   16, shape));                                // (1,(1,2))
print(idx2crd(_16{}, shape));                                // (_1,(_1,_2))
print(idx2crd(make_coord(   1,5), shape));                   // (1,(1,2))
print(idx2crd(make_coord(_1{},5), shape));                   // (_1,(1,2))
print(idx2crd(make_coord(   1,make_coord(1,   2)), shape));  // (1,(1,2))
print(idx2crd(make_coord(_1{},make_coord(1,_2{})), shape));  // (_1,(1,_2))
```

`idx2crd` 是如何计算的呢？公式如下。

```text
对于输入坐标是单个整数的情况（包含上面 tuple 展开后变成整数的情况）：
idx2crd(i, (s0, rest)) = (i % s0, idx2crd(i / s0, rest))

例如：
16 % 3 = 1, 16 / 3 = 5
所以 idx2crd(16, (3, (2, 3))) = (1, (1, 2))

对于输入坐标是 tuple 的情况：
idx2crd((i0, i1, ..., in-1), (s0, s1, ..., sn-1))
= (idx2crd(i0, s0), idx2crd(i1, s1), ..., idx2crd(in-1, sn-1))

例如：
idx2crd((1, 5), (3, (2, 3)))
= (idx2crd(1, 3), idx2crd(5, (2, 3)))
= (1, (1, 2))
```

我们可以这样理解这个公式：`idx2crd` 本质上是在做一种“混合进制拆位”。

假设 `shape = (2, 3)`，这表示：

- 第 0 维大小是 `2`
- 第 1 维大小是 `3`

如果按 CuTe 这里的规则把它压成一维，那么索引公式是：

```text
idx = c0 + 2 * c1
```

这里 `c0` 的取值范围是 `0..1`，`c1` 的取值范围是 `0..2`。

现在反过来，已知 `idx`，怎么求 `(c0, c1)`？

第一步，先求最低位 `c0`。因为 `c0` 只能在 `0..1` 里变化，所以：

```text
c0 = idx % 2
```

为什么是 `% 2`？

因为每逢加到 `2`，这一位就会“进位”到下一维。这和十进制里“个位 = 数字 % 10”是一个道理。

第二步，再求高一位 `c1`。把最低位拿掉以后，剩下的就是：

```text
c1 = idx / 2
```

所以：

```text
idx2crd(idx, (2, 3)) = (idx % 2, idx / 2)
```

这就是为什么公式里会出现：

```text
(i % s0, i / s0)
```

推广到更多维，如果 `shape = (s0, s1, s2, ...)`，那第一维就是“最低位”：

```text
c0 = i % s0
```

把这一位剥掉，剩下的编号是：

```text
i' = i / s0
```

然后继续对后面的 `shape` 做同样的事：

```text
idx2crd(i, (s0, rest)) = (i % s0, idx2crd(i / s0, rest))
```

所以这个公式不是硬背出来的，而是：

- 先拆当前这一位
- 再把剩余部分递归交给后面的维度

为什么 tuple 输入时要“逐项递归”？

如果输入本身已经是个 tuple，比如：

```text
i = (1, 5)
shape = (3, (2, 3))
```

它的意思不是“一个压扁的一维编号”，而是：

- 第一部分坐标是 `1`
- 第二维的坐标还被压成了一个 `5`

所以你不能再把整个 `(1, 5)` 当成一个整数整体去做 `%` 和 `/`，而应该：

- 第一项 `1` 对应 `shape` 的第一项 `3`
- 第二项 `5` 对应 `shape` 的第二项 `(2, 3)`

于是只能逐项处理：

```text
idx2crd((1, 5), (3, (2, 3)))
= (idx2crd(1, 3), idx2crd(5, (2, 3)))
= (1, (1, 2))
```

### 从逻辑坐标到线性索引

这部分映射主要由`stride`决定。
这个计算很简单，就是把逻辑坐标与stride做内积。
例如，对于 layout：

`(3,(2,3)):(3,(12,1))`

自然坐标 `(i,(j,k))` 会被映射成：

`i*3 + j*12 + k*1`

索引映射由 `cute::crd2idx(c, shape, stride)` 完成。它会先把输入坐标转成该 shape 对应的自然坐标（这部分同第一种映射），再与 stride 做内积：

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

## layout变换

下面的都比较简单，这里我就直接用官方的例子了

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

# 测试

这一节对应的测试名是 `layout_base`。

运行方式：

```bash
python tests/run.py layout_base
```

脚本会输出 5 个 layout 和对应的 5 个输入坐标，要求你按顺序输入这 5 个坐标对应的线性索引（空格分隔）。脚本会用 `cute` 的 `crd2idx` 结果来判分。

每次运行时，这 5 道题的题型保持不变，但具体的 shape、stride 和输入坐标会随机生成，所以题目不会固定死。如果你想复现同一套题，可以显式指定随机种子：

```bash
python tests/run.py layout_base --seed 20260409
```

测试脚本位置：

- `tests/run.py`
- `tests/layout_base/quiz.py`

依赖说明：

- 仓库里自带了 `pycute`，它是 CuTe layout algebra 的 Python 镜像实现，这套交互题默认用它来判分。
- 即使你本地安装了官方的 CuTe Python DSL，当前版本的 `cutlass.cute` 里 layout algebra 也主要要求放在 `@cute.jit` 函数里执行，不能直接当 eager Python API 来写这类命令行问答脚本；因此脚本检测到这种情况时会自动回退到 `pycute`。
- 如果你想安装官方 DSL 环境，仍然可以执行：

```bash
pip install -r requirements.txt
```

如果你使用的是 CUDA 13 环境，可以改用：

```bash
pip install -r requirements-cu13.txt
```

另外，测试脚本也支持非交互模式，便于自测：

```bash
python tests/run.py layout_base --show-answers
python tests/run.py layout_base --seed 20260409 --show-answers
```
