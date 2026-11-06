# 第11章 IQ 系列：重要性矩阵与码本量化

前面四章讲完 K-quants，你已经掌握了一整套把权重压到 4.5 bpw（`Q4_K`）、5.5 bpw（`Q5_K`）、乃至 2.625 bpw（`Q2_K`）的完整方案。那是一个逻辑自洽的世界：超级块、两级尺度、6-bit 解包、`y = d·s_j·q − dmin·m_j`——所有的误差，都被归因到"尺度和码位宽"这两样东西上。

但这一章，要做一个**范式切换**。当目标位宽降到 2 bpw 以下时，K-quants 的那套统一公式就开始失效了：用 `y = d·s·q − m` 这套"算术"去表达一个权重，无论尺度选得多好，都逃不开"码字必须均匀分布"这个限制。而 `IQ1_S` 给出的答案是——**别再用公式算了，直接查表**。它只用 1.5625 bpw（每 256 值仅 50 字节）存一个指向 2048 项码本（codebook）的索引，反量化时不再做乘加，而是"查一个预先算好的 8 值向量"。

先给你一个体量上的直觉对比：同样的权重数，`Q4_K` 需要 4.5 bpw，`IQ2_XXS` 只要 2.0625 bpw，而 `IQ1_S` 只要 1.5625 bpw。对一个 7B 模型来说，从 `Q4_K` 的约 3.8 GB 降到 `IQ1_S` 的约 1.3 GB，意味着端侧设备能多装下两倍还多的模型。代价则是：**质量不再由"算术精度"来决定，而是由"码本覆盖率"和"激活重要性"决定**——而这两者，恰恰是本章全部新概念（grid、imatrix、delta）的来源。

这一章的目标一共有三个。第一，理解**为什么仿射量化在 2 bpw 以下会失效**（11.1）。第二，掌握**imatrix 的生成与加权机制**（11.2）。第三，读透 **IQ1_S 这个最小的查表格式**的字节布局与解码循环（11.3），并对 IQ2/IQ3/IQ4 这一整个家族建立一张地图（11.4）。最后，还要在 11.5 里老老实实地交代 IQ 系列的工程代价。

照例先把几个术语统一一下。**码本（codebook / grid）**：一张查找表，每一行都是一个固定的小向量（比如 8 个整数）；量化的时候不再一个一个地去估计权重，而是看"选哪一行最像"。**索引（index）**：指向码本某一行的小整数，量化完成之后真正存进文件里的，就是它。**重要性矩阵（imatrix）**：一个和权重形状相同的浮点数组，每个位置都记录着"这个权重对模型输出有多重要"，量化时会把误差预算往重要的位置上倾斜。**加权搜索**：量化时极小化的目标，从"均匀的均方误差"改成了"按重要性加权之后的误差"。


## 11.1 范式切换：为什么要码本

### 11.1.1 仿射量化在 2 bpw 以下的失效模式

先来复习一下前面几章反复用到的仿射量化。给一个权重 \(x\)，把它量化成 \(q\) 位的整数码：

\[
q = \mathrm{clamp}(\mathrm{round}((x + m)/d),\ 0,\ 2^q - 1), \qquad \hat{x} = d\cdot q - m
\]

反量化结果的形态，是 \(2^q\) 个**等间距**的离散点：尺度 \(d\) 决定点与点的间距，偏移 \(m\) 决定起点落在哪里。对 4-bit（16 个点）、6-bit（64 个点）来说，这个均匀网格已经足够贴合权重分布了。可一旦码的位宽降到 2–3 bit，问题就暴露出来了：

```
均匀量化 vs 重尾分布的错配

  权重概率密度（重尾，中间高、两端拖长）
     │           ██
     │          ████
     │          ████
     │         ██████
     │       ██████████
     │   ██████████████████
     └───┬────┬────┬────┬────► x
         d0   d1   d2   d3

  4 个均匀码字 d0..d3 等间距排开：
  · 中间密集区只用 1–2 个码字覆盖 → 大量小权重塌缩到同一点
  · 两端拖长的离群值全被 clamp 到 d0 / d3      → 饱和误差
```

下面用数字来说话。假设有一组符合典型权重分布的值（大量接近零、少数绝对值很大）：`[0.001×30 个, 0.05×20 个, 0.9, -0.7]`，把它强行压到 2-bit 的均匀量化里：

**Python：**
```python
import numpy as np

def uniform_quantize(x, levels):
    xmin, xmax = x.min(), x.max()
    d = (xmax - xmin) / (levels - 1)
    q = np.clip(np.round((x - xmin) / d), 0, levels - 1)
    return q, d * q.astype(np.float64) + xmin

rng = np.random.default_rng(0)
x = np.concatenate([rng.normal(0, 0.01, 30), rng.normal(0, 0.1, 20), [0.9, -0.7]]).astype(np.float64)

_, xhat = uniform_quantize(x, 4)          # 2-bit → 4 级
mse = ((x - xhat) ** 2).mean()
unique = np.unique(xhat)
print(f"重建 MSE = {mse:.6f}")
print(f"只有 {len(unique)} 个不同重建值，大部分小权重被抹平：")
print(np.round(np.unique(xhat), 4))
```

跑出来的输出会告诉你：MSE 很大，而且 52 个权重被压缩成了极少数几个重建值——它们彼此之间的差异，在均匀网格下完全不可分辨。这正是"仿射量化在低位宽下失效"最直观的形态。

**C++（同一逻辑的最小实现）：**
```cpp
#include <cmath>
#include <vector>
#include <algorithm>

struct Hists { double mse = 0; int n_levels = 0; };

Hists uniform_quantize_2bit(const std::vector<double>& x) {
    auto [lo, hi] = std::minmax_element(x.begin(), x.end());
    const double xmin = *lo, xmax = *hi;
    const double d = (xmax - xmin) / 3.0;      // 4 个码字
    double mse = 0;
    std::vector<double> recon;
    for (double v : x) {
        int q = std::clamp((int)std::round((v - xmin) / d), 0, 3);
        recon.push_back(d * q + xmin);
        mse += (v - recon.back()) * (v - recon.back());
    }
    mse /= x.size();
    std::sort(recon.begin(), recon.end());
    recon.erase(std::unique(recon.begin(), recon.end()), recon.end());
    return { mse, (int)recon.size() };
}
```

**C#（.NET 5+）：**
```csharp
using System;
using System.Linq;

double[] x = new double[52]; /* 用 30+20+2 个样例行填充，见 Python 版本 */
double xmin = x.Min(), xmax = x.Max();
double d = (xmax - xmin) / 3.0;
double mse = 0;
var recon = new double[x.Length];
for (int i = 0; i < x.Length; i++)
{
    int q = Math.Clamp((int)Math.Round((x[i] - xmin) / d), 0, 3);
    recon[i] = d * q + xmin;
    mse += (x[i] - recon[i]) * (x[i] - recon[i]);
}
mse /= x.Length;
Console.WriteLine($"MSE={mse:F6} distinct={recon.Distinct().Count()}");
```

问题的根源，出在**码字的排布方式**上：均匀量化把 \(2^q\) 个点等间距地铺开，可真实权重的密度并不均匀。就算把 K-quants 那套层次化尺度搬进来，本质上也仍然是在给"均匀网格"做不同的缩放——当只剩 2 bit 时，网格实在太稀，无论缩放得多精细，都补不齐分布的形状。**所以，这个结论是结构性的，而不是参数性的：低位宽需要非均匀的重建点**。

### 11.1.2 码本 + 重要性：两个工具突破瓶颈

IQ 系列针对这两个失效点，拿出了两个工具，正好对应本章后半部分的两个主角：

1. **非线性码本（grid）**：不再把重建点固定在 `d × q − m` 这个等差数列上，而是预先准备一组"和真实权重分布更像"的小向量，把它们存成一张表。反量化也随之变成了"按索引取出一整行"——这一行里的 8 个值，可以是非均匀的、带符号的、甚至相互关联的；均匀量化做不到的事，查表却可以做到。
2. **重要性矩阵（imatrix）**：量化时不再均匀地最小化 MSE，而是按照"每个权重对模型输出的影响"去分配误差预算。误差落在不重要的权重上，顶多造成一点噪声；可一旦落到重要的权重上，就会直接改变输出的 logits 分布。

这两件事可以组合成一个目标函数：极小化按重要性加权的重建误差

\[
\min \sum_i \alpha_i (x_i - \hat{x}_i)^2
\]

这里的 \(\alpha_i\)，正是 imatrix 提供的权重。而省略号下面，藏着一个很现实的工程事实：**没有 imatrix 的 IQ 量化，质量和随机取码没什么本质区别**。所以，11.2 要先讲 imatrix 是从哪来的。


## 11.2 imatrix：重要性矩阵的生成与使用

### 11.2.1 `llama-imatrix` 的校准流程与命令

重要性矩阵并不是从模型文件里直接读出来的，而是**在量化之前，先让模型跑一遍文本，统计每个权重被激活得有多厉害**，最后得到的。这个"校准"的过程，由 `llama-imatrix` 这个工具来完成：

```bash
# 用一个 F16 模型 + 一段校准文本，生成 imatrix 文件
./llama-imatrix -m model-f16.gguf -f calibration-data.txt -ngl 99 -o imatrix.dat
```

- `-m model-f16.gguf`：指定校准用的基准模型。它必须是未量化的 F16——量化过的模型跑出来的激活统计，是没有什么意义的。
- `-f calibration-data.txt`：指定校准文本。对文本质量的要求是"和真实使用场景的分布足够接近"——通用的 wiki 文本、多语种文本、代码混合，都比单一领域的文本更稳健。实践中，常用合并后的英文维基或 `wikitext` 这类语料。
- `-ngl 99`：把尽可能多的层 offload 到 GPU 上，用来加快校准的速度。
- `-o imatrix.dat`：指定输出的 imatrix 文件名。
- 可调的参数：`-c <context>` 控制上下文长度（比如 2048/4096），`--chunks <n>` 控制处理多少个校准块。

校准的原理其实很简单：在前向传播时，统计每个权重位置上**激活值（也就是激活的幅度）的平方或加权量**，最终输出一张和每个张量形状相同的浮点矩阵。这张矩阵会在下一步量化时被读进去，告诉量化器"哪些位置上的误差是不可接受的"。

### 11.2.2 加权目标与 `vec_dot` 中的 imatrix 索引

在 `llama-quantize` 内部，imatrix 并不是以"元数据"的身份参与进来的，而是作为**加权向量**直接进入量化例程。拿 `quantize_q4_K` 来说，它的函数签名里就有一个 `quant_weights` 指针：

```c
size_t quantize_q4_K(const float * src, void * dst, int64_t nrow, int64_t n_per_row,
                     const float * quant_weights);
```

调用方传给它的，正是"imatrix 展开之后的一行"：`quant_weights + row * n_per_row`，其中 `row` 是当前正在处理的张量行号，`n_per_row` 是每行的元素个数。量化器在估计子块尺度、挑选码字的时候，会给误差乘上这个权重：

```c
// quantize_row_q4_K_impl 中的加权误差估计示意
const float * qw = quant_weights + QK_K*i + 32*j;          // 当前 32 值子块的权重
for (int l = 0; l < 32; ++l)
    weights[l] = qw[l] * sqrtf(sigma2 + x[32*j + l]*x[32*j + l]);
```

注意这里的位置关系：imatrix 是**逐权重**对齐的（`i*QK_K + l` 这个下标，正好对应该权重自己的重要性），所以误差的加权发生在矩阵乘这一层，而不是张量那一层。这一点也解释了，为什么校准阶段的文本选择如此关键——imatrix 记录下的"重要位置"，取决于校准数据到底见过什么。

对于 IQ 系列来说，imatrix 的存在是**硬性要求**，而不是可有可无的选项。一个没有 imatrix 的 IQ1_S 模型，码本选择会退化成对未加权误差的贪心近邻，质量会显著劣化。这一点，到 11.5.2 会展开细讲。


## 11.3 IQ1_S：1.5625 bpw 的查表反量化

> 说明：细纲 §11.3 曾把 IQ1_S 记为"2048 项 grid / 74 字节 / 1.5 bpw"。按 llama.cpp 主线源码（`asset/facts-verified.md`）核实，实际是 **50 字节 / 256 值 = 1.5625 bpw**。后面所有数字以此为准。

### 11.3.1 2048 项 grid：每项编码 8 个值

IQ1_S 的核心，是一张 2048 项的静态查找表，名字叫 `iq1s_grid`。表里的每一项都是一个 `uint64_t`；把它当成 8 个 `int8_t` 字节展开，就能得到一个 8 值向量——每个值都很小（通常落在 \(\{-1,0,+1\}\) 附近，偶尔也会有更大一点的）：

```c
// ggml-common.h（片段，完整 2048 项）
GGML_TABLE_BEGIN(uint64_t, iq1s_grid, NGRID_IQ1S)   // NGRID_IQ1S = 2048
    0xffffffffffffffff, 0xffffffffffffff01, 0xffffffffffff0000, 0xffffffffffff01ff,
    // ...
GGML_TABLE_END()
```

比如第一项 `0xffffffffffffffff`，展开之后就是 8 个 `-1`（所有字节都是 `0xFF`）；第二项 `0xffffffffffffff01`，展开之后是 `[1, -1, -1, -1, -1, -1, -1, -1]`（因为低字节 `0x01` 就是 `1`）。这些小向量在量化时扮演的是"模板"角色：要把 8 个权重编码成 IQ1_S，就是在 2048 个模板里找出一个"缩放之后最像"的。

为什么偏偏是 8 个值一组？因为 8 值模板刚好可以用一个 `uint64` 存下（天然对齐，SIMD 也友好），而且 8 正是这个码本的"原子"：整块 256 值被切成 32 个 8 值组，每一组靠一个 8-bit 索引定位到 2048 项网格里的某一项。可 8-bit 只有 256 种可能，显然不够，所以索引还需要再补 3 bit——这 3 bit 从 `qh` 里拿过来，高低拼起来正好 11 位，覆盖 2048 项。这个"低 8 位 + 高 3 位"的拆分，就是理解字节布局的关键。

### 11.3.2 尺度与 delta：\(dl\cdot(\mathrm{grid}[j] \pm \Delta)\)

回想一下，Q4_K 的反量化是 `y = d·sc·q − min·m`：全局尺度 `d`、子块尺度 `sc`、最小值 `m`，一套算术走下来。而 IQ1_S 不一样：它**没有那种"每个 32 值子块各自独立、一整套尺度+最小值"的两级参数**，只是用一个 32 值组共享的粗略尺度，再加一个恒定的小修正 `delta`。

具体来说，每个 32 值组（`QK_K/32 = 8` 组）都会拿 `qh[ib]`（一个 `uint16_t`）当"参数小仓库"来用：

- 高 3 位（bit 12–14）决定该组尺度 \(k \in [0,7]\)，有效尺度为
  \[
  dl = d \cdot (2k + 1)
  \]
  也就是说，块尺度实际上是按奇数倍 \(d、3d、5d、\dots、15d\) 来翻的，而不是任意的缩放。这种设计是刻意在限制尺度的分辨率——为的是省下更多 bit。
- bit 15 决定一个全局符号修正 `delta`
  \[
  \Delta = \begin{cases} -0.125 & \text{if bit15 = 1} \\ +0.125 & \text{if bit15 = 0} \end{cases}
  \]
  这个 `IQ1S_DELTA = 0.125` 的意义在于：码本模板里的值大多是 `±1、0`，如果直接乘尺度，就有可能出现"某个权重恰好是 0"的情况；而加上一个固定的 `delta` 之后，重建值永远不会落在精确的 0 上，反量化输出对下游来说也就更平滑了。
- 低 12 位（bit 0–11）被分成 4 组，每组 3 bit，充当 4 个 8 值片段的索引高 3 位。

于是，第 `ib` 组里第 `l` 个 8 值片段的完整反量化公式就是：

\[
y_j = dl \cdot \bigl(\mathrm{grid}[ \ \mathrm{idx}_l \ ]_j + \Delta \bigr),
\qquad
\mathrm{idx}_l = qs[4\cdot ib + l] \ \bigl|\  (((qh[ib] \gg 3l) \& 7) \ll 8)
\]

`idx_l` 的低 8 位来自 `qs`（每个片段占一个字节），高 3 位来自 `qh`；两者拼成一个 11-bit 的索引，再去查表。整个结构可以画成下面这样：

```
IQ1_S 的 32 值组的索引与尺度从哪来

  ┌──────── qh[ib] (uint16, 16 bit) ────────┐
  │ b15   b14 b13 b12   b11..b9 b8..b6   …  │
  │  │     └────┬─────┘      └─┬──┘         │
  │ delta     尺度 k         idx 高3位       │
  │ 符号       (2k+1)*d      (每个片段3位)    │
  └───┬──────────┬─────────────┬────────────┘
      │          │             │
      ▼          ▼             ▼
   ±0.125      dl        ┌─────┴──────┐
      │          │        │           │
      │          │        │  qs[4*ib+l]  │  (低 8 位)
      │          │        ▼           ▼
      │          │     11-bit ──► iq1s_grid  ──► int8[8]
      │          │                          │
      └────┬─────┘                          ▼
           └──────────►  y_j = dl*(grid[j] ± 0.125)
```

### 11.3.3 `block_iq1_s` 的字节布局与反量化循环

`block_iq1_s` 在内存里占 50 个字节，由静态断言锁死：

```cpp
// QK_K = 256
typedef struct {
    ggml_half  d;            // 2 字节：FP16 全局尺度
    uint8_t    qs[QK_K/8];   // 32 字节：4×8 个 8-bit 索引低半
    uint16_t   qh[QK_K/32];  // 16 字节：8 组 × uint16 参数（高 bit 的逆）
} block_iq1_s;
static_assert(sizeof(block_iq1_s) == sizeof(ggml_half) + QK_K/8 + QK_K/16,
              "wrong iq1_s block size/padding");
```

字节数和 bpw 可以这样算：

\[
\text{bpw} = \frac{50 \times 8}{256} = 1.5625 \ \text{bit/weight}
\]

在这个 161 值块里，8 个 32 值组共用同一个 `d`，组内的 `qh` 负责提供尺度、delta 和高 3 位。下面这版反量化主循环（C++，对齐 `ggml-quants.c` 的 `dequantize_row_iq1_s`）：

**C++：**
```cpp
#include <cstdint>
#include <cmath>

#define QK_K 256
#define IQ1S_DELTA 0.125f

// 2048 项网格（片段），真实实现是编译期常量表
static const uint64_t iq1s_grid[2048] = { /* ... */ };

void dequantize_row_iq1_s(const block_iq1_s * x, float * y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        const uint8_t  * qs = x[i].qs;
        const uint16_t * qh = x[i].qh;

        for (int ib = 0; ib < QK_K/32; ++ib) {            // 8 个 32 值组
            const float dl    = d * (2 * ((qh[ib] >> 12) & 7) + 1); // (2k+1)*d
            const float delta = (qh[ib] & 0x8000) ? -IQ1S_DELTA : IQ1S_DELTA;

            for (int l = 0; l < 4; ++l) {                  // 每组的 4 个 8 值片段
                const int8_t * grid = (const int8_t *)(iq1s_grid +
                    (qs[l] | (((qh[ib] >> 3*l) & 7) << 8)));  // 11-bit 索引
                for (int j = 0; j < 8; ++j) {
                    y[j] = dl * (grid[j] + delta);          // 查表 + delta
                }
                y += 8;
            }
            qs += 4;
        }
    }
}
```

这里有三个容易看岔的地方，一个一个说明：

1. **`qs` 的推进节奏，是每处理完一个 32 值组才前进 4 字节**，而不是每处理一个片段就前进一步——因为 `qs[l]` 用的是相对当前组的那个 4 字节窗口。
2. **索引是 11-bit 的**：`qs[l]`（8 位）是低半部分，`((qh[ib] >> 3*l) & 7) << 8`（3 位）是高半部分。它并不是简单的"每对字节对应一个索引"。
3. **整个 `qh[ib]` 的 16 位，被拆成了三种用途**：尺度、delta，以及索引的高 3 位。这正是"用 32 值组共享参数，来换取更低位宽"的直接体现。

**C#（.NET 5+，`unsafe` 结构体 + 指针）：**
```csharp
using System;
using System.Runtime.InteropServices;

public static class IQ1S
{
    public const int QK_K = 256;
    public const float IQ1S_DELTA = 0.125f;
    static readonly ulong[] Grid = { /* 2048 项，片段 */ };

    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public unsafe struct BlockIq1S
    {
        public ushort d;                 // FP16 位模式
        public fixed byte qs[QK_K / 8];
        public fixed ushort qh[QK_K / 32];
    }

    public static unsafe void DequantizeRow(BlockIq1S* x, float* y, int k)
    {
        int nb = k / QK_K;
        for (int i = 0; i < nb; i++)
        {
            float d = (float)BitConverter.UInt16BitsToHalf(x[i].d);
            for (int ib = 0; ib < QK_K / 32; ib++)
            {
                ushort qh = x[i].qh[ib];
                float dl = d * (2 * ((qh >> 12) & 7) + 1);
                float delta = (qh & 0x8000) != 0 ? -IQ1S_DELTA : IQ1S_DELTA;
                for (int l = 0; l < 4; l++)
                {
                    int idx = x[i].qs[ib * 4 + l] | (((qh >> (3 * l)) & 7) << 8);
                    long bits = (long)Grid[idx];         // 取回 8 值模板
                    for (int j = 0; j < 8; j++)
                    {
                        sbyte gv = (sbyte)(bits >> (8 * j)); // 逐字节解出 int8
                        y[j] = dl * (gv + delta);
                    }
                    y += 8;
                }
            }
        }
    }
}
```

**Python（`struct` 显式序列化 + 手工字节验证）：**

```python
import struct

QK_K = 256
IQ1S_DELTA = 0.125

# 与 ggml-common.h 一致的前 4 项（示意；全表 2048 项）
iq1s_grid = [
    0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFF01,
    0xFFFFFFFFFFFF0000, 0xFFFFFFFFFFFF01FF,
]

def _grid_signed(idx: int) -> list[int]:
    bits = iq1s_grid[idx]
    return [v if v < 0x80 else v - 256          # int8 解释：位模式 → 有符号
            for v in ((bits >> (8 * j)) & 0xFF for j in range(8))]

def dequantize_row_iq1_s(data: bytes, k: int) -> list[float]:
    """data: 一个超级块的 50 字节；k: 元素总数（256 的倍数）。"""
    assert k % QK_K == 0
    nb = k // QK_K
    y: list[float] = []

    for i in range(nb):
        base = i * 50
        d  = struct.unpack_from('<e', data, base)[0]                  # FP16 尺度
        qs = data[base + 2 : base + 2 + QK_K // 8]                    # 32 字节
        qh = struct.unpack_from('<8H', data, base + 2 + QK_K // 8)    # 8×uint16

        for ib in range(QK_K // 32):
            dl    = d * (2 * ((qh[ib] >> 12) & 7) + 1)
            delta = -IQ1S_DELTA if (qh[ib] & 0x8000) else IQ1S_DELTA
            for l in range(4):
                idx = qs[4 * ib + l] | (((qh[ib] >> (3 * l)) & 7) << 8)
                grid = _grid_signed(idx)
                for j in range(8):
                    y.append(dl * (grid[j] + delta))
    return y
```

**手工字节验证。** 我们来构造一个 50 字节的超级块：`d = 1.0`（FP16 位模式是 `0x3C00`，小端字节就是 `00 3C`），`qs` 除了 `qs[0] = 1` 之外全部为 0，`qh` 也全部为 0：

```python
blob = struct.pack('<e', 1.0) + b'\x01' + b'\x00' * 31 + b'\x00' * 16
y = dequantize_row_iq1_s(blob, QK_K)

# 推理推导：
#   d=1.0, qh 全 0 → 每组 dl = 1*(2*0+1) = 1，delta = +0.125
#   第 0 组 l=0：idx = 1 | 0 = 1 → grid[1] = 0xFFFFFFFFFFFFFF01
#   展开 = [1, -1, -1, -1, -1, -1, -1, -1]
#   → 前 8 值 = 1*(grid[j] + 0.125) = [1.125, -0.875 × 7]
#   其余所有片段 idx = 0 → grid = [-1 × 8] → 全 = -0.875
assert round(y[0], 4)  == 1.125
assert set(round(v, 4) for v in y[1:8]) == {round(-0.875, 4)}
assert set(round(v, 4) for v in y[8:256]) == {round(-0.875, 4)}
print("IQ1_S 手工字节验证通过，前 8 值:", [round(v, 3) for v in y[:8]])
```

这个测试的价值在于：**它把"字节 → 索引 → 模板 → 尺度"整条链路，死死钉在一个可以人工验算的结果上**。凡是改过 `qh` 位序、索引拼装或者 delta 符号的人，都会在这个断言上立刻翻车——这比任何理论讨论都更能抓住 3 细节错误。

和 K-quants 的算术循环相比，IQ1_S 的主要开销不再是乘加，而是**索引提取与查表**。`grid[j]` 本质上是一次内存/寄存器查表的结果，说它快也快、说它慢也慢，全看后端支不支持高效的向量查表指令（11.5.3 会细讲）。


## 11.4 IQ2/IQ3：不同码本容量的谱系

IQ 系列共用的是"码本 + 索引"这套心法，不过不同成员之间，在码本容量、每个模板编码几个值、符号与尺度怎么存等方面，存在系统性的差异。这一节就给出一张家族地图：不会逐字节展开，但会把每个格式的**bpw，以及它靠什么来换质量**，讲得明明白白。

### 11.4.1 IQ2_XXS/XS/S：256/512/1024 项有符号向量 grid

三个 2-bit 级成员共用"每 8 值一个模板、模板值带符号"的设计，区别在码本容量与每组尺度参数：

```
IQ2 家族的码本容量递增

  IQ2_XXS   256 项 × 8 值    66 字节/256 值 = 2.0625 bpw
  IQ2_XS    512 项 × 8 值    74 字节/256 值 = 2.3125 bpw
  IQ2_S    1024 项 × 8 值    82 字节/256 值 = 2.5625 bpw

  码本越大 → 每组可表达的"形状"越多 → 同 bit 下质量越高
  但超出 256 项后，索引需要更多 bit → 每组存储成本上升
```

拿 IQ2_XXS 举个例子，算一下它的 bpw：

\[
\text{bpw} = \frac{66 \times 8}{256} = 2.0625
\]

下面以 `IQ2_XXS` 为例（另外两个本质相同，只是参数位置不同），它的解码片段长成这样：

**C++（对照 `dequantize_row_iq2_xxs`）：**
```cpp
void dequantize_row_iq2_xxs(const block_iq2_xxs * x, float * y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;
    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;   // 同一个 8 字节的字节视图
    for (int64_t i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            memcpy(aux32, x[i].qs + 4*ib32, 2*sizeof(uint32_t));   // 8 字节读入
            const float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;  // 高 4 位粗尺度
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xxs_grid + aux8[l]);  // 逐个字节
                const uint8_t signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127];
                for (int j = 0; j < 8; ++j)
                    y[j] = db * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                y += 8;
            }
        }
    }
}
```

有两个新元素值得留意：**`ksigns_iq2xs`（128 项的符号查找表）**把每个 8 值模板的符号编码进 7 位，再用 `(aux32[1] >> 7*l) & 127` 取回来；**`kmask_iq2xs`** 则是一张 8 位掩码，逐位决定正负。这样一来，"正负号"就不必花单独的 bit 去存每一个值，而是全部合并进符号表里了。

**C#（同结构，片段）：**
```csharp
for (int ib32 = 0; ib32 < QK_K / 32; ib32++)
{
    // 该组的 8 字节：前 4 字节是 4 个 8-bit 网格索引，后 4 字节是参数
    uint a0 = (uint)(x[i].qs[ib32 * 4 + 0]) | ((uint)x[i].qs[ib32 * 4 + 1] << 16);
    uint a1 = (uint)(x[i].qs[ib32 * 4 + 2]) | ((uint)x[i].qs[ib32 * 4 + 3] << 16);
    float db = d * (0.5f + (a1 >> 28)) * 0.25f;          // 高 4 位粗尺度
    for (int l = 0; l < 4; l++)
    {
        byte gc = (byte)(a0 >> (8 * l));                 // 每字节一个 8-bit 索引
        byte sg = KSigns[(a1 >> (7 * l)) & 127];
        for (int j = 0; j < 8; j++)
            y[j] = db * Grid[gc][j] * ((sg >> j) % 2 == 1 ? -1.0f : 1.0f);
        y += 8;
    }
}
```

**Python（关键行）：**
```python
# 每 32 值组：读 8 字节参数，db 从高 4 位出
hi = struct.unpack_from('<I', qs_bytes, 4*ib32 + 4)[0]          # 组参数
db = d * (0.5 + (hi >> 28)) * 0.25
for l in range(4):
    idx  = qs_bytes[4*ib32 + l]                                  # 8-bit 网格索引
    signs = KSIGNS[(hi >> (7*l)) & 127]                          # 7-bit 符号表
    row  = [db * g * (1 if signs & (1 << j) else -1)             # 符号逐位施加
            for j, g in enumerate(_grid_signed(idx))]
    y.extend(row)
```

IQ2 的三个兄弟，在**质量/ bpw**上有着明确的阶梯，但全都共享"8 值模板 + 7 位符号表"这套心法。

### 11.4.2 IQ3_XXS：256 项 grid，每项 4 个 int8 幅度

`IQ3_XXS` 则把模板缩小到 **4 个值**，用两个 8-bit 索引去拼一个 8 值片段（每个片段 = 模板 1 的 4 个值 + 模板 2 的 4 个值）：

```
IQ3_XXS 一个 8 值片段由两个 4 值模板拼成

  IPB[0..3] = iq3xxs_grid[qs[2l+0]]    │   IPB[4..7] = iq3xxs_grid[qs[2l+1]]
  4 个 int8 幅度                        │   4 个 int8 幅度
  └──────────────┬─────────────────────┴──────────────┘
                 ▼
          db 同时缩放两组，符号各自来自 ksigns
```

它的字节布局只有两个字段：`d` + `qs[96]`（也就是 `3*QK_K/8`）。在一个 256 值块里，64 个片段索引要占去 `QK_K/4` 字节中的 3 份，剩下那 1 份的字节空间，则被 "scales_and_signs"（实际是每 32 值组一个 `uint32` 参数）拿来复用。bpw 是这样算的：

\[
\text{bpw} = \frac{98 \times 8}{256} = 3.0625
\]

> 细纲 §11.4.2 给的不等号是 3.4375 bpw——那是 **IQ3_S**（110 字节/256 值）的数值；`IQ3_XXS` 主线实际是 98 字节/3.0625 bpw。**两个格式的网格容量同是 256 项/每项 4 值，区别在尺度与符号的组织**（IQ3_S 另有独立的 `signs[32]` 和 `scales[4]`，把符号和尺度拆成显式数组）。

反量化骨架（C++，对照 `dequantize_row_iq3_xxs`）：

```cpp
void dequantize_row_iq3_xxs(const block_iq3_xxs * x, float * y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;
    uint32_t aux32;
    for (int64_t i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        const uint8_t * qs = x[i].qs;
        const uint8_t * scales_and_signs = qs + QK_K/4;   // 参数区
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            memcpy(&aux32, scales_and_signs + 4*ib32, sizeof(uint32_t));
            const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;  // 4-bit 粗尺度
            for (int l = 0; l < 4; ++l) {
                const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*l) & 127];
                const uint8_t * g1 = (const uint8_t *)(iq3xxs_grid + qs[2*l+0]);
                const uint8_t * g2 = (const uint8_t *)(iq3xxs_grid + qs[2*l+1]);
                for (int j = 0; j < 4; ++j)
                    y[j+0] = db * g1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
                for (int j = 0; j < 4; ++j)
                    y[j+4] = db * g2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
                y += 8;
            }
            qs += 8;
        }
    }
}
```

这里最值得讲的教学点是：**"256 项网格"和"每项 4 值"是一套配套的设计**——4 值模板比 8 值模板小一半，所以同样的位预算可以覆盖更多的模板；但代价是，一个片段需要两个索引，于是索引的总位宽就从 11 bit 涨到了 16 bit，质量就是这么换来的。这正好是"码本容量"和"每个模板的粒度"之间的权衡。

### 11.4.3 IQ4_NL：4-bit 非线性格点，块 32

`IQ4_NL` 是 IQ 家族里的"异类"：它**不遵守 256 值超级块**，而是回到 `Q4_0` 的 **32 值块、18 字节**布局——`d` + `qs[16]`，每个字节两个 nibble。区别只在一处：**这 16 个 nibble 不再是等间距的 -8..+7，而是映射到一张 16 项非线性格点 `kvalues_iq4nl`**：

```c
// ggml-common.h
GGML_TABLE_BEGIN(int8_t, kvalues_iq4nl, 16)
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
GGML_TABLE_END()
```

看看这张表：它不是均匀的 `-8,-7,...,7`，而是**靠近 0 的地方密集、往两端逐渐拉疏**的非线性点列。这正是 11.1.1 论证过的"低位格式需要非均匀重建点"的直接实现。

bpw：

\[
\text{bpw} = \frac{18 \times 8}{32} = 4.5
\]

它和 Q4_0 拥有相同的 bpw，但因为是格点而不是均匀码，所以**同样的 bpw 下，质量能高于 Q4_K**——这里要提醒一句：这句话成立的前提，是格点必须选对分布。解码逻辑则极短：

**C++：**
```cpp
void dequantize_row_iq4_nl(const block_iq4_nl * x, float * y, int64_t k) {
    assert(k % QK4_NL == 0);                  // QK4_NL = 32
    const int64_t nb = k / QK4_NL;
    for (int64_t i = 0; i < nb; i++) {
        const uint8_t * qs = x[i].qs;
        const float d = GGML_FP16_TO_FP32(x[i].d);
        for (int j = 0; j < QK4_NL/2; ++j) {
            y[j + 0]       = d * kvalues_iq4nl[qs[j] & 0x0F];
            y[j + QK4_NL/2] = d * kvalues_iq4nl[qs[j] >> 4];
        }
        y  += QK4_NL;
        qs += QK4_NL/2;
    }
}
```

**C#（同结构）：**
```csharp
static readonly sbyte[] KValuesIq4Nl = { -127,-104,-83,-65,-49,-35,-22,-10, 1,13,25,38,53,69,89,113 };

public static unsafe void DequantizeRowIq4Nl(BlockIq4Nl* x, float* y, int k)
{
    int nb = k / QK4_NL;
    for (int i = 0; i < nb; i++)
    {
        float d = (float)BitConverter.UInt16BitsToHalf(x[i].d);
        for (int j = 0; j < QK4_NL / 2; j++)
        {
            y[j]            = d * KValuesIq4Nl[x[i].qs[j] & 0x0F];
            y[j + QK4_NL/2] = d * KValuesIq4Nl[x[i].qs[j] >> 4];
        }
        y += QK4_NL;
    }
}
```

**Python：**
```python
KVALUES_IQ4NL = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

def dequantize_row_iq4_nl(data: bytes, k: int) -> list[float]:
    assert k % 32 == 0
    nb = k // 32
    y = []
    for i in range(nb):
        d  = struct.unpack_from('<e', data, i * 18)[0]
        qs = data[i * 18 + 2 : i * 18 + 18]
        for j in range(16):
            y.append(d * KVALUES_IQ4NL[qs[j] & 0x0F])
            y.append(d * KVALUES_IQ4NL[qs[j] >> 4])
    return y
```

到这里，IQ 家族的完整地图就可以这样收拢起来了：

```
IQ 家族地图（bpw 从低到高）

  IQ1_S    50B  /256  1.5625   8 值模板×2048 项，ql+qh，delta
  IQ2_XXS  66B  /256  2.0625   8 值模板×256 项，符号表
  IQ2_XS   74B  /256  2.3125   8 值模板×512 项
  IQ2_S    82B  /256  2.5625   8 值模板×1024 项
  IQ3_XXS  98B  /256  3.0625   4 值模板×256 项 ×2
  IQ3_S   110B  /256  3.4375   4 值模板×512 项 + 显式符号/尺度
  IQ4_NL   18B  /32   4.5      16 项非线性格点，无超级块
  IQ4_XS  136B  /256  4.25     4 值模板 + 显式 3-bit 块尺度
```

这里还有一个反直觉的点：**bpw 越低，并不代表格式越"简单"**——`IQ1_S` 的索引拼装加上 delta 修正，比 `IQ4_NL` 那种纯格点查表要复杂得多。码本容量、模板粒度、尺度表示这三者，共同决定了每个格式的工程复杂度。


## 11.5 质量与工程代价

### 11.5.1 IQ2_XXS 优于 Q2_K、IQ3_XXS 优于 Q3_K

IQ 系列存在的理由很简单：**在相同 bpw（甚至更低 bpw）下，质量比旧的 K-quants 更好**。这个论断在同类的比较中反复出现：

- `IQ2_XXS`（2.0625 bpw）的质量**优于** `Q2_K`（2.625 bpw）——位数更低，质量反而更好。
- `IQ3_XXS`（3.0625 bpw）的质量**优于** `Q3_K`（3.4375 bpw）——结论完全一样。
- `IQ4_NL`（4.5 bpw）在不少模型上优于 `Q4_K`——原因是非线性格点比均匀码字更贴合权重分布。

为什么能出现这种"降位反升质"？因为在 K-quants 里，超过一半的字节都花在了尺度/最小值这些参数上；而 IQ 把这份预算**重新分配给了码本**：模板本身已经携带了局部的形状信息，所以每个片段就可以省掉独立的子尺度。说到底，这是预算的重新分配，不是什么奇迹。

话说回来，还是要收着一点：**上面的"优于"，只是相对同一模型、同一校准数据、同一后端而言的统计规律，并不是无条件的保证**。某个具体模型的质量到底好不好，必须等实测的 PPL/KL 出来之后才能判断。这也就是为什么本书反复强调"标'约'、注明口径"。

### 11.5.2 必须提供 imatrix：量化流程变长

IQ 系列最大的工程成本，其实藏在**工具链**上：没有 imatrix 的 IQ 量化，码本选择会退化，质量显著下降。于是，量化流程变成了标准的"三步 + 一个前置"：

```
IQ 量化的完整流程

  ① 校准：   ./llama-imatrix -m model-f16.gguf -f corpus.txt -o imatrix.dat
  ② 量化：   ./llama-quantize model-f16.gguf out-iq1_s.gguf IQ1_S --imatrix imatrix.dat
  ③ 验证：   ./llama-perplexity -m out-iq1_s.gguf -f test.txt --kl-divergence-base model-f16.gguf
```

校准文本选得怎么样，会直接决定质量：单一领域的语料，会让 imatrix 只对那个领域敏感。此外，还有部分 IQ 类型（比如 `IQ1_S`、`IQ1_M`）在 `type_traits` 里是 `from_float_ref = NULL`——也就是说**它没有参考量化实现**，你没法从一个 F16 模型"原地"把它量化出来，只能靠官方的转换/量化工具链，走完整的 imatrix 流程去产出。对端侧工程师来说，"IQ 模型更小"这句话，同时意味着"制作流程更脆：少一步 imatrix，结果就不可用"。

### 11.5.3 SIMD 查表的实现难度与后端覆盖

最后一个代价，藏在性能这一侧。K-quants 的算术反量化，不过是"几条位运算 + 几条乘加"，任何后端都能照葫芦画瓢；而 IQ 的查表则要做"按索引去取寄存器或内存里的模板"，对 SIMD 指令集的依赖要强得多：

- CPU（ARM NEON / x86 AVX2）：查表可以借助 `vqtbl2q`（ARM）或者 `_mm256_shuffle_epi8`（AVX2）这类**向量查表指令**，一次处理多个值；这正是 `STRIDE` 设计要配合的对象（见 13.2.3 的三值小节，用的是同一套思路）。
- GPU：Metal / CUDA / Vulkan 对"按索引 shuffle"的支持差别很大，GPU 后端对 IQ 的支持也是**逐步合入**的——想用某个后端跑 IQ 模型之前，必须先确认那个后端确实有对应格式的内核，而不能想当然地认为"CPU 能跑 = 全平台都能跑"。

这也是为什么，很多部署项目在端侧仍然停留在 `Q4_K`/`Q5_K`：**一种格式的可用性，等于格式的字节正确性，加上目标后端的实现覆盖**。IQ 在小体积上确实有优势，但它的覆盖面和工程成本，决定了它更适合"已经选定后端、也愿意承担制作流程"的那些场景。


## 本章小结与通往 UD 的桥梁

这一章的内容，同样可以压缩成三个事实。

**第一，仿射量化在 2 bpw 以下存在结构性的失效。** 均匀网格碰到重尾分布时，要么让大量小权重塌缩在一起，要么让离群值饱和；K-quants 的层次化尺度，无非是给均匀网格换缩放，依然补不齐分布的形状。于是，IQ 转向了**码本 + 重要性**这两条路。

**第二，IQ1_S 是查表量化的最小范式。** 它占 50 字节/256 值（也就是 1.5625 bpw），核心是一张 2048 项、每项 8 个 `int8` 的 `iq1s_grid`；每个 32 值组用 `qh` 提供共享尺度 `dl`、delta 符号和高 3 位索引，而 `qs` 提供低 8 位索引；反量化全程只有"取索引 → 查模板 → 加 delta → 乘尺度"这几步。**细纲曾把它的 bpw 记为 1.5/74 字节，主线核实为 1.5625/50 字节**，本书以此为准。

**第三，IQ 的质量来自 imatrix，代价则体现在流程和后端覆盖率上。** no imatrix → 无质量；从校准、量化到验证，完整流程被拉长；SIMD 查表对不同后端的要求各不相同，GPU 的覆盖还要一步步合入。

```
从"算术反量化"到"码本查表"

  K-quants：  y = d·s·q − dmin·m   （尺度和码）
              │  乘加为主，位运算+两次乘法
  IQ 系列：   y = d·dl·(grid[idx] ± Δ)
              │  查表为主，索引+模板+一次乘法
```

下一章，我们在另一个维度上再看一次"范式切换"——主角是 `UD_IQ`。它并不发明新的格式，而是把本章讲过的 `IQ1_S`、`IQ2_XXS` 这些**已有类型，按敏感度分配到模型的不同头部**。如果说本章回答的是"一种格式如何编码"，那么第 12 章回答的则是"一个模型如何把多种格式混在一起"。