# 第11章 IQ 系列：重要性矩阵与码本量化

前四章讲完 K-quants，你已经掌握了一种把权重压到 4.5 bpw（`Q4_K`）、5.5 bpw（`Q5_K`）、乃至 2.625 bpw（`Q2_K`）的完整方案。那是一个逻辑自洽的世界：超级块、两级尺度、6-bit 解包、`y = d·s_j·q − dmin·m_j`——所有误差都被归因到"尺度和码位宽"。

但这一章要做一个**范式切换**。当目标位宽降到 2 bpw 以下时，K-quants 的统一公式开始失效：用 `y = d·s·q − m` 这套"算术"去表达一个权重，无论尺度选得多好，都受限于"均匀分布的码字"。而 `IQ1_S` 给出的答案是——**别再用公式算，直接查表**。它用 1.5625 bpw（每 256 值仅 50 字节）存储一个指向 2048 项码本（codebook）的索引，反量化时不再乘加，而是"查一个预先算好的 8 值向量"。

给你一个体量上的直觉对比：同样的权重数，`Q4_K` 需要 4.5 bpw，`IQ2_XXS` 只要 2.0625 bpw，`IQ1_S` 只要 1.5625 bpw。对一个 7B 模型，从 `Q4_K` 的约 3.8 GB 降到 `IQ1_S` 的约 1.3 GB，意味着端侧设备能多装两倍多的模型。代价是：**质量不再由"算术精度"决定，而由"码本覆盖率"和"激活重要性"决定**——这两者恰恰是本章全部新概念（grid、imatrix、delta）的来源。

本章的目标有三。第一，理解**为什么仿射量化在 2 bpw 以下失效**（11.1）。第二，掌握**imatrix 的生成与加权机制**（11.2）。第三，读透 **IQ1_S 这个最小查表格式**的字节布局与解码循环（11.3），并对 IQ2/IQ3/IQ4 家族建立一张地图（11.4）。最后在 11.5 坦诚 IQ 系列的工程代价。

先统一几个术语。**码本（codebook / grid）**：一张查找表，每一行是一个固定的小向量（比如 8 个整数），量化不再逐个估计权重，而是"选哪一行最像"。**索引（index）**：指向码本某一行的小整数，量化后真正存进文件的就是它。**重要性矩阵（imatrix）**：一个与权重同形状的浮点数组，每个位置记录"这个权重对模型输出有多重要"，量化时把误差预算向重要位置倾斜。**加权搜索**：量化时极小化的目标从"均匀的均方误差"变成"按重要性加权的误差"。


## 11.1 范式切换：为什么要码本

### 11.1.1 仿射量化在 2 bpw 以下的失效模式

先复习一下前面几章反复使用的仿射量化。给一个权重 \(x\)，量化成 \(q\) 位整数码：

\[
q = \mathrm{clamp}(\mathrm{round}((x + m)/d),\ 0,\ 2^q - 1), \qquad \hat{x} = d\cdot q - m
\]

反量化结果是 \(2^q\) 个**等间距**的离散点：尺度 \(d\) 决定间距，偏移 \(m\) 决定起点。对 4-bit（16 个点）、6-bit（64 个点）来说，这个均匀网格足够贴近权重分布。但当码位宽降到 2–3 bit 时，问题暴露出来：

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

用数字说话。假设一组服从典型权重分布（大量接近零、少数大绝对值）的值 `[0.001×30 个, 0.05×20 个, 0.9, -0.7]`，强行压到 2-bit 均匀量化：

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

输出会显示 MSE 很大，而且 52 个权重被压缩成了极少数几个重建值——它们的差异在均匀网格下完全不可分辨。这正是"仿射量化在低位宽下失效"的直观形态。

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

问题根源在于**码字的排布方式**：均匀量化把 \(2^q\) 个点等间距铺开，而真实权重的密度并不均匀。就算把 K-quants 的层次化尺度搬进来，本质仍是在给"均匀网格"不同的缩放——当只有 2 bit 时，网格太稀，无论缩放得多精细都补不齐分布的形状。**结论是结构性而非参数性的：低位宽需要非均匀的重建点**。

### 11.1.2 码本 + 重要性：两个工具突破瓶颈

IQ 系类针对这两个失效点给出两个工具，正好对应本章后半的两个主角：

1. **非线性码本（grid）**：不把重建点固定在 `d × q − m` 的等差数列，而是预先把一组"和真实权重分布更像"的小向量存成表。反量化变成"按索引取出一整行"，这一行内的 8 个值可以是非均匀的、带符号的、甚至相互关联的——均匀量化做不到的事，查表可以。
2. **重要性矩阵（imatrix）**：量化时不再均匀地最小化 MSE，而是按"每个权重对模型输出的影响"分配误差预算。误差落在不重要的权重上，只会造成一点噪声；落到重要的权重上，会直接改变输出的 logits 分布。

这两件事可以组合成一个目标函数：极小化按重要性加权的重建误差

\[
\min \sum_i \alpha_i (x_i - \hat{x}_i)^2
\]

其中 \(\alpha_i\) 是 imatrix 提供的权重。省略号之下藏着一个工程现实：**没有 imatrix 的 IQ 量化，质量和随机取码没有本质区别**。所以 11.2 先讲 imatrix 怎么来。


## 11.2 imatrix：重要性矩阵的生成与使用

### 11.2.1 `llama-imatrix` 的校准流程与命令

重要性矩阵不是从模型文件里读出来的，而是**在量子化之前，让模型跑一遍文本、统计每个权重被激活得多厉害**得到的。这个"校准"过程由 `llama-imatrix` 完成：

```bash
# 用一个 F16 模型 + 一段校准文本，生成 imatrix 文件
./llama-imatrix -m model-f16.gguf -f calibration-data.txt -ngl 99 -o imatrix.dat
```

- `-m model-f16.gguf`：校准用的基准模型，必须是未量化的 F16（量化过的模型跑出来的激活统计没有意义）。
- `-f calibration-data.txt`：校准文本。质量要求是"和真实使用场景分布接近"——通用的 wiki 文本、多语种文本、代码混合都比单一领域文本更稳健。实践常用合并后的英文维基或 `wikitext` 类语料。
- `-ngl 99`：把尽可能多的层 offload 到 GPU，加快校准。
- `-o imatrix.dat`：输出的 imatrix 文件名。
- 可调参数：`-c <context>` 上下文长度（如 2048/4096），`--chunks <n>` 处理多少个校准块。

校准的原理是：前向传播时，统计每个权重位置上**激活值（激活的幅度）的平方或加权量**，最终输出一张与每个张量同形状的浮点矩阵。这张矩阵在下一步量化时被读入，告诉量化器"哪些位置的误差不可接受"。

### 11.2.2 加权目标与 `vec_dot` 中的 imatrix 索引

在 `llama-quantize` 内部，imatrix 不是作为"元数据"参与，而是作为**加权向量**直接进入量化例程。以 `quantize_q4_K` 为例，它的签名里有一个 `quant_weights` 指针：

```c
size_t quantize_q4_K(const float * src, void * dst, int64_t nrow, int64_t n_per_row,
                     const float * quant_weights);
```

调用方传给它的正是"imatrix 展开后的一行"：`quant_weights + row * n_per_row`，其中 `row` 是当前处理的张量行号，`n_per_row` 是每行元素数。量化器在估计子块尺度、选择码字时，给误差乘上这个权重：

```c
// quantize_row_q4_K_impl 中的加权误差估计示意
const float * qw = quant_weights + QK_K*i + 32*j;          // 当前 32 值子块的权重
for (int l = 0; l < 32; ++l)
    weights[l] = qw[l] * sqrtf(sigma2 + x[32*j + l]*x[32*j + l]);
```

注意这里的位置关系：imatrix 是**逐权重**对齐的（`i*QK_K + l` 就是该权重自己的重要性），所以误差加权发生在矩阵乘层面，而不是张量层面。这也解释了为什么校准阶段的文本选择如此关键——imatrix 记录的"重要位置"取决于校准数据见过什么。

对于 IQ 系列，imatrix 的存在是**硬性要求**而非可选项。一个没有 imatrix 的 IQ1_S 模型，码本选择退化成对未加权误差的贪心近邻，质量会显著劣化。这一点在 11.5.2 会展开。


## 11.3 IQ1_S：1.5625 bpw 的查表反量化

> 说明：细纲 §11.3 曾把 IQ1_S 记为"2048 项 grid / 74 字节 / 1.5 bpw"。按 llama.cpp 主线源码（`asset/facts-verified.md`）核实，实际是 **50 字节 / 256 值 = 1.5625 bpw**。后面所有数字以此为准。

### 11.3.1 2048 项 grid：每项编码 8 个值

IQ1_S 的核心是一张 2048 项的静态查找表 `iq1s_grid`。表里每一项是一个 `uint64_t`，把它当 8 个 `int8_t` 字节展开，就得到一个 8 值向量——每个值都很小（通常落在 \(\{-1,0,+1\}\) 附近，偶尔有更大的）：

```c
// ggml-common.h（片段，完整 2048 项）
GGML_TABLE_BEGIN(uint64_t, iq1s_grid, NGRID_IQ1S)   // NGRID_IQ1S = 2048
    0xffffffffffffffff, 0xffffffffffffff01, 0xffffffffffff0000, 0xffffffffffff01ff,
    // ...
GGML_TABLE_END()
```

第一项 `0xffffffffffffffff` 展开成 8 个 `-1`（所有字节都是 `0xFF`）；第二项 `0xffffffffffffff01` 展开成 `[1, -1, -1, -1, -1, -1, -1, -1]`（低字节 `0x01` 是 `1`）。这些小向量在量化时充当"模板"：要把 8 个权重编码成 IQ1_S，就是在 2048 个模板里找一个"缩放后最像"的。

为什么是 8 个值一组？因为 8 值模板可以用一个 `uint64` 存下（天然对齐、SIMD 友好），而且 8 是这个码本的"原子"：整块 256 值被切成 32 个 8 值组，每组靠一个 8-bit 索引定位到 2048 项网格里的某一项。8-bit 只有 256 种可能，所以索引还需要补 3 bit——这 3 bit 从 `qh` 里拿，凑成 11 位覆盖 2048。这个"低 8 位 + 高 3 位"的拆分是理解字节布局的关键。

### 11.3.2 尺度与 delta：\(dl\cdot(\mathrm{grid}[j] \pm \Delta)\)

Q4_K 的反量化是 `y = d·sc·q − min·m`：全局尺度 `d`、子块尺度 `sc`、最小值 `m`，一套算术下来。IQ1_S 不一样：它**没有每个 32 值子块独立的"尺度+最小值"两级参数**，而是用一个 32 值组共享的粗尺度，再加一个恒定的小修正 `delta`。

具体地，每个 32 值组（`QK_K/32 = 8` 组）拿 `qh[ib]`（一个 `uint16_t`）当"参数小仓库"：

- 高 3 位（bit 12–14）决定该组尺度 \(k \in [0,7]\)，有效尺度为
  \[
  dl = d \cdot (2k + 1)
  \]
  所以块尺度实际上是翻了奇数倍 \(d、3d、5d、\dots、15d\)，而不是任意缩放。这刻意限制了尺度分辨率——省更多 bit。
- bit 15 决定一个全局符号修正 `delta`
  \[
  \Delta = \begin{cases} -0.125 & \text{if bit15 = 1} \\ +0.125 & \text{if bit15 = 0} \end{cases}
  \]
  这个 `IQ1S_DELTA = 0.125` 的意义是：码本模板里的值大多是 `±1、0`，直接乘尺度会产生"某个权重恰好是 0"的可能；加一个固定 `delta` 后，重建值永远不落在精确的 0 上，让反量化输出对下游更平滑。
- 低 12 位（bit 0–11）分成 4 组，每组 3 bit，作为 4 个 8 值片段的索引高 3 位。

于是第 `ib` 组内第 `l` 个 8 值片段的完整反量化公式是：

\[
y_j = dl \cdot \bigl(\mathrm{grid}[ \ \mathrm{idx}_l \ ]_j + \Delta \bigr),
\qquad
\mathrm{idx}_l = qs[4\cdot ib + l] \ \bigl|\  (((qh[ib] \gg 3l) \& 7) \ll 8)
\]

`idx_l` 的低 8 位来自 `qs`（每个片段一个字节），高 3 位来自 `qh`；两者拼成 11-bit 索引查表。整个结构可以画成：

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

`block_iq1_s` 在内存中占 50 字节，静态断言锁定：

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

字节数与 bpw 的计算：

\[
\text{bpw} = \frac{50 \times 8}{256} = 1.5625 \ \text{bit/weight}
\]

161 值块的 8 个 32 值组共用同一个 `d`，组内的 `qh` 提供尺度、delta 和高 3 位。反量化主循环（C++，对齐 `ggml-quants.c` 的 `dequantize_row_iq1_s`）：

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

这里有三个容易看岔的点，逐个说明：

1. **`qs` 的推进是每 32 值组前进 4 字节**，不是每片段一步——因为 `qs[l]` 用的是相对当前组的 4 字节窗口。
2. **索引是 11-bit**：`qs[l]`（8 位）是低半，`((qh[ib] >> 3*l) & 7) << 8`（3 位）是高半。不是简单的"每对字节一个索引"。
3. **整个 `qh[ib]` 的 16 位被拆成三份用途**：尺度、delta、索引高 3 位。这是"用 32 值组共享参数换更低位宽"的直接体现。

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

**手工字节验证。** 构造成一个 50 字节的超级块：`d = 1.0`（FP16 位模式 `0x3C00`，小端字节 `00 3C`），`qs` 全部为 0（除 `qs[0] = 1`），`qh` 全部为 0：

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

这个测试的价值在于：**它把"字节 → 索引 → 模板 → 尺度"整条链路钉死在一个可人工验算的结果上**。凡是改过 `qh` 位序、索引拼装或 delta 符号的人，都会在这个断言上立刻翻车——这比任何理论讨论都更能抓住 3 细节错误。

与 K-quants 的算术循环相比，IQ1_S 的主要开销不再是乘加，而是**索引提取与查表**。`grid[j]` 是一次内存/寄存器查表的结果，这既快又慢，取决于后端是否支持高效的向量查表指令（11.5.3 细说）。


## 11.4 IQ2/IQ3：不同码本容量的谱系

IQ 系列共用"码本 + 索引"的心法，但不同成员在码本容量、每个模板编码几个值、符号与尺度怎么存上有系统性差异。这一节给一个家族地图，不逐字节展开，但把每个格式的**bpw 和它靠什么换质量**讲清楚。

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

bpw 计算举例（IQ2_XXS）：

\[
\text{bpw} = \frac{66 \times 8}{256} = 2.0625
\]

以 `IQ2_XXS` 为例（其余两个本质相同、只是参数移位），它的解码片段是：

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

两个新元素值得注意：**`ksigns_iq2xs`（128 项符号查找表）**把每个 8 值模板的符号编码进 7 位，用 `(aux32[1] >> 7*l) & 127` 取回；**`kmask_iq2xs`** 是 8 位掩码，逐位决定正负。于是"正负号"不再花单独的 bit 存每一个值，而是合併进符号表。

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

IQ2 的三个兄弟在**质量/ bpw**上有明确阶梯，但都共享"8 值模板 + 7 位符号表"的心法。

### 11.4.2 IQ3_XXS：256 项 grid，每项 4 个 int8 幅度

`IQ3_XXS` 把模板缩到 **4 个值**，用两个 8-bit 索引拼一个 8 值片段（每片段 = 模板 1 的 4 值 + 模板 2 的 4 值）：

```
IQ3_XXS 一个 8 值片段由两个 4 值模板拼成

  IPB[0..3] = iq3xxs_grid[qs[2l+0]]    │   IPB[4..7] = iq3xxs_grid[qs[2l+1]]
  4 个 int8 幅度                        │   4 个 int8 幅度
  └──────────────┬─────────────────────┴──────────────┘
                 ▼
          db 同时缩放两组，符号各自来自 ksigns
```

字节布局只有两个字段：`d` + `qs[96]`（`3*QK_K/8`）。256 值块里，64 个片段索引占了 `QK_K/4` 字节的 3 份，剩下 1 份的字节空间被 "scales_and_signs"（实际是每 32 值组一个 `uint32` 参数）复用。bpw 计算：

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

这里的关键教学点：**"256 项网格"和"每项 4 值"是配套的设计**——4 值模板比 8 值模板小一半，所以同样的位预算能覆盖更多模板；但片段需要两个索引，于是索引总位宽从 11 bit 涨到 16 bit，质量换来了。这正是"码本容量和每个模板的粒度"之间的权衡。

### 11.4.3 IQ4_NL：4-bit 非线性格点，块 32

`IQ4_NL` 是 IQ 家族里的"异类"：它**不遵守 256 值超级块**，而是回到 `Q4_0` 的 **32 值块、18 字节**布局——`d` + `qs[16]`，每个字节两个 nibble。区别只在一处：**这 16 个 nibble 不再是等间距的 -8..+7，而是映射到一张 16 项非线性格点 `kvalues_iq4nl`**：

```c
// ggml-common.h
GGML_TABLE_BEGIN(int8_t, kvalues_iq4nl, 16)
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
GGML_TABLE_END()
```

看这张表：它不是均匀的 `-8,-7,...,7`，而是**靠近 0 密集、往两端拉疏**的非线性点列。这正是 11.1.1 论证过的"低位格式需要非均匀重建点"的直接实现。

bpw：

\[
\text{bpw} = \frac{18 \times 8}{32} = 4.5
\]

与 Q4_0 同 bpw，但因为是格点而不是均匀码，**同 bpw 下质量高于 Q4_K**——注意这句话成立的前提是格点要选对分布。解码极短：

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

至此 IQ 家族地图可以这样收拢：

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

注意一个反直觉点：**bpw 越低不代表越"简单"**——`IQ1_S` 的索引拼装和 delta 修正，比 `IQ4_NL` 的纯格点查表要复杂得多。码本容量、模板粒度、尺度表示三者共同决定了每个格式的工程复杂度。


## 11.5 质量与工程代价

### 11.5.1 IQ2_XXS 优于 Q2_K、IQ3_XXS 优于 Q3_K

IQ 系列存在的理由，就是"同 bpw（甚至更低的 bpw）下，质量比旧的 K-quants 更好"。这个论断在同类比较中反复出现：

- `IQ2_XXS`（2.0625 bpw）的质量**优于** `Q2_K`（2.625 bpw）——更低位数、更好质量。
- `IQ3_XXS`（3.0625 bpw）的质量**优于** `Q3_K`（3.4375 bpw）——同样的结论。
- `IQ4_NL`（4.5 bpw）在不少模型上优于 `Q4_K`——因为非线性格点比均匀码字更匹配权重分布。

为什么能有这种"降位反升质"？因为在 K-quants 里，超过一半的字节花在尺度/最小值参数上；而 IQ 把这份预算**重新分配给了码本**：模板本身携带了局部形状信息，所以每个片段可以省掉独立的子尺度。这是预算的重分配，不是奇迹。

话要收着说：**"优于"是相对同模型、同校准数据、同后端而言的统计规律，不是无条件保证**。具体模型的质量好坏，必须在实测 PPL/KL 出来后判断。这也是为什么本书反复强调"标'约'、注明口径"。

### 11.5.2 必须提供 imatrix：量化流程变长

IQ 系列最大的人工作成本在**工具链**：没有 imatrix 的 IQ 量化，码本选择退化，质量显著下降。于是量化流程变成了标准的"三步 + 一个前置"：

```
IQ 量化的完整流程

  ① 校准：   ./llama-imatrix -m model-f16.gguf -f corpus.txt -o imatrix.dat
  ② 量化：   ./llama-quantize model-f16.gguf out-iq1_s.gguf IQ1_S --imatrix imatrix.dat
  ③ 验证：   ./llama-perplexity -m out-iq1_s.gguf -f test.txt --kl-divergence-base model-f16.gguf
```

校准文本的选择直接决定质量：单一领域的语料会让 imatrix 只对那个领域敏感。此外，部分 IQ 类型（如 `IQ1_S`、`IQ1_M`）在 `type_traits` 中 `from_float_ref = NULL`——**没有参考量化实现**，意味着你不能从一个 F16 模型"原地"量化出它，只能靠官方转换/量化工具链经过 imatrix 流程产出。对端侧工程师来说，"IQ 模型更小"同时意味着"制作流程更脆：少一步 imatrix，结果就不可用"。

### 11.5.3 SIMD 查表的实现难度与后端覆盖

最后一个代价藏在性能侧。K-quants 的算术反量化是"几条位运算 + 几条乘加"，任何后端都能照葫芦画瓢；IQ 的查表则是"按索引取寄存器/存内存中的模板"，对 SIMD 指令集有更强的依赖：

- CPU（ARM NEON / x86 AVX2）：查表可以用 `vqtbl2q`（ARM）或 `_mm256_shuffle_epi8`（AVX2）这类**向量查表指令**一次处理多个值，这正是 `STRIDE` 设计要配合的（见第 13.2.3 的三值小节，同一套思路）。
- GPU：Metal / CUDA / Vulkan 对"按索引 shuffle"的支持差别很大，GPU 后端对 IQ 的支持是**逐步合入**的——你用某个后端跑 IQ 模型之前，必须先确认那个后端有对应格式的内核，而不是想当然认为"CPU 能跑 = 全平台能跑"。

这也是为什么很多部署项目在端侧仍停留在 `Q4_K`/`Q5_K`：**格式的可用性 = 格式的字节正确性 + 目标后端的实现覆盖**。IQ 在小体积上有优势，但它的覆盖面和工程成本决定了它更适合"已经选定后端、愿意承担制作流程"的场景。


## 本章小结与通往 UD 的桥梁

这一章可以压缩成三个事实。

**第一，仿射量化在 2 bpw 以下有结构性失效。** 均匀网格在重尾分布上要么让大量小权重塌缩，要么让离群值饱和；K-quants 的层次化尺度只是给均匀网格换缩放，补不齐分布的形状。于是 IQ 转向**码本 + 重要性**两条路。

**第二，IQ1_S 是查表量化的最小范式。** 50 字节/256 值（1.5625 bpw），核心是 2048 项、每项 8 个 `int8` 的 `iq1s_grid`；每个 32 值组用 `qh` 提供共享尺度 `dl`、delta 符号和高 3 位索引，`qs` 提供低 8 位索引；反量化只有"取索引 → 查模板 → 加 delta → 乘尺度"。**细纲把它的 bpw 记为 1.5/74 字节，主线核实为 1.5625/50 字节**，本书以此为准。

**第三，IQ 的质量来自 imatrix，代价是流程与后端覆盖率。** no imatrix → 无质量；从校准、量化到验证的完整流程变长；SIMD 查表对不同后端的要求不同，GPU 覆盖逐步合入。

```
从"算术反量化"到"码本查表"

  K-quants：  y = d·s·q − dmin·m   （尺度和码）
              │  乘加为主，位运算+两次乘法
  IQ 系列：   y = d·dl·(grid[idx] ± Δ)
              │  查表为主，索引+模板+一次乘法
```

下一章我们看另一个维度上的"范式切换"——`UD_IQ`。它不发明新格式，而是把本章的 `IQ1_S`、`IQ2_XXS` 这些**已有类型，按敏感度分配到模型的不同头部**。如果说本章回答的是"一个格式如何编码"，第 12 章回答的是"一个模型如何把多种格式混在一起"。