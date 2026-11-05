# 第7章 Q4_K：最常用的 K-quant

在上一章我们建立了 K-quants 的统一骨架：256 值的超级块、8 个 32 值子块、全局尺度 \(d\) 与全局最小值尺度 \(d_{\min}\)，以及一条通用反量化公式 \(y_i = d \cdot s_j \cdot q_i - d_{\min} \cdot m_j\)。这一章要给它填上最实在的血肉：**Q4_K**。

为什么说 Q4_K 是全书的主角？因为它是端侧部署的事实默认。一个 7B 模型量化成 `Q4_K_M` 后，体积从 FP16 的约 14 GB 缩到约 4 GB，而困惑度劣化通常只有约 0.8%，几乎看不出区别；反过来，同样约 4.5 bpw 的老格式 Q4_0 劣化约 1.5%。从 4.5 bit 的预算里多"抠"出质量，靠的不是魔法，而是把尺度参数组织得更聪明——这正是本章要拆开看的全部内容。

本章的目标是彻底读懂 `block_q4_K`：144 个字节怎么排、12 个字节的 6-bit 参数怎么交错打包、解码循环怎么跑、量化端怎么估计参数。读完你能亲手写出一个通过字节验证的 Q4_K 解码器，并且理解"解码端直接乘整数、归一化在量化端完成"这个最容易让人误会的实现细节。

先统一几个术语。**超级块（super-block）**：256 个权重组成的一个量化单元，K-quants 的最小存储单位。**子块（sub-block）**：超级块内部的 32 值切片，Q4_K 每超级块有 8 个。**尺度（scale）**：把量化码换算回原始幅度的乘数；Q4_K 有全局尺度 \(d\) 和逐子块尺度 \(s_j\) 两级。**最小值（min）**：子块的最小值偏移 \(m_j\)，配合全局 \(d_{\min}\) 使用，相当于第 5 章 Q4_1 的"显式零点"，但粒度更细。**bpw**：bits per weight，每个权重平均占多少比特。**归一化（normalization）**：把 0–63 的 6-bit 整数换算成浮点系数的过程。


## 7.1 字节布局：144 字节 / 256 值

### 7.1.1 `d`、`dmin`、`scales[12]`、`qs[128]`

`block_q4_K` 在 `ggml-common.h` 中定义如下（略有简化，去掉了 MSVC 兼容的 union）：

```cpp
#define QK_K 256
#define K_SCALE_SIZE 12

typedef struct {
    ggml_half d;                // 超级块尺度，FP16，2 字节
    ggml_half dmin;             // 超级块最小值尺度，FP16，2 字节
    uint8_t   scales[K_SCALE_SIZE]; // 12 字节：8 个 6-bit 尺度 + 8 个 6-bit 最小值（交错打包）
    uint8_t   qs[QK_K / 2];     // 128 字节：256 个 4-bit 码
} block_q4_K;

static_assert(sizeof(block_q4_K) == 2 + 2 + 12 + 128, "wrong q4_K block size/padding");
```

字段总和是 \(2 + 2 + 12 + 128 = 144\) 字节。用一张逐字节图看全局：

```
block_q4_K 的 144 字节布局（磁盘/内存中的顺序）

  偏移  0   1    2   3    4                            15   16                     143
        ┌───┬───┬───┬───┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──────────────────────────┐
        │ d │   │dmin│  │      scales[12]            │  │        qs[0..127]           │
        │1B │1B │1B │1B │  │ 12 字节：8 尺度 + 8 最小值│  │  128 字节：256 个 4-bit 码  │
        │  2 字节 FP16 │   │  （6-bit 交错打包）      │  │  每字节 2 个码，低/高半字节   │
        └───┴───┴───┴───┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──────────────────────────┘

  总计 = 4 + 12 + 128 = 144 字节 / 256 个权重
  有效位宽 = 144 × 8 / 256 = 4.5 bit/weight
```

**4.5 bpw 自己算一遍**：144 字节 × 8 bit = 1152 bit，除以 256 个权重，正好 4.5。这个数字和 Q4_0 的 4.5 bpw 完全相同——文件大小几乎一样，但质量差很多，原因就在第 4 章的结尾预告过：Q4_K 把单一尺度升级成了"两级尺度 + 最小值偏移"。

三种语言里，结构体声明与第 4 章的套路完全一致：

**C++（与 llama.cpp 一致）：**

```cpp
#define QK_K 256
#define K_SCALE_SIZE 12

typedef struct {
    uint16_t d;                  // FP16 位模式，2 字节
    uint16_t dmin;               // FP16 位模式，2 字节
    uint8_t  scales[K_SCALE_SIZE]; // 12 字节
    uint8_t  qs[QK_K / 2];       // 128 字节
} block_q4_K;

static_assert(sizeof(block_q4_K) == 2 + 2 + 12 + 128, "wrong q4_K block size/padding");
```

**C#（.NET 5+）：**

```csharp
using System.Runtime.InteropServices;

public static class Q4_K
{
    public const int QK_K = 256;
    public const int K_SCALE_SIZE = 12;

    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public unsafe struct BlockQ4K
    {
        public ushort d;                            // 2 字节
        public ushort dmin;                         // 2 字节
        public fixed byte scales[K_SCALE_SIZE];     // 12 字节
        public fixed byte qs[QK_K / 2];             // 128 字节
    }

    // sizeof(BlockQ4K) == 144
}
```

**Python：**

```python
from dataclasses import dataclass

QK_K = 256
K_SCALE_SIZE = 12

@dataclass
class BlockQ4K:
    d: bytes          # 2 字节 FP16 位模式
    dmin: bytes       # 2 字节 FP16 位模式
    scales: bytes     # 12 字节
    qs: bytes         # 128 字节

    def __post_init__(self):
        assert len(self.d) == 2 and len(self.dmin) == 2
        assert len(self.scales) == 12 and len(self.qs) == 128
```

与第 4 章相同：C++/C# 直接按结构体访问内存，Python 用 `bytes` 保存原始字节、用 `struct` 做 FP16 转换。区别只是字段从两个变成了四个。

### 7.1.2 6-bit 尺度与 6-bit 最小值的交错打包

这是 Q4_K 全章最容易踩坑的地方，也是网上资料最常写错的地方。我们先说结论，再拆字节。

**结论：12 个字节要装下 16 个 6-bit 整数**——8 个子块的尺度 \(s_0..s_7\) 和 8 个子块的最小值 \(m_0..m_7\)，每个都是 0–63 的 6-bit 整数。16 × 6 = 96 bit = 12 字节，一个不多一个不少。

那么问题来了：怎么把 96 bit 塞进 12 个字节？直觉上最自然的方式是把尺度连续放 6 字节、最小值连续放 6 字节（"前 6 字节存尺度，后 6 字节存最小值"）。**但这种连续打包不是 Q4_K 实际使用的布局。** 网上一些文章（包括某些相当流行的博客）就是这么描述的，但它与 llama.cpp 主线的参考代码不符。

llama.cpp 的真实布局来自 `get_scale_min_k4` 这个解包函数，它在 `ggml-quants.c` 里的原版是：

```cpp
static inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}
```

这里 `q` 指向 `scales[12]`，`j` 是子块索引（0–7），`*d` 解出第 `j` 个子块的尺度、`*m` 解出第 `j` 个子块的最小值。我们把它逐行拆开。

**第一组：`j < 4`（子块 0–3）**

- `*d = q[j] & 63`：子块 \(j\) 的尺度 = `scales[j]` 的低 6 位（bit 0–5）。
- `*m = q[j + 4] & 63`：子块 \(j\) 的最小值 = `scales[j+4]` 的低 6 位。

也就是说，对子块 0–3 而言：`scales[0..3]` 的低 6 位分别放 \(s_0..s_3\)，`scales[4..7]` 的低 6 位分别放 \(m_0..m_3\)。这时 `scales[0..3]` 和 `scales[4..7]` 各自的高 2 位（bit 6–7）还没被使用——它们留给第二组的"高位尾巴"。

**第二组：`j >= 4`（子块 4–7）**

- `*d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)`：子块 \(j\) 的尺度 = `scales[j+4]` 的低 4 位（bit 0–3）做低 4 位，**再拼接上** `scales[j-4]` 的高 2 位（bit 6–7）做高 2 位。
- `*m = (q[j+4] >> 4) | ((q[j] >> 6) << 4)`：子块 \(j\) 的最小值 = `scales[j+4]` 的高 4 位（bit 4–7）做低 4 位，**再拼接上** `scales[j]` 的高 2 位做高 2 位。

代入具体索引：对子块 4，`d = (scales[8] & 0xF) | ((scales[0] >> 6) << 4)`；对子块 5，用 `scales[9]` 和 `scales[1]`；子块 6 用 `scales[10]` 和 `scales[2]`；子块 7 用 `scales[11]` 和 `scales[3]`。最小值同理，用 `scales[8..11]` 的高半字节拼 `scales[4..7]` 的高 2 位。

把全部 12 个字节的角色画成一张图：

```
scales[12] 的 96 位（16 × 6 bit）分布

  字节: scales[0]      scales[1]      scales[2]      scales[3]
        ┌─────┬──┬──┐  ┌─────┬──┬──┐  ┌─────┬──┬──┐  ┌─────┬──┬──┐
        │s0   │s4│s5│  │s1   │s6│s7│  │s2   │  │  │  │s3   │  │  │
        │6位  │高│高│  │6位  │高│高│  │6位  │  │  │  │6位  │  │  │
        └─────┴──┴──┘  └─────┴──┴──┘  └─────┴──┴──┘  └─────┴──┴──┘
         bit0-5 │bit6-7              │              │              │
                │      ┌─────────────┘              │              │
                ▼      ▼                            ▼              ▼
  子块0尺度s0     子块4尺度的高2位              子块5尺度的高2位  子块6/7尺度高2位
  (低6位直存)   (低4位在scales[8])

  字节: scales[4]      scales[5]      scales[6]      scales[7]
        ┌─────┬──┬──┐  ┌─────┬──┬──┐  ┌─────┬──┬──┐  ┌─────┬──┬──┐
        │m0   │m4│m5│  │m1   │m6│m7│  │m2   │  │  │  │m3   │  │  │
        │6位  │高│高│  │6位  │高│高│  │6位  │  │  │  │6位  │  │  │
        └─────┴──┴──┘  └─────┴──┴──┘  └─────┴──┴──┘  └─────┴──┴──┘
         bit0-5 │bit6-7
                ▼
  子块0最小值m0    子块4/5/6/7最小值的高2位（低4位在scales[8..11]）

  字节: scales[8]      scales[9]      scales[10]     scales[11]
        ┌────┬────┐    ┌────┬────┐    ┌────┬────┐    ┌────┬────┐
        │s4  │m4  │    │s5  │m5  │    │s6  │m6  │    │s7  │m7  │
        │低4 │低4 │    │低4 │低4 │    │低4 │低4 │    │低4 │低4 │
        └────┴────┘    └────┴────┘    └────┴────┘    └────┴────┘
         bit0-3 bit4-7
```

读法要点总结成三条：

1. **`scales[0..3]` 的低 6 位**直接存子块 0–3 的尺度；**`scales[4..7]` 的低 6 位**直接存子块 0–3 的最小值。
2. **`scales[8..11]`** 每个字节一半给子块 4–7 的尺度（低半字节）、一半给子块 4–7 的最小值（高半字节）。
3. 子块 4–7 各自 6-bit 参数的**高 2 位**，分散存在 `scales[0..3]`（尺度用）和 `scales[4..7]`（最小值用）的 bit 6–7 上。

所以"12 字节不是 scales 和 mins 各占 6 字节"，而是**一个交错位流**：前 8 个字节的低 6 位装了一半的尺度一半的最小值，后 4 个字节各拆两半，高位尾巴又折回前面。正因为这种交错，跨实现读取 Q4_K 时**必须严格实现 `get_scale_min_k4`，不能凭"每 6 字节一组"的直觉去读**。这就是 7.2.1 要亲手写一遍的原因。


## 7.2 反量化代码路径

### 7.2.1 6-bit 解包函数：三种语言的 `get_scale_min_k4`

7.1.2 的规则可以用一个函数精确表达。C++ 用 llama.cpp 原版；C# 和 Python 逐位等价。

**C++（llama.cpp 原版）：**

```cpp
static inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}
```

**C#：**

```csharp
public static void GetScaleMinK4(int j, byte[] q, out byte d, out byte m)
{
    if (j < 4)
    {
        d = (byte)(q[j] & 63);
        m = (byte)(q[j + 4] & 63);
    }
    else
    {
        d = (byte)((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
        m = (byte)((q[j + 4] >> 4) | ((q[j]      >> 6) << 4));
    }
}
```

**Python：**

```python
def get_scale_min_k4(j: int, q: bytes) -> tuple[int, int]:
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    else:
        d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4)
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4)
        return d, m
```

三种实现逐位一致。需要理解的两个位操作：
- `x & 63`（即 `x & 0x3F`）：取低 6 位。因为 6-bit 无符号数的范围是 0–63。
- `(x >> 6) << 4`：取第 6、7 两个 bit，再把它们"搬"到第 4、5 位的位置——因为高 2 位要拼成整个 6-bit 参数的高位。

我们用一个具体的 12 字节样例验证。假设某超级块：

```
scales = 1a 0a 05 03  14 0c 07 04  8a 9b a7 b8   (十六进制)
```

取子块 0 和子块 4：

```
子块 0:  j=0 < 4
  d = scales[0] & 63 = 0x1a & 63 = 26
  m = scales[4] & 63 = 0x14 & 63 = 20

子块 4:  j=4 >= 4
  d = (scales[8] & 0xF) | ((scales[0] >> 6) << 4)
    = (0x8a & 0xF) | ((0x1a >> 6) << 4)
    = 0x0a | (0x00 << 4) = 10
  m = (scales[8] >> 4) | ((scales[4] >> 6) << 4)
    = (0x8a >> 4) | ((0x14 >> 6) << 4)
    = 0x08 | (0x00 << 4) = 8
```

在纸上走一遍 `get_scale_min_k4(4, scales)`，你会得到 `d=10, m=8`。这就是交错打包的直接体现：`scales[8]=0x8a` 的低半字节 0x0a 是子块 4 尺度的低 4 位，高半字节 0x08 是子块 4 最小值的低 4 位，而高 2 位（这里是 0）从 `scales[0]`、`scales[4]` 的 bit 6–7 补上。

### 7.2.2 子块尺度与偏移的合成：归一化在哪里发生

解包得到的是 0–63 的整数 \(s_j\)、\(m_j\)。现在要问一个关键问题：**把它们换算成浮点尺度，是除以 63 还是 64？**

细纲和一些早期文档写的是 \(\frac{d \cdot s_j}{64}\)、\(\frac{d_{\min} \cdot m_j}{64}\)，即解码端除以 64。**但 llama.cpp 主线的实际做法是：解码端不除以任何东西，直接乘 6-bit 整数；归一化（除以 63）在量化端就做完了。** 反量化公式因此是：

\[
y_i = d \cdot s_j \cdot q_i - d_{\min} \cdot m_j
\]

其中 \(d\)、\(d_{\min}\) 是 FP16 反解出的浮点值，\(s_j\)、\(m_j\) 是 6-bit 整数，\(q_i \in [0,15]\) 是 4-bit 码。没有 `/64`，也没有 `/63`。

为什么可以这样？因为量化端在写盘时，把 /63 的因子分别折进了 \(d\) 和 \(d_{\min}\)。看主线 `quantize_row_q4_K_ref` 的两个关键行：

```cpp
y[i].d    = GGML_FP32_TO_FP16(max_scale/63.f);
y[i].dmin = GGML_FP32_TO_FP16(max_min/63.f);
```

`max_scale` 是 8 个子块浮点尺度里的最大值，`max_min` 是最小值里的最大值。量化端把 `d = max_scale/63`、`dmin = max_min/63` 存进 FP16。然后对每个子块的浮点尺度 `scales[j]`，用 `ls = round(63 * scales[j]/max_scale)` 转成 0–63 的整数存进 `scales[j]`。

于是解码端算 `d * ls` 时：

\[
d \cdot l_s = \frac{\text{max\_scale}}{63} \cdot \mathrm{round}\!\left(63 \cdot \frac{\text{scales}[j]}{\text{max\_scale}}\right) \approx \text{scales}[j]
\]

`d · s_j` 就还原出了第 \(j\) 个子块的真实浮点尺度；`dmin · m_j` 同理还原出最小值。**解码端只做一次乘法和一次减法，归一化的除法成本全在量化端**——这是"解码快、量化慢"的典型权衡，因为推理时解码要做无数次，量化只做一次。

这里必须强调一点（呼应第 6 章 6.2.2）：**"除以 63 还是除以 64"取决于格式约定和量化端的一致选择**。主线 Q4_K 的约定是"量化端 /63、解码端直乘"。但不同文档、不同 fork 可能有不同写法（有的文档表述为解码端除 64）。因此跨实现读取时，**不要凭文档、要对照参考代码**——如果解码端和量化端的归一化口径不一致，所有重建值都会系统性偏移。我们在这一章以主线 `ggml-quants.c` 为准。

### 7.2.3 4-bit 码的解包与乘加循环

有了 `s_j`、`m_j`，剩下的工作几乎和第 4 章的 Q4_0 一样：从 `qs[128]` 里拆出 256 个 4-bit 码，套进 \(d \cdot s_j \cdot q_i - d_{\min} \cdot m_j\)。

差别在于**打包的分组方式**。回忆 Q4_0：每 32 值一个块，`qs[j]` 低半字节放第 \(j\) 个、高半字节放第 \(j+16\) 个。Q4_K 的 `qs[128]` 没有按"8 个子块各 16 字节"来分，而是**按 64 值一组**组织：每 64 值正好是两个子块，低半字节装第一个子块的 32 个码、高半字节装第二个子块的 32 个码。

主线 `dequantize_row_q4_K` 的循环：

```cpp
void dequantize_row_q4_K(const block_q4_K * x, float * y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const uint8_t * q = x[i].qs;
        const float d   = GGML_FP16_TO_FP32(x[i].d);
        const float min = GGML_FP16_TO_FP32(x[i].dmin);

        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l]  >> 4) - m2;
            q += 32; is += 2;
        }
    }
}
```

画成数据流图：

```
反量化一个 block_q4_K 的数据流

  144 字节
  ┌──────┬──────┬──────────┬──────────────┐
  │  d   │ dmin │ scales   │    qs[128]   │
  └──┬───┴──┬───┴────┬─────┴──────┬───────┘
     │      │        │           │
     ▼      ▼        ▼           ▼
   fp16→f32 fp16→f32 get_scale_min_k4(j)   逐字节拆 nibble
     │      │        │  ┌────────┴────────┐
     │      │      d·sc│ 低半字节        高半字节
     │      │        │  ▼                ▼
     │      │        │  d·sc·q - min·m   d·sc·q - min·m
     │      │        │  子块 is+0       子块 is+1
     │      │        └───────┬──────────┘
     └──────┴────────────────┴─────────► 256 个 float 输出
```

对比 Q4_0 的循环，**乘加部分完全一样**（一个乘法一个减法一个 nibble 提取），Q4_K 多出来的工作全部集中在尺度解包：每 64 值要调两次 `get_scale_min_k4`。后面 7.4.2 会看到，这点额外开销在矩阵乘面前通常可以忽略。

**Python 参考解码器 + 手工字节验证：**

```python
import struct

QK_K = 256
K_SCALE_SIZE = 12

def get_scale_min_k4(j, q):
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    else:
        d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4)
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4)
        return d, m

def dequantize_row_q4_K(data: bytes, k: int) -> list[float]:
    assert k % QK_K == 0
    nb = k // QK_K
    y = [0.0] * k

    for i in range(nb):
        base = i * 144
        d    = struct.unpack_from('<e', data, base + 0)[0]
        dmin = struct.unpack_from('<e', data, base + 2)[0]
        scales = data[base + 4 : base + 16]
        qs     = data[base + 16 : base + 144]

        pos = 0
        is_ = 0
        for j in range(0, QK_K, 64):          # 每 64 值一组 = 两个子块
            sc0, m0 = get_scale_min_k4(is_, scales)
            sc1, m1 = get_scale_min_k4(is_ + 1, scales)
            d1, m1_ = d * sc0, dmin * m0
            d2, m2_ = d * sc1, dmin * m1
            for l in range(32):
                y[i*QK_K + pos]     = d1 * (qs[l] & 0x0F) - m1_
                y[i*QK_K + pos + 32] = d2 * (qs[l] >> 4) - m2_
            pos += 64
            is_ += 2
    return y
```

现在手工构造一个最小的超级块来验证。为了让数值可心算，我们挑 \(d = 0.5\)、\(d_{\min} = 0.25\)，8 个尺度和最小值全部设成 0–63 内的整数：

```
构造一个验证块

  d     = 0.5      (FP16: 0x3800)
  dmin  = 0.25     (FP16: 0x3400)
  scales[0..3]     = 26, 10, 5, 3     → 子块 0-3 的尺度
  scales[4..7]     = 20, 12, 7, 4     → 子块 0-3 的最小值
  scales[8..11]    = 0x8a, 0x9b, 0xa7, 0xb8  → 子块 4-7 的参数（见 7.2.1 样例）
  qs[0] = 0x8A → 子块0的第0个码=低半字节 0xA=10，子块1的第0个码=高半字节 0x8=8
  其余 qs 全 0
```

子块 0 的合成参数：`d1 = d·sc0 = 0.5×26 = 13.0`，`m1 = dmin·m0 = 0.25×20 = 5.0`。第 0 个值的重建是：

\[
y_0 = 13.0 \times 10 - 5.0 = 125.0
\]

子块 1 的合成参数：`d2 = d·sc1 = 0.5×10 = 5.0`，`m2 = dmin·m1 = 0.25×12 = 3.0`。第 32 个值（即子块 1 第 0 个）的重建是：

\[
y_{32} = 5.0 \times 8 - 3.0 = 37.0
\]

用 Python 直接跑：

```python
import struct

# 手工构造 144 字节的 block_q4_K
data = bytearray(144)
struct.pack_into('<e', data, 0, 0.5)        # d
struct.pack_into('<e', data, 2, 0.25)       # dmin
scales = bytes([26, 10, 5, 3, 20, 12, 7, 4, 0x8a, 0x9b, 0xa7, 0xb8])
data[4:16] = scales
data[16] = 0x8A                              # qs[0]

y = dequantize_row_q4_K(bytes(data), 256)
print("y[0]  =", y[0],  "期望 125.0")
print("y[32] =", y[32], "期望 37.0")
assert abs(y[0] - 125.0) < 1e-6
assert abs(y[32] - 37.0) < 1e-6
print("验证通过 ✓")
```

如果验证失败，十有八九是 `get_scale_min_k4` 的索引写错——这正是 7.1.2 强调的交错打包在作祟。


## 7.3 量化端：参数估计与码选择

反量化假设 \(s_j\)、\(m_j\)、\(q_i\)、\(d\)、\(d_{\min}\) 都已经存在。给定 256 个 FP32 权重，量化端怎么决定这些参数？这一节读主线 `quantize_row_q4_K_ref` 的完整流程。

### 7.3.1 子块尺度与最小值的联合估计

量化端对每个超级块做四步。第一步：**对 8 个子块，各自估计一个浮点尺度 `scales[j]` 和一个浮点最小值 `mins[j]`**。主线用一个关键辅助函数 `make_qkx2_quants(32, 15, ...)`：

```cpp
for (int j = 0; j < QK_K/32; ++j) {
    float sum_x2 = 0;
    for (int l = 0; l < 32; ++l) sum_x2 += x[32*j + l] * x[32*j + l];
    float av_x = sqrtf(sum_x2/32);
    for (int l = 0; l < 32; ++l) weights[l] = av_x + fabsf(x[32*j + l]);
    scales[j] = make_qkx2_quants(32, 15, x + 32*j, weights, L + 32*j, &mins[j], Laux, -1.f, 0.1f, 20, false);
}
```

`make_qkx2_quants` 做的事情可以这样理解：它要在"一个尺度 + 一个最小值 + 15 个码字"的模型下，找到使加权误差最小的 `(scale, min)`。它先把最小值夹到非正区间（`if (min > 0) min = 0;`），然后做一个带权重的线性最小二乘：

\[
\min_{a,\,b} \sum_l w_l \bigl(x_l - (a\, q_l + b)\bigr)^2
\]

对每个候选尺度步长，计算 \(q_l = \mathrm{round}\!\left(\frac{x_l - \min}{\text{scale}}\right)\)，然后解出闭式最优的 `(scale, min)`——`weights[l] = av_x + |x[32j+l]|` 这个权重让误差估计偏向"幅度大的权重"（大权重对模型输出贡献大），这是第 2 章"加权最小二乘"思想的直接落地。

值得注意的是权重本身的构造：`av_x` 是这个子块权重的平均幅度，`|x|` 是每个权重自己的幅度。这样即使某个权重是离群大值，它的误差贡献也会被放大——量化器会"优先照顾好"大权重。

### 7.3.2 6-bit 量化与 d/dmin：两次 /63

第二步：**把 8 个浮点尺度和 8 个浮点最小值分别量化成 6-bit 整数**，并把 /63 归一化折进 `d`、`dmin`：

```cpp
float max_scale = 0, max_min = 0;
// ... 第一步里累计 8 个子块的最大 scale 和最大 min ...

float inv_scale = max_scale > 0 ? 63.f/max_scale : 0.f;
float inv_min   = max_min   > 0 ? 63.f/max_min   : 0.f;
for (int j = 0; j < QK_K/32; ++j) {
    uint8_t ls = nearest_int(inv_scale*scales[j]);
    uint8_t lm = nearest_int(inv_min*mins[j]);
    ls = MIN(63, ls);
    lm = MIN(63, lm);
    // ... 按 7.1.2 的交错规则写进 scales[12] ...
}
y[i].d    = GGML_FP32_TO_FP16(max_scale/63.f);
y[i].dmin = GGML_FP32_TO_FP16(max_min/63.f);
```

这里出现了**两次 /63**：一次是 `inv_scale = 63/max_scale`，让 `ls = round(63·scales[j]/max_scale)` 落在 0–63；另一次是 `y.d = max_scale/63`，把分母折进 FP16。二者相乘，解码端就只需要一次乘法。这正是 7.2.2 里"归一化在量化端完成"的出处。

至于 12 字节的写入，主线按 7.1.2 的规则：

```cpp
if (j < 4) {
    y[i].scales[j]   = ls;        // 子块 0-3 尺度低 6 位
    y[i].scales[j+4] = lm;        // 子块 0-3 最小值低 6 位
} else {
    y[i].scales[j+4] = (ls & 0xF) | ((lm & 0xF) << 4);  // 子块 4-7 参数的低 4 位
    y[i].scales[j-4] |= ((ls >> 4) << 6);               // 尺度的高 2 位
    y[i].scales[j-0] |= ((lm >> 4) << 6);               // 最小值的高 2 位
}
```

### 7.3.3 4-bit 码的计算与夹取

第三步：**用第一步估计出的 `(scale, min)` 计算每个权重的 4-bit 码**。主线把子块参数重新解包（用 `get_scale_min_k4`，和反量化端同一个函数），再算码：

```cpp
uint8_t sc, m;
for (int j = 0; j < QK_K/32; ++j) {
    get_scale_min_k4(j, y[i].scales, &sc, &m);
    const float d  = GGML_FP16_TO_FP32(y[i].d)  * sc;
    const float dm = GGML_FP16_TO_FP32(y[i].dmin) * m;
    if (!d) continue;
    for (int ii = 0; ii < 32; ++ii) {
        int l = nearest_int((x[32*j + ii] + dm)/d);
        l = MAX(0, MIN(15, l));
        L[32*j + ii] = l;
    }
}
```

这个公式和细纲的写法 \(q_i = \mathrm{clamp}(\mathrm{round}((x_i + m_j)/d_j),\,0,15)\) 一致，只是符号整理成了 `(x + dm)/d`（因为反量化是 \(d_j \cdot q - m_j\)，反解就是 \((x + m_j)/d_j\)）。注意这里 `nearest_int` 是四舍五入，然后 `MAX(0, MIN(15, ·))` 夹取到 [0,15]。

**`m_j` 估计偏差如何导致整体偏移**：如果量化端对某个子块的 `m_j` 估计偏大或偏小，那么 `(x + dm)/d` 会被整体平移，所有码字同时偏一个方向。因为 4-bit 只有 16 个码位，一个子块的 32 个权重共享同一组 `(d_j, m_j)`，任何偏移都会变成这个子块的系统性偏差——这正是第 2 章 2.2.2 讲的"尺度失配误差"在 K-quants 里的形态。好的一面是：因为子块只有 32 个值，局部偏移对整体的影响被限制在小范围内，比 Q4_0 的单一大尺度更可控。

最后一步：把 256 个码按 64 值一组打包进 `qs[128]`：

```cpp
uint8_t * q = y[i].qs;
for (int j = 0; j < QK_K; j += 64) {
    for (int l = 0; l < 32; ++l) q[l] = L[j + l] | (L[j + l + 32] << 4);
    q += 32;
}
```

`L[j+l]` 是前 32 个码（进低半字节），`L[j+l+32]` 是后 32 个码（进高半字节）——和反量化端 7.2.3 的读法一一对应。


## 7.4 实验：Q4_0 vs Q4_K

### 7.4.1 相同 bpw 下的逐块误差分布

Q4_0 和 Q4_K 都是 4.5 bpw，文件大小几乎一样，但误差结构完全不同。用一个 Python 脚本，把真实权重切片分别量化、统计每个子块的相对误差：

```python
import numpy as np
import struct

def quant_q4_0_block(v):
    amax = max(abs(x) for x in v)
    max_v = next(x for x in v if abs(x) == amax)
    d = max_v / -8.0
    id_ = 1.0 / d if d else 0.0
    codes = [min(15, int(x * id_ + 8.5)) for x in v]
    return [d * (c - 8) for c in codes]

def quant_q4_k_block(v):
    # 简化版：每 32 值子块用 (max-min)/15 定尺度、min 定最小值
    sub = np.array(v).reshape(8, 32)
    recon = np.zeros_like(sub, dtype=float)
    for j, s in enumerate(sub):
        lo, hi = s.min(), s.max()
        d_j = (hi - lo) / 15.0
        m_j = lo
        if d_j == 0: d_j = 1e-9
        codes = np.clip(np.round((s - m_j) / d_j), 0, 15).astype(int)
        recon[j] = d_j * codes + m_j
    return recon.reshape(-1)

# 构造一个"31 个小值 + 1 个大值"的异质超级块（复现第 4 章的场景，放 8 个子块）
rng = np.random.default_rng(0)
x = rng.normal(0, 0.01, 256)
x[::32] = 10.0                      # 每个子块塞一个大离群值

r0 = np.array(quant_q4_0_block(x))
r1 = quant_q4_k_block(x)

err0 = np.abs(r0 - x) / (np.abs(x) + 1e-9)
err1 = np.abs(r1 - x) / (np.abs(x) + 1e-9)

# 只统计 31 个"小值"的相对误差（去掉每个子块的大值）
mask = np.ones(256, dtype=bool); mask[::32] = False
print("Q4_0 小值平均相对误差:", err0[mask].mean().round(4))
print("Q4_K 小值平均相对误差:", err1[mask].mean().round(4))
```

典型结果是：**Q4_0 的小值误差是 Q4_K 的数倍甚至一个量级**。原因就是第 4 章构造过的场景：Q4_0 用一个尺度覆盖整个 32 值块，大值把尺度"绑架"了，小值全部塌缩到同一个码；Q4_K 每个子块有自己的 \(s_j, m_j\)，大值所在子块用小步长照顾大值，其他子块的尺度不被拖累。

画成直观的误差分布：

```
逐块相对误差（示意）

  Q4_0:                        Q4_K:
  误差                          误差
   │  ███                        │
   │  ███                        │  ▄▄▄
   │  ███  ▄▄                    │  ▄▄▄  ▄▄▄
   │  ███  ▄▄▄  ▄               │  ▄▄▄▄ ▄▄▄▄ ▄
   │  █████▄▄▄▄▄▄▄              │  ▄▄▄▄▄▄▄▄▄▄▄▄▄▄
   └──────────────────────►     └──────────────────────►
     低幅度区域 误差集中             误差在全范围更均匀
```

### 7.4.2 PPL、KL 与端侧 tok/s

用标准评测（如 `llama-perplexity` 与 F16 基座对比），7B 级模型上 `Q4_K_M` 的困惑度劣化**约 +0.8%**，而同为 4.5 bpw 的 `Q4_0` 约 **+1.5%**。这不是小数点的差距——在相同存储成本下，质量差距接近一倍。这些数字随模型、评测集、校准数据不同而波动，**不能跨来源直接比较**，但"Q4_K 显著优于 Q4_0"的结论在各类基准上稳定成立。

端侧实测吞吐方面：`Q4_K_M` 通常能达到与 `Q4_0` 几乎相同的 tok/s。原因是自回归解码是**带宽受限**的（第 17 章会详细分析 roofline）：两者都是 4.5 bpw，从内存读入的字节数相同，而 Q4_K 多出来的尺度解包（每 64 值两次 `get_scale_min_k4`）在矩阵乘面前通常被隐藏——ALU 在等内存时是空闲的，正好用来算解包。真正决定 tok/s 的是每权重字节数，而不是解码复杂度。


## 本章小结与通往 Q5_K 的桥梁

Q4_K 的全部内容可以压缩成四个事实。

**第一，它是"8 个 32 值子块 + 两级尺度"的超级块格式，144 字节/256 值，4.5 bpw。** `d`、`dmin` 各 2 字节 FP16，`scales[12]` 装 8 个 6-bit 尺度和 8 个 6-bit 最小值，`qs[128]` 装 256 个 4-bit 码。体积和 Q4_0 相同，质量却高得多。

**第二，12 字节是交错打包，不是 6+6。** `scales[0..3]` 低 6 位存子块 0–3 的尺度、`scales[4..7]` 低 6 位存子块 0–3 的最小值，子块 4–7 的参数拆成"低 4 位在 `scales[8..11]`、高 2 位折回前 8 字节"。一切以 `get_scale_min_k4` 为准——网上"连续 6 字节一组"的说法是错的。

**第三，解码端 \(y = d\cdot s_j\cdot q - d_{\min}\cdot m_j\)，没有 /63 也没有 /64；归一化在量化端完成。** 量化端把 \(d = \text{max\_scale}/63\)、\(d_{\min} = \text{max\_min}/63\) 折进 FP16，解码端直接乘 6-bit 整数。跨实现读取时必须以参考代码为准。

**第四，量化端用加权最小二乘联合估计每个子块的 `(s_j, m_j)`，再分别量化成 6-bit。** 权重 `av_x + |x|` 让误差预算向大权重倾斜；4-bit 码 `q = clamp(round((x+m_j)/d_j), 0, 15)`。子块尺度把"单一尺度被大值绑架"的失效模式限制在小范围内。

那 Q5_K 要做什么？Q4_K 的 4-bit 码只有 16 个重建点，有些对质量敏感的层（比如输出嵌入层）嫌它粗糙。Q5_K 的答案是：**保持子块尺度骨架完全不变，额外增加一个 `qh[32]` 的高位平面，给每个权重补第 5 个 bit**，把有效位宽从 4.5 推到 5.5。下一章就看这块"补丁"怎么贴上去，以及它为什么是 Q4_K 与 Q6_K 之间最常用的中间档。
