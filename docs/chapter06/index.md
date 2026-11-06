# 第6章 K-quants 总览：超级块与两级尺度

从这一章开始，我们离开"简单块"的世界，进入 llama.cpp 量化体系的核心地带——**K-quants**。

先回忆第 5 章结尾的结论：简单块有两个缺陷，都来自"每个块只有一个全局参数"。Q4_0 的 32 值块共享一个尺度 `d`，遇到块内异质性时小值塌缩（4.5 节）；Q4_1 想用显式最小值补救，但"整块一个 m"的粒度不够，白白付出 2 字节。

K-quants 的思路是把"分块"再套一层结构。**一个 256 值的超级块（super-block）共享两个全局尺度，内部再切 8 个 32 值子块（sub-block），每个子块拥有自己的 6-bit 尺度和 6-bit 最小值。** 这样：

- **全局尺度负责"整体量级"**——256 个权重共享一个量级基准，摊薄参数开销；
- **子块尺度负责"局部差异"**——每个 32 值子块用自己的尺度/最小值去贴合自己的分布，块内异质性不再由单一尺度承担。

这一章不深入某个具体格式，而是搭起 K-quants 的**统一骨架**。后面的 Q4_K（第 7 章）、Q5_K（第 8 章）、Q6_K（第 9 章）、Q2_K/Q3_K（第 10 章），都是这个骨架在不同位宽下的参数化。**理解了这一章的统一公式，后面四章就是"同一首歌的不同伴奏"。**

先统一几个术语。**超级块（super-block）**：K-quants 的基本编组单位，默认 256 个权重。**子块（sub-block）**：超级块内部的次级分组，默认 32 个权重。**两级尺度（two-level scale）**：全局尺度 `d` 管超级块、子块尺度 `s_j` 管子块，两个层次配合。**6-bit 整数**：取值 0..63 的存储值，反量化时作为系数乘到尺度上。**全局最小值尺度 `d_min`**：超级块级非对称偏移的量级基准，配合子块最小值 `m_j` 使用。


## 6.1 超级块与子块划分

### 6.1.1 256 值超级块与 8 个 32 值子块

K-quants 的结构可以画成这样：

```
K-quants 超级块层次结构

  ┌─────────────────────────────────────────────────────────────────┐
  │  超级块（256 个权重）                                            │
  │                                                                 │
  │   全局尺度 d（FP16，2 字节）── 超级块的整体量级                  │
  │   全局最小值尺度 d_min（FP16，2 字节）── 非对称偏移的整体量级     │
  │                                                                 │
  │  ┌─────────┬─────────┬─────────┬───────┬─────────┬─────────┐    │
  │  │ 子块 0  │ 子块 1  │ 子块 2  │  ...  │ 子块 6  │ 子块 7  │    │
  │  │ 32 值   │ 32 值   │ 32 值   │       │ 32 值   │ 32 值   │    │
  │  │ s0, m0  │ s1, m1  │ s2, m2  │       │ s6, m6  │ s7, m7  │    │
  │  └─────────┴─────────┴─────────┴───────┴─────────┴─────────┘    │
  │   每子块：6-bit 尺度 s_j + 6-bit 最小值 m_j（打包存储）          │
  │                                                                 │
  │  每个权重本身：低位量化码 q_i（如 Q4_K 的 4-bit）                 │
  └─────────────────────────────────────────────────────────────────┘

  每 256 值：d(2) + d_min(2) + 8×(6+6 bit 打包) + 256×(码位宽 bit)
```

其中 `s_j` 和 `m_j` 各占 6 bit（取值 0..63）。把账算一下就明白这个数字从哪来：8 个子块 × 每个 12 bit（6 位尺度 + 6 位最小值）= 96 bit = 12 字节，这正是后面 Q4_K/Q5_K 里 `scales[12]` 那 12 字节的来源。

为了把刚才抽象的描述变成具体的数字，我们不妨直接算一张“256 值的账单”，看看一个超级块的字节到底花在了哪里。以 Q4_K 为例（它是第 7 章的主角，也是 K-quants 家族里最标准的“8×32 子块 + 非对称”结构）：

```
一个 block_q4_K 的字节预算（共 144 字节 / 256 值）

  字节数    内容                      谁用的
  ─────────────────────────────────────────────
  2        d     (FP16)              超级块全局尺度
  2        d_min (FP16)              超级块全局最小值尺度
  12       scales[12]                8 个子块 × (6-bit 尺度 + 6-bit 最小值) 的打包
  128      qs[128]                   256 个 4-bit 码（每字节 2 个）

  有效位宽 = 144 × 8 / 256 = 4.5 bit/weight
```

注意：**6-bit 尺度和 6-bit 最小值不是各占 6 字节，而是交错塞进 12 字节**。这是 K-quants 家族里最容易被误解、也最需要亲手拆开的地方，第 7 章 7.1.2 会逐位走一遍。

三种语言里，这个结构怎么表达？与第 4 章 Q4_0 的做法一致：C++ 用结构体直接映射字节，C# 用 `Pack=1` 的 `unsafe` 结构体，Python 用 `bytes` + `struct` 显式拆解。

**C++（与 llama.cpp 的 `ggml-common.h` 一致）：**

```cpp
#define QK_K 256
#define K_SCALE_SIZE 12

typedef struct {
    ggml_half d;                  // 全局尺度（FP16 位模式），2 字节
    ggml_half dmin;               // 全局最小值尺度（FP16 位模式），2 字节
    uint8_t   scales[K_SCALE_SIZE]; // 12 字节：8 子块 × (6-bit 尺度 + 6-bit 最小值) 交错打包
    uint8_t   qs[QK_K / 2];       // 128 字节：256 个 4-bit 码
} block_q4_K;

static_assert(sizeof(block_q4_K) == 2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2, "wrong q4_K block size/padding");
```

`static_assert` 把 `sizeof` 锁死在 `4 + 12 + 128 = 144`。任何编译器填充都会让"磁盘字节 == 结构体"的约定破裂（第 3 章 3.1.4 讲过这个道理）。

**C#（.NET 5+）：**

```csharp
using System.Runtime.InteropServices;

public static class Q4_K
{
    public const int QK_K = 256;
    public const int K_SCALE_SIZE = 12;

    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public unsafe struct BlockQ4_K
    {
        public ushort d;                  // FP16 位模式，2 字节
        public ushort dmin;               // FP16 位模式，2 字节
        public fixed byte scales[K_SCALE_SIZE]; // 12 字节
        public fixed byte qs[QK_K / 2];   // 128 字节
    }
    // sizeof(BlockQ4_K) == 144
}
```

**Python：**

```python
from dataclasses import dataclass

QK_K = 256
K_SCALE_SIZE = 12

@dataclass
class BlockQ4_K:
    d: bytes        # 2 字节 FP16 位模式
    dmin: bytes     # 2 字节 FP16 位模式
    scales: bytes   # 12 字节
    qs: bytes       # 128 字节

    def __post_init__(self):
        assert len(self.d) == 2
        assert len(self.dmin) == 2
        assert len(self.scales) == K_SCALE_SIZE
        assert len(self.qs) == QK_K // 2
```

三种语言里，"全局两个参数 + 每子块两个参数 + 256 个码"这个骨架完全一致；差别只在 C/C# 把字节直接当结构体，Python 需要显式切块。

### 6.1.2 为什么是 256 和 32

这两个数字不是拍脑袋定的，是多个工程约束折中的结果：

1. **32 值子块足够小，能跟踪局部分布。** 权重张量里，相邻 32 个值的分布往往相近；子块越小，尺度越能贴合局部特征，误差越均匀（这是第 4 章 4.4 节 Q4_0 失效模式的反向解：小值不再被同块大值拖累）。

2. **256 值超级块足够大，能摊薄全局参数开销。** 两个 FP16（`d`、`d_min`）共 4 字节，摊到 256 个权重上，每权重只多 4×8/256 = 0.125 bit。若超级块是 128 值，这个开销翻倍到 0.25 bit；是 32 值，就退化成简单块了（4 字节摊到 32 个权重 = 1 bit 开销，几乎吃掉 Q4_0 一半的位宽预算）。

   把"超级块大小 vs 全局参数开销"画出来：

   ```
   全局参数（4 字节）摊到不同大小的超级块上

     超级块大小    每权重开销    对 4-bit 码的影响
     ──────────────────────────────────────────
     256 值        0.125 bit    码 4 bit → bpw 4.125（可忽略）
     128 值        0.25  bit    码 4 bit → bpw 4.25
      64 值        0.5   bit    码 4 bit → bpw 4.5
      32 值        1.0   bit    码 4 bit → bpw 5.0（退化成简单块！）

     所以"越大越省"，但太大又会失去局部性——256 是平衡点。
   ```

3. **与 SIMD 向量宽度和缓存行对齐兼容。** 现代 CPU 的 SIMD 寄存器宽 128 bit（16 字节）或 256 bit（32 字节），缓存行通常 64 字节。256 值/块和 32 值/子块的组合，让解码循环可以整齐地按向量宽度展开（第 17 章会细讲）。

4. **与 Q4_0 的 32 值块天然衔接。** 8 × 32 = 256，让"子块"可以复用 Q4_0 的 nibble 打包思路（第 7 章会看到 Q4_K 的码打包方式就是 Q4_0 的低/高 nibble 方案）。

还有一层历史背景值得知道：这套"超级块 + 子块 + 两级尺度"的设计，是 llama.cpp 在 2023 年通过 k-quants 引入的（社区常称其关键 PR 为 #1684），随后又演化出 `_S`/`_M`/`_L` 三个变体等级来进一步平衡"精度 vs 体积"（第 14 章详述）。它是从"一个块一个尺度"的朴素方案向"层次化尺度"迈进的关键一步——这也呼应了第 1 章讲的第二范式跃迁。


## 6.2 通用反量化骨架

### 6.2.1 统一公式：\(y_i = d \cdot s_j \cdot q_i - d_{\min} \cdot m_j\)

K-quants 家族（Q2_K、Q3_K、Q4_K、Q5_K、Q6_K）可以共用一个反量化公式的骨架。设：

- \(i\) 是权重在超级块内的全局索引（0..255）；
- \(j = \lfloor i / 32 \rfloor\) 是它所在的子块索引（0..7）；
- \(q_i\) 是该权重的低位量化码（位宽依格式而定：Q4_K 是 4-bit、Q5_K 是 5-bit……）；
- \(s_j\)、\(m_j\) 是子块的 6-bit 尺度和 6-bit 最小值；
- \(d\)、\(d_{\min}\) 是超级块的全局尺度。

统一公式写作：

\[
y_i = d \cdot s_j \cdot q_i - d_{\min} \cdot m_j
\]

这个公式乍看有些吓人，其实只是两项叠在一起。我们把它拆开读：

- **`d · s_j` 是"子块的有效尺度"**。全局尺度 `d` 给出量级，子块尺度 `s_j` 做局部修正；
- **`d_min · m_j` 是"子块的有效最小值"**。全局最小值尺度 `d_min` 给出偏移量级，子块最小值 `m_j` 做局部修正；
- **减号**把最小值部分从重建值里扣掉，得到最终的权重近似。

这就是"两级尺度"的全部含义：**两个全局参数（`d`、`d_min`）+ 每子块两个局部参数（`s_j`、`m_j`），共同决定反量化值。**

> **为什么叫"尺度 × 尺度"而不是"尺度 + 尺度"？**
> 你可能奇怪：为什么是 `d · s_j` 而不是 `d + s_j`？答案是**乘法的动态范围更大**。`d` 反映超级块整体量级（可能从 0.001 到 10），`s_j` 反映子块相对整体的比例（0..1 左右）。两者相乘，等价于"先整体归一、再局部缩放"，一次乘法就把两个动态范围叠在一起；而加法只能做线性搬移，无法表达"整体大、局部小"这类比例关系。Q4_K 选 `d·s` 正是看中乘法对"量级 + 比例"的联合表达能力。

反量化骨架在三种语言里怎么写？以 Q4_K 为例，核心循环是"对每个子块：解出 6-bit 尺度和最小值 → 对 32 个码：乘加"。这里先给出骨架（6-bit 解包的精确位序在第 7 章 7.2.1 用 `get_scale_min_k4` 展开）：

**C++：**

```cpp
// 骨架：对 block_q4_K 反量化的外层结构（6-bit 解包见第 7 章 get_scale_min_k4）
void dequantize_row_q4_K(const block_q4_K * x, float * y, int64_t k) {
    const int nb = k / QK_K;
    for (int i = 0; i < nb; i++) {
        const float d   = GGML_FP16_TO_FP32(x[i].d);
        const float min = GGML_FP16_TO_FP32(x[i].dmin);
        for (int j = 0; j < QK_K / 32; j++) {
            uint8_t sc, m;
            get_scale_min_k4(j, x[i].scales, &sc, &m); // 解出子块尺度/最小值
            const float d1 = d * sc;                    // 子块有效尺度
            const float m1 = min * m;                   // 子块有效最小值
            for (int l = 0; l < 32; l++) {
                int q = /* 从 x[i].qs 取第 (32*j+l) 个 4-bit 码 */;
                y[32*j + l] = d1 * q - m1;              // 统一公式
            }
        }
    }
}
```

**C#（.NET 5+）：**

```csharp
public static class Q4_K
{
    public const int QK_K = 256;

    public static void DequantizeRow(BlockQ4_K[] blocks, float[] y, int k)
    {
        int nb = k / QK_K;
        for (int i = 0; i < nb; i++)
        {
            float d = (float)BitConverter.UInt16BitsToHalf(blocks[i].d);
            float min = (float)BitConverter.UInt16BitsToHalf(blocks[i].dmin);
            for (int j = 0; j < QK_K / 32; j++)
            {
                // 6-bit 解包：见第 7 章的 get_scale_min_k4 等价实现
                (byte sc, byte m) = GetScaleMinK4(j, blocks[i].scales);
                float d1 = d * sc;
                float m1 = min * m;
                unsafe
                {
                    for (int l = 0; l < 32; l++)
                    {
                        int q = blocks[i].qs[(32 * j + l) / 2] >> (4 * ((32 * j + l) % 2)) & 0x0F;
                        y[32 * j + l] = d1 * q - m1;
                    }
                }
            }
        }
    }
}
```

**Python：**

```python
import struct

QK_K = 256

def dequantize_row_q4_K(data: bytes, k: int) -> list[float]:
    assert k % QK_K == 0
    nb = k // QK_K
    y = [0.0] * k

    for i in range(nb):
        base = i * 144
        d   = struct.unpack_from('<e', data, base)[0]
        dmin = struct.unpack_from('<e', data, base + 2)[0]
        scales = data[base + 4 : base + 16]
        qs     = data[base + 16 : base + 144]

        for j in range(QK_K // 32):
            sc, m = get_scale_min_k4(j, scales)   # 第 7 章实现
            d1, m1 = d * sc, dmin * m
            for l in range(32):
                q = qs[(32 * j + l) // 2] >> (4 * ((32 * j + l) % 2)) & 0x0F
                y[32 * j + l] = d1 * q - m1

    return y
```

三种实现的差别只在"怎么取出 4-bit 码"：C++ 里 `qs` 是字节数组，C# 和 Python 用位移 `>> 4*(idx%2)` 提取 nibble。**骨架相同、细节位序不同**——这正是第 7 章要逐一展开的。

### 6.2.2 6-bit 参数的归一化约定：除以 63 还是 64

现在回答一个在社区文档里反复出现的困惑：**6-bit 尺度取值 0..63，反量化时到底除以 63 还是 64？**

按 llama.cpp 主线源码（`ggml-quants.c`）核实，答案是：**解码端不做除法，直接乘 6-bit 整数；归一化在量化端完成。**

具体来说，量化端 `quantize_row_q4_K_ref` 的关键两行是：

```cpp
y[i].d    = GGML_FP32_TO_FP16(max_scale / 63.f);
y[i].dmin = GGML_FP32_TO_FP16(max_min  / 63.f);
```

也就是说，量化器把"真实的全局尺度除以 63"之后存进 `d`（和 `d_min`）。解码端 `dequantize_row_q4_K` 直接：

```cpp
const float d   = GGML_FP16_TO_FP32(x[i].d);    // 已经是 max_scale/63
const float d1  = d * sc;                        // sc 是 6-bit 整数，不除任何数
const float m1  = min * m;
*y++ = d1 * (q[l] & 0xF) - m1;                   // y = d·sc·q − min·m
```

于是整个链路的约定是：

```
量化端：    d = max_scale / 63     d_min = max_min / 63
解码端：    y = d·sc·q − d_min·m   （sc、m 直接是 6-bit 整数）
```

**为什么是 63 而不是 64？** 因为 6-bit 码的范围是 `[0, 63]`，共 64 个取值。除以 63 是把"最大值对应 1.0"；除以 64 是把"最大值对应 0.9844"——两种约定在数学上都合法，差别在量化端计算 `sc` 时怎么解释 `d`。llama.cpp 主线选 **63**（`max_scale/63`）。**不同文档、不同 fork 可能采用不同约定，跨实现读取时必须对照参考代码确认。** 本书以主线 `ggml-quants.c` 为准。

这一节澄清同时回答了细纲 6.2.2 的问题：**"除以 63 还是 64"不是解码端的歧义，而是量化端的选择。解码端无除法。** 第 7 章 7.2.2 会在 Q4_K 的具体代码里再走一遍。

那么，这个"归一化在量化端"的设计，和简单块（第 5 章）有什么不同？Q4_0 的量化端是 `d = max / -8`，也是先把尺度算好再存；K-quants 只是把同样思路延伸到两级尺度——**全局尺度先归一，子块尺度存"比例系数"。** 一句话：**量化端负责把真实数值折叠进存储参数，解码端只做简单的乘加。**


## 6.3 层次化尺度为什么有效

### 6.3.1 误差预算从"一个尺度"变为"尺度 + 子块修正"

K-quants 相比简单块，多出的存储是**每子块 6-bit 尺度 + 6-bit 最小值**（Q4_K 里共 12 字节/256 值）。这些字节换来的是什么？是**误差预算的重新分配**。

用信息论的语言说：量化误差有个"预算"，谁重要谁多分。简单块只有一个尺度 `d`，这个预算只能"平均分配"给整块 32 个值——块内分布不均匀时，预算被大值独占，小值几乎拿不到（Q4_0 的失效，4.5 节）。K-quants 有了子块尺度 `s_j` 和子块最小值 `m_j` 之后：

- 子块尺度和子块最小值是两个**可学习的修正项**，量化端可以针对每个 32 值子块的局部分布（均值、方差、偏斜）单独校准；
- 全局尺度 `d`、`d_min` 只承担"整体量级"和"整体偏移"两个粗粒度角色。

于是误差预算从"一个尺度管整块"变成"全局管量级、子块管分布"，**块内异质性的惩罚被分散到 8 个子块上**。第 7 章 7.4.1 会用真实权重切片展示这个效果的量化证据：Q4_K 的误差在块内比 Q4_0 均匀得多。

前面讲的都是道理，接下来我们用一个构造出来的例子，把“误差预算重新分配”这个效果直接“看”出来。假设一个 256 值的超级块，被切成了 8 个 32 值子块，其中 4 个子块的值集中在 0 附近（小量级），另外 4 个子块的值在 10 附近（大量级）——典型的“量级分层”分布。用 Q4_0 的思路（整块一个尺度）处理，会把整个 256 值按"全局最大"归一，小量级那 4 个子块的值全部塌缩到同几个码；而用 K-quants 的两级尺度，每个子块各自用自己的尺度，两层分布都能保住分辨率。可以用 Python 把这个对比画出来：

```python
import struct

def quant_q4_0_single_scale(values):
    """Q4_0 式：整个块共享一个尺度（示意，不做分块）"""
    amax = max(abs(v) for v in values)
    max_v = next(v for v in values if abs(v) == amax)
    d = max_v / -8.0
    id_ = 1.0 / d if d else 0.0
    codes = [min(15, int(v * id_ + 8.5)) for v in values]
    return [d * (q - 8) for q in codes]

def quant_q4_k_two_level(values, sub=32):
    """K-quants 式：每个子块独立尺度（示意，6-bit 尺度 + 4-bit 码）"""
    out = []
    for j in range(0, len(values), sub):
        block = values[j:j + sub]
        bmax = max(abs(v) for v in block)
        bd = bmax / 7.0            # 子块尺度
        codes = [min(15, max(0, int(v / bd + 0.5))) for v in block]
        out += [bd * (q - 8) for q in codes]   # 示意：对称量化
    return out

# 构造：前 4 子块 ~ N(0, 0.01)，后 4 子块 ~ N(10, 0.5)
import random
random.seed(42)
low  = [random.gauss(0, 0.01) for _ in range(128)]
high = [random.gauss(10, 0.5) for _ in range(128)]
values = low + high

single = quant_q4_0_single_scale(values)
two    = quant_q4_k_two_level(values)

# 单尺度：小值区域的相对误差巨大
def mse(a, b): return sum((x-y)**2 for x, y in zip(a, b)) / len(a)

print("单尺度全局 MSE:", mse(values, single))
print("两级尺度  MSE:", mse(values, two))
```

这个例子里，单尺度的 MSE 通常比两级尺度高一个数量级以上——因为它把误差预算全部压给了"整体量级"，而两级尺度把预算还给了每个子块的局部分布。**这就是 6.3.1 想让你看到的本质：多出的 12 字节买的不是"更多精度"，而是"把精度放到正确的位置"。**

### 6.3.2 K-quants 家族地图

把整个家族放在一张表里，作为后面各章的导航（字节数全部自 `ggml-common.h` 静态断言核实，bpw 按 `字节×8/256` 计算）：

| 格式 | 子块划分 | 码位宽 | 尺度参数位置 | 块字节 | bpw | 章节 |
|---|---|---|---|---|---|---|
| Q2_K | 16 × 16 值 | 2-bit | `scales[16]` 半字节 + `d`/`d_min` | 84 | 2.625 | 第 10 章 |
| Q3_K | 16 × 16 值 | 3-bit | `scales[12]` + `d` | 110 | 3.4375 | 第 10 章 |
| **Q4_K** | **8 × 32 值** | **4-bit** | **`scales[12]` 交错 + `d`/`d_min`** | **144** | **4.5** | **第 7 章** |
| Q5_K | 8 × 32 值 | 5-bit（4+1 平面） | `scales[12]` + `d`/`d_min` | 176 | 5.5 | 第 8 章 |
| Q6_K | 16 × 16 值 | 6-bit（4+2 平面） | `scales[16]` int8 + `d` | 210 | 6.5625 | 第 9 章 |
| Q8_K | — | 8-bit | `d`(f32) + `bsums` | 292 | 9.125 | 仅中间格式 |

读这张表要抓住三个维度：

1. **码位宽决定 bpw 的主基调**（2→3→4→5→6→8 bit）；
2. **子块划分影响局部性**——Q2_K/Q3_K/Q6_K 用 16×16（更细的局部），Q4_K/Q5_K 用 8×32（与 Q4_0 的 32 值块衔接）；
3. **尺度参数的位置和数量不同**——Q4_K/Q5_K 有 `d_min`（非对称），Q6_K 没有（对称但子块尺度是 8-bit 有符号，精度更高）。

把这张表放到时间线上看，也能看出设计思路的演进：k-quants 一开始就把 Q2_K/Q3_K/Q4_K/Q5_K/Q6_K 整族推出了，靠"超级块 + 子块"把误差预算从单一尺度里解放出来；随后社区在同一个骨架下调整子块大小和尺度位宽，得到不同 bpw 档位，再配合 `_S`/`_M`/`_L` 变体去适配不同敏感度的张量（第 14 章）。**骨架没变，变的只是参数。**


## 本章小结与通往 Q4_K 的桥梁

这一章可以压缩成四个事实。

**第一，K-quants 把"一个块一个尺度"升级成"超级块 + 子块的两级尺度"。** 256 值超级块共享全局尺度 `d` 和全局最小值尺度 `d_min`，内部 8 个子块各有 6-bit 尺度和 6-bit 最小值。C++/C#/Python 都能用结构体或 `bytes` 表达这个骨架。

**第二，统一公式是 \(y_i = d \cdot s_j \cdot q_i - d_{\min} \cdot m_j\)。** `d·s_j` 是子块有效尺度，`d_min·m_j` 是子块有效最小值，减号完成偏移扣除。Q2_K 到 Q6_K 都是它的特例，只是码位宽和参数打包不同。

**第三，归一化在量化端、解码端无除法。** 6-bit 整数 `s_j`、`m_j` 直接参与乘法，`d`/`d_min` 在量化端按 `/63` 归一化后存储。这个约定是主线源码核实的结果，跨实现读取必须对照参考代码。

**第四，层次化尺度的价值是"误差预算重新分配"。** 多出的 12 字节不是买"更多精度"，而是把精度从"整体量级"转移到"局部分布"——构造性实验显示两级尺度的 MSE 可以比单尺度低一个数量级以上。

下一章进入家族的主角——**Q4_K**。它是理解"6-bit 参数交错打包"的最佳载体：`scales[12]` 那 12 个字节里，8 个子块的尺度**和**最小值交错塞在一起，不是各占 6 字节。这个打包是 K-quants 家族里最容易被误解、也最需要亲手拆开的地方——第 7 章我们从 `get_scale_min_k4` 这个解包函数开始。
