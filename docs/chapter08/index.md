# 第8章 Q5_K：4-bit 主体加高位平面

上一章的 Q4_K 把有效位宽做到了 4.5 bpw，质量比同样 4.5 bpw 的 Q4_0 好一截（第 7 章已经验证过）。但有些场景仍然不满足：用户想再压一档精度，让 5-bit 甚至 6-bit 的量化质量——代价是文件大一点。Q5_K 就是 k-quant 家族在"4-bit 主体"和"更高精度"之间架起的一级台阶。

它的核心做法一句话就能概括：**Q5_K = Q4_K 的完整骨架 + 一个独立的"高位平面"**。每个权重的码仍是 4 位主体，但额外多存 1 位，让每个码从 0–15 变成 0–31，有效位宽从 4.5 bpw 升到 5.5 bpw。子块尺度、最小值偏移、6-bit 参数打包——一切都和 Q4_K 一样，只是多了一层位。

这一章你要建立起一个重要的直觉：**"加一个 bit"不是一个数学问题，而是一个打包问题**。多出来的那 1 位怎么放、放在哪个字节的哪一位，决定了格式的正确性和解码速度。Q5_K 把这 1 位集中存进一个独立的字节数组 `qh`，而不是塞进原有的 4-bit 主体里——这个"集中高位平面"的设计，是理解 Q5_K（以及后面的 Q6_K）的钥匙。

先统一几个术语。**主体（body）**：这里指存储 4-bit 码的数组 `qs`。**高位平面（high-bit plane）**：存储每个码额外那 1 位的数组 `qh`，它和 `qs` 按权重一一对应。**码（code）**：反量化时查到的整数，Q5_K 里是 5-bit，取值 0–31。**bpw**（bit per weight）：平均每个权重占用的比特数，第 1 章定义过，`bpw = 块字节数 × 8 / 每块权重数`。


## 8.1 5-bit 的组成

### 8.1.1 为什么不全用 5 个 bit 顺序存

直觉上，"每个权重 5 个 bit"最直接的做法是把 5-bit 码按顺序塞进字节流。但这么做有一个工程上的问题：5 不是 8 的约数，每 8 个 5-bit 码正好跨 5 个字节，解码时必须做"位缝"上的拼接，非常繁琐，也不利于 SIMD 向量化。

Q5_K 的答案是把 5-bit 拆成两部分：**低 4 位沿用 Q4_K 的 nibble 打包（`qs`），高 1 位单独集中存进 `qh`**。这样：

- `qs` 的处理逻辑和 Q4_K 完全一样，可以直接复用成熟代码；
- `qh` 每个字节存 8 个权重的高位，恰好 8 bit 一整字节，没有跨字节拼接；
- 解码时先分别取出低 4 位和高 1 位，拼成 5-bit 码，再套用 K-quants 公式。

用一张图看位平面：

```
一个权重的高/低平面拆分

  权重码 = 5 bit
  ┌──────────┬──────────┐
  │  高 1 bit │ 低 4 bit  │
  └──────────┴──────────┘
        │            │
        ▼            ▼
  集中存进 qh[]     分散在 qs[]（nibble 打包）
  1 字节装 8 个      1 字节装 2 个
  权重的高位        权重的低 4 位
```

后面的 Q6_K 会把"低位主体 + 高位平面"这个思路再推一步：主体仍是 4 位、高位平面变成 2 位。所以 Q5_K 是理解整个"5/6-bit 家族"的地基。

### 8.1.2 `block_q5_K` 结构体与 bpw

先看结构体。与 Q4_K 相比，`block_q5_K` 只在最后多了一个 `qh[32]`：

```c
#define QK_K 256
#define K_SCALE_SIZE 12

typedef struct {
    ggml_half d;              // 超级块尺度（FP16 位模式），2 字节
    ggml_half dmin;           // 超级块最小值尺度，2 字节
    uint8_t   scales[12];     // 子块尺度与最小值，6-bit 交错打包，12 字节
    uint8_t   qh[QK_K/8];     // 高位平面：256 个权重的高 1 位，32 字节
    uint8_t   qs[QK_K/2];     // 低位主体：256 个权重的低 4 位，128 字节
} block_q5_K;
static_assert(sizeof(block_q5_K) == 2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2 + QK_K/8,
              "wrong q5_K block size/padding");
```

字段顺序注意：`d`、`dmin` 在前，`scales`、`qh`、`qs` 在后。`scales[12]` 的打包规则与 Q4_K 完全一致（第 7 章 7.1.2 讲过：前 8 字节低 6 位存子块 0–3 的尺度和最小值，后 4 字节交错的规则），这里不再重复。

**字节总数**：

\[
4 \text{（d 与 dmin）} + 12 \text{（scales）} + 32 \text{（qh）} + 128 \text{（qs）} = 176 \ \text{字节}
\]

**有效位宽**：

\[
\text{bpw} = \frac{176 \times 8}{256} = 5.5 \ \text{bit/weight}
\]

与 Q4_K 的 4.5 相比，多出的恰好是那 1 个高位平面：

\[
4.5 \text{（Q4_K）} \xrightarrow{\ +1\ \text{bit/weight}\ } 5.5 \text{（Q5_K）}
\]

多出来的 32 字节 `qh`，平均到 256 个权重正好是 1 bit。**"加 1 bit"在字节层面就是这么直白**：多一个 32 字节的数组。


## 8.2 反量化：先合并码，再套公式

### 8.2.1 高位平面的位提取：`qh` 是按"位置"存、按"子块"分位的

这是 Q5_K 最容易写错的地方，请务必看仔细。`qh` 有 32 个字节，每个字节对应**子块内同一个位置 `l`** 的 8 个子块的高位——也就是说，`qh[l]` 的第 `k` 位（k = 0..7）是"第 `k` 个子块中、位置 `l` 的那个权重的高位"。

这里"位置"和"子块"是反直觉地颠倒的：

```
qh[l] 一个字节承载 8 个子块"同一位置"的高位

  qh[l] 的 bit   k =  7   6   5   4   3   2   1   0
                        │   │   │   │   │   │   │   │
  对应子块号      子块 7  6   5   4   3   2   1   0
  子块内位置      都是 l（0..31）

  反量化时，处理子块 k、遍历 l=0..31，
  检查 qh[l] 的第 k 位。
```

为什么这么设计？因为解码是"按子块推进"的：外层循环扫子块，内层循环扫位置。把"同一位置的高位"打包进一个字节，可以让内层循环每次只需对 `qh[l]` 做一个位测试，SIMD 也容易展开成 `and` 掩码。

ggml 的实现用一个掩码变量 `u` 来追踪"当前子块对应 `qh[l]` 的哪一位"，每处理完两个子块就左移 2 位。注意**`qh` 指针本身不推进**——永远指向超级块开头的 `qh[0..31]`，靠掩码 `u1/u2` 选择每个字节的哪一位：

```c
// 反量化一个 256 值超级块（与 ggml 的 dequantize_row_q5_K 逐行一致）
// d: 超级块尺度；min: 超级块最小值尺度；qs/qh: 低位主体与高位平面
static void dequantize_superblock_q5_K(float d, float min,
                                       const uint8_t * scales,
                                       const uint8_t * qh,
                                       const uint8_t * qs,
                                       float * y) {
    uint8_t sc, m;
    uint8_t u1 = 1, u2 = 2;                 // 掩码：bit0/bit1 → 子块 0/1
    int is = 0;                             // 子块索引
    for (int j = 0; j < QK_K; j += 64) {    // 每 64 值 = 两个 32 值子块
        get_scale_min_k4(is + 0, scales, &sc, &m);
        const float d1 = d * sc; const float m1 = min * m;
        get_scale_min_k4(is + 1, scales, &sc, &m);
        const float d2 = d * sc; const float m2 = min * m;

        for (int l = 0; l < 32; ++l) {
            y[l]      = d1 * ((qs[l] & 0xF)   + (qh[l] & u1 ? 16 : 0)) - m1;
            y[l + 32] = d2 * ((qs[l] >> 4)    + (qh[l] & u2 ? 16 : 0)) - m2;
        }
        y += 64; qs += 32; is += 2;
        u1 <<= 2; u2 <<= 2;                  // 下两个子块：bit2/bit3，bit4/bit5……
    }
}
```

关键在 `qh[l] & u1` 和 `qh[l] & u2`：

- `u1` 依次为 `1, 4, 16, 64`（即 bit0、bit2、bit4、bit6）→ 对应子块 0、2、4、6；
- `u2` 依次为 `2, 8, 32, 128`（即 bit1、bit3、bit5、bit7）→ 对应子块 1、3、5、7。

（提示：`is` 不能超过 8，子块共 8 个；`d`/`min` 已包含量化端的归一化，`sc`/`m` 直接相乘，无需再除 64——见第 7 章 7.2.2。）

三种语言的位提取逻辑完全一致：

**C++（整理为函数）：**

```cpp
// 反量化一个 256 值超级块
static void dequantize_superblock_q5_K(float d, float min,
                                       const uint8_t * scales,
                                       const uint8_t * qh,
                                       const uint8_t * qs,
                                       float * y) {
    uint8_t u1 = 1, u2 = 2;
    int is = 0;
    for (int j = 0; j < QK_K; j += 64) {
        uint8_t sc, m;
        get_scale_min_k4(is + 0, scales, &sc, &m);
        const float d1 = d * sc, m1 = min * m;
        get_scale_min_k4(is + 1, scales, &sc, &m);
        const float d2 = d * sc, m2 = min * m;

        for (int l = 0; l < 32; ++l) {
            y[l]      = d1 * ((qs[l] & 0x0F)   + (qh[l] & u1 ? 16 : 0)) - m1;
            y[l + 32] = d2 * ((qs[l] >> 4)     + (qh[l] & u2 ? 16 : 0)) - m2;
        }
        y += 64; qs += 32; is += 2;
        u1 <<= 2; u2 <<= 2;
    }
}
```

**C#（.NET 5+，用 `ReadOnlySpan` 表示两片字节）：**

```csharp
static void DequantizeSuperblockQ5K(
    float d, float min,
    ReadOnlySpan<byte> scales, ReadOnlySpan<byte> qh, ReadOnlySpan<byte> qs,
    Span<float> y)
{
    byte u1 = 1, u2 = 2;
    int is = 0;
    for (int j = 0; j < QK_K; j += 64)
    {
        GetScaleMinK4(is + 0, scales, out byte sc0, out byte m0);
        GetScaleMinK4(is + 1, scales, out byte sc1, out byte m1);
        float d1 = d * sc0, mm1 = min * m0;
        float d2 = d * sc1, mm2 = min * m1;
        for (int l = 0; l < 32; l++)
        {
            y[l]      = d1 * ((qs[l] & 0x0F) + (((qh[l] & u1) != 0) ? 16 : 0)) - mm1;
            y[l + 32] = d2 * ((qs[l] >> 4)   + (((qh[l] & u2) != 0) ? 16 : 0)) - mm2;
        }
        y = y[64..]; qs = qs[32..]; is += 2;
        u1 <<= 2; u2 <<= 2;
    }
}
```

**Python：**

```python
def get_scale_min_k4(j: int, scales: bytes) -> tuple[int, int]:
    if j < 4:
        return scales[j] & 63, scales[j + 4] & 63
    return (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4), \
           (scales[j + 4] >> 4)   | ((scales[j - 0] >> 6) << 4)

def dequantize_superblock_q5_k(d: float, min_: float,
                               scales: bytes, qh: bytes, qs: bytes) -> list[float]:
    assert len(scales) == 12 and len(qh) == 32 and len(qs) == 128
    y      = [0.0] * 256
    u1, u2 = 0x01, 0x02                # 掩码：bit0/bit1 → 子块 0/1
    is_    = 0
    for j in range(0, 256, 64):
        sc0, m0 = get_scale_min_k4(is_ + 0, scales)
        sc1, m1 = get_scale_min_k4(is_ + 1, scales)
        d1, m1v = d * sc0, min_ * m0
        d2, m2v = d * sc1, min_ * m1
        for l in range(32):
            y[j + l]      = d1 * ((qs[l] & 0x0F) + (16 if qh[l] & u1 else 0)) - m1v
            y[j + l + 32] = d2 * ((qs[l] >> 4)   + (16 if qh[l] & u2 else 0)) - m2v
        qs   = qs[32:]
        is_ += 2
        u1 <<= 2; u2 <<= 2
    return y
```

三种实现逐行对应。注意 `qh[l]` 的掩码是**每次左移 2 位**（`u1 <<= 2`），不是每次 1 位——这决定了高位平面按 stride-2 子块分组。

### 8.2.2 合并后的 5-bit 码代入 K-quants 公式

一旦把低 4 位和高 1 位合并成 5-bit 码 \(q_i \in [0,31]\)，剩下的就和 Q4_K 完全一样了。回顾第 7 章的 K-quants 统一公式（第 6 章 6.2.1）：

\[
y_i = d \cdot s_j \cdot q_i - d_{\min} \cdot m_j
\]

其中 \(s_j\) 是子块 6-bit 尺度（0–63），\(m_j\) 是子块 6-bit 最小值（0–63），它们从 `scales[12]` 里用 `get_scale_min_k4` 解出。Q5_K 只是把公式里的 \(q_i\) 从 0–15 换成 0–31：

```
Q4_K:   y = d·s_j·(4-bit 码)     - d_min·m_j      码 ∈ [0,15]
Q5_K:   y = d·s_j·(4-bit 码 + 高1位) - d_min·m_j   码 ∈ [0,31]
                                    ▲
                                    └─ qh[l] 的那 1 位，贡献 16
```

用数据流图概括 Q5_K 反量化的完整管线：

```
Q5_K 反量化一个 256 值超级块的数据流

  block_q5_K (176 字节)
  ┌────┬─────┬────────┬─────────┬──────────────┐
  │ d  │ dmin│scales  │  qh[32] │   qs[128]    │
  └─┬──┴──┬──┴───┬────┴───┬─────┴───┬──────────┘
    ▼     ▼      ▼        ▼         ▼
  fp16→f32   fp16→f32  get_scale   高位位提取    nibble 拆低 4 位
    │     │     min_k4    (每子块   (每子块 32 值
    │     │      → sc,m    1 bit)    ×低/高 nibble)
    │     │       │         │         │
    │     │       ▼         ▼         ▼
    │     │   d·sc=d_j   qh 位×16    4-bit 码
    │     │   min·m=m_j     │         │
    │     │                 └──┬──────┘
    │     │                    ▼
    │     │             5-bit 码 q_i
    │     │                    │
    │     ▼                    ▼
    │  y = d_j·q_i − m_j
    ▼
  32 个 float 输出
```

### 8.2.3 手工字节验证

打包规则对不对，最可靠的检验是构造一个已知字节串，用手算/独立实现交叉验证。构造一个简化超级块：`d = 1.0`、`dmin = 0.0`、`scales[12]` 全为 `1`（这让 `get_scale_min_k4` 解出的尺度 `sc` 恒为 1；最小值系数 `m` 在子块 0–3 为 1、子块 4–7 为 0，但 `dmin = 0` 使 `min·m` 恒为 0），`qs` 每个字节 `0x11`（低 nibble 和高 nibble 都是 1），`qh` 每个字节 `0x01`（bit0 = 1，其余为 0）。

那么：子块 0 的每个值低 4 位 = 1、高位 = 1，码 \(= 1 + 16 = 17\)，期望 \(y = 1\cdot 1\cdot 17 - 0 = 17.0\)；子块 1 的低 4 位 = 1、高位 = 0（bit1 未置位），码 = 1，期望 \(y = 1\cdot1\cdot1 - 0 = 1.0\)。

用前面的 `dequantize_superblock_q5_k` 断言：

```python
import struct

def fp16_to_f32(b: bytes) -> float:
    return struct.unpack('<e', b)[0]

d     = struct.pack('<e', 1.0)
dmin  = struct.pack('<e', 0.0)
scales = bytes([1] * 12)     # 简化：全部子块尺度/最小值系数为 1
qh     = bytes([0x01] * 32)  # 只置 bit0 → 子块 0 的高位全部为 1
qs     = bytes([0x11] * 128) # 每个字节低/高 nibble 都是 1

y = dequantize_superblock_q5_k(
    fp16_to_f32(d), fp16_to_f32(dmin), scales, qh, qs)

assert y[0]    == 1.0 * (1 + 16)        # 子块 0：低4位=1 + 高位=1 → 17
assert y[32]   == 1.0 * (1 + 0)         # 子块 1：低4位=1 + 高位=0 → 1
print("交叉验证通过：y[0] =", y[0], " y[32] =", y[32])
```

手工验证的价值在于**把你对位序的理解变成可执行的断言**。只要这里通过，整块解码的其余部分就只是"按同样的位规则重复 256 次"。


## 8.3 质量/体积权衡与代码陷阱

### 8.3.1 Q5_K_M 的 PPL 与速度

Q5_K 的有效位宽 5.5 bpw，对应的文件级策略是 `Q5_K_M`。它的质量介于 Q4_K 和 Q6_K 之间：相对 F16 的困惑度（PPL）劣化**约 +0.4%**（具体值随模型与评测集变化，不同来源的数字不可直接比较，这里是量级参考）。作为对比，第 7 章说过 Q4_K_M 约 +0.8%，下一章的 Q6_K 约 +0.2%。

```
Q5_K_M 在 k-quant 家族中的位置（示意）

  bpw      4.5        5.5       6.5
  ────────┬──────────┬──────────┬────────►
  质量     Q4_K_M     Q5_K_M     Q6_K
  PPL 劣化  ≈+0.8%    ≈+0.4%    ≈+0.2%
  文件    较小        中等       较大
```

速度方面，Q5_K_M 的解码多了一步 `qh` 位提取和加法，但这一步在矩阵乘面前通常可以忽略——第 17 章会解释，端侧推理的瓶颈主要是把权重从内存搬进计算单元（内存带宽），而不是这种每值一两次的整数位操作。因此 Q5_K_M 与 Q4_K_M 的 tok/s 差距主要来自**文件变大带来的带宽压力**，而不是解码复杂度。

它的定位是"质量敏感场景的中间选择"：当你觉得 Q4_K 的 +0.8% 略高、又不想付出 Q6_K 的文件体积时，Q5_K_M 是一个稳妥的折中。

### 8.3.2 高位平面解包中的常见位序错误

`qh` 的位序是新增格式实现里最常见的 bug 来源，而且**错位序的解码结果是"温和地错"**——不会崩溃，PPL 只是略微变差，很难排查。三类典型的错误：

1. **把 `qh` 当成"按子块分字节"**。错误做法是把 `qh[0..3]` 当作子块 0 的 32 个高位、`qh[4..7]` 当作子块 1……这是"字节=子块"的直觉，但 Q5_K 实际是"字节=位置"、"位=子块"。按错误直觉解码，每个权重拿到的是另一个子块某个位置的位，误差整体偏大且毫无规律。

2. **掩码步进方向错**。`u1`/`u2` 每 64 值左移 2 位，不是每次 1 位。如果写成 `u <<= 1`，高位平面会读到错误位。

3. **大小端混用**。`qh` 是按字节数组读的，不涉及多字节整数的字节序；但如果把 4 个 `qh` 字节当成一个 `uint32` 再移位，就引入了与实现约定有关的端序问题。

**规避方法**：用 8.2.3 的"逐值参考实现"做交叉验证——把某个超级块的字节喂给独立的 Python 参考解码器，与 ggml 的 `dequantize_row_q5_K` 输出逐值比对。第 18 章会把这一方法推广成完整验证框架，但"先构造已知字节、再断言输出"这一招，写任何新格式的当天就该用上。


## 本章小结与通往 Q6_K 的桥梁

这一章可以压缩成三个事实。

**第一，Q5_K = Q4_K 骨架 + 一个 32 字节的高位平面。** 每个权重 5 bit = 低 4 位（`qs` nibble 打包）+ 高 1 位（`qh`），结构是 `d`+`dmin`+`scales[12]`+`qh[32]`+`qs[128]`，共 176 字节，有效位宽 5.5 bpw。反量化先合并码再套公式：\(y = d \cdot s_j \cdot q_i - d_{\min} \cdot m_j\)，\(q_i \in [0,31]\)。

**第二，`qh` 的位序是"字节存位置、位存子块"。** `qh[l]` 的第 k 位是"子块 k 中位置 l 的权重的高位"，掩码 `u` 每 64 值左移 2 位。这是最常见的实现 bug 来源，必须用逐值交叉验证兜底。

**第三，Q5_K_M 是质量敏感场景的中间选择。** PPL 劣化约 +0.4%（介于 Q4_K 的 +0.8% 与 Q6_K 的 +0.2% 之间），解码开销可忽略，主要成本是文件更大带来的带宽压力。

```
高位平面思想的延续

  Q5_K:  4-bit 主体  +  1-bit 高位平面   → 5.5 bpw
  Q6_K:  4-bit 主体  +  2-bit 高位平面   → 6.5 bpw
              ▲
              └── 下一章：Q6_K 把高位平面从 1 位扩成 2 位，
                    并把子块结构从"8×32"改成"16×16"。
```

下一章的 Q6_K 会把"高位平面"这个思路用到极致：高位从 1 位变成 2 位，子块尺度从 6-bit 无符号升级成 8-bit 有符号，并且去掉最小值偏移——一个结构更简单、但精度更高的格式，也是输出嵌入层的默认选择。
