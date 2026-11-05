# 第10章 Q2_K 与 Q3_K：极端压缩的边界

第 7 到第 9 章讲的 Q4_K、Q5_K、Q6_K，主体位宽都在 4 bit 以上，属于"省空间但不伤筋动骨"的区间。这一章我们往另一个方向走到头：**2-bit 和 3-bit**。

先把吸引力讲清楚。一个 7B 模型用 Q4_K_M 大约 3.8 GB。如果每个权重只用 2 bit 呢？文件可以压到 2 GB 附近——这个体积能塞进 8 GB 显存或低配手机的 RAM 里，让之前"跑不动"的设备也能装下大模型。但 2-bit 也意味着：一个权重只有 **4 种可能的值**（0、1、2、3，或带符号的变体），信息的流失是结构性的。

本章要回答两个问题：**Q2_K 和 Q3_K 到底怎么把 2-bit / 3-bit 码装进文件里（字节布局与反量化公式）**，以及 **"再压一点"在哪里失效（为什么 Q2_K 不应该作为默认部署选择）**。第一个问题是工程细节，第二个问题是实践判断——两者合起来，就是"极端压缩的边界"。

先统一几个术语。**2-bit 码**：一个权重用一个 2 位二进制数表示，取值 0、1、2、3（本无符号整数）。**子组（subgroup）**：K-quants 中比子块更小的组织单元，Q2_K/Q3_K 把 256 值的超级块切成 16 个 16 值子组。**半字节（nibble）**：4 位，一个字节的一半。**位平面（bit-plane）**：把同一位的所有值单独抽出来存放，是拆解多 bit 码的常用手法。**PPL（perplexity，困惑度）**：衡量模型预测能力的指标，越低越好；量化后 PPL 增加越多，质量损失越大。**任务退化**：不只是"输出变糙"，而是指令跟随、推理、代码等具体能力明显下降。


## 10.1 Q2_K：16 个 16 值子组与 2-bit 主体

### 10.1.1 为什么 2-bit 不能再使用"一个子块一个尺度"的老结构

第 7 章讲过 Q4_K 的结构：256 值超级块，全局尺度 `d` 和全局最小值尺度 `dmin`，内部 8 个子块各带 6-bit 尺度和 6-bit 最小值。这个设计的前提是"每个子块有 32 个权重来摊薄尺度参数"。

到了 2-bit 有个新问题：**码本身的信息量急剧下降，尺度信息必须更精细，否则精度完全消失**。设想一个 32 值子块，若仍只有一对全局（`d`, `dmin`），那么子块内部 32 个权重的差异只能靠 2-bit 码的 4 档来表达——这 4 档之间的空隙太大，重建误差会非常高。

Q2_K 的解决办法是**把子块再切小**：256 值超级块切成 **16 个 16 值子组**，每个子组拥有一对**4-bit**的尺度系数与最小值系数（合起来正好一个字节）。子组更小，尺度就能更贴近局部分布；但单个子组只有 16 个值来摊薄参数，所以尺度只能各用 4 bit——这是"更细的粒度"与"更少的参数位宽"之间的一次权衡。

```
Q2_K 的层次结构（对比 Q4_K）

  Q4_K（8 个子块 × 32 值）:
  ┌───────────────────────────────────────────┐
  │ 全局尺度 d + 全局最小值尺度 dmin           │
  │  ┌──────┬──────┬──────┬──────┬...┬──────┐  │
  │  │子块 0│子块 1│子块 2│子块 3│   │子块 7│  │
  │  │32 值 │32 值 │32 值 │32 值 │   │32 值 │  │
  │  │6-bits│...                                   │
  │  └──────┴──────┴──────┴──────┴...┴──────┘  │
  └───────────────────────────────────────────┘

  Q2_K（16 个子组 × 16 值）:
  ┌───────────────────────────────────────────┐
  │ 全局尺度 d + 全局最小值尺度 dmin           │
  │  ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬...┬──┬──┐   │
  │  │子组0│子组1│子组2│...│子组15│             │
  │  │16值 │16值 │16值 │   │16值 │             │
  │  │4-bit│4-bit│4-bit │     4-bit │          │
  │  └──┴──┴──┴──┴──┴──┴────────────────┘   │
  │  每个子组一个字节：低半字节=尺度系数       │
  │  高半字节=最小值系数                       │
  └───────────────────────────────────────────┘
```

### 10.1.2 82 还是 84：block_q2_K 的确切布局

`block_q2_K` 的结构体定义如下：

```cpp
#define QK_K 256

typedef struct {
    uint8_t scales[QK_K/16]; // 16 字节：每个子组的尺度系数 + 最小值系数
    uint8_t qs[QK_K/4];      // 64 字节：256 个 2-bit 码，4 个/字节
    ggml_half d;             // 超级块尺度（FP16），2 字节
    ggml_half dmin;          // 超级块最小值尺度（FP16），2 字节
} block_q2_K;
```

三个字段依次排列，字段间没有填充，总大小是：

\[
4 + 16 + 64 = 84 \ \text{字节}
\]

**有效位宽**：

\[
\text{bpw} = \frac{84 \times 8}{256} = 2.625 \ \text{bit/weight}
\]

社区资料里偶尔会看到 Q2_K 记为"2.5 bpw"或"2.5625 bpw"，那是因为部分资料把变体（如 Q2_K_M）或历史版本的打包方式混进来，且尺度参数的字节数被舍入过。以主线 `ggml-common.h` 为准：**每个 256 值超级块固定 84 字节，2.625 bpw**。

字节布局整体如下：

```
block_q2_K 的 84 字节布局（磁盘/内存顺序）

  偏移  0                       15 16                      79 80   83
        ┌────────────────────────┬────────────────────────┬────────┐
        │    scales[0..15]       │      qs[0..63]         │ d, dmin│
        │  16 字节：每个子组      │  64 字节：256 个        │ 4 字节 │
        │  1 字节（4-bit+4-bit） │  2-bit 码，4 个/字节   │  2×FP16│
        └────────────────────────┴────────────────────────┴────────┘
```

### 10.1.3 半字节存储：一个字节装一对系数

`scales[i]` 这一个字节要同时装子组 \(i\) 的**尺度系数**和**最小值系数**。做法是拆分半字节：

```
scales[i] 的一个字节 = 8 bit

        bit 7 6 5 4 │ bit 3 2 1 0
        ┌───────────┼───────────┐
        │ 高 4 位    │ 低 4 位    │
        │ 最小值系数  │ 尺度系数   │
        └───────────┴───────────┘
             │            │
             ▼            ▼
         dmin 的比例      d 的比例
           系数 (0..15)    系数 (0..15)
```

- **低半字节** `scales[i] & 0x0F`：子组 \(i\) 的尺度系数 \(d_l\)，取值 0–15。
- **高半字节** `scales[i] >> 4`：子组 \(i\) 的最小值系数 \(d_h\)，取值 0–15。

注意：这里的系数的"门槛"和 Q4_K 的 6-bit `get_scale_min_k4` 完全不同。Q4_K 用 12 字节的**交错打包**装 8 对 6-bit 系数；Q2_K 用 16 字节、每个子组直接一整字节、**4-bit + 4-bit 各占半字节**——不需要任何 `get_scale_min_k4` 式的位重组。结构更简单，但系数位数更少（4-bit vs 6-bit），这是精度上的妥协。

### 10.1.4 公式与反量化循环

反量化公式是：

\[
y_i = d \cdot d_l \cdot q_i - d_{\min} \cdot d_h
\]

其中 \(q_i \in \{0,1,2,3\}\) 是 2-bit 码，\(d_l = \text{scales}[j] \& 0x0F\)，\(d_h = \text{scales}[j] \gg 4\)。把它重写成代码视角：

```cpp
dl = d * (scales[is] & 0x0F);   // 子组尺度
ml = dmin * (scales[is] >> 4);  // 子组最小值
y  = dl * ((q >> shift) & 3) - ml;
```

`(q >> shift) & 3` 从打包字节里取出 2-bit 码。因为每个字节装 4 个 2-bit 码（对应 4 个不同的权重），`shift` 依次取 0、2、4、6。

**C++（llama.cpp 的 `dequantize_row_q2_K`，逻辑整理）：**

```cpp
void dequantize_row_q2_K(const block_q2_K * x, float * y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const float d    = GGML_FP16_TO_FP32(x[i].d);
        const float dmin = GGML_FP16_TO_FP32(x[i].dmin);
        const uint8_t * q = x[i].qs;

        int is = 0;
        // 一个超级块 = 两轮，每轮 128 个值
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;                      // 2-bit 码在字节里的位偏移
            for (int j = 0; j < 4; ++j) {       // 每轮 4 次 × 2 子组 = 8 子组
                uint8_t sc = x[i].scales[is++];
                float dl = d * (sc & 0x0F);
                float ml = dmin * (sc >> 4);
                for (int l = 0; l < 16; ++l)
                    *y++ = dl * ((int8_t)((q[l]      >> shift) & 3)) - ml;

                sc = x[i].scales[is++];
                dl = d * (sc & 0x0F);
                ml = dmin * (sc >> 4);
                for (int l = 0; l < 16; ++l)
                    *y++ = dl * ((int8_t)((q[l + 16] >> shift) & 3)) - ml;

                shift += 2;                     // 换到下一位
            }
            q += 32;
        }
    }
}
```

**C#（.NET 5+）：**

```csharp
public static class Q2_K
{
    public const int QK_K = 256;

    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public unsafe struct BlockQ2_K
    {
        public fixed byte scales[QK_K / 16]; // 16
        public fixed byte qs[QK_K / 4];      // 64
        public ushort d;                     // FP16 位模式
        public ushort dmin;                  // FP16 位模式
    }

    public static void DequantizeRow(BlockQ2_K[] x, float[] y, int k)
    {
        int nb = k / QK_K;
        int yi = 0;
        for (int i = 0; i < nb; i++)
        {
            float d    = (float)BitConverter.UInt16BitsToHalf(x[i].d);
            float dmin = (float)BitConverter.UInt16BitsToHalf(x[i].dmin);
            unsafe
            {
                int is = 0;
                for (int n = 0; n < QK_K; n += 128)
                {
                    int shift = 0;
                    for (int j = 0; j < 4; j++)
                    {
                        byte sc0 = x[i].scales[is++];
                        float dl = d * (sc0 & 0x0F);
                        float ml = dmin * (sc0 >> 4);
                        for (int l = 0; l < 16; l++)
                            y[yi++] = dl * ((x[i].qs[l] >> shift) & 3) - ml;

                        byte sc1 = x[i].scales[is++];
                        dl = d * (sc1 & 0x0F);
                        ml = dmin * (sc1 >> 4);
                        for (int l = 0; l < 16; l++)
                            y[yi++] = dl * ((x[i].qs[l + 16] >> shift) & 3) - ml;

                        shift += 2;
                    }
                }
            }
        }
    }
}
```

`fixed byte scales[16]` 是 C# 的固定大小缓冲区，访问必须在 `unsafe` 上下文中；`Pack = 1` 取消对齐填充，保证结构与磁盘布局一致（`sizeof(BlockQ2_K) == 84`）。

**Python（`struct` 手工拆解）：**

```python
import struct

QK_K = 256

def dequantize_row_q2_K(data: bytes) -> list[float]:
    """data 是一个超级块的 84 字节。"""
    assert len(data) == 84
    d    = struct.unpack_from('<e', data, 80)[0]  # d 在偏移 80
    dmin = struct.unpack_from('<e', data, 82)[0]  # dmin 在偏移 82
    scales = data[0:16]
    qs     = data[16:80]

    y = []
    idx = 0
    for n in range(0, QK_K, 128):        # 两轮，各 128 值 / 32 字节
        shift = 0
        for j in range(4):               # 4 次 × 2 子组 = 8 子组
            sc = scales[idx];  idx += 1
            dl = d * (sc & 0x0F); ml = dmin * (sc >> 4)
            for l in range(16):          # 字节 0..15
                y.append(dl * ((qs[l] >> shift) & 3) - ml)

            sc = scales[idx];  idx += 1
            dl = d * (sc & 0x0F); ml = dmin * (sc >> 4)
            for l in range(16):          # 字节 16..31
                y.append(dl * ((qs[l + 16] >> shift) & 3) - ml)

            shift += 2
        qs = qs[32:]                     # 每一轮消耗 32 个 qs 字节
    return y
```

关键点是**字节下标和 shift 的对应关系**：一轮的 32 个 `qs` 字节，`qs[0..15]` 与 `qs[16..31]` 分别对应两个子组；每个字节的 4 个 2-bit 码，则靠 `shift = 0, 2, 4, 6` 四次迭代取出。三份代码的差异只体现在"语言如何表达位操作"上，数学等价：**都是每 16 值一个子组、逐个字节按 shift 取 2-bit 码、乘尺度减最小值**。

### 10.1.5 打包：4 个 2-bit 码挤进一个字节

反量化看了"取出"，量化端是反方向：如何把 4 个 0–3 的码塞进一个字节。llama.cpp 的量化参考 `quantize_row_q2_K_ref` 是把每 128 个码的 4 个分区（各 32 个）按位对齐放进 32 个字节：

```cpp
for (int j = 0; j < QK_K; j += 128) {
    for (int l = 0; l < 32; ++l) {
        qs[j/4 + l] = L[j + l]
                    | (L[j + l + 32] << 2)
                    | (L[j + l + 64] << 4)
                    | (L[j + l + 96] << 6);
    }
}
```

`L` 是 256 个 2-bit 码的数组。可以看到每个字节的低 2 位、第 3–4 位、第 5–6 位、最高 2 位分别装 4 个不同分区的值——这正好对应反量化里的 `shift = 0,2,4,6` 四次提取。**打包与解包是精确互逆的**，这正是第 7 章 Q4_K 也遵守的"先对齐再打包"约定；不同的是 Q4_K 一个字节装 2 个 4-bit 码（shift 0 和 4），Q2_K 一个字节装 4 个 2-bit 码（shift 0/2/4/6）。


## 10.2 Q3_K：3-bit 码的平面拆分

### 10.2.1 3-bit 码拆成两个平面：低 2 位 + 高 1 位

Q3_K 是 3-bit（8 个取值），但 8 不是 2 的整数次方的"整字节契合"数字——把一个字节用来装 3-bit 会浪费。所以 Q3_K 采用**平面拆分**：3-bit 码的高 1 位单独抽出来集中存放，低 2 位按"4 个/字节"存放，高位每 8 个码凑一字节。

```
Q3_K 的 3-bit 平面拆分

  3-bit 码（8 个取值 -4..3）
    高位平面 hmask（32 字节，每 byte 8 位）
    低位平面 qs（64 字节，每 byte 4 个 2-bit 码）

  单个权重 w 的 3-bit 码 = (qs 中的 2 位) + (hmask 中的 1 位)
```

`block_q3_K` 结构体：

```cpp
typedef struct {
    uint8_t hmask[QK_K/8];  // 32 字节：3-bit 码的高位平面
    uint8_t qs[QK_K/4];     // 64 字节：3-bit 码的低 2 位平面
    uint8_t scales[12];     // 12 字节：16 个 6-bit 尺度的打包（见下）
    ggml_half d;            // 超级块尺度（FP16），2 字节
} block_q3_K;
```

大小：\(32 + 64 + 12 + 2 = 110\) 字节。

**有效位宽**：

\[
\text{bpw} = \frac{110 \times 8}{256} = 3.4375 \ \text{bit/weight}
\]

note：部分文档把 Q3_K 记为 3.5 bpw，是把 110 按 112 对齐后的小数舍入；以主线 `static_assert` 的 110 字节为准，精确值是 3.4375。

### 10.2.2 6-bit 子块尺度藏在 12 字节里

Q3_K 有 16 个子组，每个子组的尺度是 **6-bit**（这一点比 Q2_K 的 4-bit 高，是对"3-bit 码本身粗糙"的一点补偿）。16 × 6 = 96 bit = 12 字节，正好装进 `scales[12]`。

但 6-bit 并不是像 Q2_K 那样"每子组一个字节"——那样需要 16 字节。12 字节装 16 个 6-bit 码，必须交错打包。llama.cpp 的解码用了一次巧妙的位重组 `unpack`：把 12 字节按 4 字节一组处理，利用对齐把 6-bit 码拼回 int8 的"低 6 位有效、多出的 2 位补偏置"，再整体减去 32 得到有符号尺度。效果等价于：**每个子组得到一个 -32..31 范围内的 int8 子块尺度**。

（实现上，`dequantize_row_q3_K` 里通过一次 12→16 字节的 `uint32_t` 位重组，把交织的 6-bit 码展开成 16 个连续的 int8，再统一减 32。这一重组对多次复用很关键，但数学核心就是上面那句话。第 18 章会用交叉验证脚本与这个位重组对拍。）

### 10.2.3 反量化公式与循环

\[
y = d \cdot (s_j - 32) \cdot \big((q_2 - 4 \cdot (1 - h))\big), \quad h \in \{0,1\}
\]

更直接地写成代码的三个部分：
1. 取子块尺度 `dl = d * (scales[is] - 32)`；
2. 低位码 `lo = (qs[byte] >> shift) & 3`；
3. 高位码 `h = (hmask[byte] & m) ? 1 : 0`，反量化值为 `dl * (lo - (h ? 0 : 4))`。

**C++（llama.cpp 参考逻辑）：**

```cpp
void dequantize_row_q3_K(const block_q3_K * x, float * y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const float d_all = GGML_FP16_TO_FP32(x[i].d);
        const uint8_t * q  = x[i].qs;
        const uint8_t * hm = x[i].hmask;
        uint8_t m = 1;   // hmask 当前位掩码（1, 2, 4, 8, ... 128）

        // 把 12 字节的 scales 解成 16 个有符号 int8 尺度（见 10.2.2）
        int8_t s16[16];
        unpack_scales(x[i].scales, s16);

        int is = 0;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                float dl = d_all * (s16[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    int lo = (q[l]      >> shift) & 3;
                    int exact = lo - ((hm[l] & m) ? 0 : 4);
                    *y++ = dl * exact;
                }
                dl = d_all * (s16[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    int lo = (q[l + 16] >> shift) & 3;
                    int exact = lo - ((hm[l + 16] & m) ? 0 : 4);
                    *y++ = dl * exact;
                }
                shift += 2;
                m <<= 1;   // 下一个 2-bit 位置 → 下一个 hmask 位
            }
            q += 32;
        }
    }
}
```

`m` 从 1 开始，在两个 128 值块内逐步左移（1→2→4→…→128），正好遍历 `hmask` 每字节的 8 个位。`(hm[l] & m) ? 0 : 4` 的含义：**高位平面为 1 时低位码 (0..3) 原样使用；高位平面为 0 时整体减 4，落进负半轴 (-4..-1)**。3-bit 码因此能表示 -4..3 共 8 个值。`unpack_scales` 的那次位重组，就是 10.2.2 里写的"低 4 位 + 高 2 位"的等效实现：它越过字节边界直接取 6-bit 码，再统一减 32 变有符号。

**C#：**

```csharp
public static partial class Q3_K
{
    public const int QK_K = 256;
    public const int QK3_K = 256;

    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public unsafe struct BlockQ3_K
    {
        public fixed byte hmask[QK_K / 8];
        public fixed byte qs[QK_K / 4];
        public fixed byte scales[12];
        public ushort d;
    }

    // 把 12 字节解成 16 个 int8 尺度（低 4 位 + 高 2 位拼 6-bit，再减 32）
    public static int[] DecodeScales(byte[] s12)
    {
        var s16 = new int[16];
        for (int i = 0; i < 4; i++)
        {
            s16[0 + i] = ((s12[i] & 0x0F) | ((s12[8 + i] & 0x03) << 4)) - 32;
            s16[4 + i] = ((s12[4 + i] & 0x0F) | ((s12[8 + i] >> 2 & 0x03) << 4)) - 32;
            s16[8 + i] = ((s12[i] >> 4)      | ((s12[8 + i] >> 4 & 0x03) << 4)) - 32;
            s16[12+ i] = ((s12[4+i] >> 4)    | ((s12[8 + i] >> 6 & 0x03) << 4)) - 32;
        }
        return s16;
    }

    public static void DequantizeRow(BlockQ3_K[] x, float[] y, int k)
    {
        int nb = k / QK_K;
        int yi = 0;
        for (int i = 0; i < nb; i++)
        {
            float d  = (float)BitConverter.UInt16BitsToHalf(x[i].d);
            var s16  = DecodeScales(ReadScales(x[i]));
            int mbit = 1, is = 0;
            byte[] q = ReadQs(x[i]);
            byte[] hm = ReadHmask(x[i]);
            for (int n = 0; n < QK_K; n += 128)
            {
                int shift = 0;
                for (int j = 0; j < 4; j++)
                {
                    float dl = d * s16[is++];
                    for (int l = 0; l < 16; l++)
                    {
                        int lo = (q[l] >> shift) & 3;
                        int hi = (hm[l] & mbit) != 0 ? 0 : 4;
                        y[yi++] = dl * (lo - hi);
                    }
                    dl = d * s16[is++];
                    for (int l = 0; l < 16; l++)
                    {
                        int lo = (q[l + 16] >> shift) & 3;
                        int hi = (hm[l + 16] & mbit) != 0 ? 0 : 4;
                        y[yi++] = dl * (lo - hi);
                    }
                    shift += 2;
                    mbit <<= 1;
                }
            }
        }
    }

    // 辅助：把 unsafe fixed 字节数组读成普通 byte[]（略）
    private static byte[] ReadHmask(BlockQ3_K b) { var r = new byte[32]; unsafe { for (int i = 0; i < 32; i++) r[i] = b.hmask[i]; } return r; }
    private static byte[] ReadQs(BlockQ3_K b) { var r = new byte[64]; unsafe { for (int i = 0; i < 64; i++) r[i] = b.qs[i]; } return r; }
    private static byte[] ReadScales(BlockQ3_K b) { var r = new byte[12]; unsafe { for (int i = 0; i < 12; i++) r[i] = b.scales[i]; } return r; }
}
```

**Python 参考实现（与 C 逐行对应）：**

```python
import struct

QK_K = 256

def decode_q3k_scales(s12: bytes) -> list[int]:
    """12 字节 → 16 个 int8 尺度（低 4 位 + 高 2 位拼 6-bit，再减 32）。"""
    out = []
    for i in range(4):
        out.append(((s12[i] & 0x0F) | ((s12[8 + i] & 0x03) << 4)) - 32)
    for i in range(4):
        out.append(((s12[4 + i] & 0x0F) | ((s12[8 + i] >> 2 & 0x03) << 4)) - 32)
    for i in range(4):
        out.append(((s12[i] >> 4)      | ((s12[8 + i] >> 4 & 0x03) << 4)) - 32)
    for i in range(4):
        out.append(((s12[4 + i] >> 4)  | ((s12[8 + i] >> 6 & 0x03) << 4)) - 32)
    return out

def dequantize_row_q3_K(data: bytes) -> list[float]:
    assert len(data) == 110
    d     = struct.unpack_from('<e', data, 108)[0]      # d 在最后
    s16   = decode_q3k_scales(data[96:108])
    qs    = data[32:96]
    hmask = data[0:32]

    y, is = [], 0
    mbit = 1                                  # hmask 位掩码：1,2,4,...128（跨 n 循环累计）
    for _n in range(0, QK_K, 128):            # 两个 128 值块
        shift = 0
        for j in range(4):                    # 每块 4 次 × 2 子组 = 8 子组
            dl = d * s16[is]; is += 1
            for l in range(16):
                lo = (qs[l] >> shift) & 3
                hi = (hmask[l] & mbit) != 0
                y.append(dl * (lo - (0 if hi else 4)))

            dl = d * s16[is]; is += 1
            for l in range(16):
                lo = (qs[l + 16] >> shift) & 3
                hi = (hmask[l + 16] & mbit) != 0
                y.append(dl * (lo - (0 if hi else 4)))

            shift += 2
            mbit <<= 1                        # 下一个 2-bit 位置 → 下一个 hmask 位
        qs = qs[32:]                          # 每块消耗 32 字节 qs
    return y
```

三份实现共享同一个核心：**子块尺度存成有符号 6-bit（减 32），3-bit 码拆成低 2 位（qs）+ 高 1 位（hmask），高位为 0 时值域落到负半区**。这个"高位平面"思路在第 8 章 Q5_K 里已经见过一次，这里手法相同，只是拆分比例从"4+1"换成了"2+1"。


## 10.3 极端压缩的边界：Q2_K 为什么不该作为默认

### 10.3.1 Q2_K 的 PPL 劣化不是平滑曲线

K-quants 从 Q4_K 到 Q5_K 再到 Q6_K，质量是**逐年平滑回升**的：每多用 1 bit，PPL 改善一点，没有骤变。下到 Q2_K 时情况变了。

实测印象（以量化 PPL 相对 F16 基准）：Q4_K 约 +0.8%，Q3_K 大概再翻一倍，而 **Q2_K 的可报告 PPL 往往要比 Q4_K 再高 2–3 倍**，并且不同的评测集会给出差异很大的数字。更关键的不是绝对数字，而是**任务表现的断崖**：

- 简单分类、短问答、检索类任务：退化相对温和（这些任务对"精确权重值"的依赖较低）；
- 多步推理、数学、代码生成、带约束的指令跟随：退化明显，有时表现为几乎不能用的水平；
- 模型越大（70B+）退化越可忍受，模型越小（7B–13B）退化越致命——因为大模型本身有更多冗余去吸收量化噪声。

用一句话概括：**Q2_K 的劣化不是"Q4_K 多一点"，而是"能力属性"层面的下降**。它更像是从"个体数值误差"变成了"结构性信息缺失"。

### 10.3.2 为什么常规部署不推荐 Q2_K

理解了误差结构，结论顺理成章：

1. **令牌级质量不可靠**：2-bit 码只有 4 档，尺度精度又受 4-bit 系数限制，任何对数值敏感的层都可能成为塌陷点；
2. **任务能力不可预测**：PPL 数字无法外推"再压一点还能用"，必须逐模型、逐任务验证（这一点在第 2 章"阈值效应"和第 13 章三值量化里会再次出现）；
3. **替代品往往更好**：同样是"塞进极限显存"，先把 KV cache 量化起来、换更小的模型、或者接受稍高的内存占用改用 Q3_K/Q4_K_M，通常比硬上 Q2_K 划算。

因此 Q2_K 的适用场景被压缩得很窄：**显存/存储极端受限、且明确接受质量下降**（比如纯文本生成做demo、低端设备跑测试）。它不是默认项，而是"不得已的最后一档"。

### 10.3.3 那么 Q3_K 呢？

Q3_K 的有效位宽 3.4375 bpw，处于"比 Q2_K 可用、比 Q4_K 小"的狭窄区间。它的 PPL 劣化一般弱于 Q2_K，多数场景下中等模型（>7B）可用，但仍明显差于 Q4_K。实用建议通常是：**如果你在 Q3_K 和 Q4_K 之间犹豫，内存允许就上 Q4_K**；只有在内存正好卡在 Q3_K 的体积上时才选 Q3_K。更深处还有 3-bit 但是用码本的 IQ3_XXS（第 11 章），它在同 bpw 下质量通常优于 Q3_K——这正是下一章要展开的范式切换。


## 本章小结与通往"码本量化"的桥梁

这一章可以压缩成三个事实。

**第一，Q2_K 把 256 值超级块切成 16 个 16 值子组，每个子组一个字节装 4-bit 尺度系数 + 4-bit 最小值系数，84 字节/256 值 = 2.625 bpw。** 更细的子组换来了更贴近局部的尺度，但系数从 Q4_K 的 6-bit 降到了 4-bit，2-bit 码也只剩 4 档。反量化公式是 \(y = d \cdot d_l \cdot q - d_{\min} \cdot d_h\)，位打包（4 个 2-bit 码/字节、shift 0/2/4/6）与 Q4_K（2 个 4-bit 码/字节）完全不同——**不能复用 Q4_K 的解包代码**。

**第二，Q3_K 用平面拆分实现 3-bit：低 2 位存进 `qs`、高 1 位存进 `hmask`，12 字节交错打包 16 个 6-bit 子块尺度，110 字节/256 值 = 3.4375 bpw。** "高位平面"的手法在第 8 章 Q5_K 已经见过，这里只是拆法从 4+1 变成 2+1。

**第三，Q2_K 的退化是结构性的，不是平滑劣化。** 2-bit 方案下 PPL 大幅上升之外，任务能力（尤其推理、代码、指令跟随）可能断崖式下降，且模型越小越明显；常规部署应把它当作极端受限时的最后一档。

注意到所有 K-quants 都用"尺度 + 偏移的算术反量化"：给一个块一个尺度，然后线性摊到每个码。但当位数压到 2 bit、甚至 1.5 bit 时，这种均匀网格开始失效——因为权重分布是重尾的，均匀的码字覆盖不了尖端。**如果改变规则：不用一个个去乘尺度，而是让每个值直接在"预先设计好的非线性码本"里查表呢？** 这就是第 11 章要讲的 IQ 系列，也是全书从"算术反量化"走向"码本查表"的范式转折点。