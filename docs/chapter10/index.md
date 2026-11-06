# 第10章 Q2_K 与 Q3_K：极端压缩的边界

第 7 到第 9 章讲的 Q4_K、Q5_K、Q6_K，主体位宽都在 4 bit 以上，属于"省空间但不伤筋动骨"的那一类。这一章我们要往另一个方向一路走到头：**2-bit 和 3-bit**。

先把它的吸引力讲清楚。一个 7B 模型用 Q4_K_M 存，大约是 3.8 GB。那要是每个权重只用 2 bit 呢？文件就能压到 2 GB 附近——这个体积足够塞进 8 GB 显存，或者低配手机的 RAM 里，让之前"跑不动"的设备也能装下大模型。不过，2-bit 也意味着：一个权重只剩 **4 种可能的值**（0、1、2、3，或者带符号的变体），信息上的流失是结构性的。

所以，本章要回答两个问题：**Q2_K 和 Q3_K 究竟是怎么把 2-bit / 3-bit 码装进文件里的（也就是字节布局与反量化公式）**；以及 **"再压一点"到底在哪里失效（为什么 Q2_K 不应该被当作默认的部署选择）**。第一个问题偏工程细节，第二个问题偏实践判断——两者合在一起，才构成了"极端压缩的边界"。

照例先把术语统一一下。**2-bit 码**：一个权重用一个 2 位的二进制数来表示，取值 0、1、2、3（这里是无符号整数）。**子组（subgroup）**：K-quants 里比子块还要小的组织单元，Q2_K/Q3_K 会把 256 值的超级块切成 16 个 16 值子组。**半字节（nibble）**：4 位，也就是一个字节的一半。**位平面（bit-plane）**：把同一位上的所有值单独抽出来存放，这是拆解多 bit 码时常用的手法。**PPL（perplexity，困惑度）**：衡量模型预测能力的指标，越低越好；量化之后 PPL 增加得越多，质量损失就越大。**任务退化**：不只是"输出变糙"，而是指令跟随、推理、代码等具体能力出现明显下降。


## 10.1 Q2_K：16 个 16 值子组与 2-bit 主体

### 10.1.1 为什么 2-bit 不能再使用"一个子块一个尺度"的老结构

第 7 章讲过 Q4_K 的结构：一个 256 值的超级块，带一个全局尺度 `d` 和一个全局最小值尺度 `dmin`，内部再分成 8 个子块，每个子块各带 6-bit 尺度和 6-bit 最小值。这个设计成立的前提，是"每个子块足足有 32 个权重，可以摊薄尺度参数"。

可一旦降到 2-bit，就冒出个新问题：**码本身能携带的信息量急剧下降，于是尺度信息必须更精细，否则精度会完全消失**。设想一个 32 值子块，如果仍然只有一对全局的（`d`, `dmin`），那么子块内部 32 个权重之间的差异，就只能靠 2-bit 码的 4 档来表达——这 4 档之间的空隙实在太大，重建误差会非常高。

Q2_K 的解决办法，是**把子块再切小一点**：把 256 值的超级块切成 **16 个 16 值子组**，每个子组都拥有一对 **4-bit** 的尺度系数和最小值系数（两者合起来，正好是一个字节）。子组变小之后，尺度就能更贴近局部的分布；可单个子组也只有 16 个值来摊薄参数了，所以尺度只能各用 4 bit——这其实就是"更细的粒度"与"更少的参数位宽"之间做的一次权衡。

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

三个字段依次排列，字段之间没有任何填充，所以总大小就是：

\[
4 + 16 + 64 = 84 \ \text{字节}
\]

**有效位宽**：

\[
\text{bpw} = \frac{84 \times 8}{256} = 2.625 \ \text{bit/weight}
\]

在社区资料里，偶尔会看到有人把 Q2_K 记成"2.5 bpw"或"2.5625 bpw"，那是因为有些资料把变体（比如 Q2_K_M）或者历史版本的打包方式混了进来，再加上尺度参数的字节数被舍入过。这里以主线的 `ggml-common.h` 为准：**每个 256 值超级块固定 84 字节，也就是 2.625 bpw**。

整个字节布局是这样的：

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

`scales[i]` 这单独一个字节，要同时装下子组 \(i\) 的**尺度系数**和**最小值系数**。具体的做法，是把一个字节拆成两个半字节来用：

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

- **低半字节** `scales[i] & 0x0F`：存放子组 \(i\) 的尺度系数 \(d_l\)，取值 0–15。
- **高半字节** `scales[i] >> 4`：存放子组 \(i\) 的最小值系数 \(d_h\)，取值 0–15。

要注意的是，这里系数的"门槛"和 Q4_K 那套 6-bit 的 `get_scale_min_k4` 完全不同。Q4_K 是用 12 个字节做**交错打包**，装下 8 对 6-bit 系数；Q2_K 则用 16 个字节，每个子组直接占满一整字节，**4-bit + 4-bit 各占一个半字节**——完全不需要 `get_scale_min_k4` 那种位重组。结构确实更简单了，但系数位数也更少（4-bit 对 6-bit），这是精度上的妥协。

### 10.1.4 公式与反量化循环

它的反量化公式是：

\[
y_i = d \cdot d_l \cdot q_i - d_{\min} \cdot d_h
\]

这里的 \(q_i \in \{0,1,2,3\}\) 是 2-bit 码，\(d_l = \text{scales}[j] \& 0x0F\)，\(d_h = \text{scales}[j] \gg 4\)。换成代码的视角来写，就是：

```cpp
dl = d * (scales[is] & 0x0F);   // 子组尺度
ml = dmin * (scales[is] >> 4);  // 子组最小值
y  = dl * ((q >> shift) & 3) - ml;
```

`(q >> shift) & 3` 做的事，就是从打包好的字节里取出 2-bit 码。因为每个字节要装 4 个 2-bit 码（对应 4 个不同的权重），所以 `shift` 会依次取 0、2、4、6。

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

这里的关键，是**字节下标和 shift 之间的对应关系**：一轮处理的那 32 个 `qs` 字节里，`qs[0..15]` 与 `qs[16..31]` 分别对应两个不同的子组；而每个字节里的 4 个 2-bit 码，则靠 `shift = 0, 2, 4, 6` 这四次迭代挨个取出来。三份代码的差异，只体现在"语言怎么表达位操作"上，数学上是等价的：**都是每 16 个值一个子组、逐个字节按 shift 取 2-bit 码、再乘尺度减最小值**。

### 10.1.5 打包：4 个 2-bit 码挤进一个字节

前面反量化讲的是"取出"，而量化端的方向正好相反：要想办法把 4 个 0–3 的码塞进同一个字节里。llama.cpp 的量化参考实现 `quantize_row_q2_K_ref` 的做法，是把每 128 个码分成 4 个分区（每区 32 个），再按位对齐放进 32 个字节：

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

这里的 `L` 是存着 256 个 2-bit 码的数组。可以看到，每个字节最低的 2 位、紧接着的第 3–4 位、第 5–6 位，以及最高的 2 位，分别装着 4 个不同分区的值——这正好对应反量化里的 `shift = 0,2,4,6` 那四次提取。**打包与解包是精确互逆的**，这正是第 7 章里 Q4_K 也遵守的"先对齐再打包"约定；区别在于，Q4_K 一个字节装的是 2 个 4-bit 码（shift 0 和 4），而 Q2_K 一个字节装的是 4 个 2-bit 码（shift 0/2/4/6）。


## 10.2 Q3_K：3-bit 码的平面拆分

### 10.2.1 3-bit 码拆成两个平面：低 2 位 + 高 1 位

Q3_K 是 3-bit 的（一共 8 个取值），但 8 这个数并不是 2 的整数次方里那种"能整字节契合"的数——拿一个字节去装 3-bit 码，难免会浪费。所以 Q3_K 采用了**平面拆分**的做法：把 3-bit 码的高 1 位单独抽出来集中存放，低 2 位按"每字节 4 个"来存，而高位则每 8 个码凑成一个字节。

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

算一下大小：\(32 + 64 + 12 + 2 = 110\) 字节。

**有效位宽**：

\[
\text{bpw} = \frac{110 \times 8}{256} = 3.4375 \ \text{bit/weight}
\]

note：有些文档会把 Q3_K 记成 3.5 bpw，那是把 110 按 112 对齐之后再做小数舍入的结果；这里以主线 `static_assert` 锁定的 110 字节为准，精确值就是 3.4375。

### 10.2.2 6-bit 子块尺度藏在 12 字节里

Q3_K 一共有 16 个子组，而每个子组的尺度是 **6-bit** 的（这一点比 Q2_K 的 4-bit 高，算是对"3-bit 码本身比较粗糙"的一点补偿）。16 × 6 = 96 bit = 12 字节，正好能塞进 `scales[12]` 里面。

不过，6-bit 并不像 Q2_K 那样"每个子组占一个字节"——那样就得要 16 个字节了。想用 12 个字节装下 16 个 6-bit 码，就必须交错打包。llama.cpp 的解码用了一次相当巧妙的位重组 `unpack`：把 12 个字节按 4 字节一组来处理，借对齐的便利，把 6-bit 码拼回 int8 的形式，做到"低 6 位有效、多出来的 2 位用来补偏置"，最后再整体减去 32，就得到了有符号尺度。效果等价于：**每个子组最终都拿到一个 -32..31 范围内的 int8 子块尺度**。

（在实现上，`dequantize_row_q3_K` 是借一次 12→16 字节的 `uint32_t` 位重组，把交织在一起的 6-bit 码展开成 16 个连续的 int8，再统一减 32。这次重组对多次复用来说很关键，不过它的数学核心，也就是上面那句话。第 18 章会写交叉验证脚本，跟这个位重组逐值对拍。）

### 10.2.3 反量化公式与循环

\[
y = d \cdot (s_j - 32) \cdot \big((q_2 - 4 \cdot (1 - h))\big), \quad h \in \{0,1\}
\]

换成更直接的写法，可以拆成代码里的三个部分：
1. 取出子块尺度 `dl = d * (scales[is] - 32)`；
2. 取出低位码 `lo = (qs[byte] >> shift) & 3`；
3. 取出高位码 `h = (hmask[byte] & m) ? 1 : 0`，反量化值就是 `dl * (lo - (h ? 0 : 4))`。

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

`m` 从 1 开始，在两个 128 值块里逐步左移（1→2→4→…→128），恰好把 `hmask` 每个字节的 8 个位遍历一遍。`(hm[l] & m) ? 0 : 4` 这句的含义是：**高位平面为 1 时，低位码 (0..3) 直接原样使用；高位平面为 0 时，就让整体减 4，落进负半轴 (-4..-1)**。靠着这一点，3-bit 码才得以表示 -4..3 这 8 个值。而 `unpack_scales` 那次位重组，就是 10.2.2 里写的"低 4 位 + 高 2 位"的等效实现：它跨过字节边界直接去取 6-bit 码，然后统一减 32，转成有符号数。

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

三份实现共享的是同一个核心：**子块尺度存成有符号的 6-bit（要减 32），3-bit 码则拆成低 2 位（qs）加上高 1 位（hmask），当高位为 0 时，值域就落到负半区**。这个"高位平面"的思路，在第 8 章 Q5_K 里已经见过一次了；这里手法相同，只是拆分比例从"4+1"换成了"2+1"。


## 10.3 极端压缩的边界：Q2_K 为什么不该作为默认

### 10.3.1 Q2_K 的 PPL 劣化不是平滑曲线

K-quants 从 Q4_K 到 Q5_K 再到 Q6_K，质量是**逐档平滑回升**的：每多用 1 bit，PPL 就改善一点，没有骤变。可一旦下到 Q2_K，情况就不一样了。

根据实测得到的印象（以量化模型的 PPL 相对 F16 基准来看）：Q4_K 大约是 +0.8%，Q3_K 大概在此基础上再翻一倍，而 **Q2_K 那些可以报告的 PPL，往往要比 Q4_K 再高出 2–3 倍**，并且不同的评测集给出的数字还会差得很多。不过，真正关键的其实不是绝对数字，而是**任务表现上的断崖**：

- 简单的分类、短问答、检索类任务：退化相对温和（这类任务对"精确权重值"的依赖本来就比较低）；
- 多步推理、数学、代码生成、带约束的指令跟随：退化很明显，有时甚至表现得几乎不能用；
- 模型越大（70B+），退化越能忍受；模型越小（7B–13B），退化越致命——因为大模型本身有更多冗余，可以去吸收量化噪声。

用一句话概括：**Q2_K 的劣化并不是"在 Q4_K 的基础上再多一点"，而是"能力属性"层面的下降**。它更像是从"个体数值上的误差"变成了"结构性的信息缺失"。

### 10.3.2 为什么常规部署不推荐 Q2_K

理解了误差的结构，结论也就顺理成章了：

1. **令牌级质量不可靠**：2-bit 码只有 4 档，尺度精度又受 4-bit 系数限制，所以任何对数值敏感的层，都可能成为塌陷点；
2. **任务能力不可预测**：光看 PPL 数字，是无法外推出"再压一点还能用"的，必须逐模型、逐任务去验证（这一点在第 2 章"阈值效应"和第 13 章三值量化里还会再次出现）；
3. **替代品往往更好**：同样是"塞进极限显存"，先把 KV cache 量化起来、换一个更小的模型，或者接受稍高一点的内存占用改用 Q3_K/Q4_K_M，通常都比硬上 Q2_K 划算。

所以说，Q2_K 的适用场景被压缩得相当窄：**显存/存储极端受限、并且明确接受质量下降**（比如拿来做纯文本生成的 demo，或者在低端设备上跑测试）。它从来不是默认项，而是"不得已时的最后一档"。

### 10.3.3 那么 Q3_K 呢？

Q3_K 的有效位宽是 3.4375 bpw，落在"比 Q2_K 可用、又比 Q4_K 小"这个狭窄的区间里。它的 PPL 劣化一般比 Q2_K 弱，多数场景下中等规模模型（>7B）都能用，但仍然明显差于 Q4_K。比较实用的建议通常是：**如果你在 Q3_K 和 Q4_K 之间犹豫，只要内存允许，就上 Q4_K**；只有内存正好卡在 Q3_K 这个体积上时，才退而求其次选 Q3_K。再往深处，还有同样是 3-bit、但改用了码本的 IQ3_XXS（第 11 章），它在相同的 bpw 下质量通常优于 Q3_K——而这，正是下一章要展开的范式切换。


## 本章小结与通往"码本量化"的桥梁

这一章的内容，可以压缩成三个事实。

**第一，Q2_K 把 256 值的超级块切成 16 个 16 值子组；每个子组用一个字节，装下 4-bit 尺度系数和 4-bit 最小值系数，于是 84 字节/256 值 = 2.625 bpw。** 更细的子组，换来的是更贴近局部的尺度；代价则是系数从 Q4_K 的 6-bit 降到了 4-bit，2-bit 码也只剩下 4 档。反量化公式是 \(y = d \cdot d_l \cdot q - d_{\min} \cdot d_h\)，位打包方式（4 个 2-bit 码/字节、shift 0/2/4/6）和 Q4_K（2 个 4-bit 码/字节）完全不同——所以**不能复用 Q4_K 的解包代码**。

**第二，Q3_K 靠平面拆分来实现 3-bit：低 2 位存进 `qs`，高 1 位存进 `hmask`；16 个 6-bit 子块尺度用 12 个字节交错打包，于是 110 字节/256 值 = 3.4375 bpw。** 这种"高位平面"的手法，在第 8 章的 Q5_K 里已经见过，这里只是拆法从 4+1 变成了 2+1。

**第三，Q2_K 的退化是结构性的，而不是平滑的劣化。** 在 2-bit 方案下，除了 PPL 会大幅上升，任务能力（尤其是推理、代码、指令跟随）还可能断崖式下降，而且模型越小越明显；所以在常规部署里，应该把它当作极端受限时的最后一档。

回过头看，所有 K-quants 用的都是"尺度 + 偏移的算术反量化"：给一个块配一个尺度，然后线性地摊到每一个码上。然而当位数被压到 2 bit、甚至 1.5 bit 时，这种均匀网格就开始失效了——因为权重分布本身是重尾的，均匀铺开的码字根本覆盖不到尖峰那一端。**那要是换一套规则呢：不再一个个去乘尺度，而是让每个值直接在一份"预先设计好的非线性码本"里查表？** 这就是第 11 章要讲的 IQ 系列，也是全书从"算术反量化"走向"码本查表"的范式转折点。