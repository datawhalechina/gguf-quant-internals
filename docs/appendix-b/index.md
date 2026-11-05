# 附录 B 字节布局图集

本附录把全书 16 个核心格式的**逐字节布局**集中展示：每个格式给一张 ASCII 布局图、C 结构体（与 llama.cpp `ggml-common.h` 一致）、Python `struct` 描述。所有字节数经 `static_assert` 与 2026-10 主线源码核实（见 `asset/facts-verified.md`、`asset/ggml-common.h`）。

通用约定：

- `d` = FP16 位模式（2 字节，小端），`dm` = `d`+`m`（或 `d`+`dmin`）共 4 字节，`d`(f32) = 4 字节 FP32。
- Python 半精度用小端格式 `'<e'`；多字节整数用 `'<H'`（u16）、`'<I'`（u32）、`'<Q'`（u64）。
- 每张图宽度≤60 字符，偏移以块起始为 0。

---

## B.1 Q4_0（type 2，32 值 / 18 字节 / 4.5 bpw）

```
偏移   0    1              17
     ┌─────┬─────────────────────────────┐
     │  d  │        qs[0..15]            │
     │ 2B  │        16 B                  │
     │ f16 │  32 个 4-bit 码（nibble）     │
     └─────┴─────────────────────────────┘
```

- 反量化：\(y_i = d\cdot(q_i - 8)\)。
- nibble 打包：`qs[j]&0xF` = 块内第 `j` 值，`qs[j]>>4` = 第 `j+16` 值。

```cpp
#define QK4_0 32
typedef struct {
    ggml_half d;              // 2 字节
    uint8_t   qs[QK4_0 / 2];  // 16 字节
} block_q4_0;
static_assert(sizeof(block_q4_0) == 18, "wrong q4_0 block size/padding");
```

```python
import struct
# 整块解析：'<e' 尺度 + 16 字节 qs
def parse_q4_0(block: bytes):
    assert len(block) == 18
    d  = struct.unpack_from('<e', block, 0)[0]
    qs = block[2:18]
    return d, qs
```

## B.2 Q4_1（type 3，32 值 / 20 字节 / 5.0 bpw）

```
偏移   0    1    2    3              19
     ┌─────┬─────┬─────────────────────────────┐
     │  d  │  m  │       qs[0..15]             │
     │ 2B  │ 2B  │        16 B                 │
     └─────┴─────┴─────────────────────────────┘
```

- 反量化：\(y_i = d\cdot q_i + m\)。

```cpp
#define QK4_1 32
typedef struct {
    ggml_half d;              // 尺度
    ggml_half m;              // 最小值
    uint8_t   qs[QK4_1 / 2];  // 16 字节
} block_q4_1;
static_assert(sizeof(block_q4_1) == 20, "wrong q4_1 block size/padding");
```

```python
def parse_q4_1(block: bytes):
    assert len(block) == 20
    d = struct.unpack_from('<e', block, 0)[0]
    m = struct.unpack_from('<e', block, 2)[0]
    qs = block[4:20]
    return d, m, qs
```

## B.3 Q5_0（type 6，32 值 / 22 字节 / 5.5 bpw）

```
偏移   0    1       4  5               21
     ┌─────┬──────────┬─────────────────────────┐
     │  d  │ qh[0..3] │       qs[0..15]         │
     │ 2B  │  4 B     │       16 B              │
     │ f16 │ 5-th 位  │ 32 个 4-bit 码           │
     └─────┴──────────┴─────────────────────────┘
```

- 5-bit 码 = `qs` 低 4 位 + `qh` 对应位（第 8 章详述）。

```cpp
#define QK5_0 32
typedef struct {
    ggml_half d;              // 2 字节
    uint8_t   qh[4];          // 第 5 bit（4 字节）
    uint8_t   qs[QK5_0 / 2];  // 16 字节
} block_q5_0;
static_assert(sizeof(block_q5_0) == 22, "wrong q5_0 block size/padding");
```

```python
def parse_q5_0(block: bytes):
    assert len(block) == 22
    d  = struct.unpack_from('<e', block, 0)[0]
    qh = block[2:6]
    qs = block[6:22]
    return d, qh, qs
```

## B.4 Q5_1（type 7，32 值 / 24 字节 / 6.0 bpw）

```
偏移   0    1    2    3       6  7               23
     ┌─────┬─────┬──────────┬─────────────────────────┐
     │  d  │  m  │ qh[0..3] │       qs[0..15]         │
     │ 2B  │ 2B  │  4 B     │       16 B              │
     └─────┴─────┴──────────┴─────────────────────────┘
```

```cpp
#define QK5_1 32
typedef struct {
    ggml_half d;              // 2 字节
    ggml_half m;              // 2 字节
    uint8_t   qh[4];          // 第 5 bit
    uint8_t   qs[QK5_1 / 2];  // 16 字节
} block_q5_1;
static_assert(sizeof(block_q5_1) == 24, "wrong q5_1 block size/padding");
```

## B.5 Q8_0（type 8，32 值 / 34 字节 / 8.5 bpw）

```
偏移   0    1                33
     ┌─────┬─────────────────────────────┐
     │  d  │     qs[0..31]  (int8)       │
     │ 2B  │         32 B                │
     └─────┴─────────────────────────────┘
```

- 反量化：\(y_i = d\cdot q_i\)（int8 有符号码，无需减偏移）。

```cpp
#define QK8_0 32
typedef struct {
    ggml_half d;          // 2 字节
    int8_t   qs[QK8_0];   // 32 字节
} block_q8_0;
static_assert(sizeof(block_q8_0) == 34, "wrong q8_0 block size/padding");
```

```python
def parse_q8_0(block: bytes):
    assert len(block) == 34
    d  = struct.unpack_from('<e', block, 0)[0]
    qs = struct.unpack_from('<32b', block, 2)
    return d, qs
```

---

## B.6 Q2_K（type 10，256 值 / 84 字节 / 2.625 bpw）

注意：**`d` 与 `dmin` 在结构体尾部**（与 Q4_K 的前置布局不同）。

```
偏移   0              15   16              79  80    81   82   83
     ┌───────────────┬────────────────────┬─────┬─────┐
     │ scales[0..15] │    qs[0..63]       │  d  │dmin │
     │ 16 B          │    64 B            │ 2B  │ 2B  │
     │ 每字节:       │  2-bit 码 4 个/字节 │     │     │
     │ 低4位=尺度系数 │                     │     │     │
     │ 高4位=最小值系数│                     │     │     │
     └───────────────┴────────────────────┴─────┴─────┘
```

- 16 子组 × 16 值；`sc = scales[is]`，`dl = d*(sc&0xF)`，`ml = dmin*(sc>>4)`。
- 2-bit 码：每组 4 字节，`shift` 依子组 0/2/4/6 步进。
- 公式：\(y = d\cdot(\text{sc}\&0xF)\cdot((q\gg\text{shift})\&3) - d_{\min}\cdot(\text{sc}\gg4)\)。

```cpp
typedef struct {
    uint8_t   scales[QK_K / 16]; // 16 字节
    uint8_t   qs[QK_K / 4];      // 64 字节
    ggml_half d;                 // 尾部
    ggml_half dmin;              // 尾部
} block_q2_K;
static_assert(sizeof(block_q2_K) == 84, "wrong q2_K block size/padding");
```

```python
def parse_q2_k(block: bytes):
    assert len(block) == 84
    scales = block[0:16]
    qs     = block[16:80]
    d      = struct.unpack_from('<e', block, 80)[0]
    dmin   = struct.unpack_from('<e', block, 82)[0]
    return scales, qs, d, dmin
```

## B.7 Q3_K（type 11，256 值 / 110 字节 / 3.4375 bpw）

注意：`hmask` 在最前、`d` 在尾部。

```
偏移   0              31   32              95  96          107  108  109
     ┌───────────────┬────────────────────┬──────────────┬─────┬─────┐
     │ hmask[0..31]  │    qs[0..63]       │ scales[0..11]│     │  d  │
     │ 32 B          │    64 B            │ 12 B (int8)  │     │ 2B  │
     │ 每字节 8 个高 1 位│ 2-bit 低 2 位     │ 尺度的 6-bit  │     │     │
     └───────────────┴────────────────────┴──────────────┴─────┴─────┘
```

- 3-bit 码 = `qs` 低 2 位 + `hmask` 高 1 位（hmask 位为 1 表示 −4 偏移）。
- 尺度：`scales[12]` 解出 16 个 int8，`dl = d*(scales[is]-32)`。
- 公式：\(y = d_l\cdot\big((q\&3) - (\mathrm{hm}\&m ? 0 : 4)\big)\)。

```cpp
typedef struct {
    uint8_t   hmask[QK_K / 8]; // 32 字节
    uint8_t   qs[QK_K / 4];    // 64 字节
    uint8_t   scales[12];      // 12 字节
    ggml_half d;               // 尾部
} block_q3_K;
static_assert(sizeof(block_q3_K) == 110, "wrong q3_K block size/padding");
```

```python
def parse_q3_k(block: bytes):
    assert len(block) == 110
    hmask  = block[0:32]
    qs     = block[32:96]
    scales = block[96:108]
    d      = struct.unpack_from('<e', block, 108)[0]
    return hmask, qs, scales, d
```

## B.8 Q4_K（type 12，256 值 / 144 字节 / 4.5 bpw）

```
偏移   0    1    2    3              15               143
     ┌─────┬─────┬────────────────┬─────────────────────────────┐
     │  d  │dmin │  scales[0..11] │         qs[0..127]          │
     │ 2B  │ 2B  │  12 B          │         128 B               │
     │     │     │ 6-bit s/m 交错 │  8 子块 × 16B，nibble 打包   │
     └─────┴─────┴────────────────┴─────────────────────────────┘
```

- 8 子块 × 32 值；`scales[12]` 交错存 8 个 6-bit 尺度 + 8 个 6-bit 最小值（`get_scale_min_k4`，见附录 A 与第 7 章）。
- 反量化：\(y_i = d\cdot s_j\cdot q_i - d_{\min}\cdot m_j\)。
- 4-bit 码：每 64 值一段，段内 `qs[l]&0xF` 与 `qs[l]>>4` 分属两个子块。

```cpp
typedef struct {
    ggml_half d;              // 2 字节
    ggml_half dmin;           // 2 字节
    uint8_t   scales[K_SCALE_SIZE]; // 12 字节
    uint8_t   qs[QK_K / 2];   // 128 字节
} block_q4_K;
static_assert(sizeof(block_q4_K) == 144, "wrong q4_K block size/padding");
```

```python
def parse_q4_k(block: bytes):
    assert len(block) == 144
    d      = struct.unpack_from('<e', block, 0)[0]
    dmin   = struct.unpack_from('<e', block, 2)[0]
    scales = block[4:16]
    qs     = block[16:144]
    return d, dmin, scales, qs
```

## B.9 Q5_K（type 13，256 值 / 176 字节 / 5.5 bpw）

```
偏移   0    1    2    3              15   16          47           175
     ┌─────┬─────┬────────────────┬──────────┬────────────────────────┐
     │  d  │dmin │  scales[0..11] │ qh[0..31]│      qs[0..127]       │
     │ 2B  │ 2B  │  12 B          │ 32 B     │      128 B            │
     └─────┴─────┴────────────────┴──────────┴────────────────────────┘
```

- = Q4_K 加一个 1-bit 高平面 `qh[]`；5-bit 码 = `qs` 低 4 位 + `qh` 对应位。
- `qh` 位序：`u1=1,u2=2`，随子块步进 `<<=2`（第 8 章）。

```cpp
typedef struct {
    ggml_half d;              // 2
    ggml_half dmin;           // 2
    uint8_t   scales[K_SCALE_SIZE]; // 12
    uint8_t   qh[QK_K / 8];   // 32（高 1 位平面）
    uint8_t   qs[QK_K / 2];   // 128
} block_q5_K;
static_assert(sizeof(block_q5_K) == 176, "wrong q5_K block size/padding");
```

## B.10 Q6_K（type 14，256 值 / 210 字节 / 6.5625 bpw）

```
偏移   0             127  128            191  192            207  208  209
     ┌──────────────┬──────────────┬──────────────┬─────┬─────┐
     │ ql[0..127]   │ qh[0..63]    │ scales[0..15]│     │  d  │
     │ 128 B        │ 64 B         │ 16 B (int8)  │     │ 2B  │
     │ 每字节 2 个低 4 位│ 每字节 4 个高 2 位 │ 8-bit 子块尺度 │     │     │
     └──────────────┴──────────────┴──────────────┴─────┴─────┘
```

- 16 子块 × 16 值；6-bit 码 = `ql` 低 4 位 + `qh` 高 2 位。
- 公式：\(y_i = d\cdot \mathrm{scales}[j]\cdot(q_i - 32)\)，无 `dmin`。

```cpp
typedef struct {
    uint8_t   ql[QK_K / 2];      // 128
    uint8_t   qh[QK_K / 4];      // 64
    int8_t    scales[QK_K / 16]; // 16（int8）
    ggml_half d;                 // 尾部
} block_q6_K;
static_assert(sizeof(block_q6_K) == 210, "wrong q6_K block size/padding");
```

```python
def parse_q6_k(block: bytes):
    assert len(block) == 210
    ql     = block[0:128]
    qh     = block[128:192]
    scales = struct.unpack_from('<16b', block, 192)
    d      = struct.unpack_from('<e', block, 208)[0]
    return ql, qh, scales, d
```

---

## B.11 IQ1_S（type 19，256 值 / 50 字节 / 1.5625 bpw）

**注意**：细纲记载为 74 字节/1.5 bpw，主线实际为 **50 字节/1.5625 bpw**（见 facts-verified）。

```
偏移   0    1             32   33                   48  49
     ┌─────┬──────────────┬────────────────────────┬───┐
     │  d  │  qs[0..31]   │  qh[0..7]  (uint16)    │   │
     │ 2B  │ 32 B         │  16 B                  │   │
     │     │ 8 值/组 索引低8位│ 每组尺度(高12位)+delta(bit15) │   │
     └─────┴──────────────┴────────────────────────┴───┘
```

- 8 组 × 32 值；`dl = d*(2*((qh[ib]>>12)&7)+1)`；`delta = qh[ib]&0x8000 ? −0.125 : +0.125`。
- 每组内 4 个 8 值子组，索引 `qs[l] | (((qh[ib]>>3l)&7)<<8)` 查 `iq1s_grid`（2048 项）。

```cpp
typedef struct {
    ggml_half d;               // 2
    uint8_t   qs[QK_K / 8];    // 32
    uint16_t  qh[QK_K / 32];   // 8 × 2 = 16
} block_iq1_s;
static_assert(sizeof(block_iq1_s) == 50, "wrong iq1_s block size/padding");
```

```python
def parse_iq1_s(block: bytes):
    assert len(block) == 50
    d  = struct.unpack_from('<e', block, 0)[0]
    qs = block[2:34]
    qh = struct.unpack_from('<8H', block, 34)   # 8 个 uint16
    return d, qs, qh
```

## B.12 IQ2_XXS（type 16，256 值 / 66 字节 / 2.0625 bpw）

```
偏移   0    1               64   65
     ┌─────┬────────────────┬───┐
     │  d  │ qs[0..31]      │   │
     │ 2B  │ 32 × uint16    │   │
     │     │ = 64 B          │   │
     └─────┴────────────────┴───┘
```

- 32 值一组；8 字节中第 2 个 uint32 的高 4 位 = 子组尺度，低 28 位 = 4 个 8 值子组的符号位。
- 每 8 值查 `iq2xxs_grid`（索引 `aux8[l]`）。

```cpp
typedef struct {
    ggml_half d;              // 2
    uint16_t  qs[QK_K / 8];   // 32 × 2 = 64
} block_iq2_xxs;
static_assert(sizeof(block_iq2_xxs) == 66, "wrong iq2_xxs block size/padding");
```

```python
def parse_iq2_xxs(block: bytes):
    assert len(block) == 66
    d  = struct.unpack_from('<e', block, 0)[0]
    qs = struct.unpack_from('<32H', block, 2)
    return d, qs
```

## B.13 IQ3_XXS（type 18，256 值 / 98 字节 / 3.0625 bpw）

**注意**：细纲写 3.4375 bpw，主线实际 IQ3_XXS = **3.0625 bpw**；3.4375 是 IQ3_S（见附录 A）。

```
偏移   0    1              96  97
     ┌─────┬──────────────┬───┐
     │  d  │ qs[0..95]    │   │
     │ 2B  │ 96 B         │   │
     │     │ 含 4 值表项索引│   │
     │     │ + 符号位/子组尺度│   │
     └─────┴──────────────┴───┘
```

- 每 32 值组：后 4 字节 = 子组尺度（高 4 位）+ 4 组符号位；前 8 字节 = 8 个 `iq3xxs_grid` 索引（每项 4 值）。

```cpp
typedef struct {
    ggml_half d;              // 2
    uint8_t   qs[3 * QK_K / 8]; // 96
} block_iq3_xxs;
static_assert(sizeof(block_iq3_xxs) == 98, "wrong iq3_xxs block size/padding");
```

## B.14 IQ4_NL（type 20，32 值 / 18 字节 / 4.5 bpw）

```
偏移   0    1              17
     ┌─────┬─────────────────────┐
     │  d  │    qs[0..15]        │
     │ 2B  │ 16 B                │
     │     │ 4-bit 索引 → kvalues_iq4nl │
     └─────┴─────────────────────┘
```

- 布局与 Q4_0 相同，但反量化是查**非线性**格点：`y = d*kvalues_iq4nl[q]`，`q` 为 4-bit 索引（第 11 章）。

```cpp
#define QK4_NL 32
typedef struct {
    ggml_half d;             // 2
    uint8_t   qs[QK4_NL / 2]; // 16
} block_iq4_nl;
static_assert(sizeof(block_iq4_nl) == 18, "wrong iq4_nl block size/padding");
```

---

## B.15 STQ1_0（fork/PR，256 值 / 42 字节 / 1.3125 bpw）

> ⚠ 此格式**不在 llama.cpp 主线**，以下布局为细纲口径，落地前必须以实际 fork/PR 文档复核（第 13 章）。

```
偏移   0                   31  32        39  40  41
     ┌────────────────────┬──────────┬─────┬─────┐
     │  qs[0..31]         │ sign[8]  │  d  │     │
     │ 32 B               │ 8 B      │ 2B  │     │
     │ 4-bit 码本索引      │ 1 位符号/权重 │ f16 │     │
     └────────────────────┴──────────┴─────┴─────┘
```

- 3:4 稀疏：每 4 权重 1 零、3 个 ±d；32 种模式用 5 bit（4-bit 码本索引 + 1-bit 符号）。
- stride-16 分组，便于 NEON `vqtbl2q`/`vdotq_s32` 无重排解码。

## B.16 PTQ1_0（fork/PR，128 值 / 28 字节 / 1.75 bpw）

> ⚠ 同上，不在主线；且 **Hadamard 折叠是必需变换**，缺失则 logits 完全错误（第 13 章）。

```
偏移   0                23  24    25  26  27
     ┌──────────────────┬────────┬─────┬─────┐
     │  base3[0..23]    │ rem[2] │  d  │     │
     │ 24 B             │ 2 B    │ 2B  │     │
     │ 5 trits/字节      │ 4 trits/字节 │ f16 │     │
     └──────────────────┴────────┴─────┴─────┘
```

- 稠密三值，base-3 编码覆盖 128 trits（24×5 + 2×4 = 128）。
- 解码：trit ∈ {−1, 0, +1}，\(y_i = d\cdot \mathrm{trit}_i\)。

---

## B.17 附录说明

1. **结构体顺序**是"磁盘顺序"：Q2_K/Q3_K/Q6_K 的 `d`（及 `dmin`）在尾部，Q4_K/Q5_K 的 `d`/`dmin` 在头部——写解析器时务必按表、不要想当然。
2. **static_assert** 是格式正确性的第一道防线（第 3、16 章）；Python 侧对应的是 `assert len(block) == N`。
3. STQ1_0/PTQ1_0 字节数为细纲口径；主线中对应位置已被 Q1_0/Q2_0（type 41/42）占用，这正是第 13/19/20 章讨论的 id 冲突实例。