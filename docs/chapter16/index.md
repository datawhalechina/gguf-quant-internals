# 第16章 llama.cpp 量化代码架构

前面几章，我们一直在"读"代码：看 `block_q4_K` 怎么布局、`get_scale_min_k4` 怎么解包、`dequantize_row_iq1_s` 怎么查表。这一章要换个视角，从"**写**代码"的角度来解剖 llama.cpp——想象一下，你要往这个项目里加一个全新的量化格式，该改哪些文件？每改一处，又会发生什么？

这可不是一个虚构的问题。第 13 章的三值格式 STQ1_0、PTQ1_0 就是活生生的例子：它们先在自己的 fork 或 PR 里诞生，然后要经过一套固定的"注册流程"，才可能被主线接受。搞懂这套流程，你不仅能读量化代码，还能安全地扩展它。

先给你一个全景式的概览。在 llama.cpp 里加一个新的量化格式，几乎必然要动到三个核心文件，外加一个测试目标：

```
想新增一个量化格式，需要动的东西

  ┌─────────────────────────────────────────────────────────┐
  │ ① ggml.h       枚举 type id、更新 GGML_TYPE_COUNT、      │
  │                ggml_ftype                               │
  │ ② ggml-common.h 定义块结构体 block_xxx + static_assert  │
  │ ③ ggml.c       在 type_traits[] 注册：块大小/类型大小/   │
  │                反量化函数指针                            │
  │ ④ ggml-quants.c 实现 quantize_row_* / dequantize_row_*  │
  │                （参考版本，保证正确性）                   │
  │ ⑤ tests        test-quantize-fns 通过（参考实现交叉验证）│
  │ ⑥ 各后端        CPU SIMD → Metal → CUDA/Vulkan          │
  └─────────────────────────────────────────────────────────┘
```

照例先把几个术语统一一下。**type id**：`ggml_type` 枚举里给每种格式分配的整数编号，GGUF 文件里每个张量的 `type` 字段就是它（见 3.2 节）。**`GGML_TYPE_COUNT`**：枚举的总长度，用来定义 `type_traits` 数组究竟有多大。**分派（dispatch）**：根据 type id 挑出合适的函数去执行，过程有点像查表。**参考实现（reference implementation）**：最简单、只求正确的实现，完全不追求速度，是验证 SIMD 优化有没有算错的基准。**`vec_dot`**：向量点积的缩写，也是量化矩阵乘的核心算子。


## 16.1 类型注册的三个文件

### 16.1.1 `ggml.h`：type id 与 `GGML_TYPE_COUNT`

第一个要改的文件，是头文件 `ggml.h`，因为**一切的入口，都是"这个格式叫什么编号"**。编号要是没注册，后面所有的表都无从查起。

`ggml.h` 里有一整段枚举，把所有类型逐项列出来。我们看一段真实的 llama.cpp 主线代码（2026 年版本）：

```c
// ggml/include/ggml.h（节选）
typedef enum ggml_type {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    // GGML_TYPE_Q4_2 = 4, support has been removed
    // GGML_TYPE_Q4_3 = 5, support has been removed
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    // ... IQ 系列、I8/I16/I32/I64、F64、BF16 ...
    GGML_TYPE_COUNT   = 43,
} ggml_type;
```

有两处细节需要留意：

**第一，被移除的格式，编号也不会复用。** `Q4_2 = 4`、`Q4_3 = 5` 的注释里写着 "support has been removed"，可 4、5 这两个编号，依然被注释"占着位"留着。为什么？因为**旧的 GGUF 文件里，可能有张量正好带着编号 4 或 5**；读取端看到它，必须能认出"这是已移除的类型"并报错，而不能误当成别的格式。一旦复用了编号，老文件的解读就会彻底错乱。这就是 3.4 节"type id 兼容性"在源码里落地的样子：编号是一种**历史契约**，只能追加，不能回收。

**第二，`GGML_TYPE_COUNT` 决定所有数组的大小。** 在 `ggml.c` 里，`type_traits` 是这样定义的：

```c
static const struct ggml_type_traits type_traits[GGML_TYPE_COUNT] = { ... };
```

`type_traits[]` 数组的长度，正好就是 `GGML_TYPE_COUNT`。新增一个格式，意味着枚举里多一项，`GGML_TYPE_COUNT` 会自动变大一格（它总是待在枚举末尾），数组也跟着扩容。反过来，如果你忘了在枚举里加类型，却直接在 `type_traits[]` 里写 `[GGML_TYPE_MINI1] = {...}`，编译就会因为"下标是未定义枚举值"而报错——**枚举就是强制注册的第一道闸门**。

除了 `ggml_type`，还有第二份枚举需要跟着同步：`ggml_ftype`。它定义在 `ggml.h`（或者 model loader 相关的头文件里），管的是"文件级量化策略"的编号——比如 `MOSTLY_Q4_K_M = 15`（见 3.2.3 节，`general.file_type` 字段就是它）。想新增一个策略（比如"主体 Q3_K + 输出 Q6_K"），就需要在 `ggml_ftype` 里分配一个新编号，并且在 `llama-quantize` 的命令行枚举里挂上对应的字符串。

### 16.1.2 `ggml-common.h`：块结构体与 `static_assert`

编号注册好了，接下来要定义的是"这个格式的块到底长什么样"。块的结构体定义在 `ggml-common.h` 里，这正是第 3 章反复强调的 POD 约束落地的地方。看真实的 `block_q4_K`：

```c
// ggml/src/ggml-common.h（节选）
typedef struct {
    GGML_EXTENSION union {
        struct {
            ggml_half d;    // 超级块尺度
            ggml_half dmin; // 超级块最小值尺度
        } GGML_COMMON_AGGR_S;
        ggml_half2 dm;
    } GGML_COMMON_AGGR_U;
    uint8_t scales[K_SCALE_SIZE]; // 12 字节：8 个子块的 6-bit 尺度与最小值（交错）
    uint8_t qs[QK_K/2];           // 128 字节：4-bit 码
} block_q4_K;
static_assert(sizeof(block_q4_K) == 2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2, "wrong q4_K block size/padding");
```

这里有两个写法值得解释一下：

- **`GGML_EXTENSION union { ... }`**：`d` 和 `dmin` 这两个 `ggml_half` 被包进了一个 union，整体又声明成 `ggml_half2`（2 字节 × 2）。这样做的目的，是允许 SIMD 后端一次读 4 个字节，把它们当成一个向量加载。`GGML_EXTENSION` 只是兼容 MSVC 的宏，不影响布局。
- **`static_assert`**：编译期的断言。`sizeof(block_q4_K)` 必须精确等于 `2+2+12+128=144`。第 3 章说过，编译器可能为了对齐而填充字节；这个断言，就把"块大小错了"从运行时崩溃，提前到了编译失败。

**为什么结构体必须定义在这里，而不是藏在实现文件里？** 因为 `ggml-common.h` 会被**所有后端**包含：CPU 的 `ggml-quants.c`、CUDA 的 `ggml-cuda.cu`、Metal 的 `ggml-metal.m`……每个后端，都要按同一份字节布局来解释数据。如果把这个结构体塞进某个后端的私有文件里，其他后端就无从得知磁盘上的那 144 个字节该怎么拆。`static_assert` 放在这里，等于对全世界宣布："我的格式就是这 144 个字节，谁都不许往里塞填充。"

### 16.1.3 `ggml.c`：`type_traits` 分派表

第三个文件是 `ggml.c`，那张决定一切的 `type_traits[]` 表就躺在里面。看真实的注册条目：

```c
// ggml/src/ggml.c（节选）
static const struct ggml_type_traits type_traits[GGML_TYPE_COUNT] = {
    [GGML_TYPE_Q4_0] = {
        .type_name      = "q4_0",
        .blck_size      = QK4_0,              // 每块 32 个元素
        .type_size      = sizeof(block_q4_0), // 每块 18 字节
        .is_quantized   = true,
        .to_float       = (ggml_to_float_t) dequantize_row_q4_0,
        .from_float_ref = (ggml_from_float_t) quantize_row_q4_0_ref,
    },
    [GGML_TYPE_F32] = {
        .type_name      = "f32",
        .blck_size      = 1,
        .type_size      = sizeof(float),
        .is_quantized   = false,
    },
    // ...
};
```

这张表把 type id 映射成了五样东西：

- `type_name`：字符串形式的名字，`llama-quantize` 的命令行、各种错误信息，全靠它。
- `blck_size`：每个块覆盖多少个浮点值。
- `type_size`：每个块占多少字节。**`bpw = type_size × 8 / blck_size`**，比如 Q4_0 就是 `18×8/32 = 4.5`（这一路我们从第 4 章算到了现在）。
- `to_float`：反量化函数指针，负责把"量化块"变成"浮点行"。
- `from_float_ref`：参考量化函数指针，负责把"浮点行"变成"量化块"。

**缺少条目会发生什么？** `type_traits[type]` 走的是纯数组下标访问，没有任何"越界检查"。如果某个 type id 在枚举里存在、却没有对应的条目，下标就会落进数组里那个默认的零值条目：`type_size = 0`、`to_float = NULL`。运行时，要么算出 0 字节这种荒谬的尺寸，要么干脆调用空指针直接崩溃。这正是 3.4 节"GGUF 可读 ≠ 构建可分派"在源码层面的原因——**一个构建能不能跑某个格式，就看它 `type_traits[]` 里到底有没有那一行。**

用一张图总结注册流程：

```
新增格式的注册流程（三个文件一条链）

  ggml.h             ggml-common.h          ggml.c
  ┌──────────────┐   ┌──────────────────┐   ┌─────────────────────┐
  │ 枚举加一项    │   │ 定义 block_mini1  │   │ type_traits[] 里     │
  │ GGML_TYPE_   │──►│ + static_assert  │──►│ [GGML_TYPE_MINI1] = │
  │ MINI1        │   │ 锁定大小          │   │ { .blck_size,       │
  │ COUNT++      │   │                  │   │   .type_size,       │
  └──────────────┘   └──────────────────┘   │   .to_float, ... }   │
                                            └─────────────────────┘
  编号（身份证）       布局（长什么样）         行为（怎么解码）
```

这三者缺一不可：没编号，别人认不出来；没布局，字节没法拆分；没注册，运行时就没有函数可调。


## 16.2 量化与反量化函数契约

### 16.2.1 `quantize_row_*` 与 `dequantize_row_*`

注册表里的函数指针，指向的都是 `ggml-quants.c` 里的实现。这一节，我们先说最基础的契约：两个方向的"行"函数。

看真实的 Q4_K 反量化函数（在 `ggml-quants.c` 里）：

```c
void dequantize_row_q4_K(const block_q4_K * x, float * y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int64_t i = 0; i < nb; i++) {
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

函数名里的那个 `row`，暗示着它的调用约定：**一次处理一行（row）**。签名统一是 `(const block_X * x, float * y, int64_t k)`，其中：

- `x`：指向一块一块的量化数据。
- `y`：输出缓冲，用来接收解出来的浮点值。
- `k`：**总元素数**（注意不是块数），并且必须满足 `k % QK == 0`——所以每个函数开头都必有 `assert(k % QK_K == 0)`。不够一块的尾巴，是不允许存在的；这是"块"这种粒度带来的硬约束（第 4 章讲过，所有张量的元素数，都必须是块大小的整数倍）。

反方向则是 `quantize_row_q4_K_ref`（名字里的 `_ref`，表示这是参考实现）：

```c
void quantize_row_q4_K_ref(const float * x, block_q4_K * y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;
    // ... 逐块估计子块尺度/最小值、量化 4-bit 码 ...
}
```

**参考实现和 SIMD 优化实现，到底是什么关系？** 每个格式通常都有至少两个版本：一个"参考"版（`*_ref`，清晰、慢、保证正确），一到多个"优化"版（AVX2、NEON，快、难读）。优化版不能"自己单独正确"——它必须和参考版产出**完全一致**的字节，否则就是解包打错了。这份一致性，靠测试来保证（16.3.1 会讲 `test-quantize-fns`）。一句话总结：**参考实现是正确性的锚，SIMD 是速度的船，船不能偏离锚。**

下面用三种语言，对照这三个概念的骨架。例子用一个假想的格式 `MINI1` 来做——别紧张，它非常简单：每块 8 个值、1 个 FP16 尺度、8 个 1-bit 码（每个值 1 位）。MINI1 的一个块只有 3 字节：2 字节尺度 + 1 字节码。

**C++（参考反量化 + 量化）：**

```cpp
#include <cassert>
#include <cstdint>
#include <cmath>

// 假设的 MINI1：8 值/块，d(fp16) + qs(8 bit)
typedef struct {
    ggml_half d;   // 2 字节
    uint8_t   qs;  // 1 字节，8 个 1-bit 码
} block_mini1;
static_assert(sizeof(block_mini1) == 3, "wrong mini1 block size/padding");

void dequantize_row_mini1(const block_mini1 * x, float * y, int64_t k) {
    assert(k % 8 == 0);
    const int64_t nb = k / 8;
    for (int64_t i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        for (int b = 0; b < 8; b++) {
            const int q = (x[i].qs >> b) & 1;   // 取第 b 位
            y[i*8 + b] = d * (2*q - 1);          // {0,1} → {-1,+1}
        }
    }
}

void quantize_row_mini1_ref(const float * x, block_mini1 * y, int64_t k) {
    assert(k % 8 == 0);
    const int64_t nb = k / 8;
    for (int64_t i = 0; i < nb; i++) {
        float amax = 0.0f;
        for (int b = 0; b < 8; b++) {
            amax = std::max(amax, std::fabs(x[i*8 + b]));
        }
        y[i].d  = GGML_FP32_TO_FP16(amax);
        y[i].qs = 0;
        for (int b = 0; b < 8; b++) {
            if (x[i*8 + b] > 0) y[i].qs |= (1u << b);  // 正→1，非正→0
        }
    }
}
```

**C#（.NET 5+）：**

```csharp
using System;
using System.Runtime.InteropServices;

public static class Mini1
{
    public const int QK_MINI1 = 8;

    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public unsafe struct BlockMini1
    {
        public ushort d;      // FP16 位模式
        public byte   qs;     // 8 个 1-bit 码
    } // sizeof == 3

    public static void DequantizeRow(BlockMini1[] x, float[] y, int k)
    {
        for (int i = 0; i < k / QK_MINI1; i++)
        {
            float d = (float)BitConverter.UInt16BitsToHalf(x[i].d);
            for (int b = 0; b < QK_MINI1; b++)
            {
                int q = (x[i].qs >> b) & 1;
                y[i * QK_MINI1 + b] = d * (2 * q - 1);
            }
        }
    }

    public static void QuantizeRow(float[] x, BlockMini1[] y)
    {
        for (int i = 0; i < x.Length / QK_MINI1; i++)
        {
            float amax = 0f;
            for (int b = 0; b < QK_MINI1; b++)
                amax = Math.Max(amax, Math.Abs(x[i * QK_MINI1 + b]));

            y[i].d = BitConverter.HalfToUInt16Bits((Half)amax);
            y[i].qs = 0;
            for (int b = 0; b < QK_MINI1; b++)
                if (x[i * QK_MINI1 + b] > 0) y[i].qs |= (byte)(1 << b);
        }
    }
}
```

**Python：**

```python
import struct

QK_MINI1 = 8

def dequantize_row_mini1(data: bytes, k: int) -> list[float]:
    assert k % QK_MINI1 == 0
    nb = k // QK_MINI1
    y = [0.0] * k
    for i in range(nb):
        base = i * 3
        d = struct.unpack_from('<e', data, base)[0]   # FP16 尺度
        qs = data[base + 2]                            # 1 字节码
        for b in range(QK_MINI1):
            q = (qs >> b) & 1
            y[i * QK_MINI1 + b] = d * (2 * q - 1)
    return y

def quantize_row_mini1(x: list[float]) -> bytes:
    assert len(x) % QK_MINI1 == 0
    nb = len(x) // QK_MINI1
    out = bytearray()
    for i in range(nb):
        amax = max(abs(v) for v in x[i*QK_MINI1:(i+1)*QK_MINI1])
        out += struct.pack('<e', amax)
        qs = 0
        for b in range(QK_MINI1):
            if x[i*QK_MINI1 + b] > 0:
                qs |= (1 << b)
        out.append(qs)
    return bytes(out)
```

这个 MINI1 虽然简陋，但它的函数签名、断言、结构体布局，和真实的 Q4_K/Q4_0 完全同构——**看懂了它，就等于看懂了 llama.cpp 里几十个量化格式的函数骨架。**

### 16.2.2 `vec_dot_*`：避免完整反量化的点积

`dequantize_row_*` 解决的是"怎么把量化块变回浮点"。可如果矩阵乘的每一步，都先把整行反量化成浮点、再算点积，那么量化全省下来的带宽就白省了——因为反量化这个过程本身，就在搬运和计算数据。llama.cpp 的核心优化是：**直接在 `vec_dot_*` 里消费量化块，边解包边乘加**。

先用文字描述一下，一次"量化点积"到底在做什么。我们要算的是浮点激活向量 \(a\) 和量化权重块 \(b\) 的内积：

```
朴素做法：先反量化再点积
  b(量化块) ──dequantize──► b'(浮点) ──►  Σ a[i] * b'[i]

优化做法：vec_dot 直接消费量化块
  a(浮点) ──────────────┐
                        ├──► vec_dot_q4_K(a, b) ──► 内积结果
  b(量化块) ────────────┘
  在块内同步做：解尺度、解 4-bit 码、乘加
```

`vec_dot_*` 的实际实现，会尽量把"解包"和"乘加"融合在一起。以 Q4_K 的 CPU 内核思路为例：对一个 256 值超级块，先解出 8 个子块的 `sc * d` 和 `m * dmin`；然后对每个 4-bit 码，把 `d1 * q - m1` 这种"乘完再加一个负偏移"的式子，拆成 `d1*q + (-m1)`——负偏移可以预先累加，于是内积就变成了"浮点 × 整数码"的乘加，最后再统一减掉。这样，SIMD 就能一次处理一大堆整数乘加。

**也正是在这里，IQ 的查表路径和 K-quants 的算术路径分道扬镳：**

- K-quants（Q4_K、Q6_K 等）走**算术路径**：解码无非是乘法和减法，SIMD 用乘加指令就能搞定。
- IQ 系列（IQ1_S、IQ2_XXS 等）走**查表路径**：解码是"取索引 → 查 `iq1s_grid` → 加符号/修正"，SIMD 需要 `vqtbl` 这类查表指令（第 11 章讲过），GPU 后端的实现难度要高得多。

这就是为什么在同一个模型里，Q4_K 在几乎所有后端上都很快，而 IQ1_S 在某些后端上却可能成为瓶颈（第 17 章会给出对比数据）。**`vec_dot` 是格式差异真正变成性能差异的地方**——反量化函数定义了正确性，`vec_dot` 定义了速度。


## 16.3 新增格式的最小步骤

把前两节的零件拼起来，就得到了一份可以照着做的清单。

### 16.3.1 结构体 → 参考量化/反量化 → type_traits → 测试

按照依赖顺序，一共六步：

```
新增 MINI1 的最小步骤

  ┌─ ① ggml.h: GGML_TYPE_MINI1 = 43, GGML_TYPE_COUNT = 44
  ├─ ② ggml-common.h: block_mini1 + static_assert(sizeof==3)
  ├─ ③ ggml-quants.c: quantize_row_mini1_ref + dequantize_row_mini1
  ├─ ④ ggml.c: type_traits[GGML_TYPE_MINI1] = { ... 注册指针 ... }
  ├─ ⑤ tests/test-quantize-fns.c: 加 MINI1 到测试表
  └─ ⑥ 用 gguf-py 交叉验证：自己写的最小 GGUF 反量化 == dequantize()
```

第 ⑤ 步的 `test-quantize-fns`，是正确性的守门员。它做的事可以用一句话概括：**对每个格式，随机生成浮点数据 → 参考量化 → 参考反量化 → 比较误差**；与此同时，还要**用 SIMD 优化版去量化 → 与参考版逐字节比对**。新增格式只需要把 `block_mini1` 的 `from_float_ref`/`to_float` 挂进测试表，测试就会自动检查两件事：

1. 参考反量化之后的误差在阈值以内（说明格式定义正确）；
2. 优化实现与参考实现的字节完全一致（说明 SIMD 没有解包错）。

这也是为什么 3.1.4 节要把"参考实现"强调得那么重——**它不只是给 CPU 用的一个慢速版本，更是所有优化版和所有测试的"标准答案"。**

第 ⑥ 步"与 gguf-py 交叉验证"，是最容易被人忽略、却最能揪出位序错误的一步。写完参考实现之后，用 Python 手工构建几个字节（就像第 7 章那样），喂给 `dequantize()`，再和自己 C++ 那套反量化输出的结果对比：

```python
import struct
from gguf import dequantize, GGMLQuantizationType

# 手工构造一个 MINI1 块：d = 2.5, qs = 0b10101010
raw = struct.pack('<eB', 2.5, 0b10101010)

# 假设 gguf 已经实现了 MINI1，反量化它
out = dequantize(raw, GGMLQuantizationType.MINI1)
print(out)   # 期待 [2.5, -2.5, 2.5, -2.5, ...]（正负交替）
```

如果两个实现的输出对不上，那几乎总是位序或者尺度合成的问题，而不是数学的问题——因为参考反量化的数学，实在是太简单了。**跨实现做逐字节比对，是揪出"位序/打包错误"最有效的手段**，这个原则到第 18 章还会完整展开。

### 16.3.2 后端扩展：CPU SIMD → Metal → CUDA/Vulkan

参考实现通过测试之后，真正的性能工作才开始。后端的扩展顺序，也很讲究：

```
后端扩展的顺序与验证闭环

  CPU 参考实现 ──test-quantize-fns──► 正确
      │
      ▼
  CPU SIMD（AVX2 / NEON）
      │  每个后端都要重新做"与参考实现字节一致"验证
      ▼
  Metal（Apple）
      ▼
  CUDA / Vulkan（NVIDIA / 通用 GPU）
```

每一步的规则都相同：**新后端的量化/反量化/`vec_dot`，必须和 CPU 参考实现逐字节一致。** 这里有一个常见误解，就是"GPU 上跑通了就算对了"——GPU 内核为了向量宽度，往往会做更激进的解包顺序调整（比如把 6-bit 位序重排），跑通了不代表没有系统性的错误；只有和参考实现逐字节比对过的结果，才算数。

这也正是第 13 章那些三值格式的处境：STQ1_0 有 ARM NEON 和 AVX2 内核，但 Metal 的覆盖还不完整；PTQ1_0 的 Hadamard 变换，还依赖着额外的 PR。**一个格式"在 CPU 测试通过了"，绝不等于"在你部署的后端就能用"**——第 15 章的部署前检查清单，会教你怎么验证这件事。


## 本章小结与通往"性能"的桥梁

这一章的内容，同样可以压缩成三条事实。

**第一，新增格式的注册，是"三文件一条链"：`ggml.h` 给编号、`ggml-common.h` 给布局、`ggml.c` 给行为。** 编号只能追加、不能回收（这是历史契约）；结构体靠 `static_assert` 锁死大小（这是 POD 约束）；`type_traits` 缺条目，运行时直接崩溃（分派表即是契约）。

**第二，函数契约由一对"行函数"加上一个点积函数构成：`quantize_row_*`/`dequantize_row_*` 保证正确性，`vec_dot_*` 决定性能。** 参考实现是正确性的锚，SIMD 是速度的船；K-quants 走算术路径，IQ 走查表路径，两者在 `vec_dot` 处分叉。

**第三，正确的扩展顺序是"参考实现 → 测试 → 逐后端验证"，而不是"一上来就写最快的内核"。** 每一步都要和参考实现、和 gguf-py 做逐字节交叉验证；三值格式当年的后端缺失，就是没把这条链走完的真实案例。

下一章，我们把视角从"代码怎么组织"，转向"代码跑得有多快"。同样是 4.5 bpw 的 Q4_0 和 Q4_K，解码开销会差多少？IQ1_S 的查表路径，在端侧会不会成为瓶颈？内存带宽、SIMD 覆盖、后端选择——这些，才是真正决定 tok/s 的变量。
