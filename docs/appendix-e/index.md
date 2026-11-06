# 附录 E llama.cpp 关键函数索引

> 本附录按格式列出 llama.cpp 生态中与量化关系最密切的三个函数族：`quantize_row_*`（量化）、`dequantize_row_*`（反量化）与 `ggml_vec_dot_*`（量化块上的直乘点积）。函数名与源文件以 llama.cpp **主线 master（2026-10 核实）** 为准。行号是写作时抓取的近似位置，会随版本漂移，使用时请以"文件名 + 函数名"为准。

## E.1 三个函数族分别做什么

先理清三个函数族的职责分工——它们的参数几乎一样，但做的事情完全不同：

| 函数族 | 方向 | 输入 | 输出 | 用途 |
|---|---|---|---|---|
| `quantize_row_*` | FP32 → 量化块 | `const float * x` | `block_* * y` | 把浮点权重变成磁盘上的量化块（第 4–10 章"量化端"） |
| `dequantize_row_*` | 量化块 → FP32 | `const block_* * x` | `float * y` | 把量化块还原成浮点数（第 4–10 章"反量化"） |
| `ggml_vec_dot_*` | 量化块 → 标量 | 两个量化块指针 | `float * s` | 不完整反量化，直接在块上做点积（第 16、17 章性能核心） |

先看以 Q4_0 为例时这三个函数的实际形态（这是理解全部命名规律的钥匙）：

```c
// ggml/src/ggml-quants.c
void quantize_row_q4_0_ref(const float * x, block_q4_0 * y, int64_t k);
void dequantize_row_q4_0(const block_q4_0 * x, float * y, int64_t k);

// ggml/src/ggml-cpu/quants.c（x 作为量化块的 vy 直乘）
void ggml_vec_dot_q4_0_q8_0(int n, float * s, size_t bs, const void * vx, size_t bx,
                            const void * vy, size_t by, int nrc);
void ggml_vec_dot_q4_0_q8_0_generic(int n, float * s, size_t bs, const void * vx, size_t bx,
                                    const void * vy, size_t by, int nrc);
```

命名规律有三点要澄清：

1. **`_ref` 后缀 = 参考实现**。`quantize_row_q4_0_ref` 是"教科书版"，正确、可读、较慢；带 imatrix 加权的 `_impl`（如 `quantize_row_q4_K_impl`）是优化版。反量化通常没有 `_ref`，因为反量化的参考实现就是完整实现。
2. **`vec_dot` 是"量化 × 量化"的直乘**。注意名字里有两个类型：`ggml_vec_dot_q4_0_q8_0` 表示"第一个操作数 vx 是 Q4_0、第二个操作数 vy 是 Q8_0"。矩阵乘时权重常常保持低 bit（Q4_0/Q4_K），而激活被打包成 Q8_0/Q8_K，所以 `_q8_0`、`_q8_K` 后缀反复出现。
3. **`_generic` 后缀 = 标量/重排后备版本**。现代 llama.cpp 里，每个点积有一个 SIMD 优化版（无后缀，如 NEON/AVX2 内联汇编）和一个 `_generic` 的"任何架构都能跑"版本，用于回退和测试基准。

## E.2 按格式的完整索引

下表汇总主线中每个量化格式对应的三个函数。`quantize_row_*_ref` 未列出时，表示该格式没有公开的参考量化实现（对应 `type_traits` 中的 `from_float_ref = NULL`，只能通过官方转换/量化流程产生）。源文件缩写：**Q** = `ggml/src/ggml-quants.c`，**C** = `ggml/src/ggml-cpu/quants.c`。

| 格式 | quantize_row（量化，Q） | dequantize_row（反量化，Q） | ggml_vec_dot（点积，C） |
|---|---|---|---|
| F16 | `ggml_fp32_to_fp16_row`（ggml.c） | `ggml_fp16_to_fp32_row` | 无（走通用路径） |
| Q1_0 | `quantize_row_q1_0_ref` | `dequantize_row_q1_0` | `ggml_vec_dot_q1_0_q8_0` |
| Q2_0 | `quantize_row_q2_0_ref` | `dequantize_row_q2_0` | `ggml_vec_dot_q2_0_q8_0` |
| Q4_0 | `quantize_row_q4_0_ref` / `_impl` | `dequantize_row_q4_0` | `ggml_vec_dot_q4_0_q8_0` (+`_generic`) |
| Q4_1 | `quantize_row_q4_1_ref` / `_impl` | `dequantize_row_q4_1` | `ggml_vec_dot_q4_1_q8_0` |
| Q5_0 | `quantize_row_q5_0_ref` / `_impl` | `dequantize_row_q5_0` | `ggml_vec_dot_q5_0_q8_0` |
| Q5_1 | `quantize_row_q5_1_ref` / `_impl` | `dequantize_row_q5_1` | `ggml_vec_dot_q5_1_q8_0` |
| Q8_0 | `quantize_row_q8_0_ref` | `dequantize_row_q8_0` | `ggml_vec_dot_q8_0_q8_0` |
| Q8_1 | `quantize_row_q8_1_ref` | （仅中间形态） | `ggml_vec_dot_q8_1_q8_1` |
| Q2_K | `quantize_row_q2_K_ref` / `_impl` | `dequantize_row_q2_K` | `ggml_vec_dot_q2_K_q8_K` |
| Q3_K | `quantize_row_q3_K_ref` / `_impl` | `dequantize_row_q3_K` | `ggml_vec_dot_q3_K_q8_K` |
| Q4_K | `quantize_row_q4_K_ref` / `_impl` | `dequantize_row_q4_K` | `ggml_vec_dot_q4_K_q8_K` |
| Q5_K | `quantize_row_q5_K_ref` / `_impl` | `dequantize_row_q5_K` | `ggml_vec_dot_q5_K_q8_K` |
| Q6_K | `quantize_row_q6_K_ref` / `_impl` | `dequantize_row_q6_K` | `ggml_vec_dot_q6_K_q8_K` |
| Q8_K | —（中间形态，见说明） | `dequantize_row_q8_K` | —（激活角色） |
| IQ1_S | `quantize_row_iq1_s_impl`（无 `_ref`） | `dequantize_row_iq1_s` | `ggml_vec_dot_iq1_s_q8_K` |
| IQ1_M | `quantize_row_iq1_m_impl`（无 `_ref`） | `dequantize_row_iq1_m` | `ggml_vec_dot_iq1_m_q8_K` |
| IQ2_XXS | `quantize_row_iq2_xxs_impl`（无 `_ref`） | `dequantize_row_iq2_xxs` | `ggml_vec_dot_iq2_xxs_q8_K` |
| IQ2_XS | `quantize_row_iq2_xs_impl`（无 `_ref`） | `dequantize_row_iq2_xs` | `ggml_vec_dot_iq2_xs_q8_K` |
| IQ2_S | `quantize_row_iq2_s_impl`（无 `_ref`） | `dequantize_row_iq2_s` | `ggml_vec_dot_iq2_s_q8_K` |
| IQ3_XXS | `quantize_row_iq3_xxs_ref`（有） | `dequantize_row_iq3_xxs` | `ggml_vec_dot_iq3_xxs_q8_K` |
| IQ3_S | `quantize_row_iq3_s_ref`（有） | `dequantize_row_iq3_s` | `ggml_vec_dot_iq3_s_q8_K` |
| IQ4_NL | `quantize_row_iq4_nl_ref`（有） | `dequantize_row_iq4_nl` | `ggml_vec_dot_iq4_nl_q8_0` |
| IQ4_XS | `quantize_row_iq4_xs_ref`（有） | `dequantize_row_iq4_xs` | `ggml_vec_dot_iq4_xs_q8_K` |
| TQ1_0 | `quantize_row_tq1_0_ref` | `dequantize_row_tq1_0` | `ggml_vec_dot_tq1_0_q8_K` |
| TQ2_0 | `quantize_row_tq2_0_ref` | `dequantize_row_tq2_0` | `ggml_vec_dot_tq2_0_q8_K` |
| MXFP4 | `quantize_row_mxfp4_ref` | `dequantize_row_mxfp4` | 后端特定 |
| NVFP4 | `quantize_row_nvfp4_ref` | `dequantize_row_nvfp4` | 后端特定 |

关于 Q8_K 的补充说明：`block_q8_K` 在结构上是"中间形态"——它把一行 FP32 量化成 `int8` 码 + 每 16 个的 `bsums`（局部和），专门作为**点积的激活侧**使用，所以它没有真正意义上的"独立量化格式"身份，也不出现在 GGUF 文件的张量类型里（它的 type id 15 仅用于运行时中间表示和部分工具）。

## E.3 quantize / dequantize 函数签名契约

所有反量化函数遵循同一签名约定（以 Q4_K 为例，`ggml/src/ggml-quants.c:1530`）：

```c
void dequantize_row_q4_K(const block_q4_K * x, float * y, int64_t k);
```

- `x`：指向一块连续的 `block_q4_K` 数组（内存里就是 GGUF 文件数据段的字节）。
- `y`：输出缓冲区，至少容纳 `k` 个 `float`。
- `k`：**总元素个数**，必须是该格式块大小 `QK` 的整数倍（函数开头 `assert(k % QK_K == 0)`）。块大小见第 2 章/附录 A。
- 返回值：无。函数把 `k` 个权重全部解包进 `y`，循环内部以块为单位推进。

量化函数的约定类似但多一层"参考/优化"：

```c
void quantize_row_q4_K_ref(const float * x, block_q4_K * y, int64_t k);          // 参考实现
static void quantize_row_q4_K_impl(const float * x, block_q4_K * y, int64_t n_per_row,
                                   const float * quant_weights);                 // imatrix 加权实现
```

`_impl` 多一个 `quant_weights` 参数（就是第 11 章的 imatrix）。没有 imatrix 时调用方传 `NULL`，`_impl` 内部退化为等价的启发式。**IQ 格式（IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/IQ2_S）只有 `_impl` 没有 `_ref`**，这正是"IQ 必须提供 imatrix 才能量化"的代码层面的原因——没有权重文件，连参考路径都不存在。

## E.4 vec_dot 的点积签名与矩阵乘

`vec_dot` 是矩阵乘的性能心脏。它的签名比前两个多出步长参数，因为点积发生在"行的某一段"上：

```c
void ggml_vec_dot_q4_K_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx,
                            const void * vy, size_t by, int nrc);
```

- `n`：要计算的点积分量个数（一段的权重数，块大小的整数倍）。
- `s`：输出，每个分量的部分和。
- `bs`：`s` 的步长；`vx`/`vy`：两个操作数；`bx`/`by`：各自行内步长。
- `nrc`：行/列计数（row-count），用于批处理多行。

**为什么叫"量化 × 量化"而不先反量化？** 第 17 章解释了行业标准技巧：矩阵乘 `Y = WX` 中，权重 `W` 是低 bit 量化（如 Q4_K），激活 `X` 被打包成 Q8_0 或 Q8_K。与其先把权重反量化成 FP32 再乘，不如在量化域直接拆包、乘加、累加，最后只把每一行的若干部分和乘以尺度。这样内存带宽减半以上，且避免了中间 FP32 张量。这也是为什么 `vec_dot` 的名字总是"某某_q8_0"或"某某_q8_K"——**激活侧通常固定用 8-bit 格式**。

`vec_dot` 的查询路径大体如下：

```
ggml_compute_forward_mul_mat (ggml.c)
  │  根据权重的 ggml_type 分派
  ▼
ggml_vec_dot_<TYPE>_<ACT>_generic  (ggml/src/ggml-cpu/quants.c，标量/重排后备)
  │  若 CPU 支持对应 SIMD 指令集
  ▼
ggml_vec_dot_<TYPE>_<ACT>          (同文件，NEON/AVX2/AVX512 内联实现)
```

## E.5 SIMD 与后端变体在哪

上面的表格列出的是"参考/C 实现"。真正的性能代码分散在各后端：

- **CPU SIMD**：`ggml/src/ggml-cpu/quants.c`（NEON/AVX2/AVX512 的 `vec_dot`、`*_generic`）。这个文件是 SIMD 优化的主战场，也是第 17 章讨论的"K-quants 位操作如何向量化"的出处。
- **CPU 张量重排**：`ggml/src/ggml-cpu/repack.cpp`（把块按 SIMD 友好的布局重排，配合 runtime repacking）。
- **CUDA/HIP（NVIDIA/AMD GPU）**：`ggml/src/ggml-cuda/` 下按格式拆文件（如 `quantize.cu`、`vec_dot.cu` 或按类型划分），每个后端有自己的 `dequantize_block` / 特殊 MUL_MAT 路径。
- **Metal（Apple）**：`ggml/src/ggml-metal/`，Metal 着色器内联实现反量化与点积。
- **Vulkan / SYCL / OpenCL 等**：`ggml/src/ggml-vulkan/`、`ggml-sycl/`、`ggml-opencl/` 各有一个镜像实现。

**后端覆盖不一致是常态**：新格式往往先做 CPU（C + SIMD），再逐步补 GPU。三值格式 STQ1_0 就是一个真实例子——主线 CPU 端有 NEON（PR #22836）与 AVX2（PR #27377）内核，但 Metal 明确"排除 STQ1_0 的 mul_mat 路径"（PR #28067）。**部署前必须确认目标后端真的有该格式的 vec_dot/dequant 内核**，而不是只看 CPU 测试通过（第 13、15 章反复强调）。

## E.6 本书主格式速查

把正文里反复出现的几个格式单独列成一张表，方便随手查阅：

| 格式 | 块大小 QK | 每块字节 | bpw | 反量化公式（简写） | 核心文件（grep 依据） |
|---|---|---|---|---|---|
| Q4_0 | 32 | 18 | 4.5 | `y = d·(q−8)` | ggml-quants.c |
| Q8_0 | 32 | 34 | 8.5 | `y = d·q` | ggml-quants.c |
| Q4_K | 256 | 144 | 4.5 | `y = d·sc·q − dmin·m` | ggml-quants.c / get_scale_min_k4 |
| Q5_K | 256 | 176 | 5.5 | 同 Q4_K，q 为 5-bit（+qh） | ggml-quants.c |
| Q6_K | 256 | 210 | 6.5625 | `y = d·sc·(q−32)` | ggml-quants.c |
| IQ1_S | 256 | 50 | 1.5625 | `y = d·dl·(grid[j]+Δ)`, 查 iq1s_grid | ggml-quants.c / ggml-common.h |
| IQ4_NL | 32 | 18 | 4.5 | `y = d·kvalues_iq4nl[q]` | ggml-quants.c / ggml-common.h |