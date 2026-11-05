# 《GGUF 量化内幕》写作风格指南与格式事实表

> 本文档是写作契约，供章节作者（含并行子代理）使用，不属于书稿正文，不要把它当章节发布。
> 权威读者参考：`ch04.md`（风格范例，逐行模仿其结构、语气、图表与代码密度）。

---

## 1. 项目规则（来自 AGENTS.md，每章必须全部满足）

1. **编写每章正文之前**先调用百度搜索（`mcp__baidu-search__search`），每章累计返回**至少 20 条结果**（可多个查询合并；若该 MCP 工具不可用，退回 `WebSearch` 凑足）。把"查询词 + 命中条数"记录在最终汇报里。
2. 读者：**编程语言与深度学习入门者，除此之外没有任何背景**。
3. 重要概念必须给 **C++ / C# / Python** 三种语言的代码。
4. 必要时用 **ASCII（box-drawing）图**解释字节布局、数据流、对比。
5. 细纲位于 `asset/detail.md`，本节编号必须与细纲对齐。

## 2. 读者画像与文风

- 不默认读者认识：位/字节、FP16/FP32、内存 vs 磁盘、mmap、矩阵乘、张量、权重、logits、PPL、KV cache、字节序、SIMD/NEON、嵌入层、FFN、QKV、校准数据、token、粒度/others。**每个专业词首次出现时给一句白话定义**（学 ch04 第 3 段"先统一几个术语"）。
- 语气：冷静、循序渐进；先给直觉（场景/图/例子），再给精确（公式/代码），绝不"突然冒出一个大概念"。
- 每章开头：用**具体数字或场景**引入问题（ch04 用"7B 模型 FP32 28GB → FP16 14GB"）；结尾用"本章小结"把已知提炼成 2–3 条事实，并递进到下一主题。
- 每节标题可以先于小节给出一个"为什么关心"。
- 加粗用于关键结论与首次定义：`**……**`。
- 公式用 $\LaTeX$：行内 `\( \)`，独立 `\[ \]`。
- 中文标点；代码注释用中文；命名与 llama.cpp/gguf 生态一致。

## 3. 章节骨架模板（从 ch04 提炼）

编号映射：细纲 `### 第 X 章` → 章节 H1 `# 第X章 <副标题>`；细纲 `#### X.m` → `## X.m`；细纲 `##### X.m.k` → `### X.m.k`。**小节标题可以改写得更具解释力**（ch04 把细纲"4.1 字节布局与结构体"改写为"4.1 先理解'块'这个概念"），但编号和覆盖范围必须对齐细纲。

统一结尾：`## 本章小结与通向<下一主题>的桥梁`——把本章压成 2–3 条"事实"，加一张回顾图，最后一句指出下一章的动机。

## 4. 三语言代码规范

- 每个重量级概念（结构体布局、反量化、量化、打包/解包、查表、正则式）同时给 C++/C#/Python。
- **C++**：与 llama.cpp 一致。FP16 用 `ggml_half`（存位模式）；转换宏 `GGML_FP16_TO_FP32 / GGML_FP32_TO_FP16`（首次出现时说明它们负责位模式↔float）。结构体必须带 `static_assert(sizeof(...) == N, ...)`。
- **C#**：.NET 5+。`System.Half`、`BitConverter.UInt16BitsToHalf / HalfToUInt16Bits`；布局用 `[StructLayout(LayoutKind.Sequential, Pack = 1)]` + `unsafe fixed` 字节数组；注明需要 `<AllowUnsafeBlocks>true</AllowUnsafeBlocks>`、命名空间 `System.Runtime.InteropServices`。
- **Python**：`struct` 模块，半精度用 `'<e'`（小端 FP16，Python 3.6+）。强调"Python 对象与磁盘字节分离，用 `bytes` + `struct` 显式序列化"。`dataclass` 可描述布局但真正读写走 `struct.unpack_from/pack`。
- 三种实现必须逐行对应、数学等价；每个格式至少给一个**校验**（`assert`、`static_assert`、交叉验证脚本）。
- 代码块前用加粗标签：`**C++：**` / `**C#（.NET 5+）：**` / `**Python：**`。

## 5. ASCII 图规范

- 用 box-drawing 字符：`┌ ┐ └ ┘ ├ ┤ ┬ ┴ ┼ │ ─`，配合 `▲ ▼` 指示流向，全宽图中的文字用中文。
- 典型图：字节布局图（偏移+字节构成）、数据流图（反量化管线）、打包映射图（位↔值）、对比图（Q4_0 vs Q4_K）、层次结构图（超级块/子块）。
- 图内数字必须真实对得上（块内字节数、偏移、位数都要如实）。宽度尽量 ≤ 60 字符。

## 6. 数字、公式与引用规范

- **bpw 一律自己算**：`bpw = 块字节数 × 8 / 每块权重数`，正文每个格式都给这一步，并注明块字节数来源（本指南 §8 或 ggml-common.h）。禁止直接抄一个对不上 block 大小的 bpw。
- 细纲里的 PPL/体积/tok/s：属于**待核实假设**。写作用"约"，并注明"特定模型与评测集下测得，不同来源不可直接比较"。若与可溯源来源冲突，以来源为准，行文用"约"降低风险；**与 ch04.md 已写的内容不要互相矛盾**（ch04 用的是绝对 PPL 增量 +0.25/+0.05，细纲 §4.4.2 用百分比 +1.5%/+0.8%——两者口径不同，需加注"绝对增量 vs 百分比，口径不同"）。
- 工具名/API/函数签名要写对：`convert_hf_to_gguf.py`、`llama-quantize`、`llama-imatrix`、`llama-perplexity`、`--imatrix`、`-ngl`、`GGUFReader`、`GGUFWriter`、`dequantize`、`GGMLQuantizationType`。
- 未合入主线的功能（三值格式、UD 类方法）：明确标注"处于 fork/PR 阶段"。

## 7. 跨章一致性约定

- 术语统一：「块」= block；「超级块」= super-block（256 值）；「子块」= sub-block（32 值）；「尺度」= scale；「反量化」= dequantize；「打包/解包」；「码」= quant code；「码本/grid」；「bit/weight (bpw)」。
- 类型与函数名一致：`block_q4_K`（大 K）、`dequantize_row_q4_0`、`scale_min_k4`。
- **`Q4_K_M` 不是一种 ggml 类型**，是量化策略/文件级名称；文件内部是混合类型（Q4_K 主体 + Q6_K 输出层 + F32 归一化）。第 3、14 章重点讲，其他章不许混淆。
- 交叉引用：「第 7 章」「见 7.3.2」，章节号严格按细纲，不许自行调整。
- type id 表只在第 3、13、19、20 章出现，其他章不展开。

## 8. 格式速查表（fact sheet）

> 标记 ⚠ 的条目**写进正文前必须用搜索/源码核实**（优先 llama.cpp 当前主线 `ggml/src/ggml-common.h` 与 `ggml/src/ggml.c` 的 `type_traits`、GGUF 官方 spec、llama.cpp README）。

### 8.1 简单块（布局已验证）
| 格式 | 每块权重 | 字节 | bpw（自算） | 公式 | 结构 |
|---|---|---|---|---|---|
| Q4_0 | 32 | 18 | 4.5 | `y = d*(q-8)` | `d`(f16) + `qs[16]` nibble |
| Q4_1 | 32 | 20 | 5.0 | `y = d*q + m` | `d` + `m`(f16) + `qs[16]` |
| Q5_0 | 32 | 22 | 5.5 | `y = d*(q5-16)` | `d` + `qh[4]` + `qs[16]` |
| Q5_1 | 32 | 24 | 6.0 | `y = d*(q5-16)+m` | `d` + `m` + `qh[4]` + `qs[16]` |
| Q8_0 | 32 | 34 | 8.5 | `y = d*q` | `d`(f16) + `qs[32]`（int8） |

### 8.2 K-quants（structural 已验证，脚标 ⚠ 待核）
统一公式：`y_i = d * s_j * q_i - dmin * m_j`（其中 `s_j = scales[j]/64`，`m_j = mins[j]/64`，⚠ 63/64 口径不同文献有出入，以 ggml-common.h 为准）。

- **Q4_K**：256 值 / **144 字节 / 4.5 bpw**。结构 `d`(2) + `dmin`(2) + `scales[12]` + `qs[128]`。8 个子块 × 32 值；子块 6-bit 尺度与 6-bit 最小值**交错打包**在 12 字节里。规范解包（必须照抄进第 7 章）：
```c
static inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}
```
即：`scales[0..3]` 低 6 位 = 子块 0..3 的尺度；`scales[4..7]` 低 6 位 = 子块 0..3 的最小值；`scales[8..11]` 低半字节 = 子块 4..7 尺度的低 4 位、高半字节 = 子块 4..7 最小值的低 4 位；高 2 位分散在 `scales[0..3]`（尺度用）与 `scales[4..7]`（最小值用）的 bit 6–7。**不是"scales 和 mins 各占 12 字节"**。4-bit 码在 `qs`，每子块 16 字节、同 Q4_0 nibble 打包。
- **Q5_K**：256 值 / **176 字节 / 5.5 bpw**。在 Q4_K 结构上加 `qh[32]`（每个权重第 5 bit，按子块 4 字节、位序 `l%8`，⚠ 位序需核对）。5-bit 码 = `qs` 低 4 位 | (`qh` 第 5 位 << 4)，其余同 Q4_K。
- **Q6_K**：256 值 / **210 字节 / 6.5625 bpw**。结构 `d`(2) + `ql[128]` + `qh[64]` + `scales[16]`（int8）。16 个子组 × 16 值；6-bit 码 = `ql` 低 4 位 + `qh` 高 2 位（⚠ 位序核对）；公式 `y = d * scales[j] * (q - 32)`（缩放常数 ⚠ 核对）。
- **Q2_K**：256 值 / **84 字节 / 2.625 bpw**。结构 `scales[16]`(每个字节低半字节=尺度系数、高半字节=最小值系数，⚠ 核对) + `d` + `dmin` + `qs[64]`。16 子组 × 16 值，2-bit 码 4 个/字节；公式 `y = d*d_l*q - dmin*d_h`（细分形式 ⚠ 以参考实现为准）。
- **Q3_K**：256 值 / sizeof **112**、type_size **110** ⚠（bpw ≈ 3.4375–3.5，两处口径都写并说明差异）。结构 `d` + `scale` + `hmask[32]` + `qs[64]` + `scales[12]`(int8)；3-bit 码 = `qs` 低 2 位 + `hmask` 1 位（⚠ 核对）。
- **Q8_K**：256 值 / 292 字节 / 9.125 bpw，仅作参考基准。

### 8.3 IQ 系列（结构已给，**bpw 全部 ⚠ 自行按源码核实**）
> 环节：Q4_K（算术反量化路径）→ IQ 系列（**码本 + imatrix 加权**，误差结构从"尺度失配"变为"最近邻选择"）。
> bpw 唯一权威：`ggml.c` `type_traits` 的 `type_size / blck_size`，`bpw = type_size*8/blck_size`。细纲 §11 中 IQ1_S"1.5 bpw"的说法有争议，第 11 章应核实后给出准确数字（IQ1_S 区块为 74 字节/256 值 = 2.3125 bpw；IQ1_M 才约 1.56 bpw），若核实后仍不一致，用"约"并加注。

- **IQ1_S**：256 值 / **74 字节**结构（2+32+8+32）= `d`(2) + `qs[32]` + `qh[8]` + `signs[32]`。32 个 8 值组；每组 11-bit 索引 = `qs` 每值 1 bit（8）+ `qh` 共享 3 bit；查 `iq1s_grid[2048]`（每项是 8 值的小整数向量）；符号位在 `signs`。公式带 `IQ1S_DELTA` 修正（⚠ 细节核对）。
- **IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S / IQ4_XS / IQ1_M / IQ2_M**：bpw 与块字节数 ⚠ 全部以源码核实，`IQ3_XXS` 细纲写 3.4375、与部分源码口径（64 字节？）矛盾，务必核实后在正文给出一致结论。
- **IQ4_NL**：32 值 / **18 字节 / 4.5 bpw**。布局同 Q4_0（`d` + `qs[16]`），但 4-bit 索引直接查**非线性**格点 `kvalues_iq4nl[16]`，`y = d * kvalues[q]`。
- grid 常量名：`iq1s_grid`、`iq2s_grid`、`iq2xxs_grid`、`kvalues_iq4nl` 等。

### 8.4 三值系列（均 ⚠ 待核实，属 fork/PR 生态前沿，用"处于演进中"语气）
- **STQ1_0**：每 4 个权重 **强制 1 个零**（3:4 稀疏），共 \(4 \times 2^3 = 32\) 种模式，5 bit/组 = 4-bit 码本索引 + 1-bit 符号。块 256 值 / **42 字节 / 1.3125 bpw**（`qs[32]` 索引 + `sign[8]` + fp16 `d`，字段名 ⚠ 核对）。64 值块内 **stride-16 分组**（\(w_0,w_{16},w_{32},w_{48}\)），便于 ARM NEON `vqtbl2q` + `vdotq_s32` 无重排解码。
- **PTQ1_0**：**稠密三值**（不强制位置），base-3 编码：24 字节装 5 trits/字节 + 2 字节余数（4 trits/字节）= 128 trits；块 128 值 / **28 字节 / 1.75 bpw**。**Hadamard 折叠**：磁盘里权重以 Hadamard 折叠形式存储，推理**必须**做激活侧变换，否则 logits 完全错误——任何一版都必须把"格式正确但缺失旋转仍然是错的"作为独立小节反复强调。
- type id：STQ1_0 资料有 42/43 两种记载、PTQ1_0 记为 143，⚠ 写作时如实呈现"id 归属未定（fork/PR）"。

### 8.5 type id 主线表（可直接引用）
0=F32, 1=F16, 2=Q4_0, 3=Q4_1, 4=Q4_2(移除), 5=Q4_3(移除), 6=Q5_0, 7=Q5_1, 8=Q8_0, 9=Q8_1, 10=Q2_K, 11=Q3_K, 12=Q4_K, 13=Q5_K, 14=Q6_K, 15=Q8_K, 16=IQ2_XXS, 17=IQ2_XS, 18=IQ3_XXS, 19=IQ1_S, 20=IQ4_NL, 21=IQ3_S, 22=IQ2_S, 23=IQ4_XS, 24=I8, 25=I16, 26=I32, 27=I64, 28=F64, 29=IQ1_M, 30=IQ2_M, 31/32/33=Q4_0_4_4/4_8/8_8, 34=TQ1_0, 35=TQ2_0, 36/37/38=IQ4_NL_4_4/4_8/8_8。

### 8.6 命令速查（工具链章节复用）
```bash
# 转换（不做量化）
python convert_hf_to_gguf.py <hf_dir> --outfile model-f16.gguf --outtype f16
# 生成 imatrix（IQ 系列必需）
./llama-imatrix -m model-f16.gguf -f calibration-data.txt -ngl 99 -o imatrix.dat
# 量化（Q4_K_M 是策略名，不是单一 ggml 类型）
./llama-quantize model-f16.gguf output-q4_k_m.gguf Q4_K_M --imatrix imatrix.dat
# PPL/KL 校验（IQ/三值格式更依赖 KL）
./llama-perplexity -m model.gguf -f test.txt --kl-divergence-base model-f16.gguf
# 目标后端起一次短生成（冒烟）
./llama-cli -m model.gguf -p "Hello" -n 64
```

### 8.7 容器事实（第 3 章）
- magic = `GGUF`（0x46554747）；当前版本 3；`tensor_count`/`kv_count` 为 u64；字符串 u64 长度前缀。
- 文件顺序：header → KV 数组 → tensor_infos 数组 →（对齐到 `general.alignment`，默认 32）→ tensor data 块。
- `tensor_infos[i].offset` 相对**数据块起点**，不是绝对文件偏移。
- mmap 零拷贝的前提：块结构体是 **POD**、无填充，磁盘字节 == 结构体内存布局。
- gguf-py：`pip install gguf`；`GGUFReader(path)`、`reader.fields`、`reader.tensors`、`tensor.data`、`tensor.tensor_type`、`gguf.dequantize(data, type)`、`GGUFWriter` 三步写（header→kv→tensor data）。

### 8.8 PPL 劣化量级参考（写作标"约"并注明口径）
Q4_0 ≈ +1.5%（或绝对 +0.25，见 ch04）、Q4_K ≈ +0.8%、Q5_K ≈ +0.4%、Q6_K ≈ +0.2%、Q8_0 ≈ 近无损（+0.1%）。ch04 已写：同一 7B 模型 Q4_0 PPL +0.25、Q4_K_M +0.05。

## 9. 附录写作规范

- 附录 A（公式速查）：一个主表按格式列：公式、尺度归一化、块字节数、bpw（自算步骤）、type id。无代码。
- 附录 B（字节布局图集）：每个格式逐字节 ASCII 图 + C 结构体 + Python `struct` 描述对照。覆盖 Q4_0、Q4_1、Q5_0、Q5_1、Q8_0、Q2_K、Q3_K、Q4_K、Q5_K、Q6_K、IQ1_S、IQ2_XXS、IQ3_XXS、IQ4_NL、STQ1_0、PTQ1_0。
- 附录 C（最小 Python 参考实现）：可独立运行的 GGUF 解析器 + Q4_0/Q4_K/Q6_K/IQ1_S/STQ1_0/PTQ1_0 解码器 + 与 `gguf-py` 的交叉验证脚本。
- 附录 D（gguf-py API 速查）：`GGUFReader`/`GGUFWriter`/`dequantize`/`GGMLQuantizationType` 调用模式与坑。
- 附录 E（函数索引）：格式 × `quantize_row_*` / `dequantize_row_*` / `vec_dot_*` × 源文件（`ggml/src/ggml.c`、`ggml/src/ggml-cpu/*` 等），可给预估路径 ⚠ 需核对。
- 附录 F（实验脚本与命令）：端到端命令序列（转换→imatrix→量化→PPL/KL→端侧基准）。
- 附录 G（术语表）：block、super-block、imatrix、grid、trit、Hadamard 折叠、stride-16 分组等。
- 附录 H（参考文献与版本）：Sherry（STQ1_0）、QuIP# 系（IQ）、Prism Hadamard、llama.cpp PR 编号与锁定版本、GGUF spec 链接；⚠ 链接与编号需搜索核实。