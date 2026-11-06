# 附录 H 参考文献与版本说明

> 本附录汇总本书写作时（2026 年 10 月）参照与引用的主要来源，并对涉及版本差异的敏感点（三值格式、type id、Hadamard 变换）给出**已核实**的说明。所有链接与编号都在写作时通过 llama.cpp / GitHub / arXiv 官方渠道核实过；由于编号会随上游滚动演进，引用前请重新复查。

## H.1 代码与格式规范（一手来源）

| 内容 | 位置 | 说明 |
|---|---|---|
| llama.cpp 官方仓库 | `https://github.com/ggml-org/llama.cpp` | 全书所有代码、命令与工具链的一手来源（2026-10 主线，master 分支） |
| GGUF 规范与 gguf-py | llama.cpp 仓库 `gguf-py/` 子目录 | 本书第 3 章、附录 D 以 gguf-py **0.19.0** 为锁定版本 |
| 量化结构体定义 | `ggml/src/ggml-common.h` | `block_q4_0`、`block_q4_K`…所有格结构体与其 `static_assert` |
| 类型特征表 | `ggml/src/ggml.c`（`type_traits[]`） | type id → 块大小/字节数/量化/反量化函数指针 |
| 量化与反量化参考实现 | `ggml/src/ggml-quants.c` | `quantize_row_*` / `dequantize_row_*` |
| CPU 点积内核 | `ggml/src/ggml-cpu/quants.c` | `ggml_vec_dot_*` / `*_generic`（NEON/AVX2） |
| 类型枚举 | `ggml/include/ggml.h` | `GGML_TYPE_*` 完整枚举与 `GGML_TYPE_COUNT` |
| 转换与工具文档 | llama.cpp `README.md`、`tools/` 目录 | `convert_hf_to_gguf.py`、`llama-imatrix`、`llama-quantize`、`llama-perplexity` 的用法 |

## H.2 学术文献（已核实 arXiv 编号）

| 工作 | 主题 | 与本书的关系 | arXiv / 链接 |
|---|---|---|---|
| QuIP：2-bit Quantization of Large Language Models With Guarantees（Chee, Cai, Kuleshov, De Sa） | 向量量化 + 龙德/非相干性 | IQ 系列码本思想的理论先导（第 11 章） | arXiv:2307.13304 |
| QuIP#：Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks（Tseng, Chee, Sun, Kuleshov, De Sa） | Hadamard 不相干 + 格码本 | IQ 码本 + Hadamard 旋转的直接来源 | arXiv:2402.04396 |
| Sherry：Hardware-Efficient 1.25-Bit Ternary Quantization via Fine-grained Sparsification | 3:4 稀疏三值、5-bit 打包 4 权重 | STQ1_0 格式的学术依据（第 13 章） | arXiv:2601.07892 |
| Prism Bonsai-2-27B 运行时支持 | 三值 + Hadamard 折叠权重 | PTQ1_0 / Bonsai-27B 的 llama.cpp 支持（见 H.3） | llama.cpp PR #29600 |

关于 **Hadamard 折叠（Prism Hadamard）**：细纲中"Prism Hadamard"指的并非单独一篇论文，而是上述 QuIP#/Sherry 生态中"对权重先做 Hadamard 旋转、再量化为三值"的做法（PTQ1_0 依赖的折叠权重方案，见 H.3 与第 13 章）。写作时在 arXiv 未检索到以"Prism Hadamard"为标题的单一论文；该称谓出现在 llama.cpp 生态的 fork/PR 语境中（"Runtime support for Prism Bonsai 2 27B"，PR #29600）。读者若要追寻原始数学，可从 QuIP#（arXiv:2402.04396）的 Hadamard Incoherence 出发。

## H.3 关键 PR 编号与状态（写作时核实）

> 状态以**2026-10-28** GitHub 查询为准。`open / merged / closed` 均如实记录；三值格式各 PR 状态变动频繁，引用前请复查。

| PR | 主题 | 状态 | 与本书关系 |
|---|---|---|---|
| ggml-org/llama.cpp#22836 | ggml-cpu：STQ1_0 三值量化 ARM NEON vec_dot kernel | open | STQ1_0 的 CPU 内核来源（第 13 章） |
| ggml-org/llama.cpp#27377 | ggml-cpu：STQ1_0 AVX2 vec_dot kernel | open | STQ1_0 x86 内核（第 13 章） |
| ggml-org/llama.cpp#23332 | CUDA：STQ1_0 反量化 kernel | open | STQ1_0 GPU 覆盖 |
| ggml-org/llama.cpp#28067 | ggml-metal：把 STQ1_0 从 Metal mul_mat 排除 | merged/closed | **Metal 不完整** 的直接证据（第 13、18 章） |
| ggml-org/llama.cpp#29672 | ggml：新增 PTQ1_0，128 值组三值 | open | PTQ1_0 主线化建议（第 13 章） |
| ggml-org/llama.cpp#29676 | cuda：新增 PTQ1_0 支持 | open | PTQ1_0 GPU |
| ggml-org/llama.cpp#29600 | Runtime support for Prism Bonsai 2 27B（Hadamard 折叠运行时） | merged/closed | **PTQ1_0 必须的激活侧变换**（第 13 章） |
| ggml-org/llama.cpp#27935 / #27765 | vulkan：TQ1_0 支持 | merged/closed | TQ1_0 后端覆盖（第 13 章对比） |
| ggml-org/llama.cpp#23439 | ggml/cpu：TQ1_0/TQ2_0 vec_dot 跳过全零尺度块 | open | TQ 系列优化 |

**写作时确认的关键事实（来自 PR #29672 提交说明，这是 PTQ1_0 布局的最权威一手依据）：**

> PTQ1_0 是 TQ1_0 的 trit 编码、但块大小为 128：24 字节按每字节 5 个 base-3 数字 + 2 字节按每字节 4 个数字，外加一个 fp16 尺度，合 1.75 bit/权重。权重被精确三值化，每 128 个值一个尺度，往返无损。而 TQ1_0 的 256 值块装不下两个尺度，做不到这点。

这段与细纲 §13.3.2 完全一致（28 字节 = 24 + 2 + 2 fp16，1.75 bpw），是附录与正文章节可以放心引用的官方描述。

**同一 PR 中最具冲击力的验证数字（Bonsai-27B，wiki.test.raw）：**

```
带 Hadamard 变换（该 PR + #29600）：
    Mean PPL(Q)/PPL(base)  : 1.000439 ± 0.000437
    缺失变换时（主线 llama.cpp 无法解码）：
    PPL 跃升至 1,258,506+
```

这两行是第 13 章"格式正确但缺失 Hadamard 变换仍然错"的最强实证——**同样的权重，同样的格式，只因缺少激活侧变换，PPL 从 10.23 量级崩到百万量级**。

## H.4 三值格式的 type id 归属（生态敏感点）

- **主线（2026-10）type id**：F32=0，F16=1，Q4_0=2，Q4_1=3，Q5_0=6，Q5_1=7，Q8_0=8，Q8_1=9，Q2_K=10，Q3_K=11，Q4_K=12，Q5_K=13，Q6_K=14，Q8_K=15，IQ2_XXS=16，IQ2_XS=17，IQ3_XXS=18，IQ1_S=19，IQ4_NL=20，IQ3_S=21，IQ2_S=22，IQ4_XS=23，TQ1_0=34，TQ2_0=35，MXFP4=39，NVFP4=40，Q1_0=41，Q2_0=42。详细表见 `asset/facts-verified.md` §1。
- **STQ1_0 / PTQ1_0 不在主线**。三值格式的 type id 在 fork/PR 语境中有多种记载（细纲提到 PTQ1_0 记为 143、STQ1_0 有 42/43 的差异）；而主线 41/42 已被 Q1_0/Q2_0 占用。Feature Request #29058 提到 **Prism PQ2_0 用 GGML type 142、PTQ1_0 用 type 143**（Ternary-Bonsai-2 模型）。**结论：遇到底层 type id 为 140+ 的模型，先确认你的构建来自哪个 fork、是否支持该 id，再谈部署**（第 13、19 章）。
- 已移除的 id：4（Q4_2）、5（Q4_3）、31–33（Q4_0_4_4/4_8/8_8）、36–38（IQ4_NL_4_4/4_8/8_8）在主线 `type_traits` 中标记为 `REMOVED`，读取遇到应视为"不再支持"而非合法格式。

## H.5 锁定版本说明

本书正文与附录锁定以下版本，避免"今天能跑、明天不跑"的歧义：

| 组件 | 锁定版本/日期 | 说明 |
|---|---|---|
| llama.cpp | master（2026-10-28 验证） | 三值格式 PR 状态以该日 GitHub 查询为准 |
| gguf-py | 0.19.0 | `GGUFReader`/`GGUFWriter`/`dequantize` API 以此版本为基准 |
| GGUF 容器版本 | v3 | `GGUFReader` 支持 v2/v3 |
| 默认对齐 | `general.alignment = 32` | 第 3 章 mmap 讨论的前提 |
| Bonsai-27B 实验 | PR #29600 / #29672 提交说明 | PPL 数据来自 wiki.test.raw 实测 |

## H.6 数据来源与复现提示

本书正文中出现的 PPL、KL、tok/s 等数字，均标注"约"并注明"某个模型/评测集下测得"。若要复现，标准命令链见附录 F；任何声称"某个格式一定多好/多差"的数字，都应在本机按"转换 → imatrix → 量化 → PPL/KL → 端侧基准"的完整流程重新验证（第 18 章），**尤其对 IQ 与三值格式，单一 PPL 数字不足以断言可用性**。