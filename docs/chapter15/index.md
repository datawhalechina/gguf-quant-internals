# 第15章 量化工具链与工作流

从第 3 章一路走到第 14 章，我们都在研究"格式"本身：Q4_0 的 18 字节块怎么解码，Q4_K 的 6-bit 尺度怎么交错打包，IQ1_S 的码本怎么查表，STQ1_0 的 3:4 稀疏又是怎么回事。但你很可能憋着一个非常实际的问题：**这些格式，到底是怎么把一个 Hugging Face 上的模型，变成我手机里能跑的那个 `.gguf` 文件的？**

这一章就来回答这个问题。它会把前文讲过的所有格式，落到一条**可执行、可复现的命令流水线**上：从 safetensors 权重出发，先转换成 F16 GGUF，再生成重要性矩阵（imatrix），然后量化成目标格式，最后做验证和部署前检查。这条流水线用到的工具其实不多——主要是 `convert_hf_to_gguf.py`、`llama-quantize`、`llama-imatrix`、`llama-perplexity` 这几个名字。不过每一步之间都有必须遵守的依赖关系，尤其是 IQ 系列和三值格式，它们对工具链还有格外多的要求。

先看整条流水线的全貌：

```
从 Hugging Face 权重到端侧部署的完整流水线

  safetensors / PyTorch 权重
  ┌───────────────────────────────┐
  │ ① convert_hf_to_gguf.py       │  ← 只做格式转换，不量化
  │    模型被"翻译"成 GGUF 容器    │
  └──────────────┬────────────────┘
                 ▼
        model-f16.gguf（F16 基线）
                 │
     ┌───────────┴───────────┐
     │ ② 需要 IQ 系列吗？     │
     │ ┌──────────────────┐  │
     │ │ ③ llama-imatrix   │  │ ← 生成权重重要性矩阵
     │ │   imatrix.dat    │  │   （IQ 量化必需）
     │ └────────┬─────────┘  │
     └──────────┼────────────┘
                ▼
     ④ llama-quantize model-f16.gguf \
        model-q4_k_m.gguf Q4_K_M      \
        --imatrix imatrix.dat   ← 有则传入
                │
                ▼
        model-q4_k_m.gguf（量化产物）
                │
     ┌──────────┴──────────┐
     │ ⑤ 验证与部署前检查   │
     │ llama-perplexity → PPL/KL
     │ gguf-py → 类型分布检查
     │ llama-cli → 冒烟生成
     └─────────────────────┘
```

这一章的很多内容，可以看作是对前文知识的"落地打包"：第 11.2 节提过 imatrix 的数学形式，第 14 章讲过 `_M` 是混合策略，这里就把它们串成可以照着敲的命令。先统一几个术语。**safetensors**：Hugging Face 生态的权重存储格式，把每个张量以二进制形式存进一个文件。**calibration data（校准数据）**：一批用于"观察"模型行为的文本，imatrix 和量化误差评估都会用到它——它不是训练数据，只是让量化器知道真实输入长什么样。**PPL（perplexity，困惑度）**：衡量模型预测能力的指标，越低越好；量化后 PPL 相对基线的抬升越小，质量损失越小。**KL 散度**：衡量两个概率分布（这里指量化前后模型的输出分布）差异的指标。**冒烟测试（smoke test）**：一次极短的运行，用来确认"能跑起来、没有明显错误"，而不是做完整评测。


## 15.1 转换与量化基础命令

### 15.1.1 `convert_hf_to_gguf.py`：从 safetensors 到 F16 GGUF

流水线的第一步是**转换**。注意，这一步不做量化。`convert_hf_to_gguf.py` 是 llama.cpp 仓库里的一个 Python 脚本，它的工作可以概括成几个动作：读取 Hugging Face 目录里的 safetensors（或 PyTorch checkpoint）权重，把它们搬进 GGUF 这个容器里，为每个张量写好第 3 章讲过的那些 `tensor_infos`（名字、维度、类型、偏移），同时把模型架构、词表等元数据也一并写进去。

最基本的用法：

```bash
python convert_hf_to_gguf.py <hf-dir> --outfile model-f16.gguf --outtype f16
```

- `<hf-dir>`：包含 `config.json`、`model.safetensors` 等文件的 Hugging Face 模型目录。
- `--outfile model-f16.gguf`：输出的文件名（不写的话，默认用模型名）。
- `--outtype f16`：指定输出的类型。默认值就是 f16，换句话说，**转换阶段产出的通常是一个 F16 GGUF**——每个权重仍然是 2 字节的浮点数，完全没有量化。

这里有一个新手最容易忽略的关键点：**这个 F16 GGUF 并不是拿来部署用的产物，而是后续所有量化的"原料"**。你可能要问：为什么不能直接从 safetensors 开始量化？原因是，量化工具统一只吃 GGUF 输入，而且 F16 本身是一个"无损的中间格式"——从 F16 再量化，和从原始浮点直接量化，得到的数值几乎一样（第 2 章讲过，F16 对权重本身已经足够精确了）。更重要的是，imatrix 也需要这个 F16 GGUF 作为输入来跑模型。所以整条流水线里永远有一个 `model-f16.gguf` 存在，后面所有命令都指向它。

```
为什么转换阶段不做量化？

  safetensors ──convert──► model-f16.gguf ──quantize──► model-q4_k_m.gguf
   (FP32/FP16)    只搬容器       (原料)              (真正量化)
                                          ▲
                              llama-imatrix 也吃这个文件做校准

  第 3 章讲过 GGUF 是"容器"：convert 只负责装箱，
  量化是装完箱之后、单独用 llama-quantize 做的一步。
```

**重要概念需要附带代码。** 转换说白了就是"搬容器"；对编译型语言来说，最直观的等价物，是一个"读入权重目录、写出 GGUF header + tensor_infos + 数据"的小工具。下面用三种语言各写一个最小示例。它们做的事情完全一样：接收一串"名字 + 浮点数组"，然后写成一个合法的 F16 GGUF 文件头和数据段。要提醒的是，这里展示的只是**概念骨架**（真正的 `convert_hf_to_gguf.py` 还要做张量命名映射、架构检测、词表处理等一大堆事），但它足够说明"容器转换"和"量化"是两码事。

**C++：**

```cpp
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// 一个最小 GGUF 写入器：只支持 F16 张量
void write_gguf_f16(const char * path,
                    const std::vector<std::string> & names,
                    const std::vector<std::vector<float>> & tensors) {
    FILE * f = fopen(path, "wb");

    // header：magic "GGUF" + version 3 + tensor_count + kv_count
    uint32_t magic = 0x46554747;                     // "GGUF"
    uint32_t vers  = 3;
    uint64_t tc    = tensors.size();
    uint64_t kc    = 0;
    fwrite(&magic, 4, 1, f);
    fwrite(&vers,  4, 1, f);
    fwrite(&tc,    8, 1, f);
    fwrite(&kc,    8, 1, f);

    // tensor_infos：name + n_dims + dims[] + type + offset
    // type 1 = GGML_TYPE_F16；offset 相对数据段起点
    uint64_t offset = 0;
    for (size_t i = 0; i < tensors.size(); i++) {
        uint64_t nlen = names[i].size();
        fwrite(&nlen, 8, 1, f);
        fwrite(names[i].data(), nlen, 1, f);

        uint32_t n_dims = 1;                          // 假设一维
        uint64_t dim0   = tensors[i].size();
        fwrite(&n_dims, 4, 1, f);
        fwrite(&dim0,   8, 1, f);

        uint32_t type = 1;                            // F16
        fwrite(&type, 4, 1, f);
        fwrite(&offset, 8, 1, f);

        offset += tensors[i].size() * 2;              // F16 = 2 字节/权重
    }
    // 对齐到 32 字节后写数据（这里假设已对齐，略去 padding）
    for (size_t i = 0; i < tensors.size(); i++)
        for (float v : tensors[i])
            fwrite(/* FP32→FP16 编码后 */ "  ", 2, 1, f); // 示意

    fclose(f);
}
```

**C#（.NET 5+）：**

```csharp
using System;
using System.IO;
using System.Text;

static class GgufWriter {
    public static void WriteF16(string path, string[] names, float[][] tensors) {
        using var w = new BinaryWriter(File.Create(path), Encoding.UTF8);
        w.Write(Encoding.ASCII.GetBytes("GGUF"));
        w.Write((uint)3);                // version
        w.Write((ulong)tensors.Length);  // tensor_count
        w.Write((ulong)0);               // kv_count

        ulong offset = 0;
        for (int i = 0; i < tensors.Length; i++) {
            byte[] nameBytes = Encoding.UTF8.GetBytes(names[i]);
            w.Write((ulong)nameBytes.Length);
            w.Write(nameBytes);

            w.Write((uint)1);                            // n_dims
            w.Write((ulong)tensors[i].Length);           // dims[0]
            w.Write((uint)1);                            // type = F16
            w.Write(offset);                             // 相对数据段偏移

            offset += (ulong)tensors[i].Length * 2;
        }
        // 数据段：每个 float 先转成船村 16 位位模式再写（示意，见下）
        for (int i = 0; i < tensors.Length; i++)
            foreach (float v in tensors[i])
                w.Write(BitConverter.HalfToUInt16Bits((Half)v)); // F16 2 字节
    }
}
```

这个 `HalfToUInt16Bits` 的作用，是把 `Half` 转成它的 16 位位模式——这正是 F16 在磁盘上的真实样子，和 llama.cpp 的 `ggml_half` 完全一致。

**Python（`struct` + `array`，最接近官方脚本的思路）：**

```python
import struct

def write_gguf_f16(path: str, names: list[str], tensors: list[list[float]]) -> None:
    with open(path, "wb") as f:
        # header（小端）
        f.write(struct.pack("<I", 0x46554747))   # "GGUF"
        f.write(struct.pack("<I", 3))            # version
        f.write(struct.pack("<Q", len(tensors))) # tensor_count
        f.write(struct.pack("<Q", 0))            # kv_count

        # tensor_infos
        offset = 0
        for name, tensor in zip(names, tensors):
            name_b = name.encode("utf-8")
            f.write(struct.pack("<Q", len(name_b)))
            f.write(name_b)
            f.write(struct.pack("<I", 1))          # n_dims
            f.write(struct.pack("<Q", len(tensor))) # dims[0]
            f.write(struct.pack("<I", 1))          # type = F16
            f.write(struct.pack("<Q", offset))     # 相对数据段偏移
            offset += len(tensor) * 2

        # 数据段：FP32 → FP16 位模式（'<e' 是小端半精度）
        for _, tensor in zip(names, tensors):
            f.write(struct.pack(f"<{len(tensor)}e", *tensor))
```

三份代码的字段一一对应，写出来的都是 `"GGUF" | version=3 | tensor_count | kv_count | 目录 | 数据` 这样的合法 v3 文件。**注意，整个转换阶段从头到尾只有 `<Q`/`<I`/`<e` 三种打包方式，看不到任何 `quantize` 的身影**——这就是"转换不等于量化"在代码层面最直接的证据。

### 15.1.2 `llama-quantize`：格式选择与输出

真正动手量化的是第二步：`llama-quantize`。它的用法是一个 `input output type` 的三段式：

```bash
./llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M
```

- `model-f16.gguf`：也就是 15.1.1 产出的那份 F16 原料。
- `model-q4_k_m.gguf`：量化结果的输出文件名。
- `Q4_K_M`：你想量到的目标格式名。

敲下回车之后，`llama-quantize` 会逐个张量地调用第 4–14 章讲过的那些参考量化函数（`quantize_row_q4_K_ref` 等），把 F16 数据压成目标格式、替每个张量填好 `ggml_type`。如果目标格式是 `Q4_K_M`，它内部实际的分配，就是第 14 章讲过的那套混合规则：主体张量量化成 Q4_K，输出层用 Q6_K，归一化层保持 F32。

这里必须再强调一次第 3、14 章反复出现的结论：**`Q4_K_M` 是一个策略名（量化策略），而不是一个 `ggml_type`**。`ggml_type` 是文件里每个张量 `type` 字段要填的单一类型编号（比如 `GGML_TYPE_Q4_K` = 12）；`Q4_K_M` 则是 `llama-quantize` 认识的一个目标格式名，它背后藏着"对每类张量分别选类型"的规则。可以对照下面这张图来理解：

```
Q4_K_M 不是一种 ggml_type

  文件名目标：        Q4_K_M        （llama-quantize 的参数）
                        │
   内部逐张量分配 ────┬─────────────────────────────────┐
                     ▼                          ▼
   主体张量: Q4_K (type id 12)    输出层: Q6_K (14)  归一化: F32 (0)
   （这些才是文件里真正出现的 ggml_type）
```

`llama-quantize` 支持很多目标格式，常用的大致分三类，正好对应本书的三个档次：

| 目标格式 | 说明 | 对应本书章节 |
|---|---|---|
| `F16` / `F32` | 不量化或几乎不量化，只做容器重排 | 第 3 章 |
| `Q4_0`、`Q8_0` | 简单块量化，实现最简单 | 第 4、5 章 |
| `Q2_K`…`Q6_K` 及 `_S`/`_M`/`_L` | K-quants 家族与混合策略 | 第 6–10、14 章 |
| `IQ1_S`、`IQ2_XXS`、`IQ4_NL` 等 | IQ 码本量化，**需要 imatrix** | 第 11 章 |

除此之外，还能用 `-t` 指定线程数、用 `-a` 指定架构等。但对本书的读者来说，最重要的选项不是这些，而是下一节的主角：`--imatrix`。


## 15.2 imatrix 与 IQ 量化工作流

### 15.2.1 校准数据准备与 `llama-imatrix` 参数

在讨论 IQ 量化之前，先确认一个前提：**K-quants 不需要 imatrix，IQ 才需要。** Q4_K 的量化器只看权重本身（块的最大值、最小值、分布），就决定尺度和码；而 IQ 系列的量化器，在"权重的绝对值"之外，还要参考"这个权重对模型输出有多重要"。用来衡量这种重要性的矩阵，就叫 **imatrix（importance matrix，重要性矩阵）**——它正是第 11.2 节那个加权最小二乘目标里的权重 \(\alpha_i\) 的具体形态。

imatrix 不能凭空生成，需要**跑一遍模型**：读一批文本进入模型，观察每个权重在计算中实际被多大的激活值流过，据此累计出每个权重的重要性。生成它的工具是 `llama-imatrix`：

```bash
./llama-imatrix -m model-f16.gguf -f calibration.txt -ngl 99 --chunks 512 -c 2048 -o imatrix.dat
```

各参数的含义：

| 参数 | 含义 | 典型值 |
|---|---|---|
| `-m` | 输入模型，必须是 F16 GGUF（原料） | `model-f16.gguf` |
| `-f` | 校准数据文件（纯文本） | `calibration.txt` |
| `-b` | 批大小（每次前向处理多少 token） | 512 |
| `-c` | 上下文长度（每次处理的序列多长） | 2048 |
| `--chunks` | 校准文本块数（重复次数总量） | 512（越大越准、越慢） |
| `-ngl` | 有多少层放到 GPU 上跑 | 99（等于全量放 GPU） |
| `-o` | 输出 imatrix 文件 | `imatrix.dat` |

**校准数据的选择直接决定 imatrix 的质量**，进而决定 IQ 量化的质量。两个要点：

1. **规模要够**。`--chunks` 太小，累计的重要性统计噪声大，量化会不稳定。实践经验中，几百个 chunk（每 chunk 一个上下文长度）是常见起点。
2. **领域要匹配部署场景**。如果模型部署后主要处理中文对话，校准数据最好也是中文对话文本，而不是英文维基百科。原因很朴素：imatrix 衡量的是"哪些权重对这个输入分布更重要"，输入分布偏了，重要性就偏了。llama.cpp 官方建议校准文本覆盖多种风格（`--file` 可以多次传入不同文件），理想情况下含代码、对话、文章等混合内容。

对 IQ 系列，这个文件几乎决定了上限——**同样一个模型、同样一个目标格式，用不用 imatrix、用什么样的校准数据，量化后的 PPL 可以拉开明显差距**。所以它不是一个"可选的优化项"，而是 IQ 工作流的事实上的标准环节。15.2.2 会看到它在量化时被显式传入。

### 15.2.2 `--imatrix` 在 `llama-quantize` 中的传入

生成 `imatrix.dat` 之后，量化时用 `--imatrix` 把它交给 `llama-quantize`：

```bash
./llama-quantize model-f16.gguf model-iq2_xxs.gguf IQ2_XXS --imatrix imatrix.dat
```

在量化器内部，imatrix 会参与每个块的尺度估计：加权目标 \(\min \sum_i \alpha_i (x_i - \hat{x}_i)^2\) 中的 \(\alpha_i\) 现在不再是均匀的 1，而是来自 imatrix 的激活重要性。放行误差预算多的权重少损失一点精度，不重要的权重多牺牲一点——这就是第 11 章那套"误差预算按重要性重新分配"思想的落地。

再强调一次这个容易踩的坑：**对 IQ 系列，这个参数不是可选的**。原因有两层：

- **工具层**：`llama-imatrix` 若不提供 imatrix，很多 IQ 格式的参考量化实现（`from_float_ref`）在主线里甚至是 `NULL`——注意第 11 章提到过，IQ 系列某些类型的量化实现依赖量化权重（quant_weights）；没有 imatrix 时 `llama-quantize` 要么拒绝量化，要么退化成明显更差的质量。
- **格式设计层**：IQ 的码本是为"加权误差"设计的。把权重本身的重要性当作常数来压，结果会系统性地在重要位置引入误差。这不只是"少了一个优化配置"，而是**没有 imatrix 就不满足这个格式的工作前提**。

```
为什么 IQ 离不开 imatrix？

  无 imatrix：                有 imatrix：
  ┌───────────────────────┐   ┌───────────────────────┐
  │ 误差均匀分摊:          │   │ 误差按重要性分摊:        │
  │ 每个权重扣同样多精度    │   │ 重要权重少扣、次要权重多扣│
  │ ███ ███ ███ ███       │   │ ██  ████  █  █████     │
  │ 错误分布平摊 → 处处中低 │   │ 错误集中在不影响输出的位│
  └───────────────────────┘   │ 置 → 整体质量更好      │
                              └───────────────────────┘
```

对于 K-quants（Q4_K_M 等），`--imatrix` 同样可以传，量化器会用它做加权估计（第 14 章提过 `quantize_row_q4_K_impl` 的 `quant_weights` 参数），通常也能带来小幅改善，但不是必需。**换句话说：传 imatrix 对 K-quants 是锦上添花，对 IQ 是没有它就没有好质量。**


## 15.3 三值格式的工具链差异

### 15.3.1 PTQ1_0 的 Hadamard 折叠依赖

第 13 章专门强调过：PTQ1_0 的磁盘权重是以 **Hadamard 折叠（Hadamard folding）** 后的形式存储的。也就是说，文件里每个三值块对应的不是原始权重，而是经过一层正交变换后的权重；推理时必须先在激活侧做对应的逆变换，计算才能正确。

这条性质对工具链有一个非常直接的后果：**PTQ1_0 的量化端不能是"拿权重直接压三值"这么简单**。它需要：

1. 先把原始权重做 Hadamard 折叠，得到折叠后的权重；
2. 把折叠后的权重量化为三值（base-3 编码）写进文件；
3. 同时，把折叠所需的元信息（变换的缩放/符号结构等）也写进文件，让推理端能重建变换。

缺了第一步或第三步，文件里的字节可能是"合法"的（解析器能读懂三值码），但推理结果完全错误——这正是"格式正确但缺失旋转仍然是错误的"那句话在工具链层面的含义。**所以使用 PTQ1_0 时的第一个检查项是：你用的量化工具（很可能是某个特定版本或 fork）是否真的执行了 Hadamard 折叠？** 不能只看它认不认得 `PTQ1_0` 这个名字。

### 15.3.2 分支/fork 与上游合并状态

第 13 章还提到一个生态现实，这里要把它明确为**操作纪律**：STQ1_0 和 PTQ1_0 **不在 llama.cpp 主线**（截至写作时，主线 type id 41/42 已被更早合入的 Q1_0/Q2_0 占用，三值格式出现在特定 fork 和未合并的 PR 里）。这意味着：

- **量化端可能不在你的 llama.cpp 构建里。** 你手头的 `llama-quantize` 是不是带 STQ1_0/PTQ1_0 支持的那个版本？需要确认，而不是假设。
- **推理端同理。** 即使你能量化出 STQ1_0 文件，运行时的 llama.cpp 构建若没有这个类型的 `dequantize_row_stq1_0` 和后端内核，加载会失败或解码错误。
- **PTQ1_0 还叠加了 Hadamard 变换 PR。** 支持 PTQ1_0 的运行时必须同时包含折叠变换的推理路径。

所以面对三值格式，工具链的黄金法则不是"格式名可识别就可量化、可运行"，而是：**找到承载该格式的具体 fork/PR，确认它包含量化端、推理端、必要的变换三步，并且把这一步记录进你的工程文档**。这也是第 3.4 节"type id 兼容性风险"的实战版——三值格式就是当下最典型的案例。


## 15.4 验证与部署前检查

### 15.4.1 类型分布检查

量化完成不等于可以部署。部署前最小检查清单的第一步，是用 gguf-py 统计一下**文件里真实出现的张量类型分布**——这能一次性抓住三类常见问题：

1. **策略是否按预期落盘**：选 `Q4_K_M` 却看到大片 F32 或 Q8_0，说明混合规则没生效或格式名敲错；
2. **有没有意外的主营类型**：文件里混进一个 `F32` 权重本来预期缩水，体积因此虚高；
3. **三值/IQ 格式的 type id 是否被识别**：如果读取方的 `GGMLQuantizationType` 不认识某个 id，这里就会现出原形。

**Python（用 gguf-py，最直接的验证脚本）：**

```python
from collections import Counter
from gguf import GGUFReader, GGMLQuantizationType

reader = GGUFReader("model-q4_k_m.gguf")

# 逐张量统计类型
dist: Counter[GGMLQuantizationType] = Counter(t.tensor_type for t in reader.tensors)

# 打印分布 + 体积占比
total_bytes = sum(t.n_bytes for t in reader.tensors)
print(f"{'类型':<20} {'张量数':>6} {'字节':>12} {'占比':>8}")
for qtype, count in dist.most_common():
    nbytes   = sum(t.n_bytes for t in reader.tensors if t.tensor_type == qtype)
    print(f"{qtype.name:<20} {count:>6} {nbytes:>12} {nbytes/total_bytes:>7.1%}")

# 快速断言：Q4_K_M 主体应是 Q4_K
assert dist[GGMLQuantizationType.Q4_K] > 0, "未发现预期的 Q4_K 张量"
```

**C++（用 llama.cpp 的 GGUF 读取接口的思想——即第 3.3.2 节 `gguf-py` 的 C 层对应物，示意）：

```cpp
#include "gguf.h"      // llama.cpp 的 GGUF 读取头文件
#include <cstdio>
#include <unordered_map>

void check_type_distribution(const char * path) {
    // llama.cpp 内部流程（示意）：gguf_init → 逐张量读 type
    struct gguf_init_params params = { /* .no_alloc=true 等 */ };
    struct gguf_context * ctx = gguf_init_from_file(path, params);

    int n = gguf_get_n_tensors(ctx);
    std::unordered_map<int, int> dist;                 // type id → 张量数
    for (int i = 0; i < n; i++) {
        int id = gguf_find_tensor(ctx, gguf_get_tensor_name(ctx, i));
        // 实际是 gguf_get tensor type；这里示意统计逻辑
        dist[id]++;
    }
    for (auto & [id, count] : dist)
        printf("type_id=%d  count=%d\n", id, count);

    gguf_free(ctx);
}
```

这段是结构示意——llama.cpp 的 C API 细节（`gguf_get_tensor_type`、按名查找等）会随版本演进，重点是**逻辑**：遍历张量 → 读出每个 `type` → 计数。这正是 `ReaderTensor.tensor_type` 在 Python 侧做的同一件事。

**C#（手写 GGUF 头解析，统计类型 id——把第 3 章的方法用到验证上）：**

```csharp
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

static class TypeCheck {
    static long ReadStr(BinaryReader r, out byte[] data) {
        long len  = r.ReadInt64();       // 字符串长度（u64）
        data = r.ReadBytes((int)len);    // 字符串本身
        return len;
    }

    public static void Main(string[] args) {
        using var r = new BinaryReader(File.OpenRead(args[0]));
        r.ReadBytes(4);        // magic "GGUF"
        r.ReadInt32();         // version
        long tensorCount = r.ReadInt64();
        long kvCount     = r.ReadInt64();

        // 跳过全部 KV（只读 key + 类型 + 按类型跳过 value）
        static int ValueSize(int t) => t switch {
            4 or 5 or 6 or 7 => 1,    // UINT8/INT8/UINT16/INT16 示意
            8 or 9 or 10     => 2,    // （演示）实际按完整类型表
            _                => 4,
        };
        for (long i = 0; i < kvCount; i++) {
            ReadStr(r, out _);
            int vtype = r.ReadInt32();
            // 简化：仅处理标量/定长；字符串与数组需递归跳过
            r.BaseStream.Seek(ValueSize(vtype), SeekOrigin.Current);
        }

        // tensor_infos：统计 type 字段
        var dist = new Dictionary<int, int>();
        for (long i = 0; i < tensorCount; i++) {
            ReadStr(r, out _);          // 张量名
            int nDims = r.ReadInt32();
            for (int d = 0; d < nDims; d++) r.ReadInt64(); // dims[]
            int type = r.ReadInt32();
            r.ReadInt64();              // offset
            dist[type] = dist.GetValueOrDefault(type) + 1;
        }
        foreach (var (t, c) in dist) Console.WriteLine($"type={t} count={c}");
    }
}
```

C# 版为了可运行做了两点简化（KV 的定长值直接 seek、未递归处理字符串/数组），但它足够演示验证思路：**类型分布检查本质上就是读 `tensor_infos` 里的 `type` 字段**。生产上直接用 gguf-py 或 llama.cpp 的 API 即可。

### 15.4.2 PPL/KL 快速验证与后端冒烟测试

类型分布对了，只说明"文件结构合法"，不说明"量化损失可接受"。第二步是质量验证。llama.cpp 提供的工具是 `llama-perplexity`，它读一段文本，计算模型在文本上的 PPL，并可与基线对比：

```bash
# 计算量化模型的 PPL，并指定"基线模型"用于 KL 散度对比
./llama-perplexity -m model-q4_k_m.gguf -f test.txt --kl-divergence-base model-f16.gguf
```

- `-f test.txt`：评测用的测试文本。**评测数据要和校准数据区分开**——用喂过 imatrix 的文本去量 PPL 会偏乐观。
- `--kl-divergence-base model-f16.gguf`：让工具同时算量化模型与 F16 基线之间的 KL 散度。

读输出时注意第 2.5 节讲的指标口径：

- **PPL**：看量化后相对基线的**抬升比例**。Q4_K_M 通常约 +0.8%、Q6_K 约 +0.2%（具体数值依赖模型与测试集，不同来源不可直接比较）。
- **KL 散度**：对 IQ/三值这类"逐 logits 形状敏感"的格式，KL 往往比单一 PPL 更能反映输出分布的偏离，因为它直接比较量化前后模型对同一输入的输出分布（见第 18.2 节）。

第三步，也是最后一道保险：**在目标后端上跑一次短生成作为冒烟测试**。这一步专门抓"加载解包错误"和"缺失运行时变换"这两类前面所有检查都发现不了的问题——可能是某类型在该后端（CPU SIMD？Metal？Vulkan？CUDA？）没有内核，也可能是 PTQ1_0 的 Hadamard 逆变换没被执行。

```bash
./llama-cli -m model-q4_k_m.gguf -p "Hello" -n 64 -ngl 99
```

正常应该看到一句合理的续写。如果输出是乱码、NaN、报 `unsupported type` 或直接崩溃，就说明某个环节（类型分派、后端内核、缺失变换）出了问题。对三值格式，这一步尤其重要——CPU 测试通过不代表设备的 GPU delegate 支持该类型。

```
部署前的最小检查清单（按顺序）

  ① 类型分布检查    gguf-py 统计 → 确认混合策略符合预期
  ② PPL/KL 快速验证 llama-perplexity → 相对 F16 基线的劣化可接受
  ③ 目标后端起一次短生成 llama-cli → 无解包错误、无缺失变换
  ④（三值格式加测） 逐格式确认量化端/推理端/变换三步的版本来源
```


## 本章小结与通向"代码架构"的桥梁

这一章把前文的格式知识落成了一条命令流水线。三个事实：

**第一，转换与量化是两步。** `convert_hf_to_gguf.py` 只做容器转换，产出 F16 GGUF 原料；`llama-quantize` 才真正量化。`Q4_K_M` 是策略名而非单一 `ggml_type`，文件内部是"Q4_K 主体 + Q6_K 输出层 + F32 归一化"的混合。

**第二，imatrix 对 IQ 系列不是可选项。** `llama-imatrix` 用 F16 原料和校准数据生成重要性矩阵，量化时用 `--imatrix` 传入；它让误差预算按权重重要性重新分配。缺了它，IQ 量化的质量会显著下降——这是格式设计的一部分。

**第三，三值和特殊格式有额外的工具链纪律。** STQ1_0/PTQ1_0 不在主线（位于 fork/PR），PTQ1_0 还叠加 Hadamard 折叠变换；必须确认工具链版本包含"量化端 + 推理端 + 变换"三步，部署前用类型分布、PPL/KL、后端冒烟测试三道检查兜底。

```
回顾：一课一图

  F16 GGUF ──imatrix──► imatrix.dat ──+──► llama-quantize ──► 部署 SIGMA
      │                  ▲            │        │
      │                  └─ 校准数据──┘        ▼
      └─────────────────────────────────── llama-perplexity / llama-cli
                                            验证与冒烟
```

到这里，你已经能把一个模型从 Hugging Face 一路量化到可部署的 GGUF 了。但也许你已经发现一个疑问：`llama-quantize` 说量化就量化，它内部到底是怎么把"目标格式名"变成"每个张量调一个量化函数"的？那些 `quantize_row_q4_K_ref`、`dequantize_row_q4_K`、`vec_dot` 函数分别在哪里定义、由谁调度？如果我自己想加一个新格式，要改哪几个文件？下一章，我们走进 llama.cpp 的量化代码架构，从"想新增一个格式该改哪些文件"的角度，把这套机制的源代码解剖给你看。