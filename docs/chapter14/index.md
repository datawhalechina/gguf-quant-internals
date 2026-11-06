# 第14章 `_S`/`_M`/`_L` 与逐张量混合精度

前十三章一直在研究"一个格式怎么工作"：Q4_0 的块、Q4_K 的超级块、IQ 的码本、STQ/PTQ 的三值。从这一章开始，我们把视角从"格式"抬升到"策略"：**一个能用的模型，并不是所有张量都套用同一种格式，而是按敏感度，给不同张量分配不同格式**。

先给你一个具体的感受。在命令行里执行 `llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M`，得到的 `model-q4_k_m.gguf` 里，可不是每一个张量都叫 `Q4_K`。事实上，文件里同时存在三种类型：

- 大部分注意力投影和 FFN 矩阵是 `Q4_K`（4-bit 主体）；
- 输出嵌入层是 `Q6_K`（6-bit，更高精度）；
- 一堆归一化层是 `F32`（32-bit，根本不量化）。

用第 3 章的 gguf-py 脚本一统计，真相立刻浮出水面：`Q4_K_M` 是一个**混合策略的名字**，而不是单一的编码格式。这也正是本章的主题：**为什么输出层要给 6-bit？为什么归一化层要保留 F32？不同位置的张量，敏感度到底差在哪里？**

先统一几个术语。**张量敏感度（tensor sensitivity）**：某个张量对量化误差的容忍程度——量化它造成的质量损失越小，说明它越"皮实"；损失越大，越"娇贵"。**logits**：模型输出的原始分数向量，softmax 之前的值，直接决定下一个 token 的概率分布。**输出嵌入层（output embedding）**：把最后一个隐藏层映射到词表大小的层，在 GPT 类模型里常与输入词嵌入共享权重（tied）。**FFN（feed-forward network）**：Transformer 块中的前馈网络，通常由 gate/up 和 down 两组投影组成。**LayerNorm**：把每个位置的向量归一化的层，对缩放非常敏感。


## 14.1 张量敏感度分层

### 14.1.1 输出嵌入层：对 logits 的直接敏感

先问一个问题：为什么 `Q4_K_M` 非要让输出嵌入层用 `Q6_K`，而不和其他层一样用 `Q4_K`？

答案藏在"输出层在模型里的地位"。绝大多数张量（比如注意力投影）参与的是**中间计算**，它们产生的一点误差，会被后面一层层的矩阵乘慢慢平均掉、稀释掉。可输出嵌入层是**最后一个线性层**：它的输出就是 logits，直接决定每个 token 的概率。换句话说，输出层的量化误差没有后续层帮它平滑，而是**直接进入最终的概率分布**。

这里可以用一个类比来理解。想象一场接力赛：前面几棒跑得快一点还是慢一点，其实没那么要紧，只要总时间对得上；真正决定名次的，是最后一棒怎么冲线。输出层就是"最后一棒"——它的一丁点偏差，都会被完整地反映到 logits 上。

量化误差对 logits 的形状影响，可以用两个指标来衡量：

- **PPL（困惑度）**：对整体的概率分布敏感。输出层一旦有误差，概率分布就会变得"更不确定"，PPL 随之上升。
- **KL 散度**：对 logits 的具体形状敏感。输出层误差会让量化后模型的 logits 偏离原始模型的 logits，哪怕 PPL 的变化并不大。

第 2.5.2 节说过，IQ 和极限格式常常用 KL 去和 base 模型做对比，原因就在这里——**输出层对 KL 的影响是超比例的**。实测经验里，仅仅把输出层从 `Q4_K` 升到 `Q6_K`，KL 散度就能明显改善，而文件体积只多出一点点（毕竟输出层只占全模型的一小部分）。这就是所谓的"把钱花在刀刃上"。

### 14.1.2 注意力 QKV、FFN gate/up/down 的差异

行内层的敏感度也不是铁板一块。Transformer 的每个块里有两大类线性投影：

1. **注意力（attention）投影**：`attn_q`、`attn_k`、`attn_v`，以及部分架构的 `attn_output`。它们把输入向量投影到查询/键/值空间。
2. **前馈网络（FFN）投影**：`ffn_gate`、`ffn_up`（放大维度）和 `ffn_down`（缩回维度）。

经验上存在一个规律：**FFN down 有时需要比 QKV 更高的位宽**。直觉解释：

- QKV 投影的误差会被注意力权重（softmax 之后的值）"过滤"掉一部分——softmax 会把大部分注意力权重压到接近零，只有少数 token 的贡献真正显著，于是 QKV 里的随机误差，很大程度上被这种门控效应钝化了。
- FFN down 则恰好相反。它直接决定块的输出 \(x + \mathrm{down}(\mathrm{gate}(x) \odot \mathrm{up}(x))\)，误差会原样叠加进残差流，再一路传递到最后的 logits。

这个差异落到实践中就是一句话：**如果一个模型想在 Q4_K 的基础上再抠一点质量，应该优先把 `ffn_down` 升级到 Q5_K 或 Q6_K，而不是去动注意力层**。这也解释了不同 `_M` 等级之间，"哪些张量升、哪些张量降"为什么会有一部分差异。

### 14.1.3 归一化与嵌入：保留 F32 的理由

第三类"娇贵"的张量，是归一化层和部分嵌入层。你可能已经注意到：为什么 `Q4_K_M` 文件里会冒出一堆 `F32`？道理很朴素——它们本来占的体积就很小，而**量化它们省下来的那几个字节，远远抵不上损失的质量**。

**LayerNorm/RMSNorm 的权重**（`attn_norm`、`ffn_norm`）：这两类层做的是逐元素归一化，每个维度的缩放因子会直接乘到归一化后的值上。所以它们对"尺度对不对"极其敏感——一旦量化把某个维度的权重从 1.001 变成 1.0，这个维度被缩放的比例就错了。更要命的是，**归一化层的张量本身极小**（形状就是隐藏维度，说白了只是一个向量），在总文件里占不到 1% 的字节，却扛着"把尺度保准"的责任。为了省下这 1% 的字节去赌质量，实在不划算。

**部分嵌入层**：词嵌入（`token_embd.weight`）在某些策略里也会保留较高的精度，甚至干脆保留 F32，原因在于嵌入向量直接决定了每个 token 的初始表征。

一句话总结这张敏感度分层表：**[输出层 > 归一化层] > [FFN down > QKV / FFN gate-up] > [结构性可压缩张量]**。越靠左越"娇贵"，越该用高位宽、甚至干脆用 F32；越靠右越"皮实"，可以放心用低位宽。

```
敏感度分层（示意）

  娇贵 ──────► 皮实（可压缩）

  输出层    归一化    FFN down   QKV /      结构性
  (logits)  (LayerNorm)         FFN gate/up   层
   │           │         │        │          │
  Q6_K/更高   F32      Q5_K/Q6_K Q4_K       Q4_K
   ｜           |         |        |          |
  直接进logits 保尺度    进残差流   被softmax  误差被后续
  不可稀释     极小张量   直接影响   钝化       计算平均
```

注意：这张图给出的只是"经验上的量级排序"，换一个架构、换一个模型，具体次序都可能微调。**敏感度分层始终是一个需要用实验去验证的假设，而不是定理**——第 18 章会具体讲怎么用实验验证它。


## 14.2 典型配置的逐张量分布

### 14.2.1 Q4_K_M：Q4_K 主体 + Q6_K 输出 + F32 归一化

现在，我们用一套真实的分析套路，来看看 `Q4_K_M` 的逐张量分布。下面的 Python 脚本按"张量名分组"来统计类型分布——它不只是数一数总量，更要让你看出其中的**规律**：

```python
from collections import defaultdict, Counter
from gguf import GGUFReader, GGMLQuantizationType

reader = GGUFReader("model-q4_k_m.gguf")

# 按张量名的"角色前缀"分组统计类型
from gguf.tensor_mapping import get_tensor_name_map  # 若可用
# 更通用的做法：按名字特征分组
groups = defaultdict(Counter)
for t in reader.tensors:
    name = t.name
    if name.startswith("output."):
        key = "output (输出层)"
    elif "norm" in name:
        key = "norm (归一化)"
    elif name.startswith("token_embd"):
        key = "token_embd (词嵌入)"
    elif "attn" in name:
        key = "attention (QKV/output 投影)"
    elif "ffn" in name:
        key = "ffn (gate/up/down)"
    else:
        key = "other"
    groups[key][t.tensor_type.name] += 1

for g, c in groups.items():
    print(f"{g:24s} " + ", ".join(f"{qt}:{n}" for qt, n in c.most_common()))
```

典型的输出（数字随模型而变，但模式稳定）：

```
output (输出层)       Q6_K:1
norm (归一化)         F32:52
attention (QKV...)    Q4_K:384
ffn (gate/up/down)    Q4_K:288
token_embd (词嵌入)   Q4_K:1
```

Python 脚本背后靠的机制，就是第 3 章讲过的"每个张量有独立的 `type` 字段"。在 C++ 和 C# 里做等价的事，无非是读出 `tensor_infos` 里那个 `type` 整数，再映射成类型名——差别只在于，是"手工映射"，还是"走 gguf-py 的枚举"。下面给出三种语言的等价视角。

**C++（手工读 type 字段并映射到类型名）：**

```cpp
#include <cstdio>
#include <cstdint>
#include <vector>
#include <string>

// 主线 ggml_type 编号 → 类型名（截取常用子集）
const char * type_name(int t) {
    switch (t) {
        case 0:  return "F32";
        case 1:  return "F16";
        case 12: return "Q4_K";
        case 13: return "Q5_K";
        case 14: return "Q6_K";
        case 29: return "IQ1_M";
        default: return "other";
    }
}

// 读取每个张量的 type 字段（假设已用 mmap 或 fread 得到 bytes）
void count_types(const uint8_t * bytes, size_t offset, size_t n_tensors) {
    std::vector<int> counts(64, 0);
    size_t off = offset;
    for (size_t i = 0; i < n_tensors; ++i) {
        // 跳过 name: u64 len + bytes
        uint64_t len;
        memcpy(&len, bytes + off, 8); off += 8 + len;
        // 跳过 n_dims: u32
        uint32_t nd;
        memcpy(&nd, bytes + off, 4); off += 4;
        // 跳过 dims: nd * u64
        off += nd * 8;
        // 读 type: u32
        uint32_t type;
        memcpy(&type, bytes + off, 4); off += 4;
        // 跳过 offset: u64
        off += 8;
        counts[type]++;
    }
    for (int t = 0; t < 64; ++t)
        if (counts[t]) printf("%-8s %d\n", type_name(t), counts[t]);
}
```

**C#（.NET 5+，同样手工解析 + 枚举映射）：**

```csharp
using System;
using System.IO;
using System.Text;

public enum GgmlType { F32 = 0, F16 = 1, Q4_K = 12, Q5_K = 13, Q6_K = 14, IQ1_M = 29 }

class TypeCounter {
    static void SkipField(BinaryReader r) {
        ulong len = r.ReadUInt64();
        r.ReadBytes((int)len);      // key
        var vtype = (int)r.ReadUInt32(); // 值类型在此处简化跳过：
        // 这里不再展开所有 GGUFValueType 的分支，仅示意跳过标量/字符串
        ulong vlen = r.ReadUInt64();
        r.ReadBytes((int)vlen);
        _ = vtype; // 完整实现见 gguf-py 的 _get_field_parts
    }

    static void ReadUInt64s(this BinaryReader r, uint n) {
        for (uint i = 0; i < n; i++) r.ReadUInt64();
    }

    public static void Main(string[] args) {
        var data = File.ReadAllBytes(args[0]);
        using var r = new BinaryReader(new MemoryStream(data));
        r.ReadBytes(4);                       // magic "GGUF"
        r.ReadUInt32();                       // version
        ulong tensors = r.ReadUInt64();       // tensor_count
        ulong kvCount = r.ReadUInt64();       // kv_count

        // 跳过 KV 数组（每个：u64 长度 + 字节 + u32 类型 + 值）
        for (ulong i = 0; i < kvCount; i++) SkipField(r);

        var counts = new int[64];
        for (ulong i = 0; i < tensors; i++) {
            ulong len = r.ReadUInt64(); r.ReadBytes((int)len);   // name
            uint nd = r.ReadUInt32(); r.ReadUInt64s(nd);        // dims
            int type = r.ReadInt32();                            // type
            r.ReadUInt64();                                     // offset
            counts[type]++;
        }
        foreach (var t in Enum.GetValues<GgmlType>())
            if (counts[(int)t] > 0) Console.WriteLine($"{t}: {counts[(int)t]}");
    }
}
```

三种实现做的是同一件事：**沿着文件顺序走一遍 tensor_infos，把每个张量的 `type` 统计进一张表**。Python 用 `GGMLQuantizationType` 枚举 + `Counter` 最省事；C++/C# 需要手工按字节跳字段（这正好巩固第 3 章的布局直觉）。

怎么读这张表？看四个要点：

1. **Q4_K 占绝对多数**——这就是所谓的"主体"。
2. **输出层单独用 Q6_K**——虽然只有 1 个张量，但它扛的是 logits 的质量。
3. **归一化层全 F32**——50+ 个张量，全是向量级的小张量。
4. **没有 Q5_K**——这提醒我们，`_M` 的"M"并不是说"每个张量都用 4.5 bpw 的 Q4_K"，而是"这样一个由规则驱动的混合"。

这正是细纲反复强调、第 3 章也预告过的那个核心事实：**_M 后缀描述的是混合策略的激进程度，而不是某一种编码格式**。

### 14.2.2 Q5_K_M、Q3_K_M 等变体的分配差异

`_S/_M/_L` 这组后缀可不是随手起的缩写。它们描述的是"混合策略的**激进程度**"：

- **S（Small）**：压缩最激进，尽量少用高位宽，所以文件最小；
- **M（Medium）**：折中，主体用目标位宽，同时保住输出层和归一化层；
- **L（Large）**：保守，让更多张量用上更高位宽，文件更大，但质量也更好。

（不同架构的 `_S/_M/_L` 具体指哪些张量，由 llama.cpp 的规则表决定，见 14.3.1。）

以 K-quants 为例，`Q3_K_S`、`Q3_K_M`、`Q3_K_L` 三者的差异大致是：**S 版几乎把所有张量都压成 Q3_K；M 版在 S 的基础上，把输出层和部分敏感层升级到 Q4_K 或 Q5_K；L 版则进一步升级更多层**。从 `Q3_K_S` 走到 `Q3_K_L`，后缀字母表达的其实是"有多少敏感张量被照顾到了"。

```
不同后缀的分配差异（示意，规则以 llama.cpp 为准）

  张量类型          Q3_K_S    Q3_K_M    Q3_K_L
  ─────────────────────────────────────────────
  主体 FFN/QKV      Q3_K      Q3_K      Q3_K
  部分敏感层        Q3_K      Q4_K      Q5_K
  输出层            Q3_K      Q4_K      Q5_K/Q6_K
  归一化            F32       F32       F32

  文件体积：  S < M < L
  预期质量：  S < M < L（通常）
```

这里有个重要提醒：**"字母越大质量就越好"并不是定理**。质量高低取决于具体的模型和评测任务。在某些模型上，S 和 M 的 PPL 差异几乎可以忽略；在另一些模型上，M 和 L 谁更好，还要看评测的是哪类任务。后缀字母只描述**策略方向**，最终的质量必须实测才作数。


## 14.3 策略生成：规则、imatrix 与搜索

### 14.3.1 llama.cpp 内置规则的确定性

`_S/_M/_L` 策略是怎么生成的？llama.cpp 靠的是一张**内置的确定性规则表**：在量化代码里（`llama-quantize` 以及各种转换逻辑中）写死了一张"张量角色 → 类型"的映射，按张量名的模式（`output.`、`norm`、`ffn_`、`attn_`、`token_embd`）和选定的策略等级（S/M/L），来决定每个张量到底用哪种 `ggml_type`。

这套方案的特点非常鲜明：

| 优点 | 缺点 |
|---|---|
| **确定性**：同一输入文件 + 同一策略，产出可复现 | **粗粒度**：只按"角色"分，不按"实际数值分布"分 |
| **零依赖**：不需要额外校准数据 | **不感知任务**：不问这个模型哪层真的更敏感 |
| **快**：量化瞬间完成 | **保守**：取的是普适经验，不是该模型的最优 |

确定性是这套方案最核心的价值——任何人都能复现出同一个 `Q4_K_M`文件。但"按角色分配"本质上只是**一个经验先验**：它默认所有 Llama 类模型的输出层都比注意力层敏感。这个假设多数时候成立，但不见得对眼前这个模型的**实际权重**成立。

### 14.3.2 UD 与外部敏感度搜索的对比

与它相对的，是另一条"外部搜索"路线（细纲 §12 里的 UD-* 方法，就是这一家族的代表）：**它不依赖内置规则，而是对每一个具体张量做敏感度评估，再用搜索算法来决定分配**。

典型流程是：

1. **敏感度评估**：用校准数据在每层上施加噪声或逐层量化，测量该层对输出（PPL/KL/损失）的影响，得到一张"张量 → 损伤"的敏感度表；
2. **分配**：在"总字节预算"约束下，用搜索/贪心算法把高位宽优先分配给敏感度高的张量；
3. **验证**：量化后用评测复核。

这类方法和内置规则的本质区别，正好可以放进同一张对比表：

| 维度 | llama.cpp 内置规则 | UD 式外部搜索 |
|---|---|---|
| 敏感度数据来源 | 无（按角色先验） | 校准数据实测 |
| 分配粒度 | 张量角色级 | 张量甚至子块级 |
| 可复现性 | 完全确定 | 依赖工具版本与校准数据 |
| 质量收益 | 普适、稳定 | 更优但依赖外部数据质量 |
| 部署代价 | 零 | 多一步量化管线 |

**那读者该怎么选？** 细纲给出了很明确的建议：先看你的部署约束。如果你要批量复现、标准化发布、把运维成本压到最低——内置规则是稳妥的默认选项；如果你追求的是"在固定体积下压出最多的质量"，并且愿意为多一步校准、搜索的流程买单——那外部敏感度搜索就值得一试。这里没有银弹，只有和约束相匹配的策略。


## 本章小结与通往"工具链"的桥梁

这一章可以概括成三个事实。

**第一，模型里各个张量的敏感度并不相等，混合精度就是"把钱花在刀刃上"。** 输出层直接影响 logits，归一化层负责保住尺度，FFN down 则直通残差流——它们都比 QKV/FFN gate-up 更需要高位宽。需要提醒的是，这个分层只是经验排序，还得靠实验去验证。

**第二，`_S/_M/_L` 描述的是混合策略的激进程度，而不是编码格式。** 一个 `Q4_K_M` 文件的内部，是"Q4_K 主体 + Q6_K 输出 + F32 归一化"这样一场规则驱动的混合；后缀字母只是在说明"有多少敏感张量被照顾到了"。而用 gguf-py 统计类型分布，正是验证这个事实的标准手段。

**第三，策略生成有两条路：一条是内置规则（确定、快、但粗），另一条是外部敏感度搜索（更优、依赖数据、可复现性差）。** 到底选哪条，取决于你的部署约束——这也正好是第 15 章"工具链"的直接输入。

下一章，我们会把所有格式都落到一条可执行的命令流水线上：从 `convert_hf_to_gguf.py` 做转换，到 `llama-quantize` 做量化，再到 `llama-imatrix` 做校准、`llama-perplexity` 做验证，最终串成一条完整的量化工作流。