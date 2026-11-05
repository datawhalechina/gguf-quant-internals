# 第14章 `_S`/`_M`/`_L` 与逐张量混合精度

前十三章一直在研究"一个格式怎么工作"：Q4_0 的块、Q4_K 的超级块、IQ 的码本、STQ/PTQ 的三值。这一章把视角从"格式"上升到"策略"：**一个可用模型，不是所有张量都用同一种格式，而是按敏感度给不同张量分配不同格式**。

先给一个具体感受。你在命令行执行 `llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M`，得到的 `model-q4_k_m.gguf` 里，并不是每一个张量都是 `Q4_K`。实际上文件里同时存在三种类型：

- 大部分注意力投影和 FFN 矩阵是 `Q4_K`（4-bit 主体）；
- 输出嵌入层是 `Q6_K`（6-bit，更高精度）；
- 一堆归一化层是 `F32`（32-bit，根本不量化）。

用第 3 章的 gguf-py 脚本一统计，真相立现——`Q4_K_M` 是一个**混合策略的名字**，不是单一的编码格式。这就是本章的主题：**为什么输出层要 6-bit？为什么归一化层保留 F32？不同位置的张量，敏感度有什么差异？**

先统一几个术语。**张量敏感度（tensor sensitivity）**：某个张量对量化误差的容忍程度——量化它造成的质量损失越小，说明它越"皮实"；损失越大，越"娇贵"。**logits**：模型输出的原始分数向量，softmax 之前的值，直接决定下一个 token 的概率分布。**输出嵌入层（output embedding）**：把最后一个隐藏层映射到词表大小的层，在 GPT 类模型里常与输入词嵌入共享权重（tied）。**FFN（feed-forward network）**：Transformer 块中的前馈网络，通常由 gate/up 和 down 两组投影组成。**LayerNorm**：把每个位置的向量归一化的层，对缩放非常敏感。


## 14.1 张量敏感度分层

### 14.1.1 输出嵌入层：对 logits 的直接敏感

先问一个问题：为什么 `Q4_K_M` 把输出嵌入层用 `Q6_K`，而不是和其他层一样用 `Q4_K`？

问题的答案藏在"输出层在模型里的地位"。绝大多数张量（比如注意力投影）参与的是**中间计算**，它们的误差会被后面一层层的矩阵乘平均、稀释。但输出嵌入层是**最后一个线性层**：它的输出就是 logits，直接影响每个 token 的概率。输出层的量化误差不会被后续层平滑，而是**直接进入最终概率分布**。

用一个类比帮助理解。想象一场接力赛，前面几棒跑得快慢都不会直接影响成绩，只要总时间对；最后一棒冲线的方式决定了最终名次。输出层就是"最后一棒"——它的一点偏差，会完整地反映在 logits 上。

量化误差对 logits 的形状影响可以用两个指标衡量：

- **PPL（困惑度）**：对整体的概率分布敏感。输出层误差会让概率分布变得"更不确定"，PPL 上升。
- **KL 散度**：对 logits 的具体形状敏感。输出层误差会让量化后模型的 logits 与原始模型的 logits 偏离，哪怕 PPL 变化不大。

第 2.5.2 节说过，IQ 和极限格式常用 KL 与 base 模型对比，正是因为这个原因——**输出层超比例地影响 KL**。实测经验中，仅把输出层从 `Q4_K` 升到 `Q6_K`，KL 散度就能显著改善，而文件体积只增加一点点（因为输出层只占全模型的一小部分）。这就是"把钱花在刀刃上"。

### 14.1.2 注意力 QKV、FFN gate/up/down 的差异

行内层的敏感度也不是铁板一块。Transformer 的每个块里有两大类线性投影：

1. **注意力（attention）投影**：`attn_q`、`attn_k`、`attn_v`，以及部分架构的 `attn_output`。它们把输入向量投影到查询/键/值空间。
2. **前馈网络（FFN）投影**：`ffn_gate`、`ffn_up`（放大维度）和 `ffn_down`（缩回维度）。

经验上存在一个规律：**FFN down 有时需要比 QKV 更高的位宽**。直觉解释：

- QKV 投影的误差会被注意力权重（softmax 之后的值）"过滤"——softmax 让大部分注意力权重趋近零，只有少数 token 的贡献显著，所以 QKV 里的随机误差很大程度被这种门控效应钝化了。
- FFN down 相反，它直接决定块输出 \(x + \mathrm{down}(\mathrm{gate}(x) \odot \mathrm{up}(x))\)，误差会原样叠加进残差流，一路传递到最后的 logits。

这个差异在细纲和社区实践里表现为：**如果一个模型要在 Q4_K 上再抠质量，优先把 `ffn_down` 升级到 Q5_K 或 Q6_K，而不是动注意力层**。这也解释了不同 `_M` 等级之间"哪些张量升/降级"的部分差异。

### 14.1.3 归一化与嵌入：保留 F32 的理由

第三类"娇贵"张量是归一化层和部分嵌入层。为什么 `Q4_K_M` 文件里有那么多 `F32`？它们其实占体积很小，但**量化它们省下的字节远不及损失的质量**。

**LayerNorm/RMSNorm 的权重**（`attn_norm`、`ffn_norm`）：这些层做的是逐元素归一化，每个维度的缩放因子直接乘到归一化后的值上。它们对"尺度正确性"极其敏感——一旦量化把某个维度的权重从 1.001 变成 1.0，这个维度被缩放的比例就错了。更关键的工程事实是：**归一化层张量极小**（形状是隐藏维度，一个向量而已），在总文件里占不到 1% 的字节，却承担着"保准尺度"的责任。为了省这 1% 去赌质量，不划算。

**部分嵌入层**：词嵌入（`token_embd.weight`）在一些策略里也保留较高精度或 F32，因为嵌入向量直接决定了每个 token 的初始表征。

一句话总结敏感度分层：**[输出层 > 归一化层] > [FFN down > QKV / FFN gate-up] > [结构性可压缩张量]**。越靠左越"娇贵"，越该用高位宽甚至 F32；越靠右越"皮实"，可以放心用低位宽。

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

注意：这张图是"经验的量级排序"，不同架构、不同模型上具体排序会有微调。**敏感度分层是需要用实验验证的假设，不是定理**——第 18 章会讲怎么用实验验证它。


## 14.2 典型配置的逐张量分布

### 14.2.1 Q4_K_M：Q4_K 主体 + Q6_K 输出 + F32 归一化

现在用一个真实套路来看 `Q4_K_M` 的逐张量分布。下面的 Python 脚本按"张量名分组"统计类型分布——不是只数总量，而是看**规律**：

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

Python 脚本背后的机制是第 3 章讲的"每个张量有独立的 `type` 字段"。C++ 和 C# 里做等价的事，就是读取 `tensor_infos` 的 `type` 整数，再映射到类型名——差别只在"手工映射"还是"走 gguf-py 的枚举"。下面给出三种语言的等价视角。

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

读这张表的方式：

1. **Q4_K 占绝对多数**——这就是"主体"。
2. **输出层单独用 Q6_K**——1 个张量，但承担 logits 质量。
3. **归一化层全 F32**——50+ 个张量，全是向量级小张量。
4. **没有 Q5_K**——`_M` 的"M"不是"每个张量都用 4.5 bpw 的 Q4_K"，而是"这样一个规则驱动的混合"。

这正是细纲反复强调、第 3 章也预告过的核心事实：**_M 后缀描述的是混合策略的激进程度，不是单一编码格式**。

### 14.2.2 Q5_K_M、Q3_K_M 等变体的分配差异

`_S/_M/_L` 这组后缀不是随意的缩写。它们描述"混合策略的**激进程度**"：

- **S（Small）**：压缩激进，尽量少用高位宽，文件最小；
- **M（Medium）**：折中，主体用目标位宽，保住输出层和归一化；
- **L（Large）**：保守，更多张量用更高位宽，文件更大但质量更好。

（不同架构的 `_S/_M/_L` 具体指哪些张量，由 llama.cpp 的规则表决定，见 14.3.1。）

以 K-quants 为例，`Q3_K_S`、`Q3_K_M`、`Q3_K_L` 的差异大致是：**S 版几乎所有张量都用 Q3_K；M 版在 S 的基础上把输出层/部分敏感层升级到 Q4_K 或 Q5_K；L 版进一步升级更多层**。从 `Q3_K_S` 到 `Q3_K_L`，"后缀字母"表达的是"有多少敏感张量被照顾到了"。

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

这里有个重要提醒：**"字母越大质量越好"不是定理**。它依赖具体模型和评测。在某些模型上 S 和 M 的 PPL 差异可以忽略，在另一些上 M 和 L 的差异因评测任务而不同。后缀字母描述的是**策略方向**，最终质量必须实测。


## 14.3 策略生成：规则、imatrix 与搜索

### 14.3.1 llama.cpp 内置规则的确定性

`_S/_M/_L` 策略是怎么生成的？llama.cpp 的做法是**内置的确定性规则表**：在量化代码里（`llama-quantize` 与转换逻辑中）有一张"张量角色 → 类型"的映射，按张量名的模式（`output.`、`norm`、`ffn_`、`attn_`、`token_embd`）和选定的策略等级（S/M/L）决定每个张量用哪种 `ggml_type`。

这套方案的特点非常鲜明：

| 优点 | 缺点 |
|---|---|
| **确定性**：同一输入文件 + 同一策略，产出可复现 | **粗粒度**：只按"角色"分，不按"实际数值分布"分 |
| **零依赖**：不需要额外校准数据 | **不感知任务**：不问这个模型哪层真的更敏感 |
| **快**：量化瞬间完成 | **保守**：取的是普适经验，不是该模型的最优 |

确定性是它的核心价值——任何人都能复现同一个 `Q4_K_M`文件。但"按角色分配"本质上是**一个经验先验**：它假设所有 Llama 类模型的输出层都比注意力层敏感。这个假设多数时候成立，却不一定对该模型的**实际权重**成立。

### 14.3.2 UD 与外部敏感度搜索的对比

与之相对的是一条"外部搜索"路线（细纲 §12 的 UD-* 方法是其家族成员）：**不靠内置规则，而是对每个具体张量做敏感度评估，再用搜索算法决定分配**。

典型流程是：

1. **敏感度评估**：用校准数据在每层上施加噪声或逐层量化，测量该层对输出（PPL/KL/损失）的影响，得到一张"张量 → 损伤"的敏感度表；
2. **分配**：在"总字节预算"约束下，用搜索/贪心算法把高位宽优先分配给敏感度高的张量；
3. **验证**：量化后用评测复核。

这类方法与内置规则的本质区别：

| 维度 | llama.cpp 内置规则 | UD 式外部搜索 |
|---|---|---|
| 敏感度数据来源 | 无（按角色先验） | 校准数据实测 |
| 分配粒度 | 张量角色级 | 张量甚至子块级 |
| 可复现性 | 完全确定 | 依赖工具版本与校准数据 |
| 质量收益 | 普适、稳定 | 更优但依赖外部数据质量 |
| 部署代价 | 零 | 多一步量化管线 |

**读者应如何选择？** 细纲给出了明确建议：看你的部署约束。如果要批量复现、标准化发布、最低运维成本——内置规则是稳妥的默认；如果追求"在固定体积下压出最多的质量"，且你愿意为多一步校准/搜索流程付账——外部敏感度搜索值得尝试。没有银弹，只有与约束匹配的策略。


## 本章小结与通往"工具链"的桥梁

这一章可以从三个事实概括。

**第一，模型里的张量敏感度不相等，混合精度是"把钱花在刀刃上"。** 输出层直接影响 logits，归一化层保尺度，FFN down 进残差流——它们比 QKV/FFN gate-up 更需要高位宽。这个分层是经验排序，需要用实验验证。

**第二，`_S/_M/_L` 描述的是混合策略的激进程度，不是编码格式。** 一个 `Q4_K_M` 文件内部是"Q4_K 主体 + Q6_K 输出 + F32 归一化"的规则驱动混合；后缀字母只是说明"多少敏感张量被照顾到了"。用 gguf-py 统计类型分布，是验证这一事实的标准手段。

**第三，策略生成有两条路：内置规则（确定、快、粗）和外部敏感度搜索（更优、依赖数据、可复现性差）。** 读者的选择取决于部署约束——这是第 15 章"工具链"的直接输入。

下一章，我们把所有格式都落到可执行的命令流水线：从 `convert_hf_to_gguf.py` 转换，到 `llama-quantize` 量化，再到 `llama-imatrix` 校准和 `llama-perplexity` 验证，串成一条完整的量化工作流。