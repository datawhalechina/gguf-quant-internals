# 第12章 UD_IQ：Unsloth Dynamic 的模型级混合

到这里，读者已经认识了两种"给模型分配合适量化类型"的方式。第一种是内置规则型，也就是第 3 章讲的 `Q4_K_M`。它的道理很朴素：llama.cpp 用一个固定规则（主体 Q4_K、输出层 Q6_K、归一化 F32）去分配所有张量，规则写死了，谁来了都按这个办。第二种靠的是格式本身的差异：第 11 章讲的 `IQ1_S`、`IQ2_XXS` 等，它们在**编码格式**这一层把每个权重压得更狠。

这一章要讲第三种，它的机制和前两种都不一样——它不是编码格式，而是**用外部搜索来决定"每个张量用哪种已有格式"**的做法，产品名叫 Unsloth Dynamic（UD，常写作 `UD-IQ1_S`，读作"UD 前缀的 IQ1_S"）。先给一个关键定性：**UD 不是一种新的 ggml 类型**。想象拆开一个 "UD-IQ1_S" 模型：代码跑起来用到的仍然是 `IQ1_S`（type id 19）、`IQ2_XXS`（16）、`Q4_K`（12）这些我们在前面章节见过的格式。UD 改的不是"每个权重怎么编码"，而是"哪些张量用 IQ1_S、哪些要用更保质量的类型"。

为什么值得专门讲一章？因为它在 2025 年初把"极限压缩"推到了新的位置：一个 70B 模型在保留大部分能力的同时，被压到了约 1/8 的体积。这个数字听起来相当诱人，但要理解这种收益从哪来、代价又在哪，必须先弄清楚三个概念：**敏感度矩阵**（那个决定"谁配什么"的依据）、**分配搜索**（如何从敏感度推出逐张量类型）、**可复现性**（为什么同名的 UD 模型可能字节级不同）。

先统一几个术语。**逐张量分配（per-tensor allocation）**：为每个张量单独挑选量化格式，而不是所有张量一刀切。**敏感度（sensitivity）**：量化某个张量后，模型输出的变化有多大；越敏感的张量越应该保留高位宽。**敏感度矩阵**：把"每个张量 × 每种候选格式"的预估损失排成一张表。**动态量化（dynamic quantization）**：在推理时按实际激活动态选择 KV cache 等部分的精度（UD 的另一半，见 12.2.1）。**可复现性（reproducibility）**：同样的输入流程能否产出完全相同的文件。


## 12.1 UD 的本质：策略而非编码

### 12.1.1 "UD-IQ1_S"内部到底是什么

先做一个反直觉的验证：把 "UD-IQ1_S" 当作一个文件来解剖。下面这个脚本，和第 3 章统计 `Q4_K_M` 类型分布的脚本几乎一样。用几乎一样的代码去解剖一个看似全新的名字，这正是想强调的第一课：**UD 模型的内部不神秘，它其实是一个混合体，装着几十种我们见过的标准 ggml 类型**。

```python
from collections import Counter
from gguf import GGUFReader

for path in ["ud-iq1_s-model.gguf", "iq1_s-plain.gguf"]:
    reader = GGUFReader(path)
    counter: Counter[str] = Counter(t.tensor_type.name for t in reader.tensors)
    print(f"\n=== {path} ===")
    for qtype, count in counter.most_common():
        print(f"  {qtype:12s} {count:5d} 张量")
```

输出会呈现出一种"金字塔"式的分布：

```
=== ud-iq1_s-model.gguf ===
  IQ1_S           196 张量     ← 大部分低敏感度张量压到 1.6 bpw
  IQ2_XXS          48 张量     ← 敏感一些的用 2.06 bpw
  IQ2_XS           18 张量
  Q4_K              8 张量     ← 很关键的张量上到 4.5 bpw
  Q6_K              2 张量     ← 输出层
  F32              70 张量     ← 归一化等
=== iq1_s-plain.gguf ===
  IQ1_S          270 张量     ← 无差别的 IQ1_S（除了 F32 归一化）
  F32             52 张量
```

对比一下，结论就出来了：**"UD-IQ1_S"里的 IQ1_S，只说明"绝大多数无关紧要的张量用 IQ1_S"，而不是"整个模型都是 IQ1_S"**。命名上它继承了"哪个格式打底"，但真实内容其实是分配策略的产物——这正是第 12.1 节要钉死的核心。

### 12.1.2 UD 和 `_M` 后缀的区别：固定规则 vs 外部搜索

乍看 `UD-IQ1_S` 和 `Q4_K_M` 很像：都是"一个名字，内部混合"。但两者的机制完全不同——一个是写死的规则，一个是现场的搜索。放在一起对比，差别最清晰：

| 维度 | `Q4_K_M`（内置规则） | `UD-IQ1_S`（外部搜索） |
|---|---|---|
| 决策依据 | llama.cpp 里写死的模板 | 模型+数据算出的敏感度矩阵 |
| 决策粒度 | 按张量角色分组（QKV/FFN/输出层/归一化） | 逐张量独立，可有几十种组合 |
| 是否可复现 | 是——任何版本`llama-quantize`结果一致 | 不确定——依赖工具版本和校准数据 |
| 依赖外部数据 | 否 | 是——需要校准数据来估计敏感度 |
| 上限/下限 | 只在使用规则内找最优 | 理论上可在全格式空间里找 |

一句话总结：**`_M` 是人写死的"手工规则"，UD 是让机器去"搜索"**。前者可复现、粗粒度、零依赖；后者更细、更贴模型，但依赖工具链和数据。

注意一个细节：UD 的策略是"外部搜索"，但最后落盘的仍是标准 GGUF + 标准 type id。也就是说，一个 UD 模型文件对 llama.cpp 而言没有任何特殊之处——**只要构建支持它内部用到的那几种格式，就能直接加载**。第 13 章讲的三值格式则是反例：它们的落盘格式本身不在主线，加载就受限。


## 12.2 UD-IQ1_S 的实测表现

### 12.2.1 89% 体积缩减 vs 约 72% top-1：一个诚实的权衡表

UD 发布时打的宣传点是"极限压缩下的质量保留"。以 70B 档模型为例，一个常用的实例数字是：**从 F16 出发，UD-IQ1_S 把文件压缩约 89%**——也就是剩下原体积的 1/9 左右；与此同时，和原始模型相比，它还能保留约 72% 的 top-1 准确率。

这两个数字要拆开看，因为它们不是同一条曲线上的点：

```
F16          100% 体积    ~100% top-1
  │
  │  ≈ 89% 体积缩减
  ▼
UD-IQ1_S     ~11% 体积    ~72% top-1（特定任务集）
  │
  ▼
IQ1_S 全量    ~11% 体积    ~60-65% top-1（同任务集，更差）
```

注意：**UD 的收益不在"把 IQ1_S 变强"，而在于"把有限的量化预算花在刀刃上"**。同样的总体积，如果均匀压成 IQ1_S，模型在敏感任务上会掉得更多；UD 的做法是保持总体积不变，把关键的少量张量"抽出来"改用更高位宽（用 Q4_K/Q6_K），从而显著改善任务保留。

诚实的注脚必须写三句：

1. **数字高度依赖模型和任务集**。前面说的 72% top-1，是特定配置下的结果，即某 70B 模型 + 某个评测集。换成别的模型、别的任务，数字都会明显浮动；如果你要写进自己的报告，请以你实测为准。
2. **KV cache 量化贡献了另一半**。UD 不只是量化权重：它还会对 KV cache 做动态量化，例如 key 用 4-bit、value 用 8-bit，并且按头缩放。弄清这一层，你就能明白"89% 体积缩减"里有一部分来自"更少的内存开销"，而不是"更小的权重文件"。
3. **极限压缩的任务退化是非线性的**。和 Q2_K（第 10 章）一样，UD 在"简短问答、检索"这类任务上相对抗压，但换到"JSON 生成、长代码、多步指令"上，退化就明显了。所以，想用单个 PPL 或单个准确率数字去外推整体表现，是危险的——必须逐任务验证。

### 12.2.2 逐张量类型检查：确认 UD 不是单一格式

12.1.1 的脚本已经完成了验证。这里再补一个更工程化的检查脚本——它会把每个张量的名字、类型、字节数都打印出来，方便你定位"到底哪个张量被升到了 Q4_K"：

```python
from gguf import GGUFReader

reader = GGUFReader("ud-iq1_s-model.gguf")

# 只看非 IQ1_S 的张量（即 UD 特意"升级"的部分）
for t in reader.tensors:
    if t.tensor_type.name != "IQ1_S":
        mb = t.n_bytes / (1024 * 1024)
        print(f"{t.name:56s} {t.tensor_type.name:10s} {mb:8.2f} MB")
```

输出大致是：

```
model.layers.0.attn_norm.weight                    F32        0.00 MB
model.layers.0.attn_q.weight                       Q4_K     18.42 MB
model.layers.0.attn_v.weight                       IQ2_XS     9.21 MB
model.layers.0.attn_k.weight                       IQ2_XXS    4.60 MB
...
model.output.weight                                Q6_K      60.2  MB
```

看两个观察点。第一，同一个 `attn` 层里的 Q、K、V，可能被分配不同的格式——**这不是笔误，而是敏感度搜索得出的结论**：某些输出投影对量化格外敏感，对应的输入投影则更耐受。第二，你也可以用这个脚本去验证任何自称 "UD" 的模型，检查它的内部到底是不是真混合，还是一种格式打天下。

正因为落盘仍用标准 type id，**任何能读 GGUF 的引擎都可以枚举这些内部类型**，完全不用依赖 Unsloth 自家的工具。下面是 C++ 与 C# 的等价枚举，写法参照第 3 章读 header 的套路——注意，我们只读目录（tensor_infos），不碰数据段：

**C++：（沿用第 3 章的 `gguf_header_t`，跳过 KV 区后逐个读 tensor_infos）**

```cpp
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>

#pragma pack(push, 1)
struct gguf_str_t   { uint64_t len; char   data[0]; };
struct gguf_ti_head_t {
    uint64_t name_len;   // name 字符串跟随其后
    uint32_t n_dims;
    // 然后是 dims[8] × n_dims、type(u32)、offset(u64)
};
#pragma pack(pop)

// 读一个带长度前缀的字符串
static std::string read_string(FILE * f) {
    uint64_t len; fread(&len, 8, 1, f);
    std::string s(len, '\0');
    fread(s.data(), 1, len, f);
    return s;
}

int main() {
    FILE * f = fopen("ud-iq1_s-model.gguf", "rb");
    uint32_t magic, version;
    uint64_t tensor_count, kv_count;
    fread(&magic,   4, 1, f); fread(&version,   4, 1, f);
    fread(&tensor_count, 8, 1, f); fread(&kv_count, 8, 1, f);

    for (uint64_t i = 0; i < kv_count; ++i) read_string(f); // 跳过 KV

    for (uint64_t i = 0; i < tensor_count; ++i) {
        std::string name = read_string(f);
        uint32_t n_dims, type;
        uint64_t offset;
        fread(&n_dims, 4, 1, f);
        std::vector<uint64_t> dims(n_dims);
        fread(dims.data(), 8, n_dims, f);
        fread(&type,   4, 1, f);
        fread(&offset, 8, 1, f);
        printf("%-56s type=%-4u\n", name.c_str(), type);
    }
    return 0;
}
```

（这里为了聚焦主题，KV 那一块用 `read_string` 简化绕过了——真实的 KV 数据还包含类型字节。完整解析见附录 C 的最小解析器。）

**C#（.NET 5+，用 `BinaryReader`）：**

```csharp
using System;
using System.IO;
using System.Text;

using var r = new BinaryReader(File.OpenRead("ud-iq1_s-model.gguf"));
string ReadStr() { int n = checked((int)r.ReadUInt64()); return Encoding.UTF8.GetString(r.ReadBytes(n)); }

r.ReadBytes(4); r.ReadUInt32();                 // magic, version
ulong tensorCount = r.ReadUInt64();
ulong kvCount     = r.ReadUInt64();
for (ulong i = 0; i < kvCount; i++) { ReadStr(); r.ReadUInt32(); ReadValue(r); }  // 简化

for (ulong i = 0; i < tensorCount; i++) {
    string name = ReadStr();
    int    nd   = r.ReadInt32();
    for (int d = 0; d < nd; d++) r.ReadUInt64();
    uint type   = r.ReadUInt32();
    r.ReadUInt64();                              // offset
    Console.WriteLine($"{name,-56} type={type}");
}

static void ReadValue(BinaryReader r) {
    // 按 GGUFValueType 分派；这里为示例略过
}
```

三种语言的核心逻辑是一致的：**跳过 KV，逐个张量读出 name / type / offset**，就能画出一张"UD 模型内部类型金字塔"。区别只在于，Python 走的是 gguf-py 的高层封装，C++/C# 走的是结构体 + `BinaryReader` 的低层拆解——但最后拿到的是同一个 GGUF 文件的同一张分布表。这个"文件可移植、读者可自证"的结论，正是第 12.3.2 节"标准 type id 落盘"的工程保障。


## 12.3 UD 的生成流程与生态风险

### 12.3.1 敏感度矩阵、分配、验证的标准流程

UD 模型不是敲一条 `llama-quantize` 命令就冒出来的，它后面是一条多步流水线。整个过程大致分三步：

```
UD 模型生成流程

  ① 敏感度估计
     F16 模型 + 校准数据
        │ 对每个张量分别量化到每个候选格式，
        │ 度量输出损失（如 KL 散度）
        ▼
     敏感度矩阵 S[张量 × 格式]

  ② 动态分配
     在"总字节数预算"约束下，从矩阵 S
     找出"预期总损失最小"的逐张量格式组合
     （本质是一个背包问题 / 贪心搜索）
        │
        ▼
     分配表：张量 → 格式

  ③ 量产与验证
     按分配表逐张量量化，合并成 GGUF
         │
         ▼
     在验证集上评估（PPL / 任务得分）
     不合格则回到 ② 调整预算
```

关键点在第 ① 步的"度量"上。UD 的敏感度，并不去算逐权重的激活值（那是第 11 章 imatrix 干的事），而是**粗粒度地测"把整个张量换成某格式、输出会变多少"**。因为这个测量要在"张量 × 格式"的矩阵上跑，数据量相当可观，实践中通常用少量校准文本，配上一个并行的量化-推理循环来完成。至于第 ② 步的搜索，那才是核心卖点：它让"总体积不变，误差最小"真正成为一个可以优化的目标，而不是拍脑袋定的规则。

### 12.3.2 跨工具可复现性问题：同名 UD ≠ 字节级复现

UD 的质量依赖两样东西：**校准数据**和**生成工具**。这两样一旦不固定，就带来一个工程上很现实的坑：

- 校准数据长什么样，直接影响敏感度估得准不准——措辞、长度、领域，每一项都会带来偏差；
- 工具版本不同，结果也可能有一丝差异：Unsloth 的 fork 和上游 llama.cpp 的量化器，对同一码表、同一 round/floor 的处理未必完全一致；
- 分配搜索本身如果没有固定随机种子，可能每次跑出不同的次优解。

结果是：**两个工具在同一个模型上生成的"UD-IQ1_S"，内部类型分布可能不同，文件也就不是字节级可复现的**。这种现象不一定是 bug——但它在工程上意味着：

1. **不能假设"文件名相同 = 文件相同"**。发布者应同时公开生成工具版本与校准数据；
2. **跨工具对比评测时，要注明谁生成的**，否则数字不可比；
3. 与第 12.1.2 表里的"可复现性"一栏呼应：**规则的确定性是 UD 用外部搜索换来的，这是代价**。

这也把全书的主题往前推了一步：从"字节布局格式"到"分配策略"，质量的锚点从"规范"转移到了"数据 + 流程"。第 20 章讨论生态治理时，会回到这一点。


## 本章小结与通往"三值极限"的桥梁

这一章可以压缩成三个事实。

**第一，UD 是策略而非编码。** "UD-IQ1_S"模型内部，是 IQ1_S、IQ2_XXS、Q4_K、Q6_K、F32 这几种标准格式的混合体；`UD` 这两个字母只说明"这些类型是被外部敏感度搜索分配出来的"。拿 gguf-py 一统计，就能打破"UD 是单一格式"的误解。

**第二，UD 权衡的本质，是"总体积不变、误差最小"的优化，而不是更狠的压缩。** 它在一个相当激进的体积预算下，只把极少数敏感的 QKV / 输出层张量留成高位宽，其余全部压到 IQ1_S，于是在"近似 F16 的 1/9 体积"下，仍然保住了约 72% 的 top-1（数字随模型、任务浮动）。代价则是依赖校准数据、工具版本，而且无法做到字节级复现。

**第三，UD 是一次"范式外"的尝试：它没有发明任何新格式，却重新组织了已有格式。** 它证明了"逐张量混合 + 外部搜索"这条路能把极限压缩推得更远——那么问题就来了：如果还想再压，编码格式这一层还有没有空间？

UD 的极限停留在大约 1.6 bpw 附近，因为它依赖的 IQ1_S 本身就是 1.56 bpw。下一章我们要把位宽推到底，进入**三值量化**：每个权重只取 \(\{-d, 0, +d\}\) 三个值，理论上可以比 1.6 bpw 更低（STQ1_0 约 1.31 bpw、PTQ1_0 约 1.75 bpw）。三值格式的挑战，不在"存"，而在"解码端需要什么额外变换"——尤其是 PTQ1_0 的 Hadamard 折叠。第 13 章会回答这个问题：格式明明正确，只缺了旋转，为什么结果仍然全错？