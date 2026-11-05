# 第18章 正确性验证与质量评估

前面 17 章我们一路解剖了 GGUF 容器的字节布局（第 3 章）、Q4_0 到 Q6_K 的算术反量化（第 4、7、8、9 章）、IQ 系列的码本查表（第 11 章）和三值格式（第 13 章）。你手里现在已经有了一整套"怎么把量化块解码回浮点"的知识。但有一个问题还没有正面回答：**我凭什么相信某个格式的实现是对的？我凭什么相信某个量化模型是能用的？**

这两个问题其实是两层东西。第一层是**正确性**：位序、打包、尺度归一化有没有错——错了就是系统性错误，整个模型都坏掉。第二层是**质量**：格式是对的，但量化损失有多大、能不能用来干活——这需要端到端的指标。很多入门者把这两层混为一谈：看到 PPL 上升了一点就觉得"质量还行"，却没发现其实是解码器写错了；或者反过来，用了正确的解码器却只看 PPL，漏掉了任务能力的断崖。

本章给出一套从"字节级"到"任务级"的验证框架，分成三层：

```
验证框架的三层递进

  ① 逐块/逐张量验证 ──► 这个格式的实现对不对？
     （参考实现交叉验证 + 误差指标）
                 │
                 ▼
  ② 全模型质量评估 ──► 量化损失有多大？能不能用？
     （PPL / KL / 任务评测）
                 │
                 ▼
  ③ 特殊格式验证 ──► 三值格式的"隐藏前提"有没有满足？
     （Hadamard 变换 / 3:4 模式完整性）
```

先统一几个术语。**参考实现（reference implementation）**：格式作者或 llama.cpp 提供的、逻辑最直白的量化/反量化函数，通常是 `quantize_row_*_ref` / `dequantize_row_*`。**交叉验证（cross-validation）**：同一份数据用两个独立实现处理，逐字节/逐值比对结果，看是否一致——这里的"独立"很关键。**最大绝对误差（max abs error）**：重建值与原始值之差的最大绝对值。**相对 L2 误差**：\(\|\hat x - x\|_2 / \|x\|_2\)，衡量整体能量级的偏离。**PPL（perplexity，困惑度）**：衡量语言模型预测能力的指标，等于交叉熵的指数，越低越好。**KL 散度（Kullback–Leibler divergence）**：衡量两个概率分布有多接近，为 0 表示完全相同。**阈值效应（threshold effect）**：某些格式的劣化不是平滑曲线，而是降到某一位宽后任务能力突然崩溃。


## 18.1 逐块与逐张量验证

### 18.1.1 参考实现交叉验证：C vs gguf-py

先说最"朴素"也最有效的第一道防线：**拿两个独立实现去比对同一个格式的解码结果**。为什么必须独立？因为如果两个实现是从同一份代码抄出来的，它们会共享同一个 bug——位序写反了、半字节高低搞反了，两边一起错，比对就失去了意义。

llama.cpp 生态里最合适的"第二实现"是 **gguf-py** 的 `dequantize` 函数。它由社区用 Python + numpy 独立实现，和 C 的 `dequantize_row_*` 是两套代码。交叉验证的思路是这样：

```
交叉验证一个格式（以 Q4_K 为例）

  ① 取一段随机或真实的权重行 x（FP32，长度是 256 的倍数）
        │
        ▼
  ② 用 llama.cpp 参考实现量化：
     quantize_row_q4_K_ref(x) ──► 得到 block_q4_K 字节
        │
        ▼
  ③ 用 gguf-py 反量化：
     dequantize(字节, Q4_K) ──► 得到重建行 x̂_py
        │
  ④ 用 C 的参考反量化：
     dequantize_row_q4_K(字节) ──► 得到重建行 x̂_c
        │
        ▼
  ⑤ 逐元素比较：x̂_py 与 x̂_c 必须完全一致（误差为 0）
```

这个流程同时验证了两件事：量化后的字节是对的（②），两个反量化实现语义一致（③ ④）。如果 ③ 和 ④ 有差异，先怀疑自己手写的那个，再怀疑 gguf-py——`dequantize` 作为生态里被千百人用过的参考，出错的概率远比你的新代码小。

下面是一个**可直接运行**的 Python 交叉验证脚本。它的巧妙之处在于：量化端直接用 gguf-py 自己的 `Q4_K.quantize()`（生成合法的 Q4_K 字节流），然后我们**手写一个反量化器**（模仿 llama.cpp 的 `dequantize_row_q4_K`，逐位规则来自第 7 章），把它与 gguf-py 的 `dequantize()` 输出逐元素比对。这就构成了"**我手写的参考实现 vs 生态参考实现**"的交叉验证——量化端两边共用，但反量化端是两套独立代码：

```python
import numpy as np
from gguf import dequantize, GGMLQuantizationType
from gguf.quants import Q4_K

QK_K = 256

def get_scale_min_k4(j, scales):
    """从 scales[12] 解出第 j 个子块的 6-bit 尺度与 6-bit 最小值。"""
    if j < 4:
        sc = scales[j] & 63
        m  = scales[j + 4] & 63
    else:
        sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4)
        m  = (scales[j + 4] >> 4) | ((scales[j]     >> 6) << 4)
    return sc, m

def dequantize_q4_k_manual(block: bytes) -> np.ndarray:
    """手写的 Q4_K 反量化参考实现（对照 llama.cpp dequantize_row_q4_K）。"""
    assert len(block) == 144
    d    = np.frombuffer(block[0:2],  dtype='<e')[0].astype(np.float32)
    dmin = np.frombuffer(block[2:4],  dtype='<e')[0].astype(np.float32)
    scales = np.frombuffer(block[4:16], dtype=np.uint8)
    qs     = np.frombuffer(block[16:144], dtype=np.uint8)

    y = np.zeros(QK_K, dtype=np.float32)
    is_ = 0
    for j in range(0, QK_K, 64):          # 每 64 值一组，对应两个子块
        sc, m = get_scale_min_k4(is_ + 0, scales)
        d1, m1 = d * sc, dmin * m
        sc, m = get_scale_min_k4(is_ + 1, scales)
        d2, m2 = d * sc, dmin * m

        g = (j // 64) * 32                # 本组在 qs 中的起始位置
        lo = qs[g:g+32] & 0x0F            # 低半字节 → 子块 is
        hi = qs[g:g+32] >> 4              # 高半字节 → 子块 is+1
        y[j:j+32]    = d1 * lo.astype(np.float32) - m1
        y[j+32:j+64] = d2 * hi.astype(np.float32) - m2
        is_ += 2
    return y

def cross_check_q4_k(n_blocks: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(QK_K * n_blocks).astype(np.float32)

    # 量化端：用 gguf-py 生成合法的 Q4_K 字节
    packed = Q4_K.quantize(x)             # shape: (n_blocks*144,) uint8

    # 路径 A：我们手写的反量化
    y_manual = np.concatenate([
        dequantize_q4_k_manual(bytes(packed[i*144:(i+1)*144]))
        for i in range(n_blocks)])

    # 路径 B：gguf-py 生态参考反量化
    y_py = dequantize(packed, GGMLQuantizationType.Q4_K)

    diff = np.abs(y_manual - y_py)
    print("最大绝对差:", diff.max())
    print("完全一致:", np.array_equal(y_manual, y_py))
    return y_manual, y_py
```

运行这个脚本，如果实现正确，`完全一致` 会打印 `True`。这里有两个要点值得展开：

- **手写的 `dequantize_q4_k_manual` 严格照抄了 llama.cpp 的位序**：`scales[12]` 的交错取法（`get_scale_min_k4`）、每 64 值一组对应两个子块（低半字节/高半字节）、反量化公式 \(y = d \cdot sc \cdot q - d_{\min} \cdot m\)（`sc`、`m` 是 6-bit 整数，归一化已在量化端折进 `d`/`dmin`，见第 7 章）。任何一处位序写错，`完全一致` 立刻变 `False`。
- **这是"格式正确性"的判据，不是"质量"判据**：反量化是确定性的算术操作——给定字节，输出唯一确定。两个实现都忠实执行格式定义，结果必然逐元素相等。所以任何非零差异都意味着至少一个实现没有忠实执行格式定义。

如果你在自己实现一个新格式时发现两者有差，几乎可以断定是你的位序写错了——先照第 7–13 章的逐位规则自查打包/解包，再对照 `ggml-quants.c` 的参考函数，最后才轮到怀疑 gguf-py。

为什么"参考实现误差为零"这么重要？因为反量化是确定性的算术操作：给定字节，输出是唯一确定的。两个实现都"对"，结果必然相同。任何不等于零的差异都意味着至少一个实现没有忠实执行格式定义。这是格式正确性的**必要条件**：连这一步都过不了，后面谈 PPL 就是空中楼阁。

### 18.1.2 最大绝对误差与相对 L2

交叉验证确认了"实现没写错"，但这还没回答"量化本身损失多大"。后者需要误差指标。llama.cpp 的 `test-quantize-fns` 会为每个格式输出几类指标，其中最有用的两个：

- **最大绝对误差**：`max_i |x_i - \hat x_i|`。它抓住最坏情况——某个离群值被严重失真。这个指标对"尺度估计失败"很敏感：如果量化器把块尺度算大了 10 倍，所有重建值都偏大，最大绝对误差立刻爆表。
- **相对 L2 误差**：\(\sqrt{\sum_i (x_i - \hat x_i)^2} / \sqrt{\sum_i x_i^2}\)。它衡量的是整体能量级的偏离，对"均匀的小噪声"更宽容。

用三语言几行就能算（逻辑完全一致：逐元素差绝对值取最大、向量差范数除以原范数）：

**Python：**

```python
import numpy as np

def error_metrics(x: np.ndarray, x_hat: np.ndarray):
    max_abs = np.max(np.abs(x - x_hat))
    rel_l2  = np.linalg.norm(x - x_hat) / np.linalg.norm(x)
    return max_abs, rel_l2
```

**C++：**

```cpp
#include <vector>
#include <cmath>
#include <algorithm>

// 返回 {max_abs, rel_l2}
std::pair<float, float> error_metrics(const std::vector<float> & x,
                                      const std::vector<float> & x_hat) {
    float max_abs = 0.0f;
    double err_sq = 0.0, x_sq = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        float e = std::fabs(x[i] - x_hat[i]);
        max_abs = std::max(max_abs, e);
        err_sq += (double)e * e;
        x_sq   += (double)x[i] * x[i];
    }
    return {max_abs, (float)(std::sqrt(err_sq) / std::sqrt(x_sq))};
}
```

**C#（.NET 5+）：**

```csharp
using System;
using System.Linq;

static (float maxAbs, float relL2) ErrorMetrics(float[] x, float[] xHat)
{
    float maxAbs = 0f;
    double errSq = 0, xSq = 0;
    for (int i = 0; i < x.Length; i++)
    {
        float e = Math.Abs(x[i] - xHat[i]);
        maxAbs = Math.Max(maxAbs, e);
        errSq += (double)e * e;
        xSq   += (double)x[i] * x[i];
    }
    return (maxAbs, (float)(Math.Sqrt(errSq) / Math.Sqrt(xSq)));
}
```

三种实现都遵循同一公式：最大绝对误差 \( \max_i |x_i - \hat x_i| \) 与相对 L2 \(\|\hat x - x\|_2 / \|x\|_2\)。

至于"阈值怎么定"，有一个实用经验：**参考实现误差必须严格为零**（这是格式定义的验证，不是质量指标）；而量化本身带来的相对 L2，Q8_0 大约在 \(10^{-3}\) 量级、Q4_K 在 \(10^{-2}\) 量级——但请不要把这些数字当作普适标准，它们强烈依赖权重分布。更重要的工作是**纵向比较**：同一个权重行分别用 Q4_0、Q4_K、Q6_K 量化，看误差随位宽下降的曲线是否平滑。如果某格式的误差异常地高于邻近位宽的格式，那多半是格式实现或参数估计有问题，而不是"它本来就这样"。

一个易犯的错误是：把 `test-quantize-fns` 的误差当作"质量判据"。不是的——它是**格式调试工具**，用于定位解包错误或尺度估计失败。真正评判模型能不能用，要看下一节的全模型指标。


## 18.2 全模型质量评估

### 18.2.1 PPL、KL 与 base 模型对比

逐块验证确认了格式正确，接下来回答"这个量化模型整体损失有多大"。最经典的指标是 **PPL（困惑度）**：用模型在测试语料上计算交叉熵，再取指数。PPL 上升得越少，说明模型分布被量化破坏得越小。llama.cpp 的 `llama-perplexity` 就是干这个的：

```bash
# 在测试语料上测 PPL
./llama-perplexity -m model-q4_k_m.gguf -f wiki.test.raw -c 512
```

但 PPL 有个盲区：**它对"分布整体形状"敏感，对"特定 token 的概率形状"不敏感**。两个模型可能 PPL 几乎一样，但对某些 token 的过度自信/不自信完全不同——而这种差异恰恰影响生成质量。这就是 **KL 散度**的用武之地：它直接比较目标模型和参考模型在每个 token 上的输出概率分布。

llama.cpp 提供了 `--kl-divergence-base` 参数，让你在测 PPL 的同时，把量化模型和参考模型（通常是 F16 原版）的逐 token 分布对比：

```bash
# 对比量化模型与 F16 参考模型的 KL 散度
./llama-perplexity -m model-q4_k_m.gguf \
    -f wiki.test.raw -c 512 \
    --kl-divergence-base model-f16.gguf
```

`--kl-divergence-base` 会额外输出目标模型相对参考模型的 KL 散度。为什么 IQ 系列和三值格式尤其依赖 KL？因为它们的误差结构是"最近邻选择"（第 11 章）或"三值化的极端压缩"（第 13 章），容易在某些 token 上出现"系统性偏向"——比如某个 token 被一致性地赋予过高概率。这种偏向对 PPL 的贡献可能很小（因为它只是众多 token 中的一个），但对 KL 的贡献却很大（因为它直接改变分布形状）。所以**对码本格式和三值格式，KL 比单一 PPL 更能暴露问题**。

需要谨慎的是：KL 要求两个模型**共享同一分词器与词表**。量化前后模型通常满足这一点（同一架构、同一 tokenizer），所以可以直接比较；但如果你拿一个完全不同架构的模型去比，KL 就没有意义了。

### 18.2.2 任务评测与长上下文测试

PPL 和 KL 说到底都是"分布距离"的标尺，它们都测不到一类退化：**任务能力**。近年的评测实践反复证明，量化模型可能出现"PPL 几乎不变、任务准确率却明显下降"的现象，尤其是在**长上下文**和**指令跟随**这两类任务上。

为什么？两个原因。第一，PPL 是**整体平均**：语料里 99% 的普通 token 把误差"摊薄"了，少数关键 token（指令边界、格式标记、长程依赖的锚点）的劣化被淹没。第二，量化噪声对**注意力机制**的影响不成比例：KV cache 量化和权重量化引入的误差会在长序列上累积，而 PPL 测不到这种累积效应。典型的做法是任务评测补刀：

- **短上下文任务**：IFEval（指令跟随）、MT-Bench（对话质量）、MMLU 类（知识问答）——各有适用的评测框架。
- **长上下文任务**：LongBench 这类基准会测 32k–128k token 的检索、多跳推理。实践发现，4-bit 模型在长上下文任务上的退化往往比短上下文明显得多。

一个值得一提的反直觉现象：**中低度量化有时在任务指标上反而"更好"**。这通常来自隐式的正则化（量化噪声恰好抑制了过拟合或抵消了某些系统误差），PPL 和任务指标都测不到这个"甜点"。所以不要把任务指标当成"越大越好"的单调标尺，而要关注**曲线形状**。

对三值格式来说，任务评测更是刚需——第 2 章提过三值的**阈值效应**：劣化不是平滑的，而是在某一位宽/模型组合下突然出现任务能力的断崖。因此"这个三值模型 PPL 还行"完全不能外推"再压一点也还能用"。**必须逐模型验证**：BitNet 系在训练时就三值感知，事后三值化的模型则完全不同。这是第 13 章适用边界的方法论前提，这里再强调一遍。


## 18.3 特殊格式的验证要点

### 18.3.1 PTQ1_0 的 Hadamard 变换正确性

三值格式 PTQ1_0（第 13 章）有一个隐藏前提：**磁盘里存的权重不是"普通的三值权重"，而是经过 Hadamard 折叠的**。推理时必须先对激活做 Hadamard 侧变换 \(y = W' \cdot (H(s \cdot P x))\)，才能得到正确的 logits。这个变换是格式定义的一部分，不是可选的加速技巧。

因此验证 PTQ1_0 模型时，除了常规的逐块误差检查，还必须做一道"变换存在性"检查：

```
PTQ1_0 验证清单（激活侧变换）

  ① 加载模型，确认权重张量确实是 Hadamard 折叠形式
     （或确认元数据声明了折叠/旋转字段）
        │
        ▼
  ② 跑一次前向，确认激活侧变换 H(s·Px) 确实被应用
        │
  ③ 关键对照：关掉变换，跑同一个输入
     → logits 应发生剧烈、系统性变化（而不是微小波动）
        │
        ▼
  ④ 只有"开变换 → 输出正常，关变换 → 输出全乱"，
     才说明变换真的在工作
```

最容易误判的场景是：**格式解析正确（字节都对）、反量化正确（逐块误差为零），但推理输出完全错误**。原因就是缺失了 Hadamard 变换。这也是细纲再三强调的工程陷阱：对 PTQ1_0，"格式正确但缺失变换"与"格式错误"在现象上一样，都表现为 logits 全错——唯一的区分手段就是上面第 ③ 步的对照实验。**只要变换被正确应用，你的模型和基座在正常输入上的表现应当接近**；如果完全不应用变换还能得到合理输出，反而说明权重根本没做折叠，或者折叠方向反了。

### 18.3.2 STQ1_0 的 3:4 模式完整性

STQ1_0 的结构约束是：**每组 4 个权重恰好 1 个零、3 个 ±d**（3:4 稀疏）。这个约束必须在解码端被验证，因为它决定了解码逻辑的成立条件。

怎么验证"每组恰好一个零"？STQ1_0 用 5 bit 编码 32 种模式（4 选 1 的位置 × 3 个 ± 符号），解码时把 5-bit 索引还原成 4 个值。验证脚本应该做两件事：

1. **模式合法性**：遍历每个 4 值组，断言其中恰好一个为零、其余三个非零且符号与符号位一致。
2. **重建数值**：用符号和位置还原出的 4 个值，乘以块尺度 d，与参考实现（或 gguf-py）比对。

```python
def check_stq_patterns(qs_index: np.ndarray, sign_bits: np.ndarray):
    """STQ1_0：断言每 4 值组恰好一个零（3:4 模式）。"""
    for idx, sgn in zip(qs_index, sign_bits):
        pos   = idx >> 3            # 3 bit：哪个位置是零
        signs = idx & 0x7           # 3 bit：其余三值的符号
        zero_at = [pos == k for k in range(4)]
        if sum(zero_at) != 1:
            raise ValueError(f"3:4 模式被破坏：{idx} 产生 {sum(zero_at)} 个零")
        # 其余位置非零
        for k in range(4):
            if not zero_at[k] and not (signs >> (k - (1 if k > pos else 0))) & 1 in (0, 1):
                ...  # 具体符号位分布以第 13 章的格式定义为准
```

上面这段是"模式检查"的骨架。为什么这很重要？因为 3:4 约束一旦被破坏（比如某组出现 2 个零），解码器按 32 种模式查表就会解错位，产生**系统性错误**——不是噪声，而是每个值都可能错。对这类格式，一个完整性的断言（"恰好一个零"）是比任何误差指标都更前置的检查：它验证的是格式的基本契约。

三值格式的验证结论可以总结成一句话：**先验证结构约束（STQ1_0 的 3:4、PTQ1_0 的 Hadamard），再谈误差和 PPL**。结构错了，后面一切指标都是假象。


## 本章小结与通向"跨引擎互操作"的桥梁

本章的验证框架可以压缩成三条事实。

**第一，正确性验证从"参考实现交叉验证"开始。** 手写的解码器要和独立实现（如 gguf-py 的 `dequantize`）逐元素比对，参考实现误差必须为零——这是格式定义正确的必要条件，比任何质量指标都更前置。`test-quantize-fns` 的最大绝对误差与相对 L2 用于调试格式实现，不用于评判质量。

**第二，质量评估要三层指标并用：PPL、KL、任务。** PPL 管分布整体，KL 管分布形状（对 IQ/三值尤其重要），任务评测管实际能力（长上下文和指令跟随最容易暴露 PPL 测不到的退化）。三值格式必须逐模型验证阈值效应，不能用单一 PPL 外推。

**第三，特殊格式有专属的"隐藏前提"检查。** PTQ1_0 必须验证激活侧 Hadamard 变换已应用（对照实验：关掉变换看输出是否全乱）；STQ1_0 必须验证每组恰好一个零。这些前提不满足时，格式字节全对也没用。

```
验证框架一图流

  字节层       参考实现交叉验证 ──► 误差指标（调试工具）
                    │
  模型层        PPL（整体） + KL（形状） + 任务（能力）
                    │
  特殊层        PTQ1_0：Hadamard 在吗？
                STQ1_0：3:4 完整吗？
```

至此，"一个 GGUF 模型对不对、能不能用"已经有了完整的判断方法。但还有一个问题悬而未决：**这些验证都假设读取端是 llama.cpp 或 gguf-py**。如果我想自己写一个 GGUF 读取器（比如用 Rust 或 JavaScript），或者换一个推理引擎，会遇到什么坑？第 19 章就来处理"跨引擎与格式互操作"——不同实现之间对"同一格式"的理解是如何分叉的。
