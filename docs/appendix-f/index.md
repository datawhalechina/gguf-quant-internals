# 附录 F 实验脚本与命令

> 本附录把正文反复出现的工具命令整理成一套**可直接照抄运行**的端到端流程：从 Hugging Face 权重出发，经过转换、imatrix、量化、PPL/KL 校验，到最后端侧冒烟。命令以 llama.cpp 主线（2026-10）与 gguf-py 0.19.0 为准。所有输出数字都是示意，实际以你的模型与设备为准。

## F.1 环境准备

llama.cpp 从源码构建（支持 CMake 或 make），构建产物在 `build/bin/`；`convert_hf_to_gguf.py` 在仓库根目录，`gguf` Python 包用 pip 安装：

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j 8
# 或者 CPU 版
# cmake -B build && cmake --build build --config Release -j 8

pip install gguf                 # gguf-py 0.19.x，第 3 章/附录 D
```

常用可执行文件（`build/bin/` 下）：

| 工具 | 作用 | 正文章节 |
|---|---|---|
| `llama-quantize` | 量化（F16 GGUF → 量化 GGUF） | 第 15 章 |
| `llama-imatrix` | 生成重要性矩阵（imatrix） | 第 11、15 章 |
| `llama-perplexity` | 计算 PPL / KL 散度 | 第 2、18 章 |
| `llama-cli` | 命令行生成（冒烟测试） | 第 18 章 |
| `llama-bench` | 吞吐基准 | 第 17 章 |

## F.2 完整流水线（一条龙脚本）

下面的 bash 脚本把六步串起来。假设已经有一个 Hugging Face 格式的模型目录 `./hf-model` 和一份纯文本校准文件 `./data/wiki_sample.txt`。

```bash
#!/usr/bin/env bash
# 环境
set -euo pipefail
LLAMA=~/llama.cpp
BIN=$LLAMA/build/bin
MODEL=hf-model          # Hugging Face 模型目录
CAL=./data/wiki_sample.txt   # 校准文本（imatrix 用）
TEST_TXT=./data/wikitext-2-test.txt  # 测试文本（PPL/KL 用）

# ① 转换：HF → F16 GGUF（只做容器转换与张量命名映射，不量化）
python3 $LLAMA/convert_hf_to_gguf.py $MODEL \
    --outfile ./model-f16.gguf --outtype f16

# ② 生成 imatrix（IQ 系列量化必需；Q4_K_M 等 K-quants 推荐）
#    --chunks 校准文本块数，越大越准但越慢；-c 上下文长度；-ngl 层数 offload 到 GPU
$BIN/llama-imatrix -m ./model-f16.gguf -f $CAL \
    -ngl 99 --chunks 512 -c 2048 -o ./imatrix.dat

# ③ 量化：F16 GGUF 里的每个张量按策略分派到不同 ggml 类型
$BIN/llama-quantize ./model-f16.gguf ./model-q4_k_m.gguf Q4_K_M \
    --imatrix ./imatrix.dat

# ④ 类型分布检查（用 gguf-py，见 F.4 的 Python 脚本）
python3 check_types.py ./model-q4_k_m.gguf

# ⑤ PPL 与 KL 校验
$BIN/llama-perplexity -m ./model-q4_k_m.gguf -f $TEST_TXT \
    --kl-divergence-base ./model-f16.gguf 2>&1 | tee ppl-log.txt

# ⑥ 部署冒烟：目标后端起一次短生成
$BIN/llama-cli -m ./model-q4_k_m.gguf -p "Hello" -n 64 -ngl 99
```

分步骤说：

**步骤①** `convert_hf_to_gguf.py` 的三个要点：
- 输出用 `--outfile`/`-o` 指定文件名，用 `--outtype` 指定精度：`f32`/`f16`/`bf16`/`q8_0`/`auto`。**`auto` 按张量原始精度决定**，通常表现为大部分 `bf16`。
- 转换阶段**不做量化**，只做容器转换与张量命名映射（HF 的 `model.layers.0.mlp.gate_proj.weight` → GGUF 的 `blk.0.ffn_gate.weight` 等）。输出 F16 GGUF 是后续量化和 imatrix 的**唯一共同起点**。
- 转换脚本按架构自动推断映射；对稀疏（MQA/GQA）模型会打印 attention 头数的协助提示，必要时用 `--ctx` 等参数显式指定。

**步骤②** `llama-imatrix` 的参数权衡：
- `--chunks`：把校准文本切成的块数。越大，重要性估计越稳定，但耗时越长；512 是常用值，质量敏感可加到 2k。
- `-c` 上下文长度：默认较小时，每块只影响前 2048 token 的激活；`-c 2048` 与模型训练长度一致时估计最准。
- `-ngl 99`：全部层 offload 到 GPU，让校准过程更快；纯 CPU 机器可省略。
- `-o` 输出 imatrix 文件路径（如 `imatrix.dat`）。
- 校准数据**领域最好与最终部署数据接近**——imatrix 的本质是"哪些权重输出影响大"，一个针对代码的模型用纯英文小说校准，界外误差会变差（第 11 章）。

**步骤③** `llama-quantize`：
- 第三参数是**策略名**（`Q4_K_M`、`Q5_K_M`、`IQ3_XXS` …），不是单一 `ggml_type`。`_S`/`_M`/`_L` 表示内部混合规则的激进程度（第 14 章）。
- `--imatrix` 传 imatrix 文件。**IQ 系列必须传**：IQ 格式没有"无 imatrix 的参考量化实现"（第 3 章/附录 E 提到 `from_float_ref = NULL`），不传 imatrix 时 `llama-quantize` 会拒绝或质量显著下降。
- 常见错误：`-ngl` 只在运行工具、不在量化时指定；量化是离线批处理，与 GPU 无关。

## F.3 PPL 与 KL：两种校验的区别

`llama-perplexity` 输出形如：

```
perplexity: 6.2345
```

`--kl-divergence-base model-f16.gguf` 会额外打印相对基础模型的 KL 散度与每种度量下的劣化百分比：

```
...
  ▏F16 (base)  : 6.0450
  ▏Q4_K_M     : 6.2345   (KL: 0.1234)   (PPL: +3.1%)
```

- **PPL（困惑度）**：衡量模型对测试文本的平均负对数似然，越低越好。PPL 按整体分布打分，对"整体丢了一点"敏感。
- **KL 散度**：量化模型 logits 相对基础模型 logits 的差异，对"输出形状/锐度变了"敏感。
- 用法建议（第 2、18 章）：**K-quants 用 PPL 劣化百分比**（如 `+0.8%`），**IQ/三值格式更依赖 KL**，因为它们的误差是"查表走偏"而非平滑劣化，单一 PPL 数字会掩盖任务能力变化。

## F.4 类型分布检查脚本（gguf-py）

量化之后第一件事是：**确认文件内部真的如策略所声明**。这个脚本直接打开 GGUF，按 `ggml_type` 统计张量：

```python
#!/usr/bin/env python3
"""check_types.py —— 打印一个 GGUF 的张量类型分布。"""
import sys
from collections import Counter
from gguf import GGUFReader

def main(path: str) -> None:
    reader = GGUFReader(path)

    arch = reader.get_field("general.architecture")
    ftype = reader.get_field("general.file_type")
    print(f"architecture = {arch.contents() if arch else '?'}")
    print(f"file_type    = {ftype.contents() if ftype else '?'}")

    counts: Counter[str] = Counter()
    by_name = {}
    for t in reader.tensors:
        counts[t.tensor_type.name] += 1
        by_name[t.name] = t.tensor_type.name

    print("\n类型分布（数量 / 占比）:")
    total = sum(counts.values())
    for qtype, n in counts.most_common():
        print(f"  {qtype:<16} {n:6d}  ({n/total:.1%})")

    # 抽样打印前几个"混合布局"的对应关系
    print("\n元组示例（前 6 个命名张量 → 类型）:")
    for i, (name, ty) in enumerate(by_name.items()):
        if i >= 6:
            break
        print(f"  {name:<45} {ty}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python check_types.py model.gguf")
        sys.exit(1)
    main(sys.argv[1])
```

运行 `python check_types.py model-q4_k_m.gguf` 的期望输出大致像：

```
architecture = llama
file_type    = 15
类型分布（数量 / 占比）:
  Q4_K              384  (87.7%)
  F32                52  (11.9%)
  Q6_K                2  ( 0.5%)
```

`file_type = 15` 是 `MOSTLY_Q4_K_M`（策略名）；内部却是三种类型并存（第 3、14 章的实锤）。

## F.5 端侧基准的命令（第 17 章）

```bash
# 通用吞吐
$BIN/llama-bench -m ./model-q4_k_m.gguf -n 32 -t 4

# 首 token 延迟 + 冷启动（第 17 章强调的两个指标）
/usr/bin/time -v $BIN/llama-cli -m ./model-q4_k_m.gguf -p "Hello" -n 1 -ngl 99 2>&1 | grep -E "elapsed|main time|Total"

# P95 延迟：多次短生成，统计耗时分布
for i in $(seq 1 10); do
    $BIN/llama-bench -m ./model-q4_k_m.gguf -n 8 -r 1 2>&1 | grep "pp512" >> bench.log
done
python3 - <<'PY'
import numpy as np
ns = [float(l.split()[-1]) for l in open('bench.log')]
print(f"mean={np.mean(ns):.2f} p95={np.percentile(ns, 95):.2f} std={np.std(ns):.2f}")
PY
```

第 17 章提醒：**端侧基准不能只看平均 tok/s**。TTFT（首 token 延迟）和 P95 延迟决定"可不可用"。不同芯片组（Snapdragon / Tensor / Dimensity / Exynos）的 GPU delegate 延迟方差可达 ±12% 到 ±52%，所以脚本里"同一模型在不同后端各跑一次"比"跑一次则极快"更有意义。

## F.6 三值格式的额外步骤（第 13 章）

STQ1_0 / PTQ1_0 不在主线（facts-verified §1）。如果你在某个 fork 构建中使用：

```bash
# 1. 确认构建支持该 type id（否则报 unsupported type）
$BIN/llama-quantize --help | grep -i "stq1\|ptq1" || echo "当前构建不支持三值格式"

# 2. PTQ1_0 需要先做 Hadamard 折叠：
#    —— 磁盘权重以折叠形式存储，推理时必须有激活侧变换
#    —— 若 fork 提供折叠脚本，在量化前对 F16 GGUF 执行
# 3. 量化后：目标后端冒烟测试（Metal / CUDA 内核可能未覆盖）
$BIN/llama-cli -m ./model-ptq1_0.gguf -p "Hello" -n 8 -ngl 99
```

**正确的三值部署 = 格式正确 + 变换正确 + 后端覆盖正确**。三者缺一 logits 都会错（第 13 章的"Hadamard 陷阱"）。

## F.7 附录小结

- **一条流水线**：`convert → imatrix → quantize → check_types → perplexity → llama-cli`。
- **两个必查**：量化后用 gguf-py 确认类型分布；部署前在目标后端跑一次短生成。
- **三组命令**：`llama-imatrix -m f16.gguf -f cal.txt -ngl 99 --chunks 512 -c 2048 -o im.dat`、`llama-quantize f16.gguf out.gguf Q4_K_M --imatrix im.dat`、`llama-perplexity -m out.gguf -f test.txt --kl-divergence-base f16.gguf`。