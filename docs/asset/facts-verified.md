# GGUF / llama.cpp 一手源码事实档案（2026-10 主线核实）

> 所有章节作者（含子代理）必须以此档案 + `asset/ggml-common.h`、`asset/ggml.c`、`asset/ggml-quants.c` 为字节级正确性的最终依据。
> 细纲 `asset/detail.md` 中的数字若与此档案冲突，**以此档案为准**，正文用"约"平滑表述，并可在章内加注差异。
> 来源：llama.cpp master `ggml/src/ggml-common.h`、`ggml/src/ggml.c`、`ggml/src/ggml-quants.c`、`ggml/include/ggml.h`，gguf-py 0.19.0。

## 1. type id（ggml.h 枚举，主线）

```
0  F32      1  F16      2  Q4_0     3  Q4_1
4  Q4_2     5  Q4_3     (4、5 已从 GGUF 文件支持中移除，enum 空位)
6  Q5_0     7  Q5_1     8  Q8_0     9  Q8_1
10 Q2_K     11 Q3_K     12 Q4_K     13 Q5_K
14 Q6_K     15 Q8_K     16 IQ2_XXS  17 IQ2_XS
18 IQ3_XXS  19 IQ1_S    20 IQ4_NL   21 IQ3_S
22 IQ2_S    23 IQ4_XS   24 I8       25 I16
26 I32      27 I64      28 F64      29 IQ1_M
30 BF16     31-33      Q4_0_4_4/4_8/8_8（已移除，转运行时重打包）
34 TQ1_0    35 TQ2_0    36-38      IQ4_NL_4_4/4_8/8_8（已移除）
39 MXFP4    40 NVFP4    41 Q1_0    42 Q2_0
```

**关键生态事实：**
- **STQ1_0 / PTQ1_0 不在 llama.cpp 主线**。type id 41/42 已被新的 Q1_0/Q2_0 占用；三值格式来自 fork/PR（细纲 §3.4.2、§13.5.1 的"id 归属"讨论即由此而来）。
- 31-33、36-38 已在 type_traits 中标记 `REMOVED, use ... with runtime repacking`。
- 每个 `ggml_type` 在 `type_traits[]` 中有：`blck_size`（每块元素数）、`type_size`（每块字节数）、`type_name`、`is_quantized`、`to_float`/`from_float_ref` 函数指针（参考实现）。

## 2. 各格式字节布局与 bpw（静态断言核实）

### 简单块
| 格式 | 常量 | 结构 | 字节 | bpw |
|---|---|---|---|---|
| Q1_0 | QK1_0=128 | `d` + `qs[128/8]` | 18 | 1.125 |
| Q2_0 | QK2_0=64 | `d` + `qs[64/4]` | 18 | 2.25 |
| Q4_0 | QK4_0=32 | `d` + `qs[32/2]` | 18 | 4.5 |
| Q4_1 | QK4_1=32 | `dm`(d+m) + `qs[16]` | 20 | 5.0 |
| Q5_0 | QK5_0=32 | `d` + `qh[4]` + `qs[16]` | 22 | 5.5 |
| Q5_1 | QK5_1=32 | `dm` + `qh[4]` + `qs[16]` | 24 | 6.0 |
| Q8_0 | QK8_0=32 | `d` + `qs[32]`(int8) | 34 | 8.5 |
| Q8_1 | QK8_1=32 | `ds`(d+s) + `qs[32]`(int8) | 36 | 9.0 |

### K-quants（QK_K=256，K_SCALE_SIZE=12）
| 格式 | 结构 | 字节 | bpw |
|---|---|---|---|
| Q2_K | `scales[16]` + `qs[64]` + `dm`(d+dmin) | 84 | 2.625 |
| Q3_K | `hmask[32]` + `qs[64]` + `scales[12]` + `d` | 110 | 3.4375 |
| Q4_K | `dm`(d+dmin) + `scales[12]` + `qs[128]` | 144 | 4.5 |
| Q5_K | `dm` + `scales[12]` + `qh[32]` + `qs[128]` | 176 | 5.5 |
| Q6_K | `ql[128]` + `qh[64]` + `scales[16]`(int8) + `d` | 210 | 6.5625 |
| Q8_K | `d`(f32) + `qs[256]`(int8) + `bsums[16]`(int16) | 292 | 9.125 |

### IQ 系列（QK_K=256）
| 格式 | 结构 | 字节 | bpw | 注释 |
|---|---|---|---|---|
| IQ2_XXS | `d` + `qs[32]`(uint16) | 66 | 2.0625 | grid 每项 8 值，256×8 |
| IQ2_XS | `d` + `qs[32]`(uint16) + `scales[8]` | 74 | 2.3125 | |
| IQ2_S | `d` + `qs[64]` + `qh[8]` + `scales[8]` | 82 | 2.5625 | |
| IQ3_XXS | `d` + `qs[96]` | 98 | 3.0625 | grid 每项 4 值 |
| IQ3_S | `d` + `qs[64]` + `qh[8]` + `signs[32]` + `scales[4]` | 110 | 3.4375 | 每项 4 值 |
| IQ1_S | `d` + `qs[32]` + `qh[8]`(uint16) | 50 | 1.5625 | 每项 8 值 |
| IQ1_M | `qs[32]` + `qh[16]` + `scales[8]` | 56 | 1.75 | 无 `d`，尺度藏在 scales |
| IQ4_NL | `d` + `qs[16]`（QK4_NL=32） | 18 | 4.5 | 查 `kvalues_iq4nl[16]` |
| IQ4_XS | `d` + `scales_h`(u16) + `scales_l[4]` + `qs[128]` | 136 | 4.25 | |

> **细纲差异**：细纲 §11.3 说 IQ1_S 为"2048 项 grid / 74 字节 / 1.5 bpw"。主线实际为 **50 字节 / 1.5625 bpw**（grid 仍是 2048 项 `iq1s_grid`，每项 int8[8]）。细纲 §11.4.2 说 IQ3_XXS 3.4375 bpw，主线实际 IQ3_XXS = 3.0625 bpw、IQ3_S = 3.4375 bpw。第 11 章必须按主线写并注明细纲数字有出入。

### 三值（不在主线，属 fork/PR；细纲口径，写作时标注"处于演进中/未合入主线"）
- TQ1_0：`qs[48]`(5 trits/字节) + `qh[4]`(每字节 4 trits) + `d`，块 256 值，54 字节，**1.6875 bpw**（主线已合入）。
- TQ2_0：`qs[64]`(2 bit/值) + `d`，块 256 值，66 字节，**2.0625 bpw**（主线已合入）。
- STQ1_0：3:4 稀疏三值，每 4 权重 1 零、32 模式、5 bit/组；细纲 §13.2.2 写 42 字节/256 值 = 1.3125 bpw。**不在主线，字段名以 fork 为准**。
- PTQ1_0：稠密三值 base-3 编码；细纲 §13.3.2 写 28 字节/128 值 = 1.75 bpw；**Hadamard 折叠**是必需变换，缺则 logits 全错。**不在主线**。

## 3. K-quants 解码精确位序（ggml-quants.c 核实）

- **Q4_K 反量化**（dequantize_row_q4_K）：`y = d*sc*q - min*m`。循环 `j` 按 64 步进，每 64 值对应两个子块（`is`、`is+1`），`qs[l]&0xF` 与 `qs[l]>>4` 分属两个子块。**sc/m 不再除以 63 或 64**（量化端已经把 /63 折进 `d`/`dmin`：`y.d = max_scale/63`, `dmin = max_min/63`）。这是细纲 §6.2.2 要澄清的点：**解码端直接乘 6-bit 整数，归一化在量化端完成**。
- `get_scale_min_k4(j, scales, &sc, &m)`：`j<4` 时 `sc=scales[j]&63`、`m=scales[j+4]&63`；`j>=4` 时 `sc=(scales[j+4]&0xF)|((scales[j-4]>>6)<<4)`、`m=(scales[j+4]>>4)|((scales[j]>>6)<<4)`。
- **Q5_K**：`qh[l]` 的 bit0/bit1（随 j 步进左移 2 位：u1=1,u2=2 再 `<<=2`）给高 4 位平面；`y = d*sc*((ql[l]&0xF)+(qh[l]&u1?16:0)) - min*m`。
- **Q6_K**：16 子块 × 16 值；`is = l/16`，sc 是 int8，`y = d*sc[is]*((6bit码)-32)`；6-bit 码 = `(ql[l]&0xF) | (((qh[l]>>k)&3)<<4)`，k 依子块位置 0/2/4/6。
- **Q2_K**：16 子组 × 16 值；`sc = scales[is]`，`dl = d*(sc&0xF)`、`ml = min*(sc>>4)`，码 `(q[l]>>shift)&3` 减 `-2`：`y = dl*((int8_t)((q[l]>>shift)&3)) - ml`（`(q&3)` 本身在 [-2,1] 语义内，`-ml` 是下限）。
- **Q3_K**：16 子组 × 16 值；3-bit 码 = `qs` 低 2 位 + `hmask` 1 位（hmask bit 为 1 表示 -4 偏移）；`scales[12]` 解码出 16 个 int8 尺度（`scales[is]-32`），`dl = d_all*(scales[is]-32)`，`y = dl*((q&3) - (hm&m ? 0 : 4))`。

## 4. IQ 解码精确位序（ggml-quants.c 核实）

- **IQ1_S**：每 32 值一组（QK_K/32=8 组）。`qh[ib]` 高 12 位含每组尺度：`dl = d*(2*((qh[ib]>>12)&7)+1)`（即尺度 ∈ {1,3,5,7,9,11,13,15}）；`delta = qh[ib]&0x8000 ? -IQ1S_DELTA : +IQ1S_DELTA`，`IQ1S_DELTA=0.125`。每 8 值查一次 `iq1s_grid`（2048 项，索引 `qs[l] | (((qh[ib]>>3*l)&7)<<8)`），`y = dl*(grid[j]+delta)`。
- **IQ2_XXS**：每 32 值组：`aux32[2]` 从 `qs+4*ib32` 取 8 字节；`db = d*(0.5+(aux32[1]>>28))*0.25`；每 8 值查 `iq2xxs_grid`（索引 `aux8[l]`，即低字节），符号来自 `ksigns_iq2xs[(aux32[1]>>7*l)&127]`，`y = db*grid[j]*(sign? -1:1)`。
- **IQ2_XS**：每 32 值组，`scales[ib32]` 低/高 4 位给两组 db；索引 `x[i].qs[4*ib32+l]&511`（9 bit），符号 `>>9`（7 bit）。
- **IQ3_XXS**：每 32 值组，`db = d*(0.5+(aux32>>28))*0.5`；每 8 值由两个 `iq3xxs_grid`（256 项，每项 4 值）拼成，符号 `ksigns_iq2xs[(aux32>>7*l)&127]`。
- **IQ4_NL**：`y[j]=d*kvalues_iq4nl[qs[j]&0xf]`，`y[j+16]=d*kvalues_iq4nl[qs[j]>>4]`。kvalues_iq4nl 是 16 项 int8 非线性格点。
- 常量：`IQ1S_DELTA=0.125f`、`IQ1M_DELTA=0.125f`、`NGRID_IQ1S=2048`；grid 表 `iq1s_grid`(uint64)、`iq2xxs_grid`、`iq2xs_grid`、`iq2s_grid`、`iq3xxs_grid`、`iq3s_grid`、`kvalues_iq4nl`(int8[16])。

## 5. gguf-py 0.19.0 API（核实）

- 安装：`pip install gguf`（可选 `gguf[gui]`）。模块名 `gguf`。
- 顶层导出：`GGUFReader`、`GGUFWriter`、`GGMLQuantizationType`、`GGUFValueType`、`dequantize`、`quantize`。
- `GGUFReader(path, mode='r'|'r+'|'c')`；`reader.fields`（OrderedDict[str, ReaderField]）；`reader.tensors`（list[ReaderTensor]）；`reader.get_field(key)`。
- `ReaderField`：NamedTuple，`offset`/`name`/`parts`/`data`/`types`，方法 `contents([index|slice])` 解码值（字符串数组/字符串/标量）。
- `ReaderTensor`：NamedTuple，`name`/`tensor_type`(GGMLQuantizationType)/`shape`(np.uint32 ndarray)/`n_elements`/`n_bytes`/`data_offset`/`data`(np.ndarray)/`field`。
- 反量化：`from gguf import dequantize; out = dequantize(tensor.data, tensor.tensor_type) -> np.ndarray`（支持 F32/F16 及所有有实现的量化类型；未实现则抛 NotImplementedError）。`quantize(data, qtype)` 同样存在。
- `GGUFWriter(path, arch, use_temp_file=True, endianess='little')`；关键方法：`add_type`/`add_architecture`/`add_quantization_version`/`add_custom_alignment(n)`/`add_file_type`/`add_*`（标量）/`add_array`/`add_tensor_info(...)`/`add_tensor(...)`/`write_header_to_file()`/`write_kv_data_to_file()`/`write_tensors_to_file()`。三步写文件：header → kv → tensor data。
- 自检工具（脚本）：`python -m gguf.scripts.gguf_dump file.gguf`（打印元数据）、`gguf_set_metadata.py`（改简单 KV）、`gguf_convert_endian.py`（端序转换）、`gguf_new_metadata.py`（复制式增删元数据，推荐给变长字段）。

## 6. 官方工具命令（convert / quantize / imatrix / perplexity）

```
# 转换（容器转换，不做量化）
python convert_hf_to_gguf.py <hf-dir> --outfile model-f16.gguf --outtype f16
# imatrix（IQ 系列量化必需；-c 上下文长度，--chunks 校准文本块数，-ngl 层数 offload 到 GPU）
./llama-imatrix -m model-f16.gguf -f calibration.txt -ngl 99 --chunks 512 -c 2048 -o imatrix.dat
# 量化（Q4_K_M 是策略名；--imatrix 传路径）
./llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M --imatrix imatrix.dat
# PPL/KL 校验（与 base 对比；--kl-divergence-base 指定参考模型）
./llama-perplexity -m model.gguf -f test.txt --kl-divergence-base model-f16.gguf
# 冒烟生成
./llama-cli -m model.gguf -p "Hello" -n 64 -ngl 99
```

## 7. 其他关键常量与事实

- `QK_K = 256`，`K_SCALE_SIZE = 12`。
- `ggml_half = uint16_t`（存 FP16 位模式）。FP16↔FP32：`GGML_FP16_TO_FP32` / `GGML_FP32_TO_FP16`。
- GGUF header：magic `"GGUF"`(0x46554747) + version(u32) + tensor_count(u64) + kv_count(u64)。字符串 = u64 长度 + 字节。KV 数组在 tensor_infos 之前。`tensor_infos[i].offset` 相对数据段起点。默认 `general.alignment = 32`。
- 结构体全是 POD，`static_assert` 锁定大小；`GGML_EXTENSION union` 用于兼容 MSVC。
- `type_traits` 对 Q1_0/Q2_0/IQ1_S/IQ1_M 等 `from_float_ref = NULL`（无参考量化实现，不可直接量化，只能转换得到）——第 11/13 章可提。
