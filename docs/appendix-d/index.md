# 附录 D gguf-py API 速查

`gguf`（也叫 gguf-py）是 GGUF 的官方 Python 库，第 3 章介绍了它的三个核心对象，本附录把它们拉通成一份**开箱即用的速查表**：常用调用模式、容易踩的坑、以及可运行的示例。

> 版本：本附录基于 gguf-py 0.19.0（2026 年主线）。
> 安装：`pip install gguf`；要可视化编辑用 `pip install gguf[gui]`（附带 Qt 界面，见官方仓库 README）。

---

## D.1 顶层导出与类型

`from gguf import ...` 可直接拿到这些名字：

| 名字 | 类型 | 作用 |
|---|---|---|
| `GGUFReader` | class | 读取 GGUF（`mode` 支持 `'r'` / `'r+'` / `'c'`） |
| `GGUFWriter` | class | 写入 GGUF（三步流程） |
| `GGMLQuantizationType` | enum | 量化类型枚举（`F32=0`、`Q4_0=2`、`Q4_K=12`、`Q6_K=14`、`IQ1_S=19`、`TQ1_0=34`……见 `asset/facts-verified.md` §1 全表） |
| `GGUFValueType` | enum | KV 值类型（`UINT8..FLOAT64`、`BOOL`、`STRING`、`ARRAY` 等） |
| `dequantize(data, qtype)` | function | 量化字节 → `np.float32` 数组 |
| `quantize(data, qtype)` | function | `np.float32` 数组 → 量化字节（存在但非无损） |

---

## D.2 `GGUFReader`：读文件

```python
from gguf import GGUFReader

reader = GGUFReader("model-q4_k_m.gguf")

# 元数据字段（ReaderField 或 None）
arch = reader.get_field("general.architecture")
print(arch.contents())               # 'llama'（字符串）

# 三个最常用的属性
reader.fields        # OrderedDict[str, ReaderField]，元数据
reader.tensors       # list[ReaderTensor]，张量列表
reader.tensor_count  # 张量个数（也可 len(reader.tensors)）

# 遍历张量
for t in reader.tensors:
    print(t.name, t.tensor_type.name, t.shape, t.n_elements, t.n_bytes)
```

**`ReaderField` 关键成员与方法：**

| 成员 | 含义 |
|---|---|
| `.name` | 字段名（不含前缀，就是 KV key） |
| `.offset` | 字段在文件中的偏移 |
| `.types` | value 类型列表 |
| `.contents([index_or_slice])` | 解码值。字符串数组传切片/索引；标量直接调用 |
| `.parts` / `.data` | 内部数据分片（一般不需要动） |

**`ReaderTensor` 关键成员（NamedTuple）：**

```python
t.name          # 张量名，如 'blk.0.attn_q.weight'
t.tensor_type   # GGMLQuantizationType 成员
t.shape         # np.ndarray（uint32），维度（注意：ggml 维度顺序是反的）
t.n_elements    # 元素总数
t.n_bytes       # 占用的字节数
t.data_offset   # 相对数据段起点的偏移
t.data          # np.ndarray（uint8 视图，指向文件映射区域）
t.field         # 对应的 ReaderField
```

> **维度顺序陷阱**：`shape` 是 ggml 的内存维度序（最后一维是行）。PyTorch/numpy 通常按反序理解。做形状还原时先 `list(t.shape)[::-1]`。

### 类型分布统计（第 3 章已有，此处给完整版）

```python
from collections import Counter
reader = GGUFReader("model.gguf")
type_counter: Counter[str] = Counter(t.tensor_type.name for t in reader.tensors)
for name, cnt in type_counter.most_common():
    print(f"{name:12s} {cnt:6d}")
```

---

## D.3 `dequantize`：反量化张量

```python
import numpy as np
from gguf import GGUFReader, dequantize

reader = GGUFReader("model.gguf")
t = reader.tensors[0]

f32 = dequantize(t.data, t.tensor_type)   # np.float32, 形状与你预期的张量维度一致
```

**三个关键认知：**

1. **签名**：`dequantize(data: np.ndarray, qtype: GGMLQuantizationType) -> np.ndarray`。`t.data` 已是 `uint8` 视图，无需再转。
2. **返回值是新数组**：反量化分配新内存，不修改原始字节。它是只读操作。
3. **不可无损往返**（第 2/19 章）：反量化后 `quantize` 回同样类型，得到的字节通常不同。量化是真有损的。

未实现的类型会抛 `NotImplementedError`——比如主线 gguf-py 没有 STQ1_0/PTQ1_0 的解码器，需要你自己写（见附录 C）或使用对应 fork。

```python
from gguf import GGMLQuantizationType
# 常见类型名速查
QG = GGMLQuantizationType
for name in ("F32", "Q4_0", "Q4_K", "Q6_K", "IQ1_S", "TQ1_0"):
    print(name, int(getattr(QG, name)))
# F32 0  Q4_0 2  Q4_K 12  Q6_K 14  IQ1_S 19  TQ1_0 34
```

---

## D.4 `GGUFWriter`：写文件

三步流程：`write_header_to_file()` → `write_kv_data_to_file()` → `write_tensors_to_file()`。**先登记无数据，再一次性落盘**，这样能正确计算每个张量的相对偏移。

```python
import numpy as np
from gguf import GGUFWriter, GGMLQuantizationType

w = GGUFWriter("example.gguf", "llama")      # path + 架构名

# KV 元数据（登记，不落盘）
w.add_name("example-model")
w.add_quantization_version(2)
w.add_architecture()
w.add_string("my.custom.key", "value")
w.add_uint32("my.num", 7)
w.add_array("my.tags", ["a", "b"])

# 张量（登记元信息 + 提供数据）
w.add_tensor(name="some.weight", tensor_type=GGMLQuantizationType.F32,
             data=np.ones((4, 4), dtype=np.float32))

# 三步落盘
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
```

`add_tensor` 接受 `data`（np.ndarray），会自动编码。若要**只登记元信息、数据稍后给**，用 `add_tensor_info(name, type, shape)` + 后面 `write_tensors_to_file`（数据通过 `.set_tensor` 或类似机制）。

**关键方法速查：**

| 方法 | 作用 |
|---|---|
| `add_type(type_name)` | 写通用类型键 |
| `add_architecture()` | 写 `general.architecture`（用构造时的架构名） |
| `add_quantization_version(v)` | 写量化版本号 |
| `add_custom_alignment(n)` | 写 `general.alignment`，**改变数据段对齐**（默认 32） |
| `add_file_type(ftype_int)` | 写 `general.file_type` |
| `add_*`（`add_uint8`/`add_int16`/`add_float32`/`add_bool`/`add_string`/...） | 标量 KV |
| `add_array(key, list)` | 数组 KV |
| `add_tensor_info(name, type, shape)` | 登记张量元信息（先于数据） |
| `add_tensor(name, tensor_type, data)` | 登记 + 提供数据 |
| `add_shard_kv_data()` | 多分片时写入分片 KV |
| 三个 `write_*_to_file()` | header → kv → tensor data，按序调用 |

### 关于 `add_custom_alignment` 的一个坑

```python
w.add_custom_alignment(64)
```

它写一个 `general.alignment = 64` 的 KV，并让数据段起点对齐到 64。**读取端必须尊重该字段**（`GGUFReader` 会自动做）。如果你自己写解析器（附录 C），别写死 32——要读 `general.alignment`。这是"偏移/对齐"系统最容易出 bug 的地方。

---

## D.5 常用调用模式

### 模式 1：读取并检查模型

```python
reader = GGUFReader("m.gguf")
print(reader.get_field("general.arch").contents())
print([t.name for t in reader.tensors[:5]])
```

### 模式 2：统计类型 + 体积

```python
from collections import Counter
reader = GGUFReader("m.gguf")
c = Counter(t.tensor_type.name for t in reader.tensors)
total = sum(t.n_bytes for t in reader.tensors)
for name, cnt in c.most_common():
    print(f"{name:12s} {cnt:6d}  {cnt*100//sum(c.values()):3d}%")
print("数据段总字节:", total)
```

### 模式 3：反量化并送入 numpy/PyTorch

```python
import torch
from gguf import GGUFReader, dequantize
reader = GGUFReader("m.gguf")
t = next(x for x in reader.tensors if x.tensor_type.name == "Q4_K")
f32 = dequantize(t.data, t.tensor_type)
tensor = torch.from_numpy(f32.copy())        # .copy() 避免共享映射内存
```

### 模式 4：写最小文件

见 D.4 的完整示例。

---

## D.6 原地修改的边界：`r+` 与复制式工具

`GGUFReader(path, mode='r+')` 允许"原地改"，但**只有定长标量安全**：

```python
reader = GGUFReader("m.gguf", mode="r+")
f = reader.get_field("general.file_type")      # 定长标量，可安全改
# f.parts[-1][0] = 7  # 更新值 → 原地写回
```

**绝不能原地改的**：字符串变长、数组增删、KV 增减。GGUF 用"偏移 + 对齐"组织，任何长度变化都会让后续偏移全部失效（第 3.3.5 节）。

官方为这类需求提供了**复制式工具**（它们重写整个文件而不是原地改）：

```bash
# 增删/修改元数据（复制式，安全）
python -m gguf.scripts.gguf_new_metadata in.gguf --put general.name "new name" -o out.gguf
# 改单个简单 KV
python -m gguf.scripts.gguf_set_metadata in.gguf   # 交互式改简单值
# 打印元数据
python -m gguf.scripts.gguf_dump in.gguf
# 端序转换（跨端序机器时）
python -m gguf.scripts.gguf_convert_endian in.gguf out.gguf
```

> 这些脚本的入口名在库里是 `gguf.scripts.gguf_dump` 等。`gguf_dump` 等价于把`reader.fields` 打印出来。

---

## D.7 常见坑速记

| 坑 | 说明 |
|---|---|
| `shape` 是 ggml 维度序 | PyTorch 视角要 `[::-1]` |
| `tensor.data` 是映射视图 | 对接 PyTorch 用 `.copy()`，避免共享内存被写坏 |
| `dequantize` 不可无损往返 | 反量化→再量化 ≠ 原始字节 |
| `r+` 只能改定长标量 | 变长 KV 必须走复制式工具 |
| 未实现类型抛 `NotImplementedError` | STQ/PTQ 等 fork 格式需自研或换 fork 库 |
| `general.alignment` 别写死 32 | 读取端必须读该字段 |

---

## D.8 与本附录其他部分的关系

- 附录 C 给出了**不依赖 gguf-py** 的最小解析器与解码器——当你需要理解字节级别、验证库的行为、或移植到其他语言时用它。
- 本附录的 `dequantize` 是"官方参考语义"，可作为附录 C 交叉验证的 ground truth（第 19.1.2 节：**任何自研读取器的偏差都应先怀疑自己**）。
- 三值格式 STQ1_0/PTQ1_0（第 13 章）在主线 gguf-py 中不存在；要读它们，须用对应 fork 的 gguf-py 或者你自己的解析器。