# 第3章 GGUF 容器与张量类型

前面的章节回答了前两个问题：为什么需要量化（第 1 章），以及量化会带来多大的误差（第 2 章）。从这一章开始，我们要回答第三个问题：**量化之后的权重，到底以什么形式存进文件、又如何在推理时被读出来？**

先给一个具体的感受。假设你下载了一个 4-bit 量化过的 7B 模型，文件约 3.8 GB。这个文件并不是一张扁平的字节大饼，而是一个有结构的“容器”（container）。它从前往后大致分成几块：开头是几十字节的说明信息，记录着格式版本、模型架构、词表大小这些“关于文件本身”的事；接着是几百个张量的目录，每个张量叫什么名字、是什么维度、用什么量化格式、数据放在文件的哪里，都写在这里；最后才是那 3.8 GB 里绝大部分的内容——真正的量化权重数据。读文件的过程，有点像按快递单上的地址去仓库取货：先看单子，再找货架，最后取到包裹。

GGUF 就是 llama.cpp 生态里使用的这种容器格式。本章的目标有两个。第一，**建立对 GGUF 的字节级直觉**。意思是你读完本章后，能亲手打开一个 `.gguf` 文件，指出每一段字节分别是什么、每个字段占几个字节。第二，**掌握用官方 Python 库 `gguf` 读写文件的全套 API**。后面章节里所有“统计某个模型张量类型分布”“反量化某个张量”的脚本，都会用到这里讲的这些接口。

开始之前，先统一几个术语。**容器（container）**：规定“元数据 + 张量目录 + 张量数据”如何排列的文件格式，它不关心张量数据具体是什么类型。**张量（tensor）**：多维数组，这里特指模型里的权重矩阵，比如注意力层的 `attn_q.weight`。**元数据（metadata）**：用键值对（key-value）方式存储的描述信息，比如模型架构名、词表大小、上下文长度。**mmap**：内存映射，把磁盘文件的一部分直接映射进进程的虚拟地址空间，让“读文件”变成“读内存”，从而避免一次整块拷贝。**POD**（Plain Old Data）：纯数据型结构体，没有构造函数、虚函数、复杂成员，内存布局与字节序列一一对应。**字节序（endianness）**：多字节整数在内存里是高位在前（大端）还是低位在前（小端），GGUF 默认小端。


## 3.1 GGUF 的二进制布局

### 3.1.1 从 magic 到数据段：文件的四个部分

一个 GGUF 文件从前往后走，依次经过四个区域。先看整张地图，再逐块拆开讲：

```
GGUF 文件的整体布局

  ┌─────────────────────────────────────────────┐
  │ ① header：magic + version + tensor_count    │
  │             + kv_count                      │
  ├─────────────────────────────────────────────┤
  │ ② KV 数组：kv_count 个键值对（元数据）        │
  ├─────────────────────────────────────────────┤
  │ ③ tensor_infos[]：tensor_count 个张量目录项  │
  ├─────────────────────────────────────────────┤
  │ ④ 对齐填充（padding，补到 alignment 整数倍） │
  ├─────────────────────────────────────────────┤
  │ ⑤ tensor data blob：所有张量的真实数据       │
  └─────────────────────────────────────────────┘
```

#### ① header（前 16 字节起）

header 的大小是固定的，里面依次放四个字段：

| 字段 | 类型 | 字节数 | 说明 |
|---|---|---|---|
| `magic` | 4 字节 | 4 | 固定为 ASCII 字符串 `"GGUF"`，十六进制 `0x46554747` |
| `version` | `uint32` | 4 | 格式版本，当前为 3 |
| `tensor_count` | `uint64` | 8 | 文件中张量的个数 |
| `kv_count` | `uint64` | 8 | 键值对（元数据条目）的个数 |

在小端字节序下，一个 GGUF v3 文件的头部十六进制通常是这个样子：

```
 47 47 55 46  03 00 00 00   xx xx xx xx xx xx xx xx   xx xx xx xx xx xx xx xx
 │  "GGUF"   │  version=3  │  tensor_count         │  kv_count            │
```

把前 4 字节解释成无符号整数，得到 `0x46554747`，转成 ASCII 就是 `GGUF`。读取时第一件事就是检查 magic：它不是 `GGUF`，就说明这个文件根本不是 GGUF 格式。接下来再检查版本，版本不在可接受范围就报错。gguf-py 的可接受版本是 `[2, GGUF_VERSION]`，也就是支持 v2 和 v3。

#### ② KV 数组：元数据区

紧接着 header 的，是 `kv_count` 个键值对。每个键值对由三部分组成：

```
单个 KV 键值对

  ┌────────────────────┬──────────────────┬──────────────────────────┐
  │ key：uint64 长度    │ value_type：uint32 │ value（按类型可变大小）    │
  │ + key 的 UTF-8 字节 │ 值类型编号         │                          │
  └────────────────────┴──────────────────┴──────────────────────────┘
```

- **key**：字符串，格式是“`uint64` 长度 + 字节本身”。拿 `general.architecture` 这个 key 举例：先写一个 `uint64`，数值是 22，再写 22 个字节的字符串内容。
- **value_type**：一个 `uint32`，告诉解析器 value 到底是什么类型。GGUF 定义了完整的类型表，包括 `UINT8/INT8/UINT16/INT16/UINT32/INT32/UINT64/INT64/FLOAT32/FLOAT64/BOOL/STRING/ARRAY` 等。
- **value**：按 value_type 解释具体内容。`STRING` 又是“长度 + 字节”的结构；`ARRAY` 会先带上元素类型和元素个数，再逐个编码数组里的元素。

一个典型 KV 键值对在文件里的样子（读法）是这样的：

```
general.architecture = "llama"
  key:     12 00 00 00 00 00 00 00 | 67 65 6e 65 72 61 6c 2e 61 72 63 68 69 74 65 63 74 75 72 65
          ^uint64 长度 = 22        ^ "general.architecture" 共 22 字节
  vtype:   0c 00 00 00   (= STRING, 12)
  value:   05 00 00 00 00 00 00 00 | 6c 6c 61 6d 61
          ^uint64 长度 = 5         ^ "llama"
```

#### ③ tensor_infos[]：张量目录

KV 数组结束之后，是 `tensor_count` 个张量目录项。每个目录项依次排列这些字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `name` | 字符串（`uint64` 长度 + 字节） | 张量名，如 `model.layers.0.attn_q.weight` |
| `n_dims` | `uint32` | 维度个数，如 2 表示二维矩阵 |
| `dims[]` | `uint64` × n_dims | 每一维的大小 |
| `type` | `uint32` | 量化类型编号（`ggml_type`），下一节专门讲 |
| `offset` | `uint64` | 该张量数据在"数据段起点"之后的字节偏移 |

请特别留意 `type` 这一项：**它让“量化”成为每个张量自己的选择，而不是整个文件共用一个统一格式**。这正是 3.2 节要展开的核心，也解释了为什么同一个文件里能混着好几种量化格式。

#### ④ 对齐填充

tensor_infos 结束后，文件会填充若干字节，让接下来的数据段起点刚好是 `alignment` 的整数倍。默认的 `alignment = 32`（对应元数据键 `general.alignment`），填充的字节通常是 0。这一步看起来不起眼，但它保证了每个张量都能从一个对齐的位置开始。

#### ⑤ tensor data blob

最后是全部张量的真实数据。第 `i` 个张量的数据，从“数据段起点 + tensor_infos[i].offset”开始，长度是“该类型每块字节数 × 块数”（由张量元素数和 `ggml_type` 一起决定）。**注意 `offset` 是相对数据段起点的偏移，而不是相对文件头的偏移**——这是最容易踩的坑，后面 3.1.3 会专门解释为什么这么设计。

### 3.1.2 用三种语言读一遍 header

光看字节是没什么感觉的，动手读一次才能建立直觉。下面用 C++、C#、Python 分别写一个“只读 header”的最小程序。三种实现的逻辑完全相同：检查 magic → 读版本 → 读两个计数，一行不多，一行不少。

**C++（直接按结构体读取，字节序假设小端）：**

```cpp
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

// 与 llama.cpp gguf.cpp 中的结构一致（注意都是小端、Packed）
struct gguf_header_t {
    uint32_t magic;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t kv_count;
};
static_assert(sizeof(gguf_header_t) == 4 + 4 + 8 + 8, "wrong gguf header size");

int main() {
    FILE * f = fopen("model.gguf", "rb");
    if (!f) { perror("open"); return 1; }

    gguf_header_t hdr;
    fread(&hdr, sizeof(hdr), 1, f);

    if (hdr.magic != 0x46554747) {           // "GGUF"
        fprintf(stderr, "not a GGUF file\n");
        return 1;
    }
    printf("version       = %u\n", hdr.version);
    printf("tensor_count  = %llu\n", (unsigned long long)hdr.tensor_count);
    printf("kv_count      = %llu\n", (unsigned long long)hdr.kv_count);

    fclose(f);
    return 0;
}
```

**C#（.NET 5+，用 `BinaryReader` 顺序读）：**

```csharp
using System;
using System.IO;
using System.Text;

class GgufHeader {
    public static void Main(string[] args) {
        using var r = new BinaryReader(File.OpenRead(args[0]));
        var magic   = r.ReadBytes(4);
        var version = r.ReadUInt32();      // 默认小端
        var tensors = r.ReadUInt64();
        var kvCount = r.ReadUInt64();

        if (Encoding.ASCII.GetString(magic) != "GGUF") {
            Console.WriteLine("not a GGUF file");
            return;
        }
        Console.WriteLine($"version       = {version}");
        Console.WriteLine($"tensor_count  = {tensors}");
        Console.WriteLine($"kv_count      = {kvCount}");
    }
}
```

**Python（用 `struct.unpack`，这是最接近“亲手拆字节”的方式）：**

```python
import struct

def read_header(path: str) -> tuple[int, int, int]:
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError("not a GGUF file")
        version      = struct.unpack("<I", f.read(4))[0]   # 小端 uint32
        tensor_count = struct.unpack("<Q", f.read(8))[0]   # 小端 uint64
        kv_count     = struct.unpack("<Q", f.read(8))[0]   # 小端 uint64
    return version, tensor_count, kv_count

version, tensor_count, kv_count = read_header("model.gguf")
print("version      =", version)
print("tensor_count =", tensor_count)
print("kv_count     =", kv_count)
```

解释一下格式化字符串：`"<I"` 表示小端（`<`）无符号整数（`I`，4 字节），`"<Q"` 表示小端无符号长整数（`Q`，8 字节）。之所以要显式写 `<`，是因为 Python 在多数平台默认就是小端，但**显式声明才不会在跨平台时出错**——这一点和 GGUF 文件自己内建的小端约定是一致的。

### 3.1.3 为什么偏移是"相对"的：mmap 与零拷贝

现在回答一个很自然的问题：为什么 `tensor_infos[i].offset` 不直接写成文件里的绝对偏移，而要绕一个弯子，用“数据段起点 + 相对偏移”？

答案藏在推理引擎的加载方式里。llama.cpp 加载 GGUF 时用的是 **mmap**：把文件整个映射进内存，得到一块从 `data_offset` 开始的连续区域。所有张量数据都落在这块区域内，各自从 `data_offset + offset[i]` 开始。用一张图把它画出来：

```
mmap 加载 GGUF：文件被映射为一块连续内存

  磁盘文件                      进程虚拟内存
  ┌──────────────┐            ┌─────────────────────────────┐
  │ header       │            │                             │
  │ KV / infos   │  mmap ───► │  这块区域被"当作内存"访问   │
  │ padding      │            │                             │
  ├──────────────┤            │  data_offset (32 对齐)      │
  │ tensor 0     │◄──────────►│  [ tensor 0 ]               │
  │ tensor 1     │            │  [ tensor 1 ]               │
  │ ...          │            │  [ ... ]                    │
  └──────────────┘            └─────────────────────────────┘
```

相对偏移的意义在于：**文件里所有张量的“位置参照系”是同一个（数据段起点），而不是不断变化的文件绝对位置**。mmap 之后，这个参照系正好就是映射区域的起点，读取者只需要做一次加法 `data_offset + offset`，就能定位任意一个张量。反过来，如果存的是绝对偏移，映射前的偏移和映射后的地址就对不上号了，还得额外记住“文件头有多长”才能换算回来。换句话说，相对偏移是更贴合 mmap 心智模型的一种表示。

### 3.1.4 为什么量化块结构体必须是 POD 且无填充

第 4 章会看到，量化数据在磁盘上是一连串紧凑的结构体（比如 `block_q4_0`，每块 18 字节）。llama.cpp 之所以能“零拷贝”地使用这些数据，前提是**结构体的内存布局和磁盘字节完全一致**——也就是说，结构体必须是 POD（Plain Old Data），并且不能有编译器偷偷填充进去的字节。

先用 C++ 的一个反例，说明“填充”的危害在哪里：

```cpp
#include <cstdint>
#include <cstdio>

struct bad_block {
    uint16_t d;   // 2 字节
    uint8_t  qs;  // 1 字节
};

int main() {
    printf("sizeof(bad_block) = %zu\n", sizeof(bad_block)); // 你猜是多少？
    return 0;
}
```

直觉上这个结构体应该是 `2 + 1 = 3` 字节，但在大多数平台上 `sizeof` 的结果是 **4**。原因在于 `uint16_t` 要求 2 字节对齐，编译器会在 `qs` 后面填充 1 个字节，把整个结构体凑成 4 字节。如果磁盘上写的是紧凑的 3 字节，而 mmap 后按 4 字节一个结构体去解释，那么第二个结构体就会从错误的位置开始——紧跟着的所有数据全部错位。

llama.cpp 的解法分成两个层面：

1. **结构体字段刻意按紧凑顺序排列**，并且每个量化结构体都配一个 `static_assert` 锁定大小。比如：
   ```cpp
   #define QK4_0 32
   typedef struct {
       ggml_half d;            // 2 字节，FP16 位模式
       uint8_t   qs[QK4_0/2];  // 16 字节，32 个 4-bit 码
   } block_q4_0;
   static_assert(sizeof(block_q4_0) == 2 + QK4_0/2, "wrong q4_0 block size/padding");
   ```
   只要 `static_assert` 在编译期能通过，就说明这个结构体没有意外填充。

2. **结构体只含标量和定长数组**，不含构造函数、虚函数、指针等任何会破坏布局的东西。所有解码逻辑都放进独立的 `dequantize_row_*` 函数里，结构体本身保持“纯数据”的身份。

对 C# 来说，对应的做法是 `[StructLayout(LayoutKind.Sequential, Pack = 1)]`（见第 4 章）；对 Python 来说，没有“结构体即内存”的概念，必须用 `struct` 和 `bytes` 显式序列化——这也是为什么从第 4 章开始，Python 参考实现总比 C/C++ 多一层“手工拆字节”的工作。


## 3.2 `ggml_type`：量化决策的下放

### 3.2.1 每个张量独立的类型字段

回过头再看一眼 tensor_infos 里那个 `type` 字段，它代表一个关键事实：**GGUF 的量化不是“文件级”属性，而是“张量级”属性**。同一个文件里，完全可以同时存在 `Q4_K`、`Q6_K`、`F32` 三种类型的张量。读取者逐个张量查看 `type`，再按类型分派到对应的解码路径。

这也解释了为什么 `Q4_K_M` 不是一种编码格式：它是对“每个张量用什么类型”这套分配规则的产品化表达。一个取名为 `Q4_K_M` 的文件，内部实际上是混合的——主体用 `Q4_K`，输出嵌入层用 `Q6_K`，归一化层用 `F32`。3.2.3 会用真实数据证明这一点。

### 3.2.2 类型分派表：`type_traits` 与 `to_float`/`from_float`

llama.cpp 在 `ggml.c` 里维护了一张 **类型特征表** `type_traits[]`，它把“类型编号”映射到“这个类型的一切信息”。下面摘了两条（做了精简）：

```c
static const struct ggml_type_traits type_traits[GGML_TYPE_COUNT] = {
    [GGML_TYPE_Q4_0] = {
        .type_name      = "q4_0",
        .blck_size      = QK4_0,          // 每块 32 个元素
        .type_size      = sizeof(block_q4_0), // 每块 18 字节
        .is_quantized   = true,
        .to_float       = (ggml_to_float_t) dequantize_row_q4_0,   // 反量化函数指针
        .from_float_ref = (ggml_from_float_t) quantize_row_q4_0_ref, // 参考量化函数指针
    },
    [GGML_TYPE_F32] = {
        .type_name      = "f32",
        .blck_size      = 1,
        .type_size      = sizeof(float),
        .is_quantized   = false,
    },
    // ...
};
```

这张表提供了两个重要的“派生量”：

- **块大小 `blck_size`** 和 **类型大小 `type_size`**：给定张量元素数，就能算出这个张量占多少字节。`bpw = type_size × 8 / blck_size`——比如 Q4_0 是 `18×8/32 = 4.5`。
- **函数指针 `to_float` / `from_float_ref`**：前者是反量化（量化块 → 浮点行），后者是量化（浮点行 → 量化块，`_ref` 表示参考实现）。推理引擎正是通过这两个指针，把“类型无关”的流程和“类型相关”的解码解耦开。

`type_traits` 还有一个重要约定：**新增格式必须同步更新这张表**。如果某个类型没有注册 `to_float`，运行时反量化就会失败，或者走到空指针。另外，一些新类型（比如 IQ1_S）在主线里 `from_float_ref = NULL`，意思是**暂时没有参考量化实现**——你没法直接从 F16 量化成它，只能通过官方的转换流程得到（第 11 章会细说）。

### 3.2.3 混合类型的实际分布：以 Q4_K_M 为例

用 gguf-py 统计一个真实的 `Q4_K_M` 模型，会得到类似下面这样的分布（具体的数字随模型不同而变，但模式是稳定的）：

| 张量组 | 常见类型 | 说明 |
|---|---|---|
| `blk.*.attn_q.weight` / `attn_k.weight` / `attn_v.weight` | `Q4_K` | 注意力投影矩阵，占体积大头 |
| `blk.*.ffn_gate.weight` / `ffn_up.weight` / `ffn_down.weight` | `Q4_K` | FFN 层 |
| `output.weight` | `Q6_K` | 输出嵌入层，用更高位宽 |
| `token_embd.weight` | `Q4_K` 或 `Q6_K` | 词嵌入，视策略而定 |
| `blk.*.attn_norm.weight` / `ffn_norm.weight` | `F32` | 归一化层，几乎不量化 |
| `blk.*.attn_output.weight` | `Q4_K` | 部分架构单独列出的输出投影 |

用一句话概括整张表：**“4-bit 主体 + 6-bit 输出层 + F32 归一化”**。这张表同时是第 14 章“逐张量混合精度”的伏笔——为什么输出层要高到 6-bit、为什么归一化层保留 F32，背后都是逐层的敏感度分析。


## 3.3 用 gguf-py 读写 GGUF

### 3.3.1 安装与三个核心对象

官方 Python 库叫 `gguf`（也叫 gguf-py），安装方式很简单：

```bash
pip install gguf
```

可选地，`gguf[gui]` 会附带一个用于可视化编辑 GGUF 的 Qt 界面。开发时也可以从 llama.cpp 仓库 clone 一份 `gguf-py` 目录，做 `pip install -e` 的 editable 安装，方便边改边用。

这个库对外主要暴露三个对象：

- **`GGUFReader`**：只读（或用 `r+` 原地改）GGUF 文件。构造时传入路径，之后通过 `.fields`（元数据字典）和 `.tensors`（张量列表）访问内容。
- **`GGUFWriter`**：写入 GGUF 文件。流程分三步：`write_header_to_file()` → `write_kv_data_to_file()` → `write_tensors_to_file()`。
- **`dequantize`**（函数）：把一个量化张量的原始字节反量化为 `float32` 的 `numpy.ndarray`。签名是 `dequantize(data: np.ndarray, qtype: GGMLQuantizationType) -> np.ndarray`。

三个对象各司其职：`GGUFReader` 负责“读”，`GGUFWriter` 负责“写”，`dequantize` 负责“把量化数据还原成浮点”。

### 3.3.2 读取元数据与遍历张量类型分布

下面这个脚本会打印关键元数据字段，并用 `Counter` 统计张量类型分布——它就是 3.2.3 那张表的来源：

```python
from collections import Counter
from gguf import GGUFReader, GGMLQuantizationType

reader = GGUFReader("model-q4_k_m.gguf")

# 关键元数据
arch = reader.get_field("general.architecture")
ftype = reader.get_field("general.file_type")
print("architecture =", arch.contents() if arch else "?")
print("file_type    =", ftype.contents() if ftype else "?")

# 张量类型分布
type_counter: Counter[GGMLQuantizationType] = Counter()
for t in reader.tensors:
    type_counter[t.tensor_type] += 1

for qtype, count in type_counter.most_common():
    print(f"{qtype.name:24s} {count:6d} 张量")
```

`reader.get_field(key)` 返回的是 `ReaderField` 或 `None`；`field.contents()` 负责把字段解码成 Python 值。`reader.tensors` 是一个 `ReaderTensor` 的列表，其中的 `tensor_type` 是 `GGMLQuantizationType` 枚举的一个成员，直接拿它当 `Counter` 的键最方便。

输出大致长这样：

```
architecture = llama
file_type    = 15
Q4_K                   384 张量
Q6_K                     2 张量
F32                     52 张量
```

注意 `file_type = 15` 对应的枚举名是 `MOSTLY_Q4_K_M`。这就是文件级的“策略名”，它只说明“整体按 Q4_K_M 的规则混合”，并不说明文件里只有一种类型。上面打印出来的分布，已经把真相说清楚了。

### 3.3.3 反量化单个张量并对接 numpy/PyTorch

反量化单个张量，是后面所有误差实验的起点。`dequantize` 接受“原始字节视图”和“类型”，返回一个 `float32` 数组：

```python
import torch
import numpy as np
from gguf import GGUFReader, dequantize

reader = GGUFReader("model-q4_k_m.gguf")

# 挑第一个 Q4_K 张量
t = next(x for x in reader.tensors if x.tensor_type.name == "Q4_K")
print("name       =", t.name)
print("type       =", t.tensor_type.name)
print("shape      =", t.shape)        # 例如 [4096 4096]
print("n_elements =", t.n_elements)
print("n_bytes    =", t.n_bytes)

# 反量化为 float32 ndarray
f32 = dequantize(t.data, t.tensor_type)
print("dequantized:", f32.shape, f32.dtype)

# 对接 numpy / PyTorch
np.save("q4k.npy", f32)
torch_tensor = torch.from_numpy(f32.copy())   # 注意 copy，避免共享内存
```

这里面有几个重要的边界，提前说清楚：

- `tensor.data` 是 `numpy.ndarray`，以 `uint8` 视图直接指向文件映射区域；`dequantize` 返回的是一个新分配的 `float32` 数组。
- **反量化是只读操作**：它能从量化块推出浮点值，但反过来不行——**量化不可无损往返**。把 `f32` 再量化回 `Q4_K`，得到的字节通常和原始字节不同，数值也略有出入。这是第 2 章误差分析的延续：4-bit 的信息量，不足以无损重建原值。
- 对接 PyTorch 时建议 `.copy()`：`torch.from_numpy` 默认与原数组共享内存，而 GGUF 的映射区域通常不应该被写。

### 3.3.4 用 `GGUFWriter` 写一个最小 GGUF

写文件的流程是“header → kv → tensor data”三步。官方 `examples/writer.py` 展示了完整流程，核心部分如下：

```python
import numpy as np
from gguf import GGUFWriter, GGMLQuantizationType

# 1. 创建 writer，声明架构名
w = GGUFWriter("example.gguf", "llama")

# 2. 写入 KV 元数据（此时只"登记"，不落盘）
w.add_name("example-model")
w.add_quantization_version(2)
w.add_architecture()

# 3. 登记一个 F32 张量的元信息
w.add_tensor_info("some.weight", GGMLQuantizationType.F32, [4, 4])

# 4. 三步落盘：header → KV → tensor data
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
```

这里有一个关键点是 `add_custom_alignment`：

```python
w.add_custom_alignment(64)   # 把对齐从默认 32 改为 64
```

它会写一个 `general.alignment = 64` 的元数据项，并直接影响数据段起点：tensor_infos 之后要填充到 64 的整数倍，才开始数据段。**读取端必须尊重这个字段**（`GGUFReader` 会自动读它），否则数据段的偏移就会算错。

另外注意，`add_tensor_info` 只登记张量的“目录项”（名字、类型、维度），真正的字节要等到 `write_tensors_to_file()` 时才写入。这个“先登记、后写数据”的设计，正是为了能在写完所有目录项之后，一次性算出每个张量的相对偏移。

### 3.3.5 原地修改元数据的边界

`GGUFReader` 支持 `mode='r+'`，允许“原地”修改。但**不是所有字段都能安全地原地改**：

- **定长标量可以**：`general.file_type`、`general.name`（若新旧长度相同）这类字段，改动不改变文件长度，原地覆盖是安全的。
- **变长字段不行**：字符串变长、数组增删，都会改变字节长度，后面 KV 和 tensor_infos 的偏移会全部失效。强行改，只会破坏整个文件的解析。

官方给出的建议是：需要改变长元数据时，用**复制式工具** `gguf_new_metadata.py` 或 `gguf_set_metadata.py`。它们的做法是把原文件复制一份，在新文件里重排元数据，而不是在原地改动。这背后还是同一个道理：**GGUF 用“偏移 + 对齐”组织数据，任何长度变化都必须重写整个目录**。


## 3.4 type id 的兼容性风险

### 3.4.1 主线 type id 与 fork/PR 自定义 id 的冲突

`type` 字段是一个整数，它必须和读取端 `type_traits[]` 的下标严格对应。llama.cpp 主线的 id 分配是全局唯一的（第 13 章的附录会给完整表），问题出在**生态**这个层面：格式创新往往先发生在 fork 或者还没合并的 PR 里，而 fork 并不会复用主线的编号管理。

于是就有了真实的兼容性风险：**“GGUF 格式可读”不等于“所有 llama.cpp 构建都能正确分派”**。你的运行库如果是老版本或者某个 fork，碰到一个它不认识的 `type` 编号，可能会直接报 `unsupported type`；更糟的情况是——如果那个编号在它的内部被解释成另一个类型，就会解码出完全错误的数值。第 16 章会给出“新增格式必须同步更新 type_traits”的实现规则，第 20 章则会讨论这条规则在生态治理上的含义。

### 3.4.2 STQ1_0 与 PTQ1_0 的 id 归属问题

举一个活生生的例子。三值量化格式 STQ1_0 和 PTQ1_0（第 13 章会详细展开）来自独立的研究性工作与相应 PR。在不同文档里，它们的 id 记录各有出入：

- PTQ1_0 的 type id 被记载为 **143**；
- STQ1_0 的 id 在不同文档中分别出现过 **42/43** 两种说法。

而 llama.cpp 主线在 2026 年的枚举里，**42 已经是 `Q2_0`（新加入的 2-bit 简单格式），41 是 `Q1_0`**。也就是说，“42/43 是 STQ1_0”这个说法即使在某个 fork 里成立，也**不能移植到主线**——因为那个编号在主线已经被占用了。同一个 id 在不同构建里指代不同格式，这正是互操作里最危险的场景。

这一节的结论，对全书的部署章节（第 15、19 章）都适用：**看到模型文件名里出现 STQ1_0 或 PTQ1_0，必须先确认你的 llama.cpp 构建是否真正支持那个 type id，再检查它支不支持该格式的后端内核**，而不是想当然地认为“格式名能识别，就能直接运行”。


## 本章小结与通往"块量化"的桥梁

这一章可以压缩成四个事实。

**第一，GGUF 是“header + KV + tensor_infos + 数据段”的结构化容器。** header 固定 16 字节（magic `"GGUF"` + version + tensor_count + kv_count），KV 存元数据，tensor_infos 是张量目录，数据段承载全部量化字节。用 C++ 结构体、C# `BinaryReader`、Python `struct` 三套代码都能读取 header，逻辑完全一致。

**第二，`tensor_infos[i].offset` 是相对数据段起点的偏移，配合 mmap 实现零拷贝加载。** 相对偏移让“文件里的位置”和“映射后的地址”只有一个参照系；`general.alignment`（默认 32）保证每个张量都能从对齐的位置开始。

**第三，量化是张量级决策，`ggml_type` + `type_traits` 表承载所有格式信息。** 一个 `Q4_K_M` 文件内部是“Q4_K 主体 + Q6_K 输出层 + F32 归一化”的混合，`_M` 描述的是分配策略而不是编码格式。用 gguf-py 的 `GGUFReader` 加上 `Counter`，可以亲手验证这一点。

**第四，POD 无填充结构体 + 全局唯一 type id，是格式正确性的两条底线。** 前者保证“磁盘字节即结构体”，后者保证“类型编号即解码路径”。STQ1_0 和 PTQ1_0 的 id 归属冲突，就是这条底线被 fork 生态冲击的典型样本。

下一章，我们终于要走进数据段内部了。那里不再有“目录”和“布局”，而是一个一个具体的量化块——从最经典的 `block_q4_0`（每 32 个权重、18 字节）开始，看它的尺度、4-bit 码和反量化公式到底是怎么工作的。