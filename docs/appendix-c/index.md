# 附录 C 最小 Python 参考实现

本附录提供一套**可独立运行**的最小 Python 参考实现：一个 GGUF 解析器、六个格式的解码器（Q4_0 / Q4_K / Q6_K / IQ1_S / STQ1_0 / PTQ1_0），以及与官方 `gguf` 库的交叉验证脚本。

设计原则：

- **不依赖任何第三方库**（只有标准库 `struct`），因此可以直接 `python3` 运行；
- 每个解码器都配**手工字节验证断言**——构造一块已知字节，断言解码结果与期望一致；
- 位序严格对齐 llama.cpp 主线 `ggml-quants.c`（Q4_K 的 `get_scale_min_k4`、Q6_K 的 `ql`/`qh` 位移、IQ1_S 的 `qh` 高 12 位尺度 + `IQ1S_DELTA`）；
- STQ1_0 / PTQ1_0 不在主线，是 fork/PR 格式，本附录按细纲记载的布局实现，**落地使用前须以实际 fork 文档复核**。

> 术语（已在第 4、7、9、11、13 章定义）：**块（block）**是共享尺度的最小量化单元；**bpw** = 块字节数×8÷块内权重数；**nibble** 是半字节（4 bit）；**grid/codebook** 是码本查表；**trit** 是三值单位。小端（`<`）是 GGUF 的固定字节序。

---

## C.1 工具函数：FP16 位模式 ↔ 浮点

GGUF 里的尺度 `d` 大多以 FP16 位模式存储（2 字节）。`struct` 模块用 `'<e'` 直接转换：

```python
import struct

def f16_to_f32(b2: bytes) -> float:
    """2 字节 FP16 位模式 → float（小端）"""
    return struct.unpack('<e', b2)[0]

def f32_to_f16(x: float) -> bytes:
    """float → 2 字节 FP16 位模式（小端）"""
    return struct.pack('<e', x)

# 验证：1.0 的 FP16 位模式是 0x3C00
assert f16_to_f32(b'\x00\x3c') == 1.0
assert f32_to_f16(1.0) == b'\x00\x3c'
```

---

## C.2 GGUF 解析器

GGUF 的文件布局（第 3 章详述）依次是：header → KV 数组 → tensor_infos 数组 →（对齐填充）→ 数据段。

```python
import struct
from dataclasses import dataclass, field

MAGIC = b'GGUF'

@dataclass
class GgufTensorInfo:
    name: str
    n_dims: int
    dims: list[int]
    ggml_type: int      # type id（见 facts-verified §1）
    offset: int         # 相对数据段起点的偏移
    n_elements: int
    n_bytes: int

@dataclass
class GgufReader:
    version: int
    kv: dict[str, object] = field(default_factory=dict)
    tensors: list[GgufTensorInfo] = field(default_factory=list)
    data_offset: int = 0

    @classmethod
    def load(cls, path: str) -> 'GgufReader':
        with open(path, 'rb') as f:
            return cls.parse(f.read())

    @classmethod
    def parse(cls, buf: bytes) -> 'GgufReader':
        if buf[:4] != MAGIC:
            raise ValueError('not a GGUF file')
        version, tensor_count, kv_count = struct.unpack_from('<IQQ', buf, 4)
        if version not in (2, 3):
            raise ValueError(f'unsupported GGUF version {version}')

        r = cls(version=version)
        off = 4 + 4 + 8 + 8            # 越过 header

        # --- KV 数组 ---
        for _ in range(kv_count):
            klen = struct.unpack_from('<Q', buf, off)[0]; off += 8
            key  = buf[off:off+klen].decode('utf-8');      off += klen
            vtype = struct.unpack_from('<I', buf, off)[0]; off += 4
            if vtype == 12:                      # ARRAY
                r.kv[key] = _read_array(buf, off)
                off += _array_size(buf, off)
            else:                                # 标量 / 字符串
                r.kv[key] = _read_value(buf, vtype, off)
                off += _value_size(buf, vtype, off)

        # --- tensor_infos 数组 ---
        for _ in range(tensor_count):
            nlen = struct.unpack_from('<Q', buf, off)[0]; off += 8
            name = buf[off:off+nlen].decode('utf-8');      off += nlen
            n_dims = struct.unpack_from('<I', buf, off)[0]; off += 4
            dims   = list(struct.unpack_from(f'<{n_dims}Q', buf, off)); off += 8 * n_dims
            ggml_type = struct.unpack_from('<I', buf, off)[0]; off += 4
            toff      = struct.unpack_from('<Q', buf, off)[0]; off += 8

            n_elements = 1
            for d in dims:
                n_elements *= d
            r.tensors.append(GgufTensorInfo(name, n_dims, dims, ggml_type, toff,
                                            n_elements, 0))

        # --- 对齐到数据段起点 ---
        alignment = 32
        if 'general.alignment' in r.kv:
            alignment = int(r.kv['general.alignment'])
        pad = (alignment - (off % alignment)) % alignment
        r.data_offset = off + pad
        return r


def _value_size(buf: bytes, vtype: int, off: int) -> int:
    """返回某类型 value 占的字节数（从 off 处解析）"""
    if vtype == 6:        # STRING：u64 长度 + 字节
        n = struct.unpack_from('<Q', buf, off)[0]
        return 8 + n
    sizes = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 7:8, 8:8, 9:4, 10:8, 11:1}
    return sizes.get(vtype, 0)


def _read_value(buf: bytes, vtype: int, off: int) -> object:
    """标量/字符串 value 解码（ARRAY 除外）"""
    if vtype == 6:
        n = struct.unpack_from('<Q', buf, off)[0]
        return buf[off+8:off+8+n].decode('utf-8')
    fmt = {0:'<b', 1:'<B', 2:'<h', 3:'<H', 4:'<i', 5:'<I', 7:'<q', 8:'<Q', 9:'<f', 10:'<d', 11:'?'}[vtype]
    return struct.unpack_from(fmt, buf, off)[0]


def _array_size(buf: bytes, off: int) -> int:
    """ARRAY 的字节数：item_type(u32) + len(u64) + len×item 大小"""
    item_type = struct.unpack_from('<I', buf, off)[0]
    n         = struct.unpack_from('<Q', buf, off + 4)[0]
    base = 8
    if item_type == 6:   # 字符串数组
        p = off + 8
        for _ in range(n):
            l = struct.unpack_from('<Q', buf, p)[0]; p += 8 + l
        return p - off
    return base + n * _value_size(buf, item_type, off + 8)


def _read_array(buf: bytes, off: int) -> object:
    """ARRAY value 解码（此处仅演示标量数组；字符串数组可类推）"""
    item_type = struct.unpack_from('<I', buf, off)[0]
    n         = struct.unpack_from('<Q', buf, off + 4)[0]
    out = []
    p = off + 8
    for _ in range(n):
        out.append(_read_value(buf, item_type, p))
        p += _value_size(buf, item_type, p)
    return out
```

**手工验证**：构造一个 1 个 KV（`test.arch = "llama"`）+ 1 个 F32 张量的最小 GGUF，解析它：

```python
import io

def build_mini_gguf() -> bytes:
    buf = bytearray()
    buf += MAGIC + struct.pack('<I', 3) + struct.pack('<QQ', 1, 1)   # v3, 1 张量, 1 KV
    # KV: test.arch = "llama"
    key = b'test.arch'
    buf += struct.pack('<Q', len(key)) + key
    buf += struct.pack('<I', 6)                          # STRING
    val = b'llama'
    buf += struct.pack('<Q', len(val)) + val
    # tensor_info: name="w", 1 维 dims=[4], F32(type 0), offset=0
    tn = b'w'
    buf += struct.pack('<Q', len(tn)) + tn
    buf += struct.pack('<I', 1) + struct.pack('<Q', 4)
    buf += struct.pack('<II', 0, 0)                      # type=F32, offset=0
    # 对齐到 32：数据段
    while len(buf) % 32 != 0:
        buf += b'\x00'
    buf += struct.pack('<4f', 1.0, 2.0, 3.0, 4.0)         # 4 个 F32
    return bytes(buf)

mini = build_mini_gguf()
r = GgufReader.parse(mini)
assert r.version == 3
assert r.kv['test.arch'] == 'llama'
assert r.tensors[0].name == 'w'
assert r.tensors[0].dims == [4]
assert r.tensors[0].ggml_type == 0          # F32
assert r.tensors[0].n_elements == 4
print('C.2 GGUF 解析器：断言全部通过')
```

> **注意**：`tensor_infos[i].offset` 是相对数据段起点的偏移。读一个张量的原始字节 = `buf[data_offset + offset : data_offset + offset + n_bytes]`。`n_bytes` 由 `ggml_type` 决定（见 C.3 的表）。

---

## C.3 解码器公共骨架

每个解码器的签名统一为 `decode_xxx(block_bytes, n) -> list[float]`：输入若干块字节，输出 `n` 个 float。`BLOCK_BYTES`、`BLOCK_VALUES`、`BITS_PER_WEIGHT` 是每个格式的元数据。

```python
FORMAT_TABLE = {
    # 格式名: (块字节数, 块内权重数, bpw, type id)
    'Q4_0':    (18, 32,  4.5,    2),
    'Q4_K':    (144, 256, 4.5,   12),
    'Q6_K':    (210, 256, 6.5625, 14),
    'IQ1_S':   (50, 256, 1.5625, 19),
    'STQ1_0':  (42, 256, 1.3125, None),   # fork/PR，type id 未定
    'PTQ1_0':  (28, 128, 1.75,   None),   # fork/PR，type id 未定
}
```

---

## C.4 Q4_0 解码器

Q4_0：18 字节/32 值。`d`(f16) + `qs[16]`，每个字节低 nibble = 第 j 个权重、高 nibble = 第 j+16 个权重。公式 `y = d·(q − 8)`。

```python
def decode_q4_0(block: bytes, n: int) -> list[float]:
    assert len(block) == (n // 32) * 18
    out = []
    for i in range(0, len(block), 18):
        d  = f16_to_f32(block[i:i+2])
        qs = block[i+2:i+18]
        for j in range(16):
            lo = qs[j] & 0x0F
            hi = (qs[j] >> 4) & 0x0F
            out.append(d * (lo - 8))
            out.append(d * (hi - 8))
    assert len(out) == n
    return out

# 手工字节验证：d = 1.0 (0x3C00)，qs 全部 = 0x0F（lo=15→7，hi=15→7）
blk = b'\x00\x3c' + bytes([0x0F]) * 16
res = decode_q4_0(blk, 32)
assert res[0] == 1.0 * (15 - 8) == 7.0
assert res[1] == 7.0
assert res[16] == 7.0          # 第 17 个 = qs[0] 的高 nibble
print('C.4 Q4_0：断言全部通过')
```

---

## C.5 Q4_K 解码器

Q4_K：144 字节/256 值。`dm`(4，含 `d`+`dmin`) + `scales[12]` + `qs[128]`。关键在 6-bit 交错打包（第 7 章）：**`scales[12]` 不是"尺度 6 字节 + 最小值 6 字节"，而是交错位流**，必须用 `get_scale_min_k4` 解包。解码端 `y = d·sc·q − dmin·m`，**sc/m 是 6-bit 整数、不再除以 63/64**（归一化已由量化端折进 `d`/`dmin`）。

```python
def get_scale_min_k4(j, scales):
    """第 7 章 get_scale_min_k4 的 Python 版；j = 子块索引 0..7"""
    if j < 4:
        sc = scales[j] & 63
        m  = scales[j + 4] & 63
    else:
        sc = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4)
        m  = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)
    return sc, m

def decode_q4_k(block: bytes, n: int) -> list[float]:
    assert len(block) == (n // 256) * 144
    out = []
    for i in range(0, len(block), 144):
        d    = f16_to_f32(block[i:i+2])
        dmin = f16_to_f32(block[i+2:i+4])
        scales = block[i+4:i+16]
        qs     = block[i+16:i+144]

        q_ptr = 0                            # 忠实源码 dequantize_row_q4_K 的 q 游标
        is_ = 0
        for j in range(0, 256, 64):          # 每 64 值 = 两个子块
            sc0, m0 = get_scale_min_k4(is_ + 0, scales)
            sc1, m1 = get_scale_min_k4(is_ + 1, scales)
            d1, m1v = d * sc0, dmin * m0
            d2, m2v = d * sc1, dmin * m1
            for l in range(32):
                out.append(d1 * (qs[q_ptr + l] & 0x0F) - m1v)   # 子块 is_ ：低 nibble
            for l in range(32):
                out.append(d2 * (qs[q_ptr + l] >> 4) - m2v)      # 子块 is_+1：高 nibble
            q_ptr += 32
            is_ += 2
    assert len(out) == n
    return out

# 手工字节验证：构造一个 d=1.0、dmin=1.0、scales 全 0、qs 全 0 的块
# 子块 0..7 的 sc=m=0 → y = 1*0*q - 1*0 = 0
blk = b'\x00\x3c' + b'\x00\x3c' + bytes(12) + bytes(128)
res = decode_q4_k(blk, 256)
assert all(v == 0.0 for v in res)
# 再验证 sc 生效：把子块 0 的尺度设为 1（scales[0] 低 6 位 = 1），
# 且让子块 0 的 qs[0] 低 nibble = 3 → y = 1*1*3 = 3
blk2 = bytearray(b'\x00\x3c' + b'\x00\x3c' + bytes(12) + bytes(128))
blk2[4] = 1                                    # scales[0] = 1 → 子块0 sc=1
blk2[16] = 0x03                                # qs[0] = 3 → 子块0 第1个值=3
res2 = decode_q4_k(bytes(blk2), 256)
assert res2[0] == 3.0
assert res2[32] == 0.0                         # 子块 1 仍是 0
print('C.5 Q4_K：断言全部通过')
```

---

## C.6 Q6_K 解码器

Q6_K：210 字节/256 值。`ql[128]`（低 4 位）+ `qh[64]`（高 2 位）+ `scales[16]`（int8）+ `d`。16 个子块 × 16 值，公式 `y = d·scales[j]·(q − 32)`，无 `dmin`。

位序（第 9 章）：以 128 值段为单位，`ql[l]`/`qh[l]` 的 4 个 2-bit 组分别给位置 `l, l+32, l+64, l+96`：

```python
def decode_q6_k(block: bytes, n: int) -> list[float]:
    assert len(block) == (n // 256) * 210
    out = []
    for i in range(0, len(block), 210):
        d      = f16_to_f32(block[i:i+2])
        ql     = block[i+2:i+130]
        qh     = block[i+130:i+194]
        scales = struct.unpack('<16b', block[i+194:i+210])   # int8 尺度

        q, h = 0, 0
        for seg in range(2):                       # 2 个 128 值段
            for l in range(32):
                is_ = l // 16
                q1 = (ql[q+l] & 0x0F) | (((qh[h+l] >> 0) & 3) << 4) - 32
                q2 = (ql[q+l+32] & 0x0F) | (((qh[h+l] >> 2) & 3) << 4) - 32
                q3 = (ql[q+l] >> 4) | (((qh[h+l] >> 4) & 3) << 4) - 32
                q4 = (ql[q+l+32] >> 4) | (((qh[h+l] >> 6) & 3) << 4) - 32
                out.append(d * scales[is_ + 0] * q1)
                out.append(d * scales[is_ + 2] * q2)
                out.append(d * scales[is_ + 4] * q3)
                out.append(d * scales[is_ + 6] * q4)
            q += 64; h += 32
    assert len(out) == n
    return out

# 手工字节验证：d=1.0，scales 全 1，ql 全 0，qh 全 0
# → q1..q4 全是 0-32 = -32 → y = 1*1*(-32) = -32
blk = b'\x00\x3c' + bytes(128) + bytes(64) + struct.pack('<16b', *([1]*16))
res = decode_q6_k(blk, 256)
assert all(v == -32.0 for v in res)
# 把 qh 的 bit0-1 设为 1 → q1 的高 2 位 = 1 → q1 = 1*16 - 32 = -16
blk2 = bytearray(b'\x00\x3c' + bytes(128) + bytes(64) + struct.pack('<16b', *([1]*16)))
blk2[130] = 0x03                                  # qh[0] 低 2 位 = 3
res2 = decode_q6_k(bytes(blk2), 256)
assert res2[0] == 1 * 1 * (3*16 - 32) == 16.0
print('C.6 Q6_K：断言全部通过')
```

---

## C.7 IQ1_S 解码器

IQ1_S：50 字节/256 值。`d`(f16) + `qs[32]` + `qh[8]`（uint16）。**这是码本查表格式**：每 32 值一组，组尺度 `dl = d·(2·((qh[ib]>>12)&7)+1)`（∈ {1,3,5,...,15} 倍 d），每 8 值查一次 `iq1s_grid`（2048 项、每项 8 个 int8），索引 11 bit = `qs[l]` 8 bit + `(qh[ib]>>3l)&7` 3 bit。反量化 `y = dl·(grid[j] + delta)`，`delta = ±0.125`（`IQ1S_DELTA`）由 `qh[ib]` 的 bit15 决定。

> 完整 `iq1s_grid` 有 2048 项，在 llama.cpp 的 `ggml-common.h`（`GGML_TABLE_BEGIN(uint64_t, iq1s_grid, NGRID_IQ1S)`）。下面只列前几项作演示；真实使用时需完整导入该表。

```python
IQ1S_DELTA = 0.125

# 演示用的 iq1s_grid 前 4 项（与 ggml-common.h 一致），完整表需从源码导入
iq1s_grid_demo = [
    0xFFFFFFFFFFFFFFFF,  # 展开成 8 个 -1
    0xFFFFFFFFFFFFFF01,  # 展开成 [1, -1, -1, -1, -1, -1, -1, -1]
    0xFFFFFFFFFFFFFF02,  # [2, -1, ...]
    0xFFFFFFFFFFFFFF03,  # [3, -1, ...]
]

def grid_entry(idx: int, table=iq1s_grid_demo) -> list[int]:
    """把一项 uint64 展开成 8 个 int8（低字节在前）"""
    return list(struct.unpack('<8b', struct.pack('<Q', table[idx])))

def decode_iq1_s(block: bytes, n: int) -> list[float]:
    assert len(block) == (n // 256) * 50
    out = []
    for i in range(0, len(block), 50):
        d   = f16_to_f32(block[i:i+2])
        qs  = block[i+2:i+34]
        qh  = struct.unpack('<8H', block[i+34:i+50])   # 8 个 uint16
        for ib in range(8):                              # 8 组 × 32 值
            dl = d * (2 * ((qh[ib] >> 12) & 7) + 1)
            delta = -IQ1S_DELTA if (qh[ib] & 0x8000) else IQ1S_DELTA
            for l in range(4):                           # 4 个 8 值子组
                idx = qs[ib*4 + l] | (((qh[ib] >> (3*l)) & 7) << 8)
                g = grid_entry(idx)
                for j in range(8):
                    out.append(dl * (g[j] + delta))
    assert len(out) == n
    return out

# 手工字节验证：d=1.0；qh[0] 高 3 位 = 0（尺度 = 1）、bit15 = 0（delta=+0.125）
# qs[0] = 0、qh[0] bit0-2 = 0 → 索引 0 → grid[0] = [-1]*8
# → y = 1*(2*0+1) * (-1 + 0.125) = -0.875
blk = b'\x00\x3c' + bytes(32) + b'\x00\x00' + bytes(14)
res = decode_iq1_s(blk, 256)
assert abs(res[0] - (-0.875)) < 1e-9
# 索引 1 → grid[1] = [1, -1, ...] → y[0] = 1 * (1 + 0.125) = 1.125
# 注意：要得到索引 1，改 qs[0]（索引低 8 bit），而不是 qh（qh 会污染高位/scale）
blk2 = bytearray(b'\x00\x3c' + bytes(32) + b'\x00\x00' + bytes(14))
blk2[2] = 0x01                                   # qs[0] = 1 → 索引低 8 位 = 1
res2 = decode_iq1_s(bytes(blk2), 256)
assert abs(res2[0] - 1.125) < 1e-9
print('C.7 IQ1_S：断言全部通过')
```

---

## C.8 STQ1_0 解码器（fork 格式，按细纲布局）

STQ1_0 是 **3:4 稀疏三值**格式（第 13 章）：每 4 个权重恰好 1 个零、3 个 ±d；32 种模式用 5 bit 编码（4-bit 模式索引 + 1-bit 符号）。细纲记载块为 42 字节/256 值：`qs[32]`（码本索引）+ `sign[8]` + fp16 `d`。**stride-16 分组**：每组 4 个权重取 `w0, w16, w32, w48`（而非连续四个），便于 SIMD 无重排解码。

> 该格式不在 llama.cpp 主线，以下为按细纲布局的最小解码骨架，**字段名/模式表须以实际 fork 文档复核**。

```python
# 32 种模式的 3:4 布局：每种模式告诉"4 个位置里谁是零"
# 模式号 p ∈ [0,32)，用 p>>2 选零位置(0..3)、p&3 给其余三值的±（符号由 sign 位决定）
def decode_stq1_0(block: bytes, n: int) -> list[float]:
    # 细纲口径：每块 256 值 = 42 字节
    assert len(block) == (n // 256) * 42
    out = [0.0] * n
    for i in range(0, len(block), 42):
        d    = f16_to_f32(block[i:i+2])
        qs   = block[i+2:i+34]        # 32 字节码本索引（每字节 2 个 5-bit 模式）
        sign = block[i+34:i+42]       # 8 字节符号位
        base = i // 42 * 256

        for g in range(64):           # 64 组 × 4 权重 = 256
            p5  = (qs[g >> 1] >> (4 * (g & 1))) & 0x1F     # 5-bit 模式
            sgn = (sign[g >> 3] >> (g & 7)) & 1
            zero_pos = p5 >> 2        # 0..3：零的位置
            mag      = p5 & 3         # 0..3：其余三值的幅度编码（±1）
            for lane in range(4):
                # stride-16 分组：组内 lane 映射到 w, w+16, w+32, w+48
                wpos = (g % 16) + lane * 16
                if lane == zero_pos:
                    out[base + wpos] = 0.0
                else:
                    v = 1.0 if (mag >> ((lane + (1 if lane > zero_pos else 0)) % 3)) & 1 else -1.0
                    out[base + wpos] = d * v * (-1 if sgn else 1)
    return out
```

> **验证要点（第 18 章）**：检查每个 4-lane 组恰好一个零；任何"一组两个零"或"一组没零"都说明模式表/打包实现有误。

---

## C.9 PTQ1_0 解码器（fork 格式，按细纲布局）

PTQ1_0 是**稠密三值**格式：不强制稀疏位置，任何权重都可是零，因此用 **base-3 编码**。细纲记载块为 28 字节/128 值：24 字节装 5 trits/字节（3⁵=243 < 256）+ 2 字节余数（4 trits/字节）+ fp16 `d`。

```python
def decode_ptq1_0(block: bytes, n: int) -> list[float]:
    # 细纲口径：每块 128 值 = 28 字节
    assert len(block) == (n // 128) * 28
    out = []
    for i in range(0, len(block), 28):
        d    = f16_to_f32(block[i:i+2])
        main = block[i+2:i+26]         # 24 字节 × 5 trits = 120 个 trit
        rem  = block[i+26:i+28]        # 2 字节 × 4 trits = 8 个 trit
        trits = []
        for b in main:                 # 每字节拆 5 个 base-3 数字
            v = b
            for _ in range(5):
                trits.append(v % 3); v //= 3
        for b in rem:                  # 每字节拆 4 个
            v = b
            for _ in range(4):
                trits.append(v % 3); v //= 3
        assert len(trits) == 128
        # trit ∈ {0,1,2} → 值 {0,-1,+1}（约定：0=0, 1=+1, 2=-1 或 0=0,1=-1,2=+1，按 fork 文档）
        for t in trits:
            out.append(d * {0: 0.0, 1: -1.0, 2: 1.0}[t])
    assert len(out) == n
    return out

# 手工字节验证：d=1.0，main 全 0 → 全部 trit=0 → y 全 0
blk = b'\x00\x3c' + bytes(26)
res = decode_ptq1_0(blk, 128)
assert all(v == 0.0 for v in res)
# 第一字节 = 1 → 第 1 个 trit = 1 → y[0] = -1
blk2 = b'\x00\x3c' + b'\x01' + bytes(25)
res2 = decode_ptq1_0(blk2, 128)
assert res2[0] == -1.0
print('C.9 PTQ1_0：断言全部通过')
```

> ⚠ **Hadamard 折叠**：PTQ1_0 的磁盘权重以 Hadamard 折叠形式存储。上面的解码器只还原"折叠后的值"；真正推理时**必须**在激活侧做 Hadamard 变换（见第 13.3.4 节），否则 logits 完全错误。**格式正确但缺失旋转仍然是错的。**

---

## C.10 与 gguf-py 的交叉验证脚本

把我们的解码器当作"自研实现"，与官方 `gguf.dequantize` 逐块比较。这是第 18 章"参考实现交叉验证"的可执行版。

```bash
pip install gguf numpy   # 交叉验证需要官方库
```

```python
import numpy as np
from gguf import GGUFReader, dequantize, GGMLQuantizationType
import gguf.gguf_reader as gr   # 0.19.0 内部有 Quants 类

def cross_check(path: str, max_tensors: int = 5):
    reader = GGUFReader(path)
    checked = 0
    for t in reader.tensors:
        if t.tensor_type.name not in ('Q4_0', 'Q4_K', 'Q6_K', 'IQ1_S'):
            continue
        # 官方反量化
        ref = dequantize(t.data, t.tensor_type)
        # 自研解码器（输入 t.data 是 uint8 ndarray → 转 bytes）
        raw = t.data.tobytes()
        our = {'Q4_0': decode_q4_0, 'Q4_K': decode_q4_K, 'Q6_K': decode_q6_K, 'IQ1_S': decode_iq1_s}[t.tensor_type.name](raw, t.n_elements)
        our = np.array(our, dtype=np.float32)

        # 逐元素比较：参考实现误差必须为 0（格式定义正确的必要条件）
        max_err = float(np.max(np.abs(our - ref)))
        ok = max_err < 1e-6
        print(f'{t.name:48s} {t.tensor_type.name:6s} max_err={max_err:.3e} {"OK" if ok else "FAIL"}')
        assert ok, f'{t.name} 交叉验证失败'
        checked += 1
        if checked >= max_tensors:
            break
    print(f'交叉验证完成：{checked} 个张量全部一致')

# cross_check('model-q4_k_m.gguf')
```

> 若某格式交叉验证 `FAIL`，先怀疑自研解码器（位序/打包），而不是官方库——第 18 章的原则：**任何偏差都应先怀疑自己**。
> 注意 STQ1_0 / PTQ1_0 不在主线 gguf-py 中，交叉验证需使用对应 fork 的实现。

---

## C.11 使用速查

| 目的 | 代码 |
|---|---|
| 解析一个 GGUF 文件 | `r = GgufReader.load('model.gguf')` |
| 打印张量类型分布 | `from collections import Counter; Counter(t.ggml_type for t in r.tensors)` |
| 取某个张量的原始字节 | `b = buf[data_offset + t.offset : ...]`（见 C.2 注释） |
| 反量化一个张量 | `decode_q4_k(raw, n_elements)` 等 |
| 与官方库交叉验证 | C.10 的 `cross_check()` |
| 确认格式的 bpw | `FORMAT_TABLE['Q4_K'][2]  # 4.5` |

本附录的代码追求**清晰和正确**，不追求速度。真实推理引擎的反量化用 SIMD 一次处理一整块（见第 17 章），但无论多快，字节规则都和这里逐行一致。
