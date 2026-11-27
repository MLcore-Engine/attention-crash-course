# vLLM KV Cache 源码逐行学习

先按下面四课读一遍 A 的生命周期，每课都给出固定版本的原始代码、准确行号、逐行解释和代入的数值。主线为 vLLM v0.11.0 的 V1、CUDA、BF16、full attention，包含 FlashAttention 的实际页表读取实现。

| 顺序 | 要弄懂的问题 | 阅读 |
|---|---|---|
| 1 | GPU 上的缓存由哪一行创建，每个块到底占多少字节 | [显存里到底放了什么](lessons/01-显存里到底放了什么.md) |
| 2 | 17 个 token 怎样领到两块，生成到第几个 token 才会再领一块 | [请求怎样领到块](lessons/02-请求怎样领到块.md) |
| 3 | position 怎样变成 slot，再变成 K/V 的字节地址，读取怎样查块表 | [从位置算到显存地址](lessons/03-从位置算到显存地址.md) |
| 4 | 相同前缀怎样共享，请求结束后引用、数据和 hash 分别怎样变化 | [前缀复用、释放与抢占](lessons/04-前缀复用释放与抢占.md) |
| 5 | 定位其他后端、混合缓存、MLA、量化、迁移与上游测试 | [源码全景与验证](lessons/05-源码全景与验证.md) |

这一遍固定每块 16 个 token、8 个块号、两层、两个 KV 头、head_dim=32。A 的 17 个输入 token 先领到 1、2 号块，B 复用 A 的第一页、另领尾页，随后追到 A 增长、A/B 结束、C 命中和 D 淘汰。每一步都能在 CPU 运行时看到块表与引用计数。

```bash
python3 examples/vllm_kv_cache/verify_sources.py
python3 examples/vllm_kv_cache/lifecycle_demo.py
```

第二个命令直接从本地原始文件提取并执行上游的 CPU 块池、双向队列及 hash 函数。它用 Python 数组演示 GPU 地址与 K/V 数值，源投影和 CUDA kernel 由教学数据模型表示。本机验证覆盖 CPU 逻辑与地址断言。

两份完整源码快照在 [upstream/source](upstream/source) 和 [upstream/flash-attention](upstream/flash-attention)，共 4,024 个文件。版本、Git blob、SHA-256 和依赖提交记录在 [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)。本课程沿用现有教程的 v0.11.0，源码行号和行为以该提交为准。

## 既有的 custom-layout 地址例子

This small study follows the **vLLM v0.11.0** CUDA paged-attention path. The
`upstream/` directory contains the corresponding upstream files, kept with the
Apache-2.0 license. The Python demo is an address-mapping model; it does not
compile or execute CUDA.

## Start Here

Run the CPU-only demonstration from the repository root:

```bash
python examples/vllm_kv_cache/kv_cache_demo.py
```

The example uses a block size of 4, a block table `[5, 1]`, and five tokens.
Thus logical positions 0-3 occupy physical block 5, and position 4 starts in
physical block 1. Its slot mapping is `[20, 21, 22, 23, 4]`.

## Follow the Source

1. [`upstream/block_table.py`](upstream/block_table.py), `compute_slot_mapping`
   (line 76) computes `slot = physical_block * block_size + offset`.
2. [`upstream/cache_kernels.cu`](upstream/cache_kernels.cu),
   `reshape_and_cache_kernel` (line 217) writes new K/V to the custom layout.
   `reshape_and_cache_flash_kernel` (line 282) writes to the NHD or HND layout.
3. [`upstream/attention_kernels.cuh`](upstream/attention_kernels.cuh),
   `paged_attention_kernel` (line 85) maps logical blocks through the block
   table and reads the K/V values for attention.

## What the Writer Does

The incoming tensors have logical shape:

```text
key, value: [num_tokens, num_kv_heads, head_size]
```

For each token, `slot_mapping` gives the destination slot. CUDA decodes it:

```text
block_idx    = slot_idx / block_size
block_offset = slot_idx % block_size
```

In the custom paged-attention path, the destination layouts are intentionally
different:

```text
K: [num_blocks, num_kv_heads, head_size/x, block_size, x]
V: [num_blocks, num_kv_heads, head_size,   block_size]
```

For a source element `key[token, head, dim]`, let
`x_idx = dim // x` and `x_offset = dim % x`. Its cache coordinates are:

```text
K: [block_idx, head, x_idx, block_offset, x_offset]
V: [block_idx, head, dim,   block_offset]
```

In the CUDA kernel these coordinates become flat addresses. The K index is
equivalent to:

```text
((((block_idx * num_kv_heads + head) * (head_size/x) + x_idx)
   * block_size + block_offset) * x + x_offset)
```

The value index is:

```text
(((block_idx * num_kv_heads + head) * head_size + dim)
  * block_size + block_offset)
```

For BF16, the kernel uses `x = 16 / sizeof(cache_t) = 8`. The K layout keeps
`x` adjacent elements together for vector loads, then places the block's token
offset in the next dimension. The V layout makes token offsets contiguous for
each value dimension, matching the weighted sum over tokens. So “reshape” here
means **scatter each element into a kernel-specific address layout**, not
calling a tensor `reshape()` and not repeatedly transposing the whole cache.

The demo models this exact custom layout with `HEAD_SIZE = X = 8`. It writes
through `slot_mapping`, then reads back by logical position and checks every
K/V element against the original input.

## How the Reader Finds K/V

`paged_attention_kernel` iterates over the request's logical blocks. For each
logical block index, it gets `physical_block_number = block_table[block_idx]`.
The token's offset inside that block plus the KV head selects the cached data.
The K pointer arithmetic in the source follows the layout above; V reads use
`row_idx * BLOCK_SIZE + physical_block_offset` for each output dimension.

The mathematical operation remains:

```text
scores = Q @ K.T / sqrt(head_size)
output = softmax(scores) @ V
```

Only the address translation changes. There is no need to concatenate paged
blocks into one contiguous sequence first. The demo also shows GQA mapping:
with 4 query heads and 2 KV heads, query head 3 reads KV head
`3 // (4 // 2) = 1`.

## Important: There Is More Than One Layout

Do not mix the writer and reader from different backends:

- The custom `reshape_and_cache_kernel` layout above matches
  `attention_kernels.cuh`'s `paged_attention_kernel`.
- `reshape_and_cache_flash_kernel` instead supports **NHD**
  `[blocks, block_size, heads, head_size]` and **HND**
  `[blocks, heads, block_size, head_size]`. The selected FlashAttention kernel
  consumes that backend's layout. It does not use the custom K/V layout above.

This is why “the vLLM KV cache shape” has no single universal answer: inspect
the selected attention backend and its matching cache writer/reader.

## Source Provenance

The files in `upstream/` are from the `vllm-project/vllm` **v0.11.0** tag:

- `csrc/cache_kernels.cu`
- `csrc/attention/attention_kernels.cuh`
- `vllm/v1/worker/block_table.py`

Their original copyright and license notices are retained. See
[`upstream/LICENSE`](upstream/LICENSE) for the Apache-2.0 license.
