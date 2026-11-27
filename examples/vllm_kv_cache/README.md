# vLLM KV Cache: From Slot Mapping to Attention Reads

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
