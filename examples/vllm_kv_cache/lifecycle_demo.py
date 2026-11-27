"""Trace real v0.11.0 BlockPool methods and a CPU model of paged GPU addresses.

Run with Python 3.10+. GPU values use Python floats. BF16 byte counts describe
the modeled GPU layout. The program executes upstream CPU block management,
while GPU allocation, model projections and CUDA kernels are modeled.
"""

from __future__ import annotations

from math import exp, isclose, sqrt
from types import SimpleNamespace

from upstream_cpu import load_upstream_cpu


BLOCK_SIZE = 16
NUM_BLOCKS = 8
NUM_LAYERS = 2
NUM_KV_HEADS = 2
HEAD_SIZE = 32
SOURCE_LINE = {
    "allocate": "single_type_kv_cache_manager.py:123-131 → block_pool.py:257-284",
    "write": "block_table.py:107-113 → cache_kernels.cu:292-327",
    "touch": "block_pool.py:322-336",
    "free": "single_type_kv_cache_manager.py:164-171 → block_pool.py:338-353",
}


class PagedKV:
    """Flat CPU buffers using the NHD and HND writer's element offsets."""

    def __init__(self) -> None:
        elements = NUM_BLOCKS * BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE
        self.nhd = [[None] * elements for _ in range(NUM_LAYERS * 2)]
        self.hnd = [[None] * elements for _ in range(NUM_LAYERS * 2)]

    @staticmethod
    def offset(block: int, offset: int, head: int, dim: int, layout: str) -> int:
        if layout == "NHD":
            return ((block * BLOCK_SIZE + offset) * NUM_KV_HEADS + head) * HEAD_SIZE + dim
        return ((block * NUM_KV_HEADS + head) * BLOCK_SIZE + offset) * HEAD_SIZE + dim

    @staticmethod
    def source_value(token: int, layer: int, kind: int, head: int, dim: int) -> float:
        return token + layer * 100 + kind * 10 + head + dim / 16

    def write(self, block_table: list[int], position: int, token: int) -> None:
        logical_block, offset = divmod(position, BLOCK_SIZE)
        block = block_table[logical_block]
        for layer in range(NUM_LAYERS):
            for kind in range(2):
                buffer = layer * 2 + kind
                for head in range(NUM_KV_HEADS):
                    for dim in range(HEAD_SIZE):
                        value = self.source_value(token, layer, kind, head, dim)
                        self.nhd[buffer][self.offset(block, offset, head, dim, "NHD")] = value
                        self.hnd[buffer][self.offset(block, offset, head, dim, "HND")] = value

    def read(self, table: list[int], position: int, layer: int, kind: int, head: int) -> list[float]:
        logical_block, offset = divmod(position, BLOCK_SIZE)
        block = table[logical_block]
        result = []
        for dim in range(HEAD_SIZE):
            nhd = self.nhd[layer * 2 + kind][self.offset(block, offset, head, dim, "NHD")]
            hnd = self.hnd[layer * 2 + kind][self.offset(block, offset, head, dim, "HND")]
            assert nhd is not None and nhd == hnd
            result.append(nhd)
        return result


def main() -> None:
    upstream = load_upstream_cpu()
    pool = upstream.BlockPool(NUM_BLOCKS, enable_caching=True)
    storage = PagedKV()
    requests = {}

    def request(name: str, tokens: list[int]) -> SimpleNamespace:
        parent_hash = None
        hashes = []
        for start in range(0, len(tokens) - BLOCK_SIZE + 1, BLOCK_SIZE):
            parent_hash = upstream.hash_block_tokens(
                upstream.sha256, parent_hash, tokens[start:start + BLOCK_SIZE], None
            )
            hashes.append(parent_hash)
        return SimpleNamespace(
            request_id=name, all_token_ids=tokens, block_hashes=hashes, lora_request=None
        )

    def block_ids(name: str) -> list[int]:
        return [block.block_id for block in requests[name]]

    def check_state(label: str, expected_free: list[int]) -> None:
        free = pool.free_block_queue.get_all_free_blocks()
        assert [block.block_id for block in free] == expected_free
        assert len(free) == pool.get_num_free_blocks()
        assert pool.null_block.block_id == 0 and pool.null_block.is_null
        assert all(block.ref_cnt == 0 and not block.is_null for block in free)
        assert all(block.ref_cnt >= 0 for block in pool.blocks)
        for block in pool.blocks[1:]:
            expected_refs = sum(block in blocks for blocks in requests.values())
            assert block.ref_cnt == expected_refs
        print(f"\n{label}")
        print("  请求块表", {name: block_ids(name) for name in requests})
        print("  空闲队列", expected_free)
        print("  活跃引用", {block.block_id: block.ref_cnt for block in pool.blocks if block.ref_cnt})
        print("  前缀索引块", [block.block_id for block in pool.blocks if block.block_hash])

    check_state("0. 原始 BlockPool 初始化，0 号作为 null block", list(range(1, 8)))
    bytes_per_layer_page = 2 * BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE * 2
    assert bytes_per_layer_page == 4096
    print(f"  BF16 每层每块 {bytes_per_layer_page} B，跨 {NUM_LAYERS} 层共 8192 B")

    tokens_a = list(range(1, 18))
    requests["A"] = pool.get_new_blocks(2)
    assert block_ids("A") == [1, 2]
    for position, token in enumerate(tokens_a):
        storage.write(block_ids("A"), position, token)
    req_a = request("A", tokens_a)
    pool.cache_full_blocks(req_a, requests["A"], 0, 1, BLOCK_SIZE, 0)
    check_state("1. A prefill 17 个 token，ceil(17/16)=2 块", [3, 4, 5, 6, 7])
    print("  对照源码", SOURCE_LINE["allocate"])
    print("  position 16 → logical block 1 → physical block 2 → slot 32")

    tokens_a.append(900)
    storage.write(block_ids("A"), 17, 900)
    check_state("2. 下一轮处理 A 已采样的 token 900，position 17 → slot 33", [3, 4, 5, 6, 7])
    print("  对照源码", SOURCE_LINE["write"])

    tokens_b = list(range(1, 17)) + [777]
    req_b = request("B", tokens_b)
    hit = pool.get_cached_block(req_b.block_hashes[0], [0])
    assert hit is not None and hit[0] is requests["A"][0]
    pool.touch((hit,))
    requests["B"] = hit + pool.get_new_blocks(1)
    storage.write(block_ids("B"), 16, tokens_b[16])
    assert pool.blocks[1].ref_cnt == 2
    check_state("3. B 命中前 16 个 token，共享 1 号块，尾部另领 3 号块", [4, 5, 6, 7])
    assert storage.read(block_ids("A"), 16, 0, 0, 0)[0] == 17
    assert storage.read(block_ids("B"), 16, 0, 0, 0)[0] == 777
    print("  A position 16 的 K[0,0]=17，B 同一逻辑位置为 777，各写自己的尾块")

    for position in range(18, 33):
        needed = (position + 1 + BLOCK_SIZE - 1) // BLOCK_SIZE
        if needed > len(requests["A"]):
            requests["A"].extend(pool.get_new_blocks(needed - len(requests["A"])))
        token = 900 + position - 17
        tokens_a.append(token)
        storage.write(block_ids("A"), position, token)
    req_a = request("A", tokens_a)
    pool.cache_full_blocks(req_a, requests["A"], 1, 2, BLOCK_SIZE, 0)
    assert block_ids("A") == [1, 2, 4]
    check_state("4. A 长到 33 个已计算 token，position 32 才新增 4 号块", [5, 6, 7])

    query = [1 / (100 * HEAD_SIZE)] * HEAD_SIZE
    q_head, kv_head = 3, 3 // (4 // NUM_KV_HEADS)
    keys = [storage.read(block_ids("B"), pos, 0, 0, kv_head) for pos in range(len(tokens_b))]
    values = [storage.read(block_ids("B"), pos, 0, 1, kv_head) for pos in range(len(tokens_b))]
    scores = [sum(q * k for q, k in zip(query, key)) / sqrt(HEAD_SIZE) for key in keys]
    exponentials = [exp(score - max(scores)) for score in scores]
    weights = [value / sum(exponentials) for value in exponentials]
    output = [
        sum(weight * value[dim] for weight, value in zip(weights, values))
        for dim in range(HEAD_SIZE)
    ]
    contiguous_values = [
        [storage.source_value(token, 0, 1, kv_head, dim) for dim in range(HEAD_SIZE)]
        for token in tokens_b
    ]
    expected = [
        sum(weight * value[dim] for weight, value in zip(weights, contiguous_values))
        for dim in range(HEAD_SIZE)
    ]
    assert all(isclose(actual, wanted) for actual, wanted in zip(output, expected))
    for position, token in enumerate(tokens_b):
        for layer in range(NUM_LAYERS):
            for kind in range(2):
                for head in range(NUM_KV_HEADS):
                    actual = storage.read(block_ids("B"), position, layer, kind, head)
                    assert actual == [
                        storage.source_value(token, layer, kind, head, dim)
                        for dim in range(HEAD_SIZE)
                    ]
    print(f"  Q head {q_head} → KV head {kv_head}，分页 attention 输出", [round(x, 4) for x in output])

    pool.free_blocks(reversed(requests.pop("A")))
    check_state("5. A 结束，倒序释放 4、2、1；1 号块仍被 B 使用", [5, 6, 7, 4, 2])
    pool.free_blocks(reversed(requests.pop("B")))
    check_state("6. B 结束，1 号块 ref_cnt=0，仍可命中", [5, 6, 7, 4, 2, 3, 1])
    old_value = storage.nhd[0][storage.offset(2, 0, 0, 0, "NHD")]
    assert old_value == 17

    assert pool.get_num_free_blocks() == 7
    hit = pool.get_cached_block(req_b.block_hashes[0], [0])
    assert hit is not None
    assert pool.get_num_free_blocks() == 7  # Lookup alone does not pin a block.
    pool.touch((hit,))
    assert pool.get_num_free_blocks() == 6
    requests["C"] = hit + pool.get_new_blocks(1)
    storage.write(block_ids("C"), 16, tokens_b[16])
    check_state("7. C 再命中 1 号块，先 touch 移出空闲队列，再领 5 号尾块", [6, 7, 4, 2, 3])
    print("  对照源码", SOURCE_LINE["touch"])

    requests["D"] = pool.get_new_blocks(5)
    assert block_ids("D") == [6, 7, 4, 2, 3]
    assert pool.blocks[2].block_hash is None
    assert pool.get_cached_block(req_a.block_hashes[1], [0]) is None
    assert storage.nhd[0][storage.offset(2, 0, 0, 0, "NHD")] == old_value
    tokens_d = list(range(1000, 1080))
    for position, token in enumerate(tokens_d):
        storage.write(block_ids("D"), position, token)
    assert storage.nhd[0][storage.offset(2, 0, 0, 0, "NHD")] == 1048
    req_d = request("D", tokens_d)
    pool.cache_full_blocks(req_d, requests["D"], 0, 5, BLOCK_SIZE, 0)
    check_state("8. D 复用空闲块，领到 2 号时撤销旧 hash，写入时覆盖旧 K/V", [])
    try:
        pool.get_new_blocks(1)
    except ValueError:
        print("  空池再领 1 块 → ValueError；真实 KVCacheManager 会先检查并返回 None")
    else:
        raise AssertionError("An exhausted block pool accepted another allocation")

    pool.free_blocks(reversed(requests.pop("D")))
    pool.free_blocks(reversed(requests.pop("C")))
    check_state("9. 所有请求退出，7 个可用块全部归还", [3, 2, 4, 7, 6, 5, 1])
    assert pool.get_usage() == 0
    assert pool.reset_prefix_cache()
    assert all(block.block_hash is None for block in pool.blocks)
    assert pool.get_cached_block(req_b.block_hashes[0], [0]) is None
    assert storage.nhd[0][storage.offset(2, 0, 0, 0, "NHD")] == 1048
    check_state("10. 重置前缀索引后不再命中，缓冲区数值等待后续覆盖", [3, 2, 4, 7, 6, 5, 1])
    print("\n上游 CPU 块管理、NHD/HND 地址、逐元素读回和 attention 数值断言全部通过。")


if __name__ == "__main__":
    main()
