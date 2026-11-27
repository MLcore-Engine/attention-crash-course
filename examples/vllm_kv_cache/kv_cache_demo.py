"""CPU-only address-mapping demo for vLLM v0.11.0 paged KV cache."""

from math import exp, sqrt


BLOCK_SIZE = 4
NUM_BLOCKS = 8
NUM_KV_HEADS = 2
NUM_Q_HEADS = 4
HEAD_SIZE = 8
X = 8  # 16 bytes / 2-byte cache element, as in the BF16 CUDA path.


def zeros(shape: tuple[int, ...]) -> list:
    if len(shape) == 1:
        return [0.0] * shape[0]
    return [zeros(shape[1:]) for _ in range(shape[0])]


def main() -> None:
    # Each token has [num_kv_heads, head_size] K and V values.
    num_tokens = 5
    keys = [
        [[float(t * 0.1 + h * 0.02 + d * 0.01) for d in range(HEAD_SIZE)]
         for h in range(NUM_KV_HEADS)]
        for t in range(num_tokens)
    ]
    values = [
        [[float(t * 10 + h * 2 + d * 0.1) for d in range(HEAD_SIZE)]
         for h in range(NUM_KV_HEADS)]
        for t in range(num_tokens)
    ]

    # Logical blocks 0 and 1 live in physical blocks 5 and 1.
    block_table = [5, 1]
    positions = list(range(num_tokens))
    slot_mapping = [
        block_table[pos // BLOCK_SIZE] * BLOCK_SIZE + pos % BLOCK_SIZE
        for pos in positions
    ]
    assert slot_mapping == [20, 21, 22, 23, 4]

    # vLLM's custom paged-attention layout:
    # K [blocks, kv_heads, head_size/x, block_size, x]
    # V [blocks, kv_heads, head_size, block_size]
    key_cache = zeros((NUM_BLOCKS, NUM_KV_HEADS, HEAD_SIZE // X,
                       BLOCK_SIZE, X))
    value_cache = zeros((NUM_BLOCKS, NUM_KV_HEADS, HEAD_SIZE, BLOCK_SIZE))

    # Simulate reshape_and_cache_kernel's scalar address mapping.
    for token_idx, slot in enumerate(slot_mapping):
        if slot < 0:
            continue
        block_idx, block_offset = divmod(slot, BLOCK_SIZE)
        for head_idx in range(NUM_KV_HEADS):
            for dim in range(HEAD_SIZE):
                key_cache[block_idx][head_idx][dim // X][block_offset][dim % X] = (
                    keys[token_idx][head_idx][dim]
                )
                value_cache[block_idx][head_idx][dim][block_offset] = (
                    values[token_idx][head_idx][dim]
                )

    def read_kv(logical_pos: int, kv_head: int) -> tuple[list[float], list[float]]:
        logical_block, block_offset = divmod(logical_pos, BLOCK_SIZE)
        physical_block = block_table[logical_block]
        k = [
            key_cache[physical_block][kv_head][d // X][block_offset][d % X]
            for d in range(HEAD_SIZE)
        ]
        v = [
            value_cache[physical_block][kv_head][d][block_offset]
            for d in range(HEAD_SIZE)
        ]
        return k, v

    # Read back the logical sequence through block_table, not by physical order.
    for pos in positions:
        for head in range(NUM_KV_HEADS):
            k, v = read_kv(pos, head)
            assert k == keys[pos][head]
            assert v == values[pos][head]

    # Q head 3 maps to KV head 1 for GQA (4 Q heads / 2 KV heads).
    q_head = 3
    kv_head = q_head // (NUM_Q_HEADS // NUM_KV_HEADS)
    query = [0.1 * (d + 1) for d in range(HEAD_SIZE)]
    scores: list[float] = []
    retrieved_values: list[list[float]] = []
    for pos in positions:
        k, v = read_kv(pos, kv_head)
        scores.append(sum(q * x for q, x in zip(query, k)) / sqrt(HEAD_SIZE))
        retrieved_values.append(v)

    max_score = max(scores)
    weights_unnormalized = [exp(score - max_score) for score in scores]
    weight_sum = sum(weights_unnormalized)
    weights = [weight / weight_sum for weight in weights_unnormalized]
    output = [
        sum(weights[t] * retrieved_values[t][d] for t in positions)
        for d in range(HEAD_SIZE)
    ]

    print("pos | logical block | physical block | offset | slot")
    for pos, slot in zip(positions, slot_mapping):
        logical_block, offset = divmod(pos, BLOCK_SIZE)
        print(f"{pos:>3} | {logical_block:>13} | {block_table[logical_block]:>14} "
              f"| {offset:>6} | {slot}")
    print(f"\nQ head {q_head} uses KV head {kv_head}.")
    print("Attention weights:", [round(weight, 4) for weight in weights])
    print("Attention output:", [round(value, 2) for value in output])
    print("All cache write/read assertions passed.")


if __name__ == "__main__":
    main()
