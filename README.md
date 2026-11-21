# attention-crash-course

用一个 **8 维隐藏层、4 个 Q 头、2 个 KV 头的玩具模型（TinyToy）**，把注意力机制和 LLM 推理效率的计算一笔一笔算清楚。

所有形状关系、字节账、FLOPs 公式都与真实模型（Llama-3、Qwen 等）完全一致，只是规模缩小到手算可核对的量级：

| 量 | TinyToy | Llama-3-70B |
|---|---|---|
| `hidden_size` | 8 | 8192 |
| `num_attention_heads` | 4 | 64 |
| `num_key_value_heads` | 2 | 8（GQA，group_size=8） |
| `head_dim` | 2 | 128 |
| `num_hidden_layers` | 2 | 80 |
| KV / token | 32 B | 320 KiB |

## 仓库内容

### [`LLM推理全链路-从玩具模型到GPU水位.md`](./LLM推理全链路-从玩具模型到GPU水位.md)

主线文档。回答一个问题：**为什么 LLM 推理的瓶颈不在算力，而在带宽？**

| 章 | 主题 |
|---|---|
| 0 | 导读：为什么是这个顺序 |
| 1 | 注意力的头与维度：宽度守恒、多头为什么免费、GQA 分组、共享 K/V 为什么不塌缩 |
| 2 | 一个 token 的两张账单：`2N` FLOPs 与 KV 字节的五因子拆解 |
| 3 | KV cache 的物理存储：为什么存 K/V 不存 Q、PagedAttention、prefix caching |
| 4 | GPU 硬件与 Roofline：HBM / SM / Tensor Core、`AI ≈ 2M`、decode 的物理下界 |
| 5 | Prefill 与 Decode：两种负载的形状差异 |
| 6 | cached token 的真实成本：算力 5× / 并发 3× / 带宽 1× |
| 7 | vLLM 调度器与排队：五道准入闸门 |
| 8 | 水位：算力 / 带宽 / KV / 槽位四维取 max |
| 附录 | TinyToy 完整规格、Llama-3-70B 对照、公式速查卡、术语表、易错点清单 |

### [`TinyToy_LLM推理全链路.ipynb`](./TinyToy_LLM推理全链路.ipynb)

把上面每个数字跑成代码，并逐项断言。**已完整执行过一遍，输出保存在 notebook 里**，可以直接读，也可以自己重跑。

覆盖内容：

- **第 1 章**：唯一的 8 维 `h` 手工投影出 `q/k/v` → 分成 4 个 Q head / 2 个 KV head → GQA 分组注意力 → prefill → 逐 token decode → KV cache 字节账
- **第 2 章**：用 `nn.Linear` 统计参数量，验证 `2N = 2304`
- **第 3 章**：真正按 block 分配的 `PagedKVCache`，验证并发 4 → 8、利用率 37.5% → 75%
- **第 4 章**：算术强度与 Roofline 拐点；本机 GEMM 的 M–吞吐趋势
- **第 5 章**：完整 TinyToy 前向，验证 **prefill 与 decode 数值等价**
- **第 6~8 章**：cached token 倍数、五道闸门、四维水位

## 运行

notebook 在 `torch 2.6.0` + Python 3.11 下验证通过：

```bash
conda create -n minimind python=3.11
conda activate minimind
pip install torch jupyter
jupyter lab TinyToy_LLM推理全链路.ipynb
```

CPU 就能跑完（玩具模型一次 forward 只有 2304 FLOPs）。除了 4.7 节那个 GEMM 基准，其余全是确定性数值，不需要 GPU。

> 文档约定：所有**绝对时间**都没有意义，只验证**公式与比例关系**。

## 建议阅读路径

1. 先读文档第 0~2 章，建立「一个 token 花多少算力、占多少字节」的直觉；
2. 打开 notebook 的 `## 2`、`## 3` 两节，逐格跑，对照文档第 1 章的数值追踪；
3. 再回文档第 3~4 章，理解为什么 KV cache 会把推理变成带宽瓶颈；
4. 最后读第 5~8 章，落到 vLLM 的调度与水位。

## 本节要带走的三句话

1. **多头是免费的**：切头不改变 FLOPs，只改变「同时能看多少种关系」。
2. **GQA 省的是字节，不是计算**：它提高算术强度，让带宽瓶颈晚一点到来。
3. **decode 是带宽受限的**：每生成一个 token 都要把**全部权重**从 HBM 读一遍，这是物理下界，不是实现问题。
