# LLM 推理技术速查表

> 面向推理系统的学习、选型与排障。按原 cheatsheet 的速查形式重新组织，内容独立整理，不替代论文或引擎文档。
>
> **更新基准：2026-11-25**。技术与性能结论依赖模型、硬件、引擎版本和流量；部署前应以目标环境实测为准。
>
> 版本快照：vLLM `v0.34.0`（11-16）、SGLang `v0.5.23`（11-02）、TensorRT-LLM `v1.3.0`（11-20）；FlashAttention-4 `beta40`（11-18，预发布）。快照只用于标记查阅基线，不代表所有环境都应升级到这些版本。

## 0. 一眼看懂推理

```text
请求 → 分词 → 调度
              ├─ Prefill：处理输入 token，产生首个输出 token，并写入 KV
              └─ Decode：逐步生成 token，反复读取历史 KV
                         ↓
               流式输出 → 请求结束 → 释放请求引用
```

核心差异：**Prefill 通常计算密集，Decode 常受权重/KV 的显存带宽和 KV 容量限制。** 对标准全注意力，Prefill 的注意力计算随输入长度近似二次增长；每个 Decode token 都要读取可见历史 KV，成本随上下文长度增长。实际瓶颈会随模型、批量和硬件变化。

## 1. 指标与容量公式

| 指标 | 含义 | 常见影响因素 |
|---|---|---|
| TTFT | 请求到首个输出 token 的时间 | 排队、Prefill、Prefix Cache 命中、网络 |
| ITL / TPOT | 相邻输出 token 的间隔 / 平均每输出 token 时间 | Decode 吞吐、并发、上下文长度、调度 |
| 吞吐 | 系统或单请求每秒处理的 token 数 | 批量、模型、量化、硬件利用率 |
| SLO | 延迟或吞吐目标；延迟常看 p95/p99 | 负载分布与尾延迟，不应只看平均值 |

全注意力 Transformer 中，单条序列每新增一个 token 的 KV 大小近似为：

$$
S_{\mathrm{KV/token}}=2\,L\,H_{\mathrm{kv}}\,d_{\mathrm{head}}\,b
$$

$L$ 是层数，$H_{kv}$ 是 KV 头数，$d_{head}$ 是每头维度，$b$ 是每元素字节数；系数 2 对应 K 和 V。该估算不含块对齐、量化 scale、元数据，也不适用于直接套算 MLA 等不同 KV 表示。

$$
N_{\text{resident tokens}}\lesssim
\frac{M_{\mathrm{KV\ pool}}}{S_{\mathrm{KV/token}}}
$$

$M_{\mathrm{KV\ pool}}$ 是扣除权重、运行时工作区和其他显存预留后实际可用的 KV 池。张量并行/流水并行下需按每卡实际分片或复制方式计算，不能简单把全模型公式除以卡数。

## 2. 三种“缓存”别混为一谈

| 机制 | 解决什么问题 | 省下什么 | 不会自动省下什么 |
|---|---|---|---|
| Paged KV / PagedAttention | 如何分配、寻址和访问 KV 块 | 连续预留导致的浪费；便于动态增长与共享 | Attention 数学计算、有效上下文的 KV 读取 |
| Prefix Caching | 重复前缀能否复用已有 KV | 命中部分的 Prefill 重算 | Decode 对历史 KV 的读取；未命中后缀的计算 |
| Semantic / response cache | 相似问题能否复用答案或检索结果 | 可能跳过模型生成 | 语义误命中风险；它不是 KV Cache |

**PagedAttention 是 KV 的内存管理/访问方式；Prefix Caching 是 KV 的跨请求复用策略。两者可以同时启用。** 分块存储并不等于前缀缓存，前缀命中也不要求所有引擎都使用同一种 attention kernel。

**版本注意：**vLLM [当前设计文档](https://docs.vllm.ai/en/latest/design/paged_attention/)将旧的 PagedAttention kernel 讲解标为历史实现；现在分页 KV 由 KV cache manager 和所选 attention backend 协作处理。vLLM [`v0.34.0` 发布说明](https://github.com/vllm-project/vllm/releases/tag/v0.34.0)将默认 `--block-size` 从 16 改为 64，ROCm paged attention 仍用 16，显式参数也可覆盖。block size 是引擎/后端配置，不是 PagedAttention 的普适常数。

Prefix Cache 命中通常要求 token 前缀及其缓存上下文兼容（例如模型权重、适配器、多模态输入和位置/缓存配置）。它不是“语义相似就命中”。具体哈希键、块对齐、部分块策略和淘汰行为由引擎实现决定。以 vLLM 为例，前缀块通过 hash 与引用计数共享；不应笼统称为 copy-on-write。多租户部署还应了解 `cache_salt` 等隔离机制及其适用边界。混合 Attention/Mamba 模型还可能缓存 SSM state，命中粒度和 checkpoint 规则与普通 KV 不同。

**评估命中价值：**看命中 token 数占输入 token 数的比例、TTFT、GPU KV 占用、淘汰率和排队变化；单看请求命中率可能误导。缓存复用能省计算，但被缓存的 KV 仍占显存，并在 Decode 时参与注意力。

## 3. 降低 KV 成本

| 方法 | 主要作用 | 代价 / 注意点 |
|---|---|---|
| GQA / MQA | 减少 KV 头数，降低每 token KV 字节数 | 需模型架构支持；不是推理时无损改配置 |
| MLA | 用潜在表示压缩 KV 状态 | 需模型与内核支持，不能按普通 K/V 张量公式直接估算 |
| KV 量化（生产常见 FP8 / INT8；研究含 2-bit 等） | 降低 KV 占用、提高可容纳 token 数 | 量化误差、scale 元数据与读写内核影响结果；低比特论文结果不代表引擎普遍支持 |
| 滑动窗口 / 混合注意力 | 限制部分层可见的历史范围 | 依赖模型训练时的注意力结构，可能改变长上下文能力 |
| KV offload / 分层缓存 | 将暂不活跃的 KV 放到 CPU、主机内存或远端存储 | 受 PCIe/网络带宽、拷贝和调度影响，可能增加延迟 |

例如 SGLang [HiCache](https://docs.sglang.io/docs/advanced_features/hicache_design.md) 将 KV 放在 GPU、主机 DRAM 与外部存储等层级；其 GPU 与主机层默认是实例私有，只有配置为共享后端的外部层可跨实例复用。分层缓存扩展的是可复用容量，不等于这些 KV 与 GPU HBM 同速。

> 权重低比特量化与 KV Cache 量化是两件事：前者主要改变模型权重的存储/计算路径，后者压缩上下文状态。两者可以组合，但收益与质量都要分别测量。

## 4. Attention Kernel 与调度

- **FlashAttention**：通过分块和在线 softmax 降低 HBM 与片上存储之间的数据搬运；目标是精确计算注意力，不是近似注意力。不同版本、GPU 和形状下的收益不同。FlashAttention-4 的 [论文](https://arxiv.org/abs/2603.05451)面向 Blackwell；截至本表日期，代码仓库最新是 [`beta40` 预发布](https://github.com/Dao-AILab/flash-attention/releases)，不能把论文基准当作所有硬件上的生产结果。
- **Paged Attention Kernel**：在分页 KV 布局上执行注意力；分页本身是数据布局/寻址，具体 kernel 可由不同后端实现。不要把某个引擎历史上的 kernel 描述当成所有后端的当前实现。
- **Continuous / in-flight batching**：每个迭代动态加入新请求、移除已完成请求，提高批次利用率；调度仍需兼顾公平性、尾延迟和显存上限。
- **Chunked Prefill**：把长输入拆成较小块，与 Decode 交错调度，控制 Prefill 对已有请求 ITL/TPOT 的干扰；块太小可能增加调度开销并拉长 TTFT。像 `max_num_batched_tokens` 这样的预算值是引擎配置，需按流量调优。
- **Prefill-Decode Disaggregation**：把 Prefill 与 Decode 放到不同资源池，可分别调 TTFT/ITL 并控制尾延迟；需要转移 KV，且 vLLM [当前文档](https://docs.vllm.ai/en/latest/features/disagg_prefill/)明确说明该功能本身**不提升吞吐**。它是实验性且版本相关的部署选项，应比较 SLO/goodput、传输成本和资源利用率。
- **KV-aware routing**：将请求路由到有可复用前缀或 KV 的实例，可能减少 Prefill 重算；同时要防止热点实例过载，需一起观察命中率和负载均衡。

## 5. 解码与量化

### Speculative Decoding

草稿模型或其他提议器先提出若干 token，目标模型并行验证；也可使用 n-gram / lookup 等免训练提议器。合适的验证算法可保持目标分布，但**不保证加速**。

主流引擎文档已列出 EAGLE 系列（含 [EAGLE-3](https://arxiv.org/abs/2503.01840)）、MTP、draft model、n-gram / suffix 等方法，但模型、硬件和后端支持并不一致。论文或厂商基准中的倍数只适用于对应配置；方法清单可查 [vLLM speculative decoding 文档](https://docs.vllm.ai/en/latest/features/speculative_decoding/)并以实际版本为准。

重点观测：每次验证接受的 token 数、草稿成本、验证成本、额外显存与端到端 ITL。草稿质量低、输出短、模型已很快或批量很大时，收益可能消失。

### 权重量化

- **W8 / FP8 / W4 等**表示权重或激活的数值格式/位宽；具体含义因量化方案而异。
- 低位宽可降低模型显存和部分带宽压力；吞吐是否提升取决于 GPU 指令、内核、批量和算子支持。
- 比较方案时同时记录质量指标、TTFT、ITL/TPOT、吞吐、显存和长上下文表现，不要只比较模型文件大小。

## 6. 按症状排查

| 症状 | 优先检查 | 可尝试的方向 |
|---|---|---|
| TTFT 高、ITL 正常 | 队列、Prefill 长度、前缀命中、Prefill 算力 | Prefix Cache、chunked prefill、Prefill 批次/资源 |
| ITL 高、TTFT 正常 | Decode 批量、上下文长度、KV 读取带宽 | GQA/MLA 架构、KV 量化、调度与 kernel |
| 并发上不去 / OOM | 权重和工作区余量、KV/token、最大上下文、KV 池与块浪费 | 减少 KV/token、调整池大小/并发限制、offload |
| Prefix 命中率高但延迟无改善 | 命中 token 比例、排队、Decode、KV 占用和 cache eviction | 以 TTFT 和系统级吞吐验证，不只看命中率 |
| 长 Prefill 时 Decode 卡顿 | Prefill 是否抢占迭代预算 | chunked prefill；有隔离需求时评估 P/D 分离 |
| 量化后更慢或质量下降 | 内核是否命中、算子 fallback、长上下文误差 | 换目标硬件支持的格式并做端到端对照 |

## 7. 选型检查清单

1. 收集真实输入/输出长度分布、并发、请求重复前缀比例和 p95/p99 SLO。
2. 先算每 token KV 和权重/工作区余量，再估算可承载的 active tokens。
3. 按瓶颈选择：重复长前缀优先验证 Prefix Cache；KV 容量不足看 GQA/MLA/KV 量化；长 Prefill 干扰 Decode 看 chunked prefill 或资源分离。
4. 固定模型、硬件、引擎版本和请求轨迹，对照测 TTFT、ITL/TPOT、吞吐、KV 水位、命中 token、淘汰与质量。
5. 扫描并发和输入长度，不要用单请求短 prompt 的结果代表生产负载。

## 8. 参考资料

**本仓库与基础论文**

- 原始概念速查：[Transformer 与大语言模型 cheatsheet（PDF）](./cheatsheet-transformers-large-language-models.pdf)
- 本仓库深入讲解：[LLM 推理全链路：从玩具模型到 GPU 水位](./LLM推理全链路-从玩具模型到GPU水位.md)
- 本仓库深入讲解：[PagedAttention 与 Prefix Caching：原理与 vLLM 实现](./PagedAttention与PrefixCaching-原理与vLLM实现.md)
- [PagedAttention 论文](https://arxiv.org/abs/2309.06180) · [SGLang / RadixAttention 论文](https://arxiv.org/abs/2312.07104) · [DeepSeek-V2 / MLA 论文](https://arxiv.org/abs/2405.04434)
- [FlashAttention-2 论文](https://arxiv.org/abs/2307.08691) · [FlashAttention-4 论文](https://arxiv.org/abs/2603.05451) · [KIVI 低比特 KV 量化论文](https://arxiv.org/abs/2402.02750)
- [Speculative Decoding 论文](https://arxiv.org/abs/2211.17192) · [EAGLE-3 论文](https://arxiv.org/abs/2503.01840)

**引擎文档与版本快照**

- [vLLM 自动前缀缓存](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching.html) · [PagedAttention 当前实现说明](https://docs.vllm.ai/en/latest/design/paged_attention/) · [推测解码方法清单](https://docs.vllm.ai/en/latest/features/speculative_decoding/)
- [vLLM Prefill/Decode 分离文档](https://docs.vllm.ai/en/latest/features/disagg_prefill/)（实验性；文档说明不提升吞吐）· [SGLang HiCache 设计](https://docs.sglang.io/docs/advanced_features/hicache_design.md)
- [vLLM `v0.34.0` 发布说明](https://github.com/vllm-project/vllm/releases/tag/v0.34.0) · [SGLang `v0.5.23`](https://github.com/sgl-project/sglang/releases/tag/v0.5.23) · [TensorRT-LLM `v1.3.0`](https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.3.0)
- [FlashAttention 发布版本](https://github.com/Dao-AILab/flash-attention/releases)（`beta40` 于 2026-11-18 发布，属预发布）

> vLLM `latest` 文档是开发预览，配置项与默认值会变化。涉及部署时，请按目标引擎的固定版本查阅文档和发布说明。
