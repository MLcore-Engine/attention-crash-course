# PagedAttention 与 Prefix Caching：原理与 vLLM / SGLang 实现

> 配套主线文档：[LLM推理全链路-从玩具模型到GPU水位.md](file:///Users/jyxc-dz-0100664/Documents/Qoder/2026-11-19/69b4e9d0/LLM推理全链路-从玩具模型到GPU水位.md) 第 3 章。
> 主线文档的 §3.6 / §3.7 是**结论版**；这一份是**推导版 + 源码版**。
> 本文所有数字都由附录 A / E 的脚本实际执行验证过（共 **79 + 33** 项断言全部通过），vLLM 源码锁定 **v0.11.0**，SGLang 源码锁定 **v0.5.2**。
> 第一至第六部分是 vLLM 的推导与源码；**第七部分（§28–§35）是 SGLang 的对照实现**，用来回答「同样的前缀复用，换一种数据结构会长什么样」。

---

## 0 导读

### 0.1 这份教程要回答的五个问题

主线文档告诉你「PagedAttention 把利用率从 37.5% 提到 75%」。这句话是对的，但它跳过了五个真正需要理解的地方：

| # | 问题 | 本文位置 |
|---|---|---|
| 1 | 分页**凭什么**能省显存？它省的是哪一种浪费，哪一种浪费它反而引入了？ | 第一部分（§1–§4） |
| 2 | K/V 在物理上不连续了，attention 的数学**为什么一点都不变**？ | 第二、三部分（§5–§11） |
| 3 | vLLM 里到底是谁在什么时候调用了谁，一个块从领取到归还经过了哪些字段？ | 第四部分（§12–§16） |
| 4 | Prefix caching 命中之后，**哪些成本真的消失了，哪些一分钱没省**？为什么明明整块缓存了却还要重算一整块？ | 第五部分（§17–§25） |
| 5 | 同样的机制换一套数据结构（SGLang 的基数树）会长什么样？哪些不变量不变？ | 第七部分（§28–§35） |

### 0.2 贯穿全文的玩具规格：TinyToy

和主线文档完全一致的规格，小到可以手算：

| 符号 | 含义 | TinyToy | Llama-3-70B |
|---|---|---|---|
| `H` | `hidden_size` | 8 | 8192 |
| `H_q` | `num_attention_heads` | 4 | 64 |
| `H_kv` | `num_key_value_heads` | 2 | 8 |
| `d` | `head_dim = H / H_q` | 2 | 128 |
| `L` | `num_hidden_layers` | 2 | 80 |
| — | GQA `group_size = H_q / H_kv` | 2 | 8 |
| — | 数据类型 | BF16（2 B） | BF16（2 B） |

KV 字节账（每个数字都写出乘法）：

```
每 token 每层 KV = 2(K和V) × H_kv × d × sizeof(dtype)
TinyToy         = 2 × 2 × 2 × 2 B = 16 B
每 token 全部层  = L × 16 B = 2 × 16 = 32 B

Llama-3-70B     = 2 × 8 × 128 × 2 B = 4096 B / 层
                = 80 × 4096 B = 327,680 B = 320 KiB
```

KV 池与分块：

| 量 | TinyToy | Llama-3-70B |
|---|---|---|
| KV 池 | 1024 B | 50 GiB |
| `block_size` `B` | 4 token | 16 token |
| 每块字节 | `4 × 32 = 128 B` | `16 × 327,680 = 5,242,880 B = 5 MiB` |
| 块数 | `1024 / 128 = 8` | `50 GiB / 5 MiB = 10,240` |
| token 容量 | `8 × 4 = 32` | `10,240 × 16 = 163,840` |

> **先埋一个伏笔**：上面这个「8 块」是理想化的算术模型。真实 vLLM 里 `block_id = 0` 会被一个特殊的 null block 无条件占掉，**实际可分配的是 7 块**。这个 off-by-one 在 §15 专门讲，它直接修正了主线文档「8 并发」的说法。

### 0.3 先立规矩：三本账，分母不同

这是全文最容易出错的地方，先把它钉死。任何「利用率 = X%」的说法，都必须先问**分母是什么**。

| 账本 | 定义 | 4 个长度为 3 的序列（`B=4`，8 块池） |
|---|---|---|
| ① 有效 KV | 真正装了 token 的槽位数 | `4 × 3 = 12` token |
| ② 已分配容量 | `⌈L/B⌉ × B` 求和 | `4 × ⌈3/4⌉ × 4 = 4 × 4 = 16` token |
| ③ 整个池容量 | `块数 × B` | `8 × 4 = 32` token |

于是有两个完全不同、都正确的「利用率」：

```
① / ② = 12 / 16 = 75%      ← 已分配的块里，有多少装了真东西（衡量尾块浪费）
① / ③ = 12 / 32 = 37.5%    ← 整个池子有多少装了真东西（衡量池子有多满）
```

**75% 和 37.5% 不矛盾，它们回答的是两个不同的问题。** 主线文档里「利用率从 37.5% 提到 75%」这句话，严格说是把两个不同分母的比值放在一起对比，容易让人误以为「分页让池子变满了」。分页真正做的是消除了**预留造成的浪费**，让同样多的有效 KV 占更少的块。§4 会把这件事算清楚。

### 0.4 边界声明

- vLLM 源码版本 **v0.11.0**，V1 引擎。V0 的 `BlockSpaceManager` 是另一套实现，本文不涉及。
- SGLang 源码版本 **v0.5.2**（`radix_cache.py` / `allocator.py` / `memory_pool.py` / `schedule_batch.py`）。SGLang 的 `page_size` 默认是 1，本文在对照示例里取 4 是为了和 vLLM 的 `block_size=4` 同尺度；两者不是同一个参数。
- **版本提醒**：写作时 vLLM 最新为 0.34.x、SGLang 最新为 0.5.23。本文锁定 v0.11.0 / v0.5.2，是为了让行号和类名可复现。SGLang 新版已把 `RadixCache` 重构出 `RadixKey` / `UnifiedRadixCache`，机制仍是基数树 + 叶子 LRU，但 API 有变化；vLLM 新版仍保留 `block_pool.py` 的块池结构，细节有演进。
- 本文只讲 **full attention**（`FullAttentionManager`）。sliding window、Mamba、hybrid 多 KV cache group 的差异只在需要时点一句，不展开。
- 张量并行 / DCP（decode context parallel）会改变 `block_size` 的有效值和 slot 计算，§6.3 给出真实代码分支，但不展开分布式布局。
- **所有绝对时间都没有意义**，本文只验证公式、比例和机制。
- 本文**不含**性能实测数字。「分页让吞吐提高 N 倍」这类结论依赖模型、负载、backend，本文不给。

---

# 第一部分 原理：分页到底解决了什么

## 1 显存里到底放什么

### 1.1 一个 token 的 KV 长什么样

先明确一件容易混淆的事：**KV cache 里存的是每一层的 K 和 V 投影结果，不是 hidden state。**

TinyToy 一个 token 在第 0 层产生：

```
h        : [8]                 ← hidden state，不存
q = h·Wq : [4, 2]  = 8 个数     ← Q，不存（用完即弃，§1.3）
k = h·Wk : [2, 2]  = 4 个数     ← 存
v = h·Wv : [2, 2]  = 4 个数     ← 存
```

存下来的是 `4 + 4 = 8` 个数，BF16 每个 2 B → **16 B**。两层 → **32 B**。这就是 §0.2 那个 `2 × H_kv × d × 2B` 的来源。

注意 `k` 的第一维是 `H_kv = 2` 而不是 `H_q = 4`。**GQA 省的就是这里**：4 个 Q 头分成 2 组，每组 2 个 Q 头共享同一份 K/V。如果不用 GQA（`H_kv = H_q = 4`），每层就是 `2 × 4 × 2 × 2 = 32 B`，翻倍。

### 1.2 为什么 K/V 必须留着，Q 不用留

decode 到第 `t` 步时，要算的是：

```
输出[t] = softmax( q[t] · K[0..t]ᵀ / √d ) · V[0..t]
```

| 张量 | 是否需要历史 | 原因 |
|---|---|---|
| `q[t]` | 不需要 | 只有当前 token 的 query 参与计算，`q[0..t-1]` 已经算完输出了，再也用不到 |
| `K[0..t]` | **需要全部** | 当前 query 要和**每一个**历史 key 做点积 |
| `V[0..t]` | **需要全部** | softmax 权重是对**每一个**历史 value 加权 |

所以 KV cache 的大小随序列长度**线性增长**，而 Q 不占历史空间。这是「为什么长上下文吃显存」的根因。

### 1.3 一个具体的物理排布

TinyToy，`B = 4`，一块 = `4 token × 32 B = 128 B`。逻辑上一块的内容是：

```
块内偏移 0:  token 的 [层0 K(4数) 层0 V(4数) 层1 K(4数) 层1 V(4数)]  = 32 B
块内偏移 1:  下一个 token 的同样 32 B
块内偏移 2:  ...
块内偏移 3:  ...
                                                          合计 128 B
```

> **真实实现比这个复杂**：vLLM 不会把「所有层」放在一个块里。每一层有自己的 KV tensor，块号在各层之间是**共享的同一套逻辑编号**，但物理张量是 per-layer 的。而且 K 张量为了访存合并还做了维度拆分。§7 讲真实的两种布局。这里先用「一块 = 一个 token 的全部 KV」的简化模型建立直觉。

## 2 三种分配策略的对照实验

现在做一个真正的对照实验。场景固定：**4 个请求，实际长度都是 3，`max_model_len` 允许到 8。**

### 2.1 策略 A：连续分配 + 按 max_len 预留

每个请求进来就按「它将来可能长到的最大长度」划一段连续显存。

```
池子 32 槽（1024 B）
请求1: [槽 0..7]   实际用 0,1,2      浪费 3,4,5,6,7
请求2: [槽 8..15]  实际用 8,9,10     浪费 11..15
请求3: [槽16..23]  实际用16,17,18    浪费 19..23
请求4: [槽24..31]  实际用24,25,26    浪费 27..31
```

| 账本 | 数值 |
|---|---|
| 有效 KV | `4 × 3 = 12` token |
| 已分配容量 | `4 × 8 = 32` token |
| 整个池 | 32 token |
| ①/② | `12/32 = 37.5%` |
| ①/③ | `12/32 = 37.5%` |
| 空闲块 | **0** |

**第 5 个请求进不来**，尽管池子里有 20 个槽是空的。这 20 个槽被「预留」锁住了。

> **重要限定**：按 `max_len` 预留是**一种基线策略**，常见于早期/朴素实现，用来保证「请求一旦开始就绝不会中途 OOM」。**它不是所有非分页实现的通性。** 说「不用分页就必然浪费 60%+」是不严谨的——浪费多少完全取决于预留策略。本文把它当作一个**明确的对照组**，而不是「非分页」的定义。

### 2.2 策略 B：连续分配 + 按实际长度精确分配

不预留，来多少给多少，但要求物理连续。

```
请求1: [槽 0..2]
请求2: [槽 3..5]
请求3: [槽 6..8]
请求4: [槽 9..11]
空闲:  [槽12..31]  连续 20 槽
```

| 账本 | 数值 |
|---|---|
| 有效 KV | 12 token |
| 已分配容量 | 12 token |
| ①/② | **100%** |
| ①/③ | `12/32 = 37.5%` |
| 空闲 | 20 槽，连续 |

**这个静态场景下策略 B 完胜分页**（分页要 16 槽，它只要 12 槽）。这一点必须承认：**分页不是无条件更省。**

策略 B 的问题出在**动态**：序列要增长，而它右边紧邻着别人的数据，长不了。要么一开始就预留（退化成策略 A），要么搬家（重分配 + 拷贝，代价随长度线性增长）。

### 2.3 策略 C：固定块分页

按需领块，一块 4 槽，逻辑上连续、物理上任意。

```
请求1: block_table = [0]      → 物理块0 的槽 0,1,2   （槽3 空着）
请求2: block_table = [1]      → 物理块1 的槽 0,1,2
请求3: block_table = [2]
请求4: block_table = [3]
空闲块: 4,5,6,7                共 4 块
```

> **模型约定**：第一到第三部分（§1–§11）用**理想 8 块模型**，块 0..7 全部可分配，为的是算术干净。
> 第四部分起（§12 之后）切换到**真实 vLLM 语义**，块 0 被 null block 占掉，可用块只有 1..7。
> 两者的差别就是 §15 要讲的那个 off-by-one，切换点会明确标注。

| 账本 | 数值 |
|---|---|
| 有效 KV | 12 token |
| 已分配容量 | `4 × ⌈3/4⌉ × 4 = 16` token |
| ①/② | `12/16 = **75%**` |
| ①/③ | `12/32 = 37.5%` |
| 空闲块 | 4 块 = 16 槽 |

### 2.4 三策略并排

| | A 预留 max_len | B 精确连续 | C 固定块分页 |
|---|---|---|---|
| 已分配槽位 | 32 | **12** | 16 |
| 有效/已分配 | 37.5% | **100%** | 75% |
| 能否增长到 4 | 能（预留够了） | **不能**（右边被占） | 能（块内还有空位） |
| 能否增长到 8 | 能 | 不能 | 能（再领 1 块） |
| 增长是否需要拷贝 | 否 | **是** | **否** |
| 第 5 个请求能否进来 | **不能** | 能 | 能 |
| 支持跨请求共享 | 否 | 否 | **是**（§22） |

**结论**：分页买到的是「**按需增长且不搬家**」+「**可共享**」，代价是「**每序列最多 `B−1` 个槽的尾块浪费**」。它不是把浪费降到零，而是把**不可控的、随 `max_len` 放大的浪费**换成**可控的、上界为 `B−1` 的浪费**。

## 3 内碎片与外碎片：两个不同的病

上面的实验只展示了内碎片。外碎片需要「有请求结束」才会出现，这是分页第二个不可替代的价值。

### 3.1 内碎片（internal fragmentation）

**分给你的空间里，你没用上的部分。** 策略 A 的预留、策略 C 的尾块，都是内碎片。

```
策略 A：给你 8 槽，你用 3 → 内碎片 5 槽/请求
策略 C：给你 4 槽，你用 3 → 内碎片 1 槽/请求
```

### 3.2 外碎片（external fragmentation）

**空闲总量够，但没有一段足够长的连续空间。** 这是变长连续分配的专属病。

具体实验（池 32 槽，策略 B）：

```
初始:  A[0..2]  B[3..7]  C[8..9]  D[10..15]  空闲[16..31]
B 结束 → 释放 [3..7]   5 槽
D 结束 → 释放 [10..15]  6 槽

现在的空洞:  [3..7]=5   [10..15]=6   [16..31]=16
空闲总量 = 5 + 6 + 16 = 27 槽
最大连续空洞 = 16 槽
```

来一个需要 **20 个连续槽**的请求：

| 判据 | 结果 |
|---|---|
| 空闲总量 `27 ≥ 20` | 满足 |
| 最大连续空洞 `16 ≥ 20` | **不满足** |
| 能否分配 | **不能** |

**27 个槽空着，却装不下一个 20 槽的请求。** 这就是外碎片。

同样的场景，分页（`B=4`）：20 token 需要 `⌈20/4⌉ = 5` 块，而 27 个空闲槽相当于 6 块多的容量，块与块之间**不要求相邻**，随便凑 5 块就行。**外碎片被彻底消除**——因为分配单位固定，任何空闲块都能满足任何请求。

### 3.3 为什么「固定大小」是关键

这是操作系统页式虚拟内存的核心洞察，直接搬过来：

| 操作系统 | vLLM |
|---|---|
| 虚拟页号 | 逻辑块号 `t // B` |
| 页表 | `block_table` |
| 物理页帧号 | 物理块号 |
| 页内偏移 | `t % B` |
| 固定页大小（4 KiB） | 固定 `block_size`（如 16 token） |
| 按需分页（demand paging） | 长到边界才领新块 |
| 共享内存页 / COW | prefix caching 的块共享 |

**类比成立的部分**：地址翻译、按需分配、消除外碎片、共享。
**类比不成立的部分**（必须说清，否则会误导）：

| OS 有 | vLLM 有没有 |
|---|---|
| 缺页中断（page fault） | **没有**。块不够就是不够，调度器直接不调度这个请求或抢占别人（§16），不会「触发中断去换页」 |
| 换出到磁盘（swap） | V1 默认**没有**。抢占策略是 RECOMPUTE（丢掉重算），不是换出 |
| 写时复制（COW） | **没有**。共享的块是**只读前缀**，谁都不会去改它，所以不需要 COW |
| TLB | **没有**。每次都要读 `block_table`（§11） |

## 4 尾块浪费的上界，以及它**不能**推出什么

### 4.1 上界证明

长度 `L`，块大小 `B`：

```
所需块数   = ⌈L / B⌉
已分配槽位 = ⌈L / B⌉ × B
尾块浪费   = ⌈L / B⌉ × B − L
```

设 `L = qB + r`，`0 ≤ r < B`：
- `r = 0`：`⌈L/B⌉ = q`，浪费 `= qB − qB = 0`
- `r > 0`：`⌈L/B⌉ = q+1`，浪费 `= (q+1)B − (qB+r) = B − r`，而 `1 ≤ r ≤ B−1` → 浪费 `∈ [1, B−1]`

```
∴ 尾块浪费 ∈ [0, B−1]，与 L 无关。
```

`B = 4` 时逐个验证（脚本对 `L = 0..39` 全部断言过）：

| `L` | `⌈L/4⌉` | 已分配 | 浪费 |
|---|---|---|---|
| 3 | 1 | 4 | **1** |
| 4 | 1 | 4 | **0** |
| 5 | 2 | 8 | **3** |
| 6 | 2 | 8 | 2 |
| 7 | 2 | 8 | 1 |
| 8 | 2 | 8 | 0 |

**关键性质：浪费的上界只由 `B` 决定，不随序列变长而增长。** 这和策略 A 的预留浪费（随 `max_len` 增长）是本质区别。

### 4.2 这个上界**不能**推出的三件事

这是主线文档里一个需要纠正的推理。原文说：

> ~~「100 个序列最多浪费 1500 槽，占池子 0.9%，所以利用率 99%」~~

**这个推理是错的。** 上界 `B−1` 只约束**尾块浪费**，它不说明：

| 它不说明 | 为什么 |
|---|---|
| 池子有多满 | 分母是「整个池」，而上界管的是「已分配容量里的空隙」。100 个序列可能只占了池子的 10%，也可能装不下 |
| 这些序列是否装得下 | 需要块数 = `Σ⌈Lᵢ/B⌉`，可能远超池子块数（§26 有真实反例） |
| 有效/整个池的比值 | 那是另一本账（§0.3 的 ①/③） |

正确的说法只能是：

```
有效 KV / 已分配容量 ≥ (Σ Lᵢ) / (Σ Lᵢ + n×(B−1))
```

当 `Σ Lᵢ` 很大时这个比值趋近 1。**它衡量的是「分出去的块内部有多紧凑」，不是「池子利用率」。**

### 4.3 「8 并发」需要加的限定

主线文档说分页后并发从 4 提到 8。在理想 8 块模型下：

| 状态 | 块数 | 空闲块 |
|---|---|---|
| 8 个请求各长 3，各持 1 块 | 8 | **0** |

必须加两条限定：

1. **它们都能长到 4**（`⌈4/4⌉ = 1`，块内还有空位，不需要新块）。
2. **任何一个想长到 5 就必须等**（`⌈5/4⌉ = 2`）。8 个请求全部长到 5 需要 `8 × 2 = 16` 块 > 8 块。此时调度器要么让部分请求等着，要么抢占（§16）。

所以「8 并发」是**「8 个请求同时持有块」**，不是「8 个请求都能自由长到 `max_len`」。真实 vLLM 里还要再减 1（§15）。

---

# 第二部分 地址翻译：逻辑到物理

## 5 `block_table` 与 `locate(t)`

### 5.1 表是什么

每个请求有一张一维整数表，下标是**逻辑块号**，值是**物理块号**。

```
请求 A，长度 5，B = 4：
  逻辑块 0 → 物理块 5
  逻辑块 1 → 物理块 1
  block_table = [5, 1]
```

表长 `⌈5/4⌉ = 2`。物理块 5 和 1 在显存里离得很远，**没有任何关系**，只有这张表把它们串成「A 的第 0..7 个槽」。

### 5.2 四步定位

```python
block_size = 4
block_table = [5, 1]

def locate(t):
    logical_block = t // block_size          # 在第几个逻辑块
    offset        = t % block_size           # 块内第几个位置
    physical_block = block_table[logical_block]   # 查表
    slot = physical_block * block_size + offset   # 全局槽号
    return physical_block, offset, slot
```

逐 token 追踪（脚本已断言）：

| `t` | `t//4` | `t%4` | `block_table[·]` | `slot = phys×4+off` |
|---|---|---|---|---|
| 0 | 0 | 0 | 5 | `5×4+0 = 20` |
| 1 | 0 | 1 | 5 | `5×4+1 = 21` |
| 2 | 0 | 2 | 5 | `5×4+2 = 22` |
| 3 | 0 | 3 | 5 | `5×4+3 = 23` |
| 4 | **1** | **0** | **1** | `1×4+0 = **4**` |

**注意 `t=3 → slot 23`，`t=4 → slot 4`。逻辑上相邻的两个 token，物理槽号从 23 跳到 4。** 这就是「不连续」的具体含义。

### 5.3 增长时表怎么变

A 从 3 长到 5：

| 长度 | `⌈L/4⌉` | 已有块数 | 新增块数 `max(0, 需要−已有)` | `block_table` |
|---|---|---|---|---|
| 3 | 1 | 1 | 0 | `[5]` |
| 4 | 1 | 1 | **0** | `[5]` |
| 5 | 2 | 1 | **1** | `[5, 1]` |
| 8 | 2 | 2 | 0 | `[5, 1]` |
| 9 | 3 | 2 | 1 | `[5, 1, ?]` |

**表是 append-only 的**：只在尾部追加，中间已写入的物理块号**永不改变**。这个性质非常重要，它是 §24「不去重」和 GPU 侧增量更新的前提。

真实源码就是这几行（`single_type_kv_cache_manager.py:109`）：

```python
def allocate_new_blocks(self, request_id, num_tokens):
    req_blocks = self.req_to_blocks[request_id]
    num_required_blocks = cdiv(num_tokens, self.block_size)
    num_new_blocks = num_required_blocks - len(req_blocks)
    if num_new_blocks <= 0:
        return []
    new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
    req_blocks.extend(new_blocks)          # ← append-only
    return new_blocks
```

## 6 slot 号不是字节地址

### 6.1 Python 侧：算出 slot

`block_table.py:107`，真实的向量化写法（非 DCP 分支）：

```python
block_table_indices = (req_indices * self.max_num_blocks_per_req +
                       positions // self.block_size)
block_numbers = self.block_table.np.ravel()[block_table_indices]
block_offsets = positions % self.block_size
np.add(block_numbers * self.block_size, block_offsets,
       out=self.slot_mapping.np[:req_indices.shape[0]])
```

三点值得注意：

1. **`block_table` 是二维定长矩阵** `[max_num_reqs, max_num_blocks_per_req]`，不是「每请求一个变长 list」。所以要先用 `req_indices × 行宽 + 逻辑块号` 拍平成一维下标。这是为了能用一次 numpy 花式索引处理整个 batch。
2. **`positions` 是逻辑位置**，不是 batch 内的 token 下标。源码注释明确警告：

   > `NOTE(woosuk): We can't simply use token_indices // block_size here because M (max_model_len) is not necessarily divisible by block_size.`

   一个 batch 里混着多个请求的多个 token，`token_indices` 是 batch 内序号，跟「这个 token 在它自己序列里的第几位」完全无关。必须用 `positions`。
3. **输出是 `slot_mapping`，长度 = batch 里的 token 数**，`dtype=int64`。它是「**槽号**」，不是字节偏移。

### 6.2 GPU 侧：把 slot 拆回 (块, 偏移)

写入 kernel `reshape_and_cache_flash_kernel`（`csrc/cache_kernels.cu:282`）：

```cpp
const int64_t token_idx = blockIdx.x;              // 一个 CUDA block 处理一个 token
const int64_t slot_idx  = slot_mapping[token_idx];
if (slot_idx < 0) return;                          // -1 = padding，跳过
const int64_t block_idx    = slot_idx / block_size;
const int64_t block_offset = slot_idx % block_size;
...
cache_t* key_dst = key_cache + block_idx * block_stride
                             + block_offset * page_stride;
```

**这里有一个漂亮的对称性**：

```
Python:  slot = physical_block × B + offset      （合成）
CUDA:    physical_block = slot / B               （拆回）
         offset         = slot % B
```

Python 侧刚算出来的 `(块, 偏移)` 被编码成一个整数传给 GPU，GPU 再解码回 `(块, 偏移)`。为什么要这样绕？因为**传一个 `[num_tokens]` 的一维 int64 数组，比传两个数组省一半带宽和一半 kernel 参数**，而且编解码是两条整数指令，几乎免费。脚本对全部 32 个槽断言了这个往返恒等式。

`slot_idx < 0` 的分支也是真实存在的：padding token 和 DCP 下「不属于本 rank」的 token 都写 `-1`。

### 6.3 DCP 分支（点一句）

`block_table.py:84` 起，当 `dcp_world_size > 1`：

```python
virtual_block_size = self.block_size * self.dcp_world_size
block_table_indices = req_indices * max_num_blocks_per_req + positions // virtual_block_size
virtual_block_offsets = positions % virtual_block_size
mask = virtual_block_offsets % self.dcp_world_size == self.dcp_rank
block_offsets = virtual_block_offsets // self.dcp_world_size
slot_mapping = np.where(mask, block_numbers * self.block_size + block_offsets, -1)
```

token `i` 的 KV 存在 `dcp_rank == i % world_size` 的那张卡上，交错存放。不属于本 rank 的位置写 `-1`，被 §6.2 的分支跳过。**这也解释了为什么 `KVCacheManager.__init__` 里会有 `self.block_size *= dcp_world_size`**：逻辑块要覆盖 `world_size × block_size` 个 token。

## 7 真实的两种物理布局

这一节直接回应主线文档里「只多一次间址」这个过于简化的说法。**代价取决于布局，而 vLLM 里有两套不同的布局。**

### 7.1 布局一：custom paged attention kernel 用的

`csrc/cache_kernels.cu:217` 的注释写得很明确：

```
key_cache   : [num_blocks, num_heads, head_size/x, block_size, x]
value_cache : [num_blocks, num_heads, head_size,   block_size]
```

其中 `x = 16 / sizeof(cache_t)`（`attention_kernels.cuh:195`）。BF16 是 2 B → **`x = 8`**。

**K 和 V 的布局不一样**，K 多了一个「把 `head_size` 拆成 `head_size/x` 组、每组 `x` 个元素」的维度。为什么？

因为一个 thread group 要一次读 **16 字节**（GPU 访存的友好粒度）。把 `x` 个连续的 `head_size` 元素放在最内层，就能让 `physical_block_offset * x` 这个偏移直接落到一段 16 B 对齐的连续内存上：

```cpp
const cache_t* k_ptr = k_cache
    + physical_block_number * kv_block_stride    // 跳到那个物理块
    + kv_head_idx * kv_head_stride               // 跳到那个 KV 头
    + physical_block_offset * x;                 // 跳到那个 token 的 16B 片段
```

V 不需要这个拆分，因为 V 的读取模式是「按 `head_size` 维度累加」，不是「按 token 做点积」。

### 7.2 布局二：FlashAttention / FlashInfer backend 用的

同一个文件 `csrc/cache_kernels.cu:285`，另一个 kernel：

```
key_cache : NHD 或 HND
  NHD = [num_blocks, block_size, num_heads, head_size]
  HND = [num_blocks, num_heads, block_size, head_size]
```

代码里用 `is_contiguous_heads = (head_stride == head_size)` 来区分这两种。

### 7.3 所以「分页的代价」到底是什么

| 说法 | 评价 |
|---|---|
| ~~「只多一次间址」~~ | **过于简化，不成立**。间址次数、访存模式、是否合并，都取决于 `B`、布局、backend |
| 正确说法 | 每个 **(序列, 逻辑块)** 需要一次 `block_table` 查表；块内 `B` 个 token 共享这次查表；此外还有 layout 相关的 stride 乘法和可能的非合并访存 |

摊薄关系（脚本已算）：

| `B` | 一次查表摊到多少 token |
|---|---|
| 4 | 4 |
| 16 | **16** |
| 128 | 128 |

**这就是 `block_size` 的核心权衡**：

```
B 小 → 尾块浪费小（上界 B−1），但查表次数多、每次访存连续段短
B 大 → 查表摊得薄、访存连续段长，但尾块浪费大
```

vLLM 默认 `block_size = 16` 是这个权衡的折中，不是随便取的。

---

# 第三部分 attention 的数学为什么一点都不变

## 8 数学不变性

### 8.1 要证明的命题

**分页只改变 K/V 的物理存放位置，不改变 (a) 取到哪些 K/V、(b) 它们的逻辑顺序、(c) causal mask、(d) 任何一步算术。因此输出逐位相同。**

### 8.2 关键：softmax 的分母跨越**全部**可见位置

decode 第 `t` 步，单个头：

```
s_j = q · k_j / √d          for j = 0..t
w_j = exp(s_j) / Σ_{m=0..t} exp(s_m)      ← 分母是对全部 t+1 个位置求和
out = Σ_{j=0..t} w_j · v_j
```

**分母 `Σ_{m=0..t}` 是一个跨全序列的全局量。** 只要：
- 参与求和的 `j` 集合不变（causal mask 保证 `j ≤ t`）
- 每个 `s_j` 的值不变（同样的 `q`、同样的 `k_j`）

那么 `w_j` 不变，`out` 不变。**物理位置根本不在这个公式里出现。**

### 8.3 数值验证

TinyToy，`H_q=4, H_kv=2, d=2`，GQA 组大小 2 → Q 头 0,1 用 KV 头 0。取序列长 5，最后一个 token 的 query `q = [1.0, 0.5]`，KV 头 0 的数据：

| `t` | `k_t` | `v_t` |
|---|---|---|
| 0 | `[0.5, 1.0]` | `[1, 2]` |
| 1 | `[1.0, 0.0]` | `[3, 4]` |
| 2 | `[0.0, 0.5]` | `[5, 6]` |
| 3 | `[1.0, 1.0]` | `[7, 8]` |
| 4 | `[0.5, 0.5]` | `[9, 10]` |

`√d = √2 ≈ 1.41421`。分页版用 `block_table = [5, 1]`：`t=0..3` 在物理块 5，`t=4` 在物理块 1。

脚本结果：

```
连续版输出 = [5.041254658459493, 6.041254658459493]
分页版输出 = [5.041254658459493, 6.041254658459493]
逐位完全相同（== 判定，不是近似）
```

**不是「误差很小」，是「完全相同」。** 因为浮点运算的**顺序都没变**——kernel 仍然按逻辑顺序 `j = 0,1,2,...` 累加，只是取数地址不同。

## 9 kernel 里是怎么做到的

`csrc/attention/attention_kernels.cuh`，`paged_attention_kernel`。四个关键动作：

### 9.1 拿到本序列的表头

```cpp
const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;   // :202
```

和 §6.1 Python 侧的二维拍平完全对应。

### 9.2 一个 warp 负责一个逻辑块

```cpp
for (int block_idx = start_block_idx + warp_idx;
     block_idx < end_block_idx;
     block_idx += NUM_WARPS) {                                    // :222
    const int64_t physical_block_number =
        static_cast<int64_t>(block_table[block_idx]);              // :252
```

- **查表发生在块粒度**，一个 warp 一次查表，然后处理这个块里的 `B` 个 token。这是 §7.3 摊薄关系的代码依据。
- `static_cast<int64_t>` 不是多余的。源码注释：

  > `The block number is stored in int32. However, we cast it to int64 because int32 can lead to overflow when this variable is multiplied by large numbers (e.g., kv_block_stride).`

  真实模型 `kv_block_stride` 很大，`int32 × 大数` 会溢出。这是一个真实的工程坑。

### 9.3 逻辑下标用于 mask，物理块号用于取数

```cpp
const int physical_block_offset = (thread_group_idx + i * WARP_SIZE) % BLOCK_SIZE;
const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;   // :263 ← 逻辑位置
...
const bool mask = token_idx >= seq_len;                                 // :297
logits[token_idx - start_token_idx] = mask ? 0.f : qk;                   // :298
```

**两个坐标系严格分工**：

| 用途 | 用哪个坐标 |
|---|---|
| 算访存地址 | `physical_block_number`（物理） |
| 算 causal mask、写 `logits` 下标 | `token_idx = block_idx × B + offset`（**逻辑**） |

`logits` 数组按**逻辑位置**索引。所以不管物理块怎么乱序，`logits[0..seq_len-1]` 里躺着的永远是逻辑顺序的分数。**这是不变性在代码层面的落点。**

### 9.4 softmax 跨 warp 归约 → 全局

```cpp
qk_max = fmaxf(qk_max, fmaxf(...));           // 线程组内
for (int mask = WARP_SIZE/2; mask >= THREAD_GROUP_SIZE; mask /= 2)
    qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));   // warp 内 :309
if (lane == 0) red_smem[warp_idx] = qk_max;
__syncthreads();
qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;          // :319
for (int mask = NUM_WARPS/2; mask >= 1; mask /= 2)
    qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));   // 跨 warp :321
```

`qk_max` 先在 warp 内归约，再通过 shared memory **跨所有 warp** 归约，得到**整个序列**的最大值。`exp_sum` 同理。

**所以：不同 warp 处理不同物理块，但它们贡献到同一个 softmax 分母。** 这就是 §8.2 那个「全局分母」在硬件上的实现。分页没有把 attention 切成独立的块级 attention。

## 10 反例：如果按块分别 softmax 再相加，会错多少

这一节用数值说明「为什么不能图省事按块算」。

错误做法：块 0（`t=0..3`）内部做一次 softmax 得 `out₀`，块 1（`t=4`）内部做一次 softmax 得 `out₁`，然后平均。

用 §8.3 完全相同的数据，脚本结果：

| 做法 | 输出 |
|---|---|
| **正确**（全局 softmax） | `[5.041255, 6.041255]` |
| **错误**（按块 softmax 后平均） | `[6.607260, 7.607260]` |
| 最大绝对误差 | **1.566005** |

误差不是小数点后的噪声，是**量级上的错**。原因很直白：

- 块 1 只有 1 个 token，它内部 softmax 的权重必然是 `1.0`，于是 `out₁ = v₄ = [9, 10]`。
- 正确算法里 `t=4` 的权重只是 5 个位置中的一个，远小于 1。
- 错误做法把「最后一个块」的权重放大到了 `1/块数`，完全脱离了真实注意力分布。

**块数越多、尾块越空，这个错误越离谱。** 这也说明为什么 §9.4 的跨 warp 归约是必需的，不是优化。

> 顺带区分清楚：**FlashAttention 的分块是沿 softmax 的在线归约（online softmax）做的，它用 running max / running sum 精确地合成分母，数学上等价于全局 softmax。** 那和这里的「按块独立 softmax 再平均」是两件完全不同的事。前者对，后者错。

## 11 分页没有省掉的东西

诚实清单。分页是**显存管理**技术，不是**计算或带宽**优化。

| 项目 | 分页是否改变 |
|---|---|
| attention 的 FLOPs | **不变**。同样的点积、同样的 softmax |
| 每步 decode 要读的 KV 字节数 | **不变**。`L` 个 token 的 KV 一个字节都不能少读 |
| 每步 decode 要读的权重字节数 | **不变**。全部权重仍然每步读一遍 |
| 权重的 HBM 带宽下界 | **不变** |
| KV 显存的**浪费** | **减少**（消除预留浪费 + 外碎片） |
| 可并发的请求数 | **增加**（同样显存装更多有效 KV） |
| 跨请求共享前缀 | **新增能力**（第五部分） |
| 访存地址计算 | **增加**（查表 + stride 乘法，§7.3） |
| kernel 复杂度 | **增加** |

**一句话**：分页让你**装得下更多请求**，从而把 batch 做大，从而摊薄权重读取——**吞吐的提升是「batch 变大」带来的，不是「分页本身让计算变快」**。这个因果链不能跳。

---

# 第四部分 生命周期与 vLLM 实现

## 12 三个真实数据结构

### 12.1 `KVCacheBlock`：一个块的全部元数据

`kv_cache_utils.py:170`，字段就这几个：

```python
block_id: int                                    # 0 .. num_gpu_blocks-1
ref_cnt: int = 0                                 # 引用计数
_block_hash: Optional[BlockHashWithGroupId] = None    # 只有满块且被缓存时才有
prev_free_block: Optional["KVCacheBlock"] = None  # 空闲链表前驱
next_free_block: Optional["KVCacheBlock"] = None  # 空闲链表后继
is_null: bool = False                            # 是否是 null block
```

三个关键观察：

1. **没有「属于哪个请求」字段。** 归属关系存在管理器的 `req_to_blocks: dict[str, list[KVCacheBlock]]` 里，不在块上。这正是共享的前提——一个块可以被多个请求的 list 同时引用。
2. **`ref_cnt` 和 `block_hash` 是两个独立维度。** 一个块可以 `ref_cnt = 0`（没人用）但 `block_hash` 有值（内容还可复用）。这就是「僵尸块」，§22 详述。
3. **`block_hash` 的 setter 带断言**：

   ```python
   @block_hash.setter
   def block_hash(self, block_hash):
       assert self.block_hash is None, "The block already has a hash. This should not happen."
   ```

   一个块的 hash 只能被设置一次，必须先 `reset_hash()` 才能再设。这保证了「块内容 ↔ hash」的绑定不会悄悄改变。

### 12.2 `FreeKVCacheBlockQueue`：手写双向链表

`kv_cache_utils.py:216`。**为什么不用 Python 的 `deque`？** 源码注释直接回答：

> `We implement this class instead of using Python builtin deque to support removing a block in the middle of the queue in O(1) time.`

因为 prefix caching 命中一个僵尸块时，要把它**从队列中间**摘走（§22 的 `touch`）。`deque` 做不到 O(1) 中间删除。

实现细节：
- 链表节点就是 `KVCacheBlock` 自己的 `prev_free_block` / `next_free_block` 字段，**不额外分配任何 Python 对象**（注释明说是为了追上 C++ 实现的 deque 性能）。
- 有 `fake_free_list_head` / `fake_free_list_tail` 两个哨兵（`block_id = -1`），保证真实块永远有前驱和后继，消掉边界分支。
- `num_free_blocks` 单独维护，`popleft` / `remove` / `append` 各自增减。

**队列的语义顺序**（源码注释）：

> `1. The least recent used block is at the front (LRU).`
> `2. If two blocks have the same last accessed time (allocated by the same sequence), the one with more hash tokens (the tail of a block chain) is at the front.`

即：**队首 = 最先被淘汰**。而且同一条 hash 链里，**尾块比头块先淘汰**——因为保住头块才能保住「前缀」的价值（§17）。

### 12.3 `BlockHashToBlockMap`：hash → 块

`block_pool.py:21`。类型是：

```python
dict[BlockHashWithGroupId, Union[KVCacheBlock, dict[int, KVCacheBlock]]]
```

两个设计点：

1. **`Union` 是为了减少 GC 压力**（注释 NOTE #2）。绝大多数 hash 只对应一个块，就直接存块对象，不建内层 dict；只有出现重复才升级成 dict。
2. **不去重**（注释 NOTE #1）：

   > `We currently don't de-duplicate the blocks in the cache... This is because we want to make sure the allocated block IDs won't change so that block tables are append-only.`

   §24 展开。

## 13 完整调用链

一次调度里，KV 相关的调用顺序（全部可在源码中逐行核对）：

```
Scheduler.schedule()
│
├─ [新请求] KVCacheManager.get_computed_blocks(request)          kv_cache_manager.py:154
│     └─ coordinator.find_longest_cache_hit(block_hashes, num_tokens-1)
│           └─ FullAttentionManager.find_longest_cache_hit()      single_type_...:252
│                 └─ BlockPool.get_cached_block(hash, group_ids)  block_pool.py:163
│                       └─ BlockHashToBlockMap.get_one_block()
│
└─ KVCacheManager.allocate_slots(request, num_new_tokens, ...)    kv_cache_manager.py:193
      ├─ coordinator.remove_skipped_blocks()      ← sliding window 才有事做
      ├─ coordinator.get_num_blocks_to_allocate()                 single_type_...:59
      ├─ if 需要 > BlockPool.get_num_free_blocks():  return None   ← 准入闸门
      ├─ BlockPool.touch(new_computed_blocks)                     block_pool.py:322
      ├─ coordinator.save_new_computed_blocks()                   single_type_...:88
      ├─ coordinator.allocate_new_blocks()                        single_type_...:109
      │     └─ BlockPool.get_new_blocks(k)                        block_pool.py:257
      │           ├─ FreeKVCacheBlockQueue.popleft_n(k)
      │           ├─ _maybe_evict_cached_block(blk)  ← 惰性淘汰      block_pool.py:286
      │           └─ blk.ref_cnt += 1
      └─ coordinator.cache_blocks(request, num_tokens_to_cache)
            └─ BlockPool.cache_full_blocks()                      block_pool.py:188

[请求结束] KVCacheManager.free(request)                           kv_cache_manager.py:306
      └─ SingleTypeKVCacheManager.free()                          single_type_...:156
            └─ BlockPool.free_blocks(reversed(req_blocks))        block_pool.py:338

[GPU 侧] BlockTable.append_row(block_ids, row_idx)                block_table.py:49
         BlockTable.compute_slot_mapping(req_indices, positions)  block_table.py:76
         BlockTable.commit_block_table / commit_slot_mapping  ← 拷到 GPU
         reshape_and_cache_flash_kernel(...)   ← 写 K/V           cache_kernels.cu:282
         paged_attention_kernel(...)           ← 读 K/V           attention_kernels.cuh:85
```

**注意 CPU / GPU 的分工**：上面整条链**全在 CPU 上用 Python 跑**，产出两张表（`block_table`、`slot_mapping`），然后 `commit_*` 拷到 GPU。GPU kernel 只认这两张整数表，完全不知道「请求」「hash」「引用计数」这些概念。**块管理是 CPU 的事，GPU 只负责按表访存。**

### 13.1 `allocate_slots` 的四段布局

源码里画了这张图（`kv_cache_manager.py:221`），值得原样理解：

```
-----------------------------------------------------------------------
| < computed > | < new computed > |    < new >    | < pre-allocated > |
-----------------------------------------------------------------------
|                  < required >                   |
--------------------------------------------------
|                    < full >                  |
------------------------------------------------
                                  | <new full> |
                                  --------------
```

| 段 | 含义 |
|---|---|
| `computed` | 之前已经算过、已持有块的 token |
| `new computed` | **本次 prefix cache 新命中**的 token |
| `new` | 本次真的要算的 token |
| `pre-allocated` | 投机解码预留（`num_lookahead_tokens`，如 EAGLE 的草稿 token） |
| `full` | 满块的部分，只有这部分会被 hash 并缓存 |

`pre-allocated` 那段解释了为什么 `allocate_slots` 有 `num_lookahead_tokens` 参数：投机解码要先把草稿 token 的槽位占上，但**它们不能被缓存**（可能被拒绝）。所以有这一行：

```python
num_tokens_to_cache = min(num_computed_tokens + num_new_tokens, request.num_tokens)
```

注释说得很清楚：只缓存「finalized」的 token。

## 14 TinyToy 逐步追踪

现在把上面所有机制串成一个可以手算的完整故事。池 8 块，`B = 4`，`block_id 0` 是 null block（§15），可用块 1..7。

### 14.1 初始状态

```
blocks:      0(null) 1  2  3  4  5  6  7
ref_cnt:     -       0  0  0  0  0  0  0
free_queue:  [1, 2, 3, 4, 5, 6, 7]      num_free = 7
cache:       {}
usage:       1 - 7/7 = 0.0
```

### 14.2 A 进来，prompt 长 3

```
需要块数 = ⌈3/4⌉ = 1
get_new_blocks(1) → popleft → 块1，ref_cnt: 0→1
A.block_table = [1]
free_queue = [2,3,4,5,6,7]              num_free = 6
```

**此时 A 没有任何 hash。** 因为 `cache_blocks` 算的是 `num_full_blocks = 3 // 4 = 0`。**不满一块，不进缓存。**

```
cache = {}
```

### 14.3 A decode 到长度 4

```
需要块数 = ⌈4/4⌉ = 1，已有 1 → 新增 0     ← 不领新块
num_full_blocks = 4 // 4 = 1 → 块1 满了
cache_full_blocks: 块1.block_hash = hA0
cache = {hA0: {1: 块1}}
```

| 时刻 | A 长度 | 块数 | free | cache 大小 |
|---|---|---|---|---|
| 刚进来 | 3 | 1 | 6 | **0** |
| decode 一步 | 4 | 1 | 6 | **1** |

**这一步是理解 prefix caching 粒度的关键**：块满了才有 hash，才可能被别人复用。

### 14.4 A 长到 5

```
需要块数 = ⌈5/4⌉ = 2，已有 1 → 新增 1
get_new_blocks(1) → 块2，ref_cnt 0→1
A.block_table = [1, 2]
free_queue = [3,4,5,6,7]                num_free = 5
```

slot_mapping（脚本断言）：

| `t` | 逻辑块 | 偏移 | 物理块 | slot |
|---|---|---|---|---|
| 0 | 0 | 0 | 1 | 4 |
| 1 | 0 | 1 | 1 | 5 |
| 2 | 0 | 2 | 1 | 6 |
| 3 | 0 | 3 | 1 | 7 |
| 4 | 1 | 0 | **2** | **8** |

### 14.5 B 进来，前 4 个 token 和 A 相同

```
B.block_hashes = [hA0, ...]        ← hash 链算出同一个 hA0
get_computed_blocks → find_longest_cache_hit → 命中块1
touch([块1]):  块1.ref_cnt 1 → 2   （ref≠0，不在空闲队列里，无需摘除）
B 还需要 ⌈5/4⌉ - 1(命中) - 0(已有) = 1 块
get_new_blocks(1) → 块3
B.block_table = [1, 3]             ← 和 A 共享物理块 1
free_queue = [4,5,6,7]              num_free = 4
```

三本账（脚本断言）：

| 量 | 数值 |
|---|---|
| 在用物理块 | 3（块 1, 2, 3） |
| A+B 有效 token | `5 + 5 = 10` |
| 已分配槽位 | `3 × 4 = 12` |
| 如果**不共享**需要块数 | `2 + 2 = 4` |
| 共享省下 | **1 块 = 128 B** |

**注意「有效 10 / 已分配 12」这个比值在共享下会失真**：块 1 里的 4 个 token 被两个请求各算一次「有效」。共享场景下更该看的是**物理块占用**，而不是有效 token 比值。这是三本账之外的第四种视角。

### 14.6 A 结束

`SingleTypeKVCacheManager.free` 的关键一行：

```python
ordered_blocks = reversed(req_blocks)      # ← 逆序！尾块先释放
self.block_pool.free_blocks(ordered_blocks)
```

`free_blocks` 做两件事：

```python
for block in blocks_list: block.ref_cnt -= 1          # 全部先减
self.free_block_queue.append_n([b for b in blocks_list
                                if b.ref_cnt == 0 and not b.is_null])   # 只有归零的入队
```

追踪：

| 块 | 减之前 ref | 减之后 ref | 是否入空闲队列 |
|---|---|---|---|
| 块2（尾，先处理） | 1 | **0** | **是** |
| 块1（头，后处理） | 2 | **1** | 否（B 还在用） |

```
free_queue = [4,5,6,7,2]        ← 块2 追加在尾部
num_free = 5
块1.ref_cnt = 1，hash 仍是 hA0，B 继续正常读
```

**为什么要逆序？** 因为入队顺序 = 淘汰优先级的反序（队首先淘汰）。逆序释放让**尾块排在头块前面**，于是淘汰时先丢尾块、后丢头块。保住头块 = 保住可复用的前缀。这正是 §12.2 那条 LRU 规则的实现手段。

## 15 null block：8 块池子只有 7 块能用

这是读源码才能发现、而且**直接修正主线文档**的一个细节。

`BlockPool.__init__`（`block_pool.py:154`）：

```python
# To represent a placeholder block with block_id=0.
# The ref_cnt of null_block is not maintained, needs special care to
# avoid freeing it.
self.null_block = self.free_block_queue.popleft()
self.null_block.is_null = True
```

**无条件执行，与 `enable_caching` 无关。** 后果：

| 量 | 理想算术模型 | 真实 vLLM |
|---|---|---|
| `num_gpu_blocks` | 8 | 8 |
| 可分配块 | 8 | **7** |
| token 容量 | 32 | **28** |
| 单块请求最大并发 | 8 | **7** |

脚本验证：`Pool(8)` 上 `get_new_blocks(7)` 成功且 `num_free` 归 0，再要 1 块直接 `raise ValueError`。

`get_usage()` 也相应减 1（`block_pool.py:400`）：

```python
total_gpu_blocks = self.num_gpu_blocks - 1      # Subtract 1 to account for null block.
return 1.0 - (self.get_num_free_blocks() / total_gpu_blocks)
```

**null block 是干什么的？** 给 sliding window / chunked local attention 当占位符。窗口外的块被释放后，`block_table` 那个位置不能留空洞（表是定长矩阵，且逻辑下标要能算），就填 null block 的 id。`find_longest_cache_hit` 的文档里有例子：

> `sliding window manager should return a list like ([NULL, NULL, KVCacheBlock(7), KVCacheBlock(8)]) for block size 4 and sliding window 8`

所以 full attention 用不到它，但**它照样占着一块**。

> **对主线文档的修正**：§3.6 第七步 / 第八步说的「8 并发」，在真实 vLLM 的 8 块池里应该是 **7**。主线文档的 8 块模型是为了算术干净而做的理想化，本身没错，但落到真实实现要减 1。真实模型 10,240 块 → 10,239 可用，相对误差 0.01%，可以忽略；玩具模型 8 块 → 7 块，相对误差 12.5%，**不能忽略**。

## 16 准入闸门与抢占

### 16.1 闸门就一个 if

`kv_cache_manager.py:271`：

```python
if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
    return None
```

**返回 `None` 就是「这一步装不下」。** 注意 `get_num_free_blocks()` 返回的是**空闲队列长度**，而队列里**包含僵尸块**（§22）。所以这个闸门比「真正空白的块」宽松——这是好事，僵尸块本来就可以随时征用。

### 16.2 `None` 之后发生什么：抢占

`scheduler.py:254` 起，RUNNING 请求的分配在一个 `while True` 里：

```python
while True:
    new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens, ...)
    if new_blocks is None:
        # 默认 FCFS 策略：running 按到达顺序排，pop() 取队尾 = 最晚进来的
        # PRIORITY 策略下换成 max(self.running, key=lambda r: (r.priority, r.arrival_time))
        preempted_req = self.running.pop()          # ← 弹出队尾
        self.kv_cache_manager.free(preempted_req)
        preempted_req.status = RequestStatus.PREEMPTED
        preempted_req.num_computed_tokens = 0       # ← 归零！
        self.waiting.prepend_request(preempted_req)
        if preempted_req == request:
            can_schedule = False                    # 抢占到自己头上，放弃
            break
        continue                                    # 重试分配
```

四个要点：

| 要点 | 含义 |
|---|---|
| `self.running.pop()` | 弹**队尾** = 最晚进来的 = 优先级最低。不是随机踢 |
| `num_computed_tokens = 0` | **RECOMPUTE 策略**：已算的全部作废，重新排队从头 prefill |
| `waiting.prepend_request` | 插到等待队列**头部**，下次优先恢复 |
| `preempted_req == request` | 如果轮到自己也抢不到，说明池子真的满了，本步不调度 |

**`num_computed_tokens = 0` 是 thrash 的根源**：被抢占的请求下次回来要重新 prefill 整个 prompt。如果负载持续超载，就会出现「A 抢占 B → B 回来抢占 C → C 回来抢占 A」的循环，算力全花在重算上。

**但这里有一个 prefix caching 带来的缓解**：被抢占请求的块虽然 `ref_cnt` 归零，**hash 还在**（成了僵尸块）。它重新排队时 `get_computed_blocks` 可能命中自己刚留下的僵尸块，从而跳过大部分 prefill。**能命中多少，取决于这些僵尸块有没有被别人淘汰掉。** 这是 prefix caching 一个不太被提及的实际收益。

这条路径可以在源码里闭合验证——`scheduler.py:380`：

```python
# Get already-cached tokens.
if request.num_computed_tokens == 0:
    new_computed_blocks, num_new_local_computed_tokens = \
        self.kv_cache_manager.get_computed_blocks(request)
```

抢占恰好把 `num_computed_tokens` 置成 `0`（`:276`），**所以被抢占的请求恢复时必然会走一次缓存查找**。两处代码合起来构成完整链路：`free` 不清 hash（§22.1）→ 置 0 → 查缓存 → 可能命中自己的僵尸块。

### 16.3 `get_num_blocks_to_allocate` 的精确算法

`single_type_kv_cache_manager.py:59`：

```python
num_required_blocks = cdiv(num_tokens, self.block_size)
num_new_blocks = (num_required_blocks - len(new_computed_blocks)
                  - len(self.req_to_blocks[request_id]))
num_evictable_computed_blocks = sum(
    blk.ref_cnt == 0 and not blk.is_null for blk in new_computed_blocks)
return num_new_blocks + num_evictable_computed_blocks
```

第三项 `num_evictable_computed_blocks` 很微妙，§23 专门讲。

---

# 第五部分 Prefix Caching

## 17 为什么必须是「前缀」，不能是任意子串

### 17.1 因果注意力的约束

token `t` 的 K/V 是这样算出来的：

```
h_t^(0) = embedding(token_t)
h_t^(l) = TransformerLayer_l( h_t^(l-1),  {h_j^(l-1) : j ≤ t} )
k_t^(l), v_t^(l) = 投影( h_t^(l-1) )
```

**`h_t` 依赖于 `t` 之前的所有 token**（通过 attention）。所以：

> **`k_t` / `v_t` 的值，由「从位置 0 到位置 t 的整个 token 序列」唯一决定，而不是由 `token_t` 单独决定。**

举例：`token_t = "bank"` 这个 token 本身一样，但

```
"river bank"      的 bank → K/V 是一组值
"investment bank" 的 bank → K/V 是另一组值
```

**所以能复用的单位必须是「从位置 0 开始的一段完整前缀」，不能是中间任意一段。** 这不是工程限制，是数学约束。

### 17.2 hash 链把这个约束编码进去

vLLM 的做法（`kv_cache_utils.py:547`）：

```python
def hash_block_tokens(hash_function, parent_block_hash, curr_block_token_ids, extra_keys=None):
    if not parent_block_hash:
        parent_block_hash = NONE_HASH
    curr_block_token_ids_tuple = tuple(curr_block_token_ids)
    return BlockHash(hash_function(
        (parent_block_hash, curr_block_token_ids_tuple, extra_keys)))
```

**哈希的输入是一个三元组**：`(父块哈希, 本块 token ids, 额外键)`。

递归展开：

```
h₀ = H(NONE_HASH, tokens[0:B],   extra)
h₁ = H(h₀,        tokens[B:2B],  extra)
h₂ = H(h₁,        tokens[2B:3B], extra)
```

`h₂` 里嵌套了 `h₁`，`h₁` 里嵌套了 `h₀` → **`h₂` 唯一确定了 `tokens[0:3B]` 的全部内容**。

脚本验证的两个性质：

| 性质 | 验证 |
|---|---|
| 尾块 token 完全相同、前缀不同 → hash 不同 | `[1,2,3,4, 9,9,9,9]` vs `[5,6,7,8, 9,9,9,9]`，第 2 块 hash **不同** |
| 第一块的父哈希是 `NONE_HASH` | 断言通过 |

**这就是「为什么只能复用前缀」在代码里的落点**：hash 链让「前缀不同」必然导致「hash 不同」，从而不可能误命中。

### 17.3 `extra_keys`：三个必须区分的维度

`kv_cache_utils.py:517`：

```python
extra_keys = lora_extra_keys + mm_extra_keys + cache_salt_keys
```

| 来源 | 为什么必须进 hash |
|---|---|
| **LoRA ID** | 同样的 token，挂不同 LoRA adapter，权重不同 → K/V 不同 |
| **多模态输入哈希** | 同样的占位 token，图片/音频不同 → K/V 不同 |
| **`cache_salt`** | 显式隔离。只在**第一个块**加入（`start_token_idx == 0`），但因为链式传递，**影响后续所有块** |

`cache_salt` 是多租户场景的安全边界：不同租户传不同 salt，即使 prompt 完全相同也不会互相命中，避免通过缓存命中时序侧信道推断别人的 prompt。

脚本验证：同样 4 个 token，`extra=None` vs `extra=("lora-7",)` → hash **不同**。

### 17.4 hash 是什么时候算的

**不是命中查找时算，是请求创建和追加 token 时就算好了。** `block_pool.py:202` 注释：

> `The block hashes values are computed by the Request object immediately when it is created and when new tokens are appended.`

`get_request_block_hasher`（`kv_cache_utils.py:576`）增量计算：

```python
start_token_idx = len(request.block_hashes) * block_size    # 从已有链尾接着算
prev_block_hash_value = request.block_hashes[-1] if request.block_hashes else None
while True:
    end_token_idx = start_token_idx + block_size
    if end_token_idx > num_tokens:
        break                        # ← We only hash full blocks
    ...
```

`start_token_idx = len(block_hashes) × B` 这一行说明链是**增量延长**的，不重算已有的。

## 18 只有满块才有 hash

上面那个 `break` 就是全部规则。后果：

| 序列长度（`B=4`） | 满块数 `L//4` | 有 hash 的 token | 无 hash 的尾部 token |
|---|---|---|---|
| 3 | 0 | **0** | 3 |
| 4 | 1 | 4 | 0 |
| 5 | 1 | 4 | **1** |
| 7 | 1 | 4 | **3** |
| 8 | 2 | 8 | 0 |

**尾块永远不进缓存。** 因为它还会变——下一个 token 就要写进去，内容变了 hash 就失效了。给一个会变的块算 hash 毫无意义。

`cache_blocks`（`single_type_kv_cache_manager.py:133`）：

```python
num_cached_blocks = self.num_cached_block[request.request_id]
num_full_blocks = num_tokens // self.block_size        # ← 向下取整
self.block_pool.cache_full_blocks(..., num_cached_blocks, num_full_blocks, ...)
self.num_cached_block[request.request_id] = num_full_blocks
```

`num_cached_block` 记录「已经 hash 过几块」，避免重复 hash。`cache_full_blocks` 里还有 `assert blk.block_hash is None`——配合 §12.1 的 setter 断言，双重保证一个块不会被 hash 两次。

**实际影响**：`B = 16` 时，一个 prompt 无论多长，末尾最多 15 个 token 永远无法被复用。短 prompt 场景下这个损失比例可观：prompt 长 20，`B=16` → 只有 16 个 token 可缓存，**4 个（20%）不行**。

## 19 命中查找：沿链走，一 miss 即停

`FullAttentionManager.find_longest_cache_hit`（`single_type_kv_cache_manager.py:252`）：

```python
max_num_blocks = max_length // block_size
for block_hash in itertools.islice(block_hashes, max_num_blocks):
    # block_hashes is a chain of block hashes. If a block hash is not
    # in the cached_block_hash_to_id, the following block hashes are
    # not computed yet for sure.
    if cached_block := block_pool.get_cached_block(block_hash, kv_cache_group_ids):
        for computed, cached in zip(computed_blocks, cached_block):
            computed.append(cached)
    else:
        break
if use_eagle and computed_blocks[0]:
    for computed in computed_blocks:
        computed.pop()
```

四点：

1. **`break` 而不是 `continue`。** 注释解释了正确性：hash 链意味着「第 `i` 块没命中 → 第 `i+1` 块不可能命中」。因为第 `i+1` 块的 hash 里嵌了第 `i` 块的 hash，如果第 `i` 块的内容从没被缓存过，第 `i+1` 块的 hash 也就不可能存在于缓存中。**所以命中一定是连续前缀，不会出现「命中块 0 和块 2 但跳过块 1」。**
2. **`max_num_blocks = max_length // block_size`** — 向下取整，§20 的坑之一。
3. **`use_eagle` 时 `pop()` 掉最后一个命中块** — 投机解码需要最后一块的 hidden states 给 draft head，必须重算。
4. **多 KV cache group 时要求全部 group 都命中**（`get_cached_block` 里任一 group miss 就 `return None`）。

另外 `get_computed_blocks` 开头还有一个直接跳过的条件（`kv_cache_manager.py:169`）：

```python
if (not self.enable_caching
        or (request.sampling_params is not None
            and request.sampling_params.prompt_logprobs is not None)):
    return self.create_empty_block_list(), 0
```

**请求了 `prompt_logprobs` 就完全不用 prefix caching。** 因为要返回每个 prompt token 的 logprob，就必须真的跑一遍前向拿到 logits，缓存命中跳过计算就拿不到了。

## 20 两个真实的坑：明明缓存了，却要重算一整块

这是本文最想讲清楚、而主线文档 §3.7 完全没提的机制。

### 20.1 坑的来源：两行代码

**第一行**（`kv_cache_manager.py:174`）：

```python
# NOTE: When all tokens hit the cache, we must recompute the last token
# to obtain logits. Thus, set max_cache_hit_length to prompt_length - 1.
# This can trigger recomputation of an entire block, rather than just
# the single last token, because allocate_slots() requires
# num_computed_tokens to be block-size aligned. Removing this limitation
# could slightly improve performance in the future.
max_cache_hit_length = request.num_tokens - 1
```

**第二行**（`single_type_kv_cache_manager.py:271`）：

```python
max_num_blocks = max_length // block_size        # 向下取整
```

两行叠加的效果：

```
命中 token 数 = min( (num_tokens - 1) // B ,  已缓存满块数 ) × B
```

**为什么必须留一个 token 不算？** 因为模型的输出是「下一个 token 的 logits」，它来自**最后一个 prompt token** 的前向。如果全部命中、一个 token 都不算，就没有任何前向发生，拿不到 logits，无法采样出第一个输出 token。

**为什么这会损失一整块？** 因为 `num_computed_tokens` 必须块对齐（命中是以整块为单位的），`(num_tokens - 1) // B` 的向下取整会把「差一个 token 满块」的情况整块砍掉。

### 20.2 具体数值（脚本全部断言）

`B = 4`，假设**前缀全部已在缓存中**：

| prompt 长度 | `num_tokens - 1` | `// 4` | 命中 token | **必须重算** |
|---|---|---|---|---|
| 4 | 3 | 0 | **0** | **4（全部！）** |
| 5 | 4 | 1 | 4 | 1 |
| 7 | 6 | 1 | 4 | 3 |
| **8** | **7** | **1** | **4** | **4（整整一块）** |
| 9 | 8 | 2 | 8 | **1** |
| 16 | 15 | 3 | 12 | 4 |
| 17 | 16 | 4 | 16 | **1** |

**看第 8 行和第 9 行**：

```
prompt 长 8：缓存里有全部 2 块，却只命中 1 块 → 重算 4 个 token
prompt 长 9：缓存里有 2 块，命中 2 块       → 重算 1 个 token
```

**prompt 多一个 token，重算量从 4 降到 1。** 长度恰好是 `B` 的整数倍时最吃亏。

真实模型 `B = 16`：

| prompt 长度 | 命中 | 重算 |
|---|---|---|
| 2048（= 128×16） | **2032** | **16** |
| 2049 | 2048 | **1** |

**一个 2048 token 的 prompt 完全命中缓存，仍然要 prefill 16 个 token。** 这不是 bug，源码注释明确承认这是当前限制（`Removing this limitation could slightly improve performance in the future`）。

### 20.3 实践含义

| 场景 | 影响 |
|---|---|
| 系统提示词长度恰好是 `B` 的倍数 | 每次命中都白算 `B` 个 token |
| 命中率统计 | `prefix_cache_stats.hits` 记的是 `num_new_computed_tokens`，已扣除这部分，所以**统计是诚实的** |
| 想优化 | 把 prompt 长度调成 `kB + 1` 而不是 `kB`（如果业务允许） |

## 21 命中省了什么、没省什么

主线文档这里已经修正过一次（用户指出「即使命中 cache 这个 cache 也是占用了显存的」，是对的）。这里给完整版本：

| 成本项 | 命中后 | 说明 |
|---|---|---|
| 命中部分的 **prefill FLOPs** | **真的省了** | 唯一真正归零的项 |
| 命中部分的 **prefill 激活显存** | **真的省了** | 不算就没有激活 |
| **新增 KV 显存** | **0** | 复用已有物理块，`ref_cnt += 1` |
| **KV 池占用** | **照常占** | 那些块本来就在池子里，不会因为命中而释放 |
| **decode 时读 KV 的 HBM 带宽** | **照常读** | 每个 decode step 都要读全部可见 KV |
| **decode 时读权重的 HBM 带宽** | **照常读** | 与缓存无关 |
| **计入 TPM / 计费 token** | **照常计** | 命中 token 仍是请求的 input token |
| **KV 池的准入名额** | **可能要占** | 命中的是僵尸块时要占，§23 |
| **首 token 延迟（TTFT）** | **降低** | prefill 变短 |

**「零成本」只能指 prefill FLOPs 这一项。** 说「cached token 免费」是错的。

### 21.1 命中率与 KV 水位的两种运行态

主线文档提过需要限定，这里说清楚：

| 运行态 | 命中率↑ 的效果 |
|---|---|
| **池子未满**（有空闲块） | 更多请求能被接纳 → 在用块↑ → **KV 水位↑** |
| **池子已满**（空闲块 ≈ 0） | 命中率↑ 让新请求少领块 → 同样池子装更多请求 → **KV 水位不变（已满），但并发↑、吞吐↑** |

所以「命中率↑ → KV 水位↑」只在**未满**时成立。满载时水位已经顶到 100%，命中率的作用是**提高单位显存的请求吞吐**，不是继续抬水位。

## 22 refcount、僵尸块、惰性淘汰、LRU

这是 prefix caching 实现里最精巧的部分。

### 22.1 核心洞察：释放 ≠ 清除

`free_blocks`（`block_pool.py:338`）只做两件事：减 `ref_cnt`、把归零的块**追加进空闲队列**。

**它不清除 `block_hash`，也不从 `cached_block_hash_to_block` 里删除。**

于是出现「僵尸块」：

```
ref_cnt = 0          ← 没有任何请求在用
block_hash = hX0     ← 内容仍然有效、仍然可查
在空闲队列里          ← 随时可以被征用
K/V 数据还在显存里    ← 没被擦除
```

**这是有意的设计**：请求结束了，但它算出来的前缀对别人可能还有用。留着，直到真的有人需要这块显存。

### 22.2 完整实验（脚本逐步断言）

占满全部 7 个可用块，全部缓存，然后全部释放：

```python
p2 = Pool(8)
X = p2.get_new_blocks(7)                      # 块 1..7
p2.cache_full(X, ["hX0".."hX6"], 0, 7)
p2.free_blocks(reversed(X))
```

结果：

| 观测 | 值 |
|---|---|
| 各块 `ref_cnt` | 全 **0** |
| 各块 `block_hash` | **全部保留** `hX0..hX6` |
| `num_free_blocks` | **7** |
| `get_usage()` | **0.0** |
| `get_cached("hX0")` | **仍能查到块 1** |
| 空闲队列顺序 | `[7, 6, 5, 4, 3, 2, 1]` |

**两个反直觉但正确的点**：

1. **`usage = 0.0`，但缓存里有 7 个块的内容。** 「KV cache 使用率」这个指标**把僵尸块算作空闲**。所以看到 `gpu_cache_usage` 低，不代表缓存里没东西。
2. **队列顺序是分配顺序的逆序** `[7,6,...,1]`。因为 `reversed` 释放 + 追加队尾。块 7 是链尾 → 最先淘汰；块 1 是链头 → 最后淘汰。**保住前缀。**

### 22.3 命中僵尸块 = 复活

`touch`（`block_pool.py:322`）：

```python
for block in blocks_per_group:
    # ref_cnt=0 means this block is in the free list (i.e. eviction
    # candidate), so remove it.
    if block.ref_cnt == 0 and not block.is_null:
        self.free_block_queue.remove(block)      # ← O(1) 中间摘除
    block.ref_cnt += 1
```

**这就是 §12.2 手写双向链表的唯一理由。** 追踪：

| 步骤 | 块1 ref | 空闲队列 | `num_free` |
|---|---|---|---|
| 释放后 | 0 | `[7,6,5,4,3,2,1]` | 7 |
| `touch([块1])` | **1** | `[7,6,5,4,3,2]` | **6** |

块 1 从队列**中间**被摘走，`num_free` 减 1。**它从「可淘汰」变回「在用」，K/V 一个字节都没动。**

### 22.4 惰性淘汰（lazy eviction）

淘汰**不在释放时发生**，而在**别人领块时**发生。`get_new_blocks`（`block_pool.py:257`）：

```python
ret = self.free_block_queue.popleft_n(num_blocks)
if self.enable_caching:
    for block in ret:
        self._maybe_evict_cached_block(block)     # ← 淘汰就在这里
        assert block.ref_cnt == 0
        block.ref_cnt += 1
```

`_maybe_evict_cached_block`（`block_pool.py:286`）：

```python
block_hash = block.block_hash
if block_hash is None: return False              # 没 hash，不用淘汰
if self.cached_block_hash_to_block.pop(block_hash, block.block_id) is None:
    return False
block.reset_hash()                               # 清掉 hash
return True
```

追踪（接 §22.3，队列 `[7,6,5,4,3,2]`）：

```
get_new_blocks(1) → popleft → 块7
  块7.block_hash = hX6 → 从 cache 里 pop 掉 → reset_hash()
  块7.ref_cnt = 1
```

| 观测 | 值 |
|---|---|
| 被征用的块 | **7**（队首 = LRU） |
| 块7 的 hash | **None** |
| `get_cached("hX6")` | **None**（淘汰了） |
| `get_cached("hX0")` | **块1**（还在） |
| cache 大小 | 7 → **6** |

**为什么惰性更好？** 如果释放时就清除，那「请求结束 → 立刻有相同前缀的新请求进来」这个极常见的模式就完全命中不了。惰性淘汰把「清除」推迟到**真的有显存压力**的那一刻，最大化复用窗口。

### 22.5 从未用过的块比缓存块先被淘汰

另一个实验（脚本断言）：

```
Pool(8)，领 2 块（块1,2），缓存，释放
领块后队列:  [3,4,5,6,7]
释放后队列:  [3,4,5,6,7,2,1]     ← 释放的块追加在尾部
再领 1 块 → 块3                ← 不是块2！
cache 大小仍为 2                ← 缓存完好
```

**因为「释放的块追加到队尾」，所以队列里天然形成三段优先级**：

```
[从未用过的块]  →  [较久前释放的块]  →  [最近释放的块]
   最先淘汰                                最后淘汰
```

从未用过的块没有缓存价值，优先消耗它们，**缓存被完整保留**。这是「append 到队尾」这个看似平凡的选择带来的实际收益。

### 22.6 三种块状态

| 状态 | `ref_cnt` | `block_hash` | 在空闲队列 | 能否被命中 | 能否被征用 |
|---|---|---|---|---|---|
| **在用** | `> 0` | 有（满块）或无 | 否 | **能** | 否 |
| **僵尸** | `0` | **有** | **是** | **能** | **能** |
| **空白** | `0` | 无 | 是 | 否 | 能 |
| null | 不维护 | 无 | 否 | 否 | 否 |

`reset_prefix_cache`（`block_pool.py:355`）要求**所有块都已释放**才能成功：

```python
num_used_blocks = self.num_gpu_blocks - self.get_num_free_blocks()
if num_used_blocks != 1:      # The null block is always marked as used
    logger.warning(...); return False
```

那个 `!= 1` 又是 null block。**用途**：RLHF 里权重更新后，旧 K/V 全部失效，必须整体作废。

## 23 命中僵尸块也要占准入名额

回到 §16.3 那个微妙的第三项：

```python
num_evictable_computed_blocks = sum(
    blk.ref_cnt == 0 and not blk.is_null for blk in new_computed_blocks)
return num_new_blocks + num_evictable_computed_blocks
```

源码注释：

> `If a computed block of a request is an eviction candidate (in the free queue and ref_cnt == 0), it will be changed from a free block to a computed block when the request is allocated, so we also count it as needed to be allocated.`

**逻辑**：命中一个僵尸块，`touch` 会把它从空闲队列摘走 → `num_free` 减 1。**从池子的角度看，这和领一个新块一样消耗了一个空闲名额。** 所以准入检查必须把它算进去，否则会超分配。

脚本验证（`B=4`，请求 8 token，命中 2 块）：

| 命中的块状态 | `num_new_blocks` | `num_evictable` | **总需求** |
|---|---|---|---|
| 2 个都是**僵尸**（ref=0） | `2 − 2 − 0 = 0` | **2** | **2** |
| 2 个都**在用**（ref=1） | `2 − 2 − 0 = 0` | **0** | **0** |

**同样命中 2 块，一个要占 2 个名额，一个占 0 个。** 这就是为什么「命中率高」不等于「一定能调度进来」——如果命中的全是僵尸块而池子已经紧绷，照样可能被拒。

这也修正了一个可能的误解：**prefix caching 不是无条件降低显存压力**。它降低的是「新增块的领取」，但命中僵尸块时压力仍在。真正零压力的命中，是命中**别的活跃请求正在用的块**（§14.5 的 A/B 共享）。

## 24 不去重：同一个 hash 可以对应多个物理块

`BlockHashToBlockMap` 的 NOTE #1：

> `We currently don't de-duplicate the blocks in the cache, meaning that if a block becomes full and is cached, we don't check if there is already an identical block in the cache. This is because we want to make sure the allocated block IDs won't change so that block tables are append-only.`

脚本验证：

```python
Y = p3.get_new_blocks(1); p3.cache_full(Y, ["dup"], 0, 1)   # 块1
Z = p3.get_new_blocks(1); p3.cache_full(Z, ["dup"], 0, 1)   # 块2
p3.cache["dup"] → {1: 块1, 2: 块2}      ← 两个块，同一个 hash
get_cached("dup") → 块1                 ← 返回第一个
```

**为什么会出现重复？** 两个请求同时进来，前缀相同，但都还没缓存过。各自 `get_computed_blocks` 都 miss，各自领块、各自算、各自满块后 `cache_full_blocks`。于是同内容两份。

**为什么不去重？** 因为去重意味着「把请求 A 的 `block_table` 里的块号改成 B 的块号」，这违反 **append-only**（§5.3）。而 append-only 是 GPU 侧增量更新的前提——`BlockTable.append_row` 只往行尾写，不改动已写入的部分：

```python
def append_row(self, block_ids, row_idx):
    start = self.num_blocks_per_row[row_idx]
    self.num_blocks_per_row[row_idx] += num_blocks
    self.block_table.np[row_idx, start:start + num_blocks] = block_ids
```

如果允许改中间，就得处理「GPU 正在用这一行读 K/V」的同步问题。**vLLM 选择用一点显存冗余换取简单性和无锁。**

代价：并发冷启动时会有重复块，浪费一些显存。收益：`block_table` 永远只在尾部增长。

## 25 Prefix caching 的完整成本收益表

| 维度 | 收益 | 代价 |
|---|---|---|
| prefill 算力 | 命中部分归零 | 尾块 + §20 的对齐损失仍要算 |
| TTFT | 显著降低 | — |
| KV 显存 | 活跃请求间共享，真省 | 僵尸块占着池子（但可被征用） |
| 冷启动重复块 | — | 不去重带来的冗余（§24） |
| CPU 开销 | — | 每请求算 hash 链、查表、维护链表 |
| 命中率不确定性 | — | 依赖负载的前缀重复度，无重复则纯开销 |
| 抢占恢复 | 可能命中自己的僵尸块，少重算 | 僵尸块可能已被淘汰 |
| 多租户隔离 | `cache_salt` 提供隔离 | 加 salt 就无法跨租户复用 |

**什么时候 prefix caching 收益最大**：长系统提示词 + 高并发 + 多轮对话。三者都指向「大量请求共享同一长前缀」。

**什么时候几乎没收益**：每个请求 prompt 都完全不同（如批量翻译互不相关的短句）。此时 hash 计算和链表维护是纯开销，`enable_prefix_caching=False` 更合适。

---

# 第六部分 放大到真实模型

## 26 Llama-3-70B 对照

### 26.1 规格与账本

| 量 | 计算 | 结果 |
|---|---|---|
| KV / token | `2 × 80 × 8 × 128 × 2 B` | `327,680 B = 320 KiB` |
| KV 池 | 假设 | `50 GiB` |
| `block_size` | vLLM 默认 | `16` |
| 每块字节 | `16 × 327,680` | `5,242,880 B = 5 MiB` |
| 块数 | `50 GiB / 5 MiB` | `10,240` |
| 可用块（减 null） | `10,240 − 1` | `10,239` |
| token 容量 | `10,240 × 16` | `163,840` |

### 26.2 两个必须纠正的推理

**错误推理一**：「100 个 2048-token 请求，分页后能装下」

```
100 × 2048 × 327,680 B = 65,536,000,000 B = 62.5 GiB  >  50 GiB
```

**根本装不下。** 需要 `100 × ⌈2048/16⌉ = 100 × 128 = 12,800` 块 > 10,240 块。分页消除的是浪费，**不是把池子变大**。

**错误推理二**：「80 个请求刚好填满，它们都能继续 decode」

```
80 × ⌈2048/16⌉ = 80 × 128 = 10,240 块    ← 刚好填满，空闲 0
```

每个请求长到 2049：

```
80 × ⌈2049/16⌉ = 80 × 129 = 10,320 块  >  10,240
```

**多出 80 块，一块都拿不到。** 此时 §16.2 的抢占必然发生。**「刚好填满」是一个不稳定状态**：任何请求再 decode 一个 token 就会触发抢占。

实际部署要留 headroom，不能把 `gpu_memory_utilization` 调到刚好装满目标并发。

### 26.3 「16×」是容量比，不是加速比

```
2048 / 128 = 16
```

这个 16 的含义是：**如果按 `max_len = 2048` 预留，一个请求的预留额度够 16 个实际长 128 的请求用。**

| 它是 | 它不是 |
|---|---|
| KV **容量**维度的并发上界之比 | 吞吐提升 16 倍 |
| 假设「实际长度 128、预留 2048」这个特定对照 | 端到端延迟改善 16 倍 |
| 只考虑 KV 一个维度 | 忽略了算力、带宽、`max_num_seqs` 等其他闸门 |

**真实吞吐提升取决于负载**：如果本来就都是长请求（实际长度接近 `max_len`），预留浪费本来就小，分页收益也小。而且并发上去之后，瓶颈可能立刻转移到算力或带宽——那是主线文档第 8 章「四维水位取 max」要处理的事。

## 27 常见误解清单

| # | 误解 | 纠正 | 依据 |
|---|---|---|---|
| 1 | 分页把利用率提到 99% | 尾块上界 `B−1` 只约束「已分配容量内的空隙」，不说明池子有多满，也不说明装不装得下 | §4.2 |
| 2 | 75% 和 37.5% 矛盾 | 分母不同：已分配容量 vs 整个池 | §0.3 |
| 3 | 不用分页就必然浪费 60%+ | 按 `max_len` 预留是**一种基线策略**，不是非分页实现的通性 | §2.1 |
| 4 | 分页无条件比连续分配省 | 静态精确连续分配可能更省；分页的价值在**动态增长不搬家** + **无外碎片** + **可共享** | §2.4 |
| 5 | 分页只多一次间址 | 每 (序列, 块) 一次查表，摊到 `B` 个 token；还有 layout 相关 stride 和合并性问题，两套布局不同 | §7.3 |
| 6 | 分页让 attention 变快 | FLOPs、KV 读字节、权重读字节**全不变**。吞吐提升来自 batch 变大 | §11 |
| 7 | 8 块池能放 8 个并发 | 真实 vLLM 里 null block 占掉块 0，只有 **7** | §15 |
| 8 | 8 并发都能自由长到 `max_len` | 只能各自在块内长到 4；想到 5 必须等新块 | §4.3 |
| 9 | 100×2048 分页后装得下 | 62.5 GiB > 50 GiB，装不下 | §26.2 |
| 10 | 刚好填满还能继续 decode | 80 请求全长到 2049 需 10,320 块 > 10,240，必然抢占 | §26.2 |
| 11 | 16× 是吞吐提升 | 是 KV **容量**维度的并发比 | §26.3 |
| 12 | cached token 零成本 | 只有 prefill FLOPs 归零；KV 显存、decode 带宽、TPM 照常 | §21 |
| 13 | 命中率↑ 一定让 KV 水位↑ | 只在池子未满时成立 | §21.1 |
| 14 | 整块缓存了就能整块跳过 | `(num_tokens−1) // B` 会砍掉一整块，长 2048 全命中仍要算 16 个 | §20 |
| 15 | 命中就不占显存名额 | 命中僵尸块要占（`num_evictable_computed_blocks`） | §23 |
| 16 | `usage` 低说明缓存空 | 僵尸块被算作空闲，`usage=0` 时缓存可能是满的 | §22.2 |
| 17 | 可以复用任意相同子串 | 因果注意力下 K/V 依赖全部前缀，hash 链强制前缀匹配 | §17 |
| 18 | 释放块就清除了缓存 | 释放只减 ref + 入队；淘汰是**领块时惰性**发生的 | §22.4 |
| 19 | 分页 = OS 分页，有缺页和 swap | 没有缺页中断、V1 默认无 swap（RECOMPUTE）、无 COW、无 TLB | §3.3 |
| 20 | 按块 softmax 再合并等价 | 数值反例最大误差 1.566，量级错误 | §10 |

---

# 第七部分 对照实现：SGLang 的 RadixCache

## 28 先厘清：哪些是「代码」，哪些是「模型」

这一节先把上一版里容易被误读的地方钉死。

| 位置 | 性质 | 本次处理 |
|---|---|---|
| §12–§13、§16、§19–§20 的源码片段 | 逐行摘自 **vLLM v0.11.0** | 保留，附录 B 补正一处路径 |
| §14 的 TinyToy 生命周期 | 附录 A 的**语义等价移植**，不是 import vLLM 执行 | 保留；§34 用 SGLang 走同一道题 |
| §22 的僵尸块 / 惰性淘汰实验 | 同上，是 `BlockPool` 的重写 | 保留 |
| 「125 项断言全部通过」 | 计数错误：附录 A 实际只有 **79** 个 `assert` | 已改 |
| `vllm/v1/core/block_table.py` | 路径错误：实际在 `vllm/v1/worker/block_table.py` | 已改 |

**必须说清的一点**：TinyToy 的数字不是 vLLM 跑出来的。它是把 `KVCacheBlock` / `FreeKVCacheBlockQueue` / `BlockPool` 的语义等价重写成约 150 行 Python 后跑出来的。这样做的好处是不用 GPU、可以手算；代价是读者无法仅凭数字确认「移植是否忠实」。所以下面直接把两套实现的源码贴出来，逐段对应；附录 E 再给一个可执行对照模型。

**§14 的每一步仍然能在源码里找到依据**，不是凭空的算术：

| 原文 §14 步骤 | 源码位置 | 关键条件 |
|---|---|---|
| A 长 3，不缓存 | `single_type_kv_cache_manager.py:133` | `num_full_blocks = 3 // 4 = 0` |
| A 长 4，缓存块 1 | `block_pool.py:188` `cache_full_blocks` | `num_full_blocks = 4 // 4 = 1` |
| A 长 5，领块 2 | `single_type_kv_cache_manager.py:109` | `cdiv(5,4) - 1 = 1` |
| B 命中块 1 | `single_type_kv_cache_manager.py:252` | `islice` 沿 hash 链走，miss 即 `break` |
| A 结束，逆序释放 | `single_type_kv_cache_manager.py:168` | `ordered_blocks = reversed(req_blocks)` |
| 块 1 `ref=1`，不入空闲队列 | `block_pool.py:338` | 只 append `ref_cnt == 0` 的块 |
| 块 0 被 null block 占掉 | `block_pool.py:157` | `self.null_block = self.free_block_queue.popleft()` |

**不能从这些数字推出的东西**，原文 §4.2 / §26.2 已经列过；这里只补一条：**TinyToy 的「8 块」是 `num_gpu_blocks` 的理想算术，不是 `get_usage` 的分母。** 真实 `get_usage()` 的分母是 `num_gpu_blocks − 1`（`block_pool.py:400`）：8 块池的初始值是 `1 − 7/7 = 0`，7 块可用块全部分配时才是 `1 − 0/7 = 1.0`。

**两套设计的一句话版本**：

| | vLLM v0.11.0 | SGLang v0.5.2 |
|---|---|---|
| 一句话 | 固定块 + 块级哈希链 + 双向空闲链表 | token 页 + 基数树 + 叶子 LRU 堆 |
| 前缀的 key | 块内容的哈希（父哈希链式嵌套） | 原始 token 序列（树路径） |
| 前缀的 value | 物理块号 | KV slot 下标 |
| 每请求映射 | `block_table`：逻辑块号 → 物理块号 | `req_to_token`：token 位置 → KV slot |

两套实现解决的是同一个问题：**在「KV 必须按 token 连续逻辑顺序参与 attention」和「物理显存只能按固定单位分配」之间做地址翻译，并把重复前缀的物理 KV 复用出去。** 差异在于它们选择了不同的中间层：vLLM 选择「块表」，SGLang 选择「token 索引 + 基数树」。

**源码怎么复现**（只取需要的目录，不拉全仓库）：

```bash
git clone --depth 1 --branch v0.11.0 --filter=blob:none --sparse \
  https://github.com/vllm-project/vllm.git vllm-0.11.0
cd vllm-0.11.0 && git sparse-checkout set vllm/v1/core vllm/v1/worker csrc

git clone --depth 1 --branch v0.5.2 --filter=blob:none --sparse \
  https://github.com/sgl-project/sglang.git sglang-0.5.2
cd sglang-0.5.2 && git sparse-checkout set \
  python/sglang/srt/mem_cache python/sglang/srt/managers
```

本文引用的所有 `:行号` 都以这两个 tag 为准。

---

## 29 SGLang 的两级内存结构

### 29.1 `ReqToTokenPool`：一行就是一个请求的 token→slot 表

`python/sglang/srt/mem_cache/memory_pool.py:55`：

```python
class ReqToTokenPool:
    """A memory pool that maps a request to its token locations."""

    def __init__(self, size, max_context_len, device, enable_memory_saver):
        ...
        self.req_to_token = torch.zeros(
            (size, max_context_len), dtype=torch.int32, device=device
        )
        self.free_slots = list(range(size))

    def write(self, indices, values):
        self.req_to_token[indices] = values

    def alloc(self, need_size: int) -> List[int]:
        if need_size > len(self.free_slots):
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return select_index

    def free(self, free_index: Union[int, List[int]]):
        if isinstance(free_index, (int,)):
            self.free_slots.append(free_index)
        else:
            self.free_slots.extend(free_index)
```

关键点：

1. **`req_to_token` 是 `int32` 的二维定长矩阵**，一行一个请求，列是 token 在该请求里的逻辑位置，值是 KV 池里的**槽号**。
2. 它存的**不是块号**。第 `i` 列直接就是第 `i` 个 token 的 KV 在池子里的下标，attention 拿到这一行就能 gather，不需要 `block_id * block_size + offset` 的二次换算。
3. 前缀命中后，SGLang 把命中的槽号**直接写进新请求的行**（`schedule_batch.py:1193`）：

```python
if pre_len > 0:
    self.req_to_token_pool.write(
        (req.req_pool_idx, slice(0, pre_len)), req.prefix_indices
    )
```

这就是「共享」在 SGLang 里的全部动作：**把同一批槽号复制进另一个请求的行**。没有引用计数块、没有 block table 的物理块号。

### 29.2 `MHATokenToKVPool`：扁平的 per-layer KV 池

`memory_pool.py:449`：

```python
# [size, head_num, head_dim] for each layer
# The padded slot 0 is used for writing dummy outputs from padded tokens.
self.k_buffer = [
    torch.zeros(
        (self.size + self.page_size, self.head_num, self.head_dim),
        dtype=self.store_dtype,
        device=self.device,
    )
    for _ in range(self.layer_num)
]
self.v_buffer = [...]
```

和 vLLM 的「块张量」不同，SGLang 的池子是**按 token 铺平**的：`[size + page_size, head_num, head_dim]`。第 0 行是 padding 专用槽，和 vLLM 的 null block 一样属于「无条件保留的 0 号位」，但用途不同：

| | vLLM | SGLang |
|---|---|---|
| 0 号位 | null block，给 sliding window / local attention 当占位符 | padding slot，给 padded token 写 dummy 输出 |
| 可用量 | `num_gpu_blocks − 1` | `size`（索引起始为 1） |

### 29.3 allocator：`free_pages` 栈 + `release_pages` 延迟回收

`allocator.py:118`：

```python
class TokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    def clear(self):
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.free_pages = torch.arange(
            1, self.size + 1, dtype=torch.int64, device=self.device
        )
        ...
        self.release_pages = torch.empty((0,), dtype=torch.int64, device=self.device)

    def alloc(self, need_size: int):
        if self.need_sort and need_size > len(self.free_pages):
            self.merge_and_sort_free()
        if need_size > len(self.free_pages):
            return None
        select_index = self.free_pages[:need_size]
        self.free_pages = self.free_pages[need_size:]
        return select_index

    def free(self, free_index: torch.Tensor):
        ...
        if self.need_sort:
            self.release_pages = torch.cat((self.release_pages, free_index))
        else:
            self.free_pages = torch.cat((self.free_pages, free_index))
```

三点：

1. **分配单位是 token**（`page_size = 1` 时）。`free_pages` 就是一个可用槽号栈，`alloc(n)` 取前 `n` 个。
2. **`free` 不一定立刻回到 `free_pages`**。`need_sort` 为真时先进 `release_pages`，等 `merge_and_sort_free()` 再合并排序。这和 vLLM 的「释放只减 ref、淘汰延迟到领块时」是同一个思想：**释放动作与可复用动作解耦**。
3. `page_size > 1` 时换成 `PagedTokenToKVPoolAllocator`（`allocator.py:419`）：

```python
self.num_pages = size // page_size
...
def alloc(self, need_size: int):
    num_pages = need_size // self.page_size
    ...
    out_pages = self.free_pages[:num_pages]
    self.free_pages = self.free_pages[num_pages:]
    out_indices = (
        out_pages[:, None] * self.page_size
        + torch.arange(self.page_size, device=self.device)
    ).reshape(-1)
    return out_indices
```

它仍返回**扁平的槽号数组**，只是保证一个请求拿到的页是整页对齐、页内连续的。

### 29.4 与 vLLM `block_table` 的根本区别

| | vLLM | SGLang |
|---|---|---|
| 每请求映射 | `block_table[row, logical_block] = physical_block` | `req_to_token[row, position] = slot` |
| slot 从哪来 | GPU 侧 `physical_block * B + offset` | 直接就是数组里的值 |
| 分配单位 | 固定 `B` token 的块 | `page_size` 个 token（默认 1） |
| 映射的稳定性 | **append-only**，已写入的块号不改 | 每次 chunk / 命中后可以整段重写 |
| 共享的动作 | `ref_cnt += 1`，块号不变 | 把同一批槽号复制进另一行 |
| 0 号位 | null block | padding slot |

**「映射是否可变」是后面所有设计差异的根**：

- vLLM 的 `block_table` 要增量拷到 GPU，且 GPU 可能正在读这一行，所以块号一旦写下就不能改 → 不能去重、不能把两个同内容块合并。
- SGLang 的 `req_to_token` 是纯 CPU 侧的行，重写一行只影响下一次 forward 的 gather 索引 → 可以自由 split、合并、复制。

---

## 30 基数树：`RadixCache` 的数据结构

### 30.1 `TreeNode`

`radix_cache.py:43`：

```python
class TreeNode:
    def __init__(self, id: Optional[int] = None):
        self.children = defaultdict(TreeNode)
        self.parent: TreeNode = None
        self.key: List[int] = None
        self.value: Optional[torch.Tensor] = None
        self.lock_ref = 0
        self.last_access_time = time.monotonic()

        self.hit_count = 0
        # indicating the node is locked to protect from eviction
        self.host_ref_counter = 0
        # store the host indices of KV cache
        self.host_value: Optional[torch.Tensor] = None
        # store hash values of each pages
        self.hash_value: Optional[List[str]] = None

    @property
    def evicted(self):
        return self.value is None

    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time
```

**`key` 和 `value` 是同一段 token 的两种表示**：

```
key   = [token_id, ...]        这条边覆盖的 token 序列
value = [kv_slot,  ...]        这些 token 的 KV 在池子里的槽号，长度与 key 相同
```

树根 `root.key = []`、`root.value = []`、`root.lock_ref = 1`（`radix_cache.py:158` 附近）。根本身永远不淘汰。

### 30.2 子节点怎么索引：按「第一页」分叉

`radix_cache.py:138`：

```python
if self.page_size == 1:
    self.key_match_fn = _key_match_page_size1
    self.get_child_key_fn = lambda key: key[0]
else:
    self.key_match_fn = partial(_key_match_paged, page_size=page_size)
    self.get_child_key_fn = lambda key: tuple(key[:page_size])
```

- `page_size = 1`：每个 token 一个分叉键，树可以按 token 粒度共享。
- `page_size = 4`：每个节点至少一页，分叉键是「前 4 个 token 的 tuple」。**这是后端 kernel 的页对齐要求，不是树的要求。**

`_key_match_paged` 只按整页比较：

```python
def _key_match_paged(key0, key1, page_size):
    min_len = min(len(key0), len(key1))
    i = 0
    while i < min_len:
        if key0[i : i + page_size] != key1[i : i + page_size]:
            break
        i += page_size
    return i
```

所以 `page_size > 1` 时，**匹配长度天然是页的整数倍**，不会出现「共享 3 个 token、第 4 个不共享」的中间状态。

---

## 31 命中：`match_prefix`

`radix_cache.py:158`：

```python
def match_prefix(self, key: List[int], **kwargs) -> MatchResult:
    if self.disable or len(key) == 0:
        return MatchResult(...)

    if self.page_size != 1:
        page_aligned_len = len(key) // self.page_size * self.page_size
        key = key[:page_aligned_len]

    value, last_node = self._match_prefix_helper(self.root_node, key)
    if value:
        value = torch.cat(value)
    else:
        value = torch.empty((0,), dtype=torch.int64, device=self.device)
    return MatchResult(
        device_indices=value,
        last_device_node=last_node,
        last_host_node=last_node,
    )
```

`_match_prefix_helper`（`radix_cache.py:367`）：

```python
def _match_prefix_helper(self, node: TreeNode, key: List):
    node.last_access_time = time.monotonic()
    child_key = self.get_child_key_fn(key)
    value = []
    while len(key) > 0 and child_key in node.children.keys():
        child = node.children[child_key]
        child.last_access_time = time.monotonic()
        prefix_len = self.key_match_fn(child.key, key)
        if prefix_len < len(child.key):
            new_node = self._split_node(child.key, child, prefix_len)
            value.append(new_node.value)
            node = new_node
            break
        else:
            value.append(child.value)
            node = child
            key = key[prefix_len:]
            if len(key):
                child_key = self.get_child_key_fn(key)
    return value, node
```

逐步拆开：

1. **页对齐**：`len(key) // page_size * page_size`。这是 `(num_tokens−1)//B` 在 SGLang 里的等价物 —— 只匹配整页。
2. **按第一页找孩子**：`get_child_key_fn(key)` 取 `key[0]` 或 `tuple(key[:page_size])`。
3. **逐节点匹配**：`prefix_len` 是该节点 key 与查询 key 的公共前缀长度（页对齐）。
4. **部分命中就分裂**：如果 `prefix_len < len(child.key)`，说明查询只命中了这个节点的前半段。`_split_node` 把原节点一分为二，前半段成为新节点，查询返回新节点。
5. **一 miss 即停**：`while` 条件不满足就退出。和 vLLM 的 `break` 一样，命中永远是连续前缀。
6. **返回值是槽号**：`value` 是每个命中节点的 `value` 拼接，最终是 `[kv_slot, ...]`，直接写进 `req.prefix_indices`。

### 31.1 为什么必须分裂

设 `page_size = 4`，树里已经有一条边：

```
root -- key=[1,2,3,4,5,6,7,8] value=[11..18] --> leaf
```

来了一个查询 `[1,2,3,4,9]`：

- 第一页 `(1,2,3,4)` 命中。
- 第二页 `(5,6,7,8)` vs `(9,...)` 不匹配 → `prefix_len = 4 < 8`。
- `_split_node` 把原边拆成：
  ```
  root -- key=[1,2,3,4] value=[11,12,13,14] --> new_node
  new_node -- key=[5,6,7,8] value=[15,16,17,18] --> old leaf
  ```
- 查询返回 `[11,12,13,14]`，`last_node = new_node`。

**分裂不复制 KV 数据**，只改 `key` / `value` 的切片边界。原 leaf 的 `value` 变成 `[15..18]`，新节点的 `value` 是 `[11..14]`，物理槽号一个都没动。这是基数树能支持「任意长度前缀共享」的关键。

---

## 32 写入：`insert` / `cache_finished_req` / `cache_unfinished_req`

### 32.1 `_insert_helper`

`radix_cache.py:411`：

```python
def _insert_helper(self, node: TreeNode, key: List, value):
    node.last_access_time = time.monotonic()
    if len(key) == 0:
        return 0

    child_key = self.get_child_key_fn(key)
    total_prefix_length = 0
    while len(key) > 0 and child_key in node.children.keys():
        node = node.children[child_key]
        node.last_access_time = time.monotonic()
        prefix_len = self.key_match_fn(node.key, key)
        total_prefix_length += prefix_len
        key = key[prefix_len:]
        value = value[prefix_len:]

        if prefix_len < len(node.key):
            new_node = self._split_node(node.key, node, prefix_len)
            node = new_node

        if len(key):
            child_key = self.get_child_key_fn(key)

    if len(key):
        new_node = TreeNode()
        new_node.parent = node
        new_node.key = key
        new_node.value = value
        node.children[child_key] = new_node
        self.evictable_size_ += len(value)
    return total_prefix_length
```

两个语义要点：

1. **返回值 `total_prefix_length` 是「插入前已经存在的公共前缀长度」**，不是「插入后的总长度」。调用方靠它判断哪些 KV 槽是重复的、可以立刻释放。
2. **只有真正新增的后缀才 `evictable_size_ += len(value)`**。已存在的节点不重复计入。`value` 的所有权从请求转移给树节点 —— 树「持有一份引用」，这就是 `cache_finished_req` 里那句注释 `Radix Cache takes one ref in memory pool` 的含义。

### 32.2 `cache_finished_req`：请求结束时把前缀交给树

`radix_cache.py:203`（删去 disable 分支和空行）：

```python
def cache_finished_req(self, req: Req):
    token_ids = (req.origin_input_ids + req.output_ids)[:-1]
    kv_indices = self.req_to_token_pool.req_to_token[
        req.req_pool_idx, : len(token_ids)
    ]

    if self.page_size != 1:
        page_aligned_len = len(kv_indices) // self.page_size * self.page_size
        page_aligned_kv_indices = kv_indices[:page_aligned_len].to(
            dtype=torch.int64, copy=True
        )
        self.token_to_kv_pool_allocator.free(kv_indices[page_aligned_len:])
    else:
        page_aligned_len = len(kv_indices)
        page_aligned_kv_indices = kv_indices.to(dtype=torch.int64, copy=True)

    # Radix Cache takes one ref in memory pool
    new_prefix_len = self.insert(
        token_ids[:page_aligned_len], page_aligned_kv_indices
    )
    self.token_to_kv_pool_allocator.free(
        kv_indices[len(req.prefix_indices) : new_prefix_len]
    )

    self.req_to_token_pool.free(req.req_pool_idx)
    self.dec_lock_ref(req.last_node)
```

四步，每一步都对应一个易错点：

| 步骤 | 代码 | 含义 |
|---|---|---|
| 1 | `(origin_input_ids + output_ids)[:-1]` | **最后一个 token 没有 KV**，必须丢掉。对应 vLLM 的 `num_tokens - 1` |
| 2 | `free(kv_indices[page_aligned_len:])` | 页对齐之外的尾巴立刻还给分配器，不进树 |
| 3 | `insert(...)` | 页对齐的前缀交给树；树从此持有这些槽号 |
| 4 | `free(kv_indices[len(prefix_indices):new_prefix_len])` | 这次重算出来、但树里已经有的**重复槽号**立刻释放，避免泄漏 |

第 4 步是最容易看漏的：`insert` 返回的是「已有前缀长度」，所以 `[len(req.prefix_indices), new_prefix_len)` 这一段是「本次算出来的、但和树里重复的」槽号。树只保留一份，多出来的必须 free。

### 32.3 `cache_unfinished_req`：chunked prefill 的中间态

`radix_cache.py:240`（删去 disable 分支）：

```python
def cache_unfinished_req(self, req: Req, chunked=False):
    token_ids = req.fill_ids
    kv_indices = self.req_to_token_pool.req_to_token[
        req.req_pool_idx, : len(token_ids)
    ]

    if self.page_size != 1:
        page_aligned_len = len(kv_indices) // self.page_size * self.page_size
        page_aligned_kv_indices = kv_indices[:page_aligned_len].to(
            dtype=torch.int64, copy=True
        )
    else:
        page_aligned_len = len(kv_indices)
        page_aligned_kv_indices = kv_indices.to(dtype=torch.int64, copy=True)
    page_aligned_token_ids = token_ids[:page_aligned_len]

    # Radix Cache takes one ref in memory pool
    new_prefix_len = self.insert(
        page_aligned_token_ids, page_aligned_kv_indices, chunked=chunked
    )
    self.token_to_kv_pool_allocator.free(
        kv_indices[len(req.prefix_indices) : new_prefix_len]
    )

    # The prefix indices could be updated, reuse it
    new_indices, new_last_node, _, _ = self.match_prefix(page_aligned_token_ids)
    self.req_to_token_pool.write(
        (req.req_pool_idx, slice(len(req.prefix_indices), len(new_indices))),
        new_indices[len(req.prefix_indices) :],
    )

    self.dec_lock_ref(req.last_node)
    self.inc_lock_ref(new_last_node)

    if self.page_size != 1:
        req.prefix_indices = torch.cat(
            [new_indices, kv_indices[len(new_indices) :]]
        )
    else:
        req.prefix_indices = new_indices
    req.last_node = new_last_node
```

和 `cache_finished_req` 的差别：

- 请求还没结束，所以 `fill_ids` 全部参与（没有 `[:-1]`）。
- 插入后**重新 `match_prefix`**，因为 `insert` 可能刚分裂了节点，`last_node` 变了。
- 把新命中的槽号写回 `req_to_token` 行，然后 `dec` 旧锁、`inc` 新锁。
- `req.prefix_indices` 更新为「树里命中的部分 + 还没进树的尾巴」。

**调用时机很重要**：`cache_unfinished_req` 只在 **prefill / chunk 结束**时调用（`scheduler_output_processor_mixin.py:94`），**不是每个 decode step 都调用**。decode 过程中树不更新，直到请求结束才由 `cache_finished_req` 一次性插入。这和 vLLM 每个调度步 `allocate_slots → cache_blocks` 的节奏不同，也是 §34 里 A 的 prompt 取 4 而不是 3 的原因。

### 32.4 与 vLLM「不去重」的对照

vLLM 的 `BlockHashToBlockMap` 有一条明确的 NOTE #1：

> We currently don't de-duplicate the blocks in the cache... This is because we want to make sure the allocated block IDs won't change so that block tables are append-only.

SGLang 则相反：`_split_node` 会把一条边拆开，`cache_finished_req` 会把重复的 KV 槽立刻 free。原因在 §29.4 已经说了：

```
vLLM: block_table 要增量拷 GPU，块号不能改 → 不去重
SGLang: req_to_token 是 CPU 侧的可变行 → 可以 split / 去重
```

**这不是谁更先进，而是映射结构的约束不同。** vLLM 用一点显存冗余换「块表 append-only + 无锁增量更新」；SGLang 用树操作换「按 token/页粒度去重」。

---

## 33 锁定与淘汰

### 33.1 `lock_ref`：沿路径保护前缀

`radix_cache.py:319`：

```python
def inc_lock_ref(self, node: TreeNode):
    if self.disable:
        return 0
    delta = 0
    while node != self.root_node:
        if node.lock_ref == 0:
            self.evictable_size_ -= len(node.value)
            self.protected_size_ += len(node.value)
            delta -= len(node.value)
        node.lock_ref += 1
        node = node.parent
    return delta

def dec_lock_ref(self, node: TreeNode):
    if self.disable:
        return 0
    delta = 0
    while node != self.root_node:
        if node.lock_ref == 1:
            self.evictable_size_ += len(node.value)
            self.protected_size_ -= len(node.value)
            delta += len(node.value)
        node.lock_ref -= 1
        node = node.parent
    return delta
```

**锁是沿「节点 → 根」的整条路径加的**，不是只锁叶子。因为要保住一个前缀，必须同时保住它的所有祖先。`lock_ref` 从 0 变 1 时，该节点的 KV 从 `evictable_size_` 移到 `protected_size_`；减回 0 时再移回来。

这和 vLLM 的 `ref_cnt` 是同一个角色：

| | vLLM | SGLang |
|---|---|---|
| 计数对象 | 物理块 | 树节点（一段 token） |
| 加锁动作 | `touch(blocks)`：`ref_cnt += 1`，从空闲队列摘除 | `inc_lock_ref(node)`：沿路径 `lock_ref += 1` |
| 解锁动作 | `free_blocks`：`ref_cnt -= 1`，归零入队 | `dec_lock_ref(node)`：沿路径 `lock_ref -= 1` |
| 保护单位 | 单块 | 路径上的所有节点 |

### 33.2 `evict`：叶子最小堆，按 `last_access_time` 淘汰

`radix_cache.py:294`：

```python
def evict(self, num_tokens: int):
    if self.disable:
        return

    leaves = self._collect_leaves()
    heapq.heapify(leaves)

    num_evicted = 0
    while num_evicted < num_tokens and len(leaves):
        x = heapq.heappop(leaves)

        if x == self.root_node:
            break
        if x.lock_ref > 0:
            continue

        self.token_to_kv_pool_allocator.free(x.value)
        num_evicted += len(x.value)
        self._delete_leaf(x)

        if len(x.parent.children) == 0:
            heapq.heappush(leaves, x.parent)

        self._record_remove_event(x)
```

四个要点：

1. **只有叶子可淘汰**。内部节点还被至少一个孩子引用，删掉它会把孩子的前缀打断。
2. **堆的排序键是 `last_access_time`**，`TreeNode.__lt__` 定义了它。`match_prefix` / `insert` 都会刷新路径上的时间戳。
3. **`lock_ref > 0` 直接跳过**。被活跃请求占用的前缀不会被淘汰。
4. **删掉叶子后，如果父节点变成新叶子，把父节点压回堆**。这样一次 `evict` 可以连续回收一整条链。

`_delete_leaf`（`radix_cache.py:462`）只做两件事：从父节点的 `children` 里摘掉自己、`evictable_size_ -= len(node.key)`。**它不碰 `value` 指向的 KV 数据** —— 数据已经在上一行 `free(x.value)` 里还给分配器了。

### 33.3 什么时候触发淘汰

`schedule_batch.py:1785`：

```python
def _evict_tree_cache_if_needed(self, num_tokens: int):
    ...
    if self.token_to_kv_pool_allocator.available_size() < num_tokens:
        if self.tree_cache is not None:
            self.tree_cache.evict(num_tokens)
```

`alloc_token_slots`（`schedule_batch.py:979`）在真正 `alloc` 之前先调用它：

```python
def alloc_token_slots(self, num_tokens: int, backup_state: bool = False):
    self._evict_tree_cache_if_needed(num_tokens)
    ...
    out_cache_loc = self.token_to_kv_pool_allocator.alloc(num_tokens)
```

所以 SGLang 的淘汰时机是：**分配前，先看够不够；不够就按 LRU 淘汰树叶子，再分配。**

vLLM 的淘汰时机是：**分配中，`get_new_blocks` 弹出的块如果是缓存块，顺手 `_maybe_evict_cached_block`。** 两者都是「只有真的需要显存时才清除」，区别只是「先腾后分」还是「边分边腾」。

---

## 34 同一道题的 SGLang 追踪

为了和 §14 对照，沿用同一组请求、同样的 `page_size = 4`，但池子缩小到 **8 个可用槽**（槽 0 保留给 padding，可用 1..8），这样能真正触发淘汰。

> **和 §14 的一处必要差异**：§14 里 A 的 prompt 是 3，靠 decode 补满第 4 个 token。SGLang 的树**不在每个 decode step 更新**：`cache_unfinished_req` 在 prefill / chunk 结束时调用（`scheduler_output_processor_mixin.py:94`），`cache_finished_req` 在请求结束时调用（同文件 `:90`）。所以这里把 A 的 prompt 改成 4，让第一页在 prefill 结束时就进树；否则要等 A 结束，B 才可能命中。

| 步骤 | 动作 | `req_to_token` | 树（key → value, lock） | 空闲槽 | evictable / protected |
|---|---|---|---|---|---|
| 0 | 初始 | — | 空 | 1..8 | 0 / 0 |
| 1 | A prefill 4 token | A: `[1,2,3,4]` | `[1,2,3,4] → [1,2,3,4], lock=1` | 5..8 | 0 / 4 |
| 2 | A decode 到 5 | A: `[1,2,3,4,5]` | 不变（decode 不更新树） | 6..8 | 0 / 4 |
| 3 | B prompt 5，前 4 个相同 | B: `[1,2,3,4,6]` | 同一节点，`lock=2` | 7,8 | 0 / 4 |
| 4 | A 结束（已算 5） | A 行释放 | 节点仍在，`lock=1` | 5,7,8 | 0 / 4 |
| 5 | C 要 4 槽 | — | `evict(4)` 跳过 `lock=1` 的节点 | 5,7,8 | 0 / 4 |
| 6 | B 结束（已算 5） | B 行释放 | 节点 `lock=0` | 5,6,7,8 | 4 / 0 |
| 7 | C 要 8 槽 | — | `evict(8)` 删掉叶子，回收 `[1..4]` | 1..8 | 0 / 0 |

四个值得停下来看的时刻：

**第 1 步：prefill 结束就插入整页。** `cache_unfinished_req` 的 `page_aligned_len = 4 // 4 * 4 = 4`，正好一整页，`insert` 后 `inc_lock_ref` 把这条路径从 evictable 移到 protected。

**第 2 步：decode 不更新树。** 这一点和 vLLM 明显不同：vLLM 每个调度步的 `allocate_slots` 都会调用 `cache_blocks`，块一满就进 hash 表；SGLang 的 `cache_unfinished_req` 只在 prefill/chunk 结束时调用，decode 期间树不更新（这是实现选择，不是语义要求，但确实省掉了每步的树操作）。所以第 5 个 token 的 KV 槽（槽 5）只属于 A，不进树。**这是 SGLang 版的「尾块浪费」：不满一页的尾巴不参与共享。**

**第 3 步：B 只领 1 个新槽。** `match_prefix` 返回 4 个槽号，`extend_input_len = 5 - 4 = 1`，分配器只分配 1 个槽（槽 6）。B 的行是 `prefix_indices + [6]`。共享的 4 个槽号是**复制**进 B 的行的，不是引用一个块对象。

**第 5 步：活跃前缀会挡住淘汰。** C 需要 4 槽，空闲只有 3 槽；`evict(4)` 唯一的叶子是那个 `lock_ref = 1`（B 还在用）的节点，直接 `continue`，一个字节都回收不了。SGLang 此时会报 OOM / 触发 retract，而不是把 B 正在读的 KV 抢走。**这和 vLLM 的「命中僵尸块要占准入名额」是同一类约束的两个方向**：vLLM 是「命中也要占名额」，SGLang 是「锁住就不让淘汰」。

**第 6 步：B 结束后才能回收。** B 的 `dec_lock_ref` 把 `lock_ref` 降到 0，节点从 protected 变回 evictable。C 要 8 槽时，`evict(8)` 删掉叶子、free 掉 `[1,2,3,4]`，空闲回到 8。

### 34.1 和 §14 的 vLLM 追踪并排

| | vLLM §14 | SGLang §34 |
|---|---|---|
| 树/哈希的更新时机 | **每个调度步**的 `allocate_slots` → `cache_blocks` | prefill/chunk 结束、请求结束 |
| A 长 5 时占的物理量 | 2 块（8 槽，有效 5） | 5 槽（其中 4 槽同时被树和 B 引用） |
| B 命中 4 token 后新增 | 1 块（4 槽） | 1 槽 |
| 共享的单位 | 物理块 | KV 槽号数组 |
| A 结束后 | 块 2 释放，块 1 ref=1 | 行释放，尾槽释放，树节点 lock 2→1 |
| B 结束后 | 块 1 ref=0，变僵尸块（hash 还在） | 节点 lock=0，变 evictable |
| 淘汰单位 | 块 | 树叶子（一段 token） |
| 淘汰后 hash/树结构 | `reset_hash()`，hash 从 map 删除 | `_delete_leaf`，节点从父节点摘除 |

**结论：两套实现的「生命周期」形状是一样的**——分配 → 命中/共享 → 加锁 → 释放/解锁 → 延迟淘汰。差别只在「计数对象」是块还是树节点。

---

## 35 两套实现的取舍

### 35.1 总表

| 维度 | vLLM v0.11.0 | SGLang v0.5.2 |
|---|---|---|
| 分配单位 | 固定 `block_size`（默认 16） | `page_size`（默认 1；部分 backend 强制 16/32/64/128） |
| 序列→物理 | `block_table` 块号数组 + `slot_mapping` | `req_to_token` 逐 token 槽号 |
| 前缀索引 | 块级哈希链（父哈希 + 本块 token + extra_keys） | 基数树（原始 token 序列） |
| 命中粒度 | 整块 | 整页 |
| 共享方式 | `ref_cnt += 1`，块号不变 | 槽号数组复制进另一行 |
| 去重 | **不做**（append-only 约束） | **做**（`_split_node` + 释放重复槽） |
| 淘汰结构 | 双向空闲链表（队首 LRU） | 叶子最小堆（`last_access_time`） |
| 淘汰时机 | `get_new_blocks` 内部，边分边腾 | `alloc_token_slots` 之前，先腾后分 |
| 元数据开销 | 每块一个 `KVCacheBlock` | 每段 token 一个 `TreeNode` |
| 0 号位 | null block（sliding window 占位） | padding slot（dummy 输出） |
| 对后端的要求 | paged attention kernel 读 block table | attention 后端读 `kv_indices` |
| 典型优势 | 块表简单、GPU 增量更新无锁、适合固定块 kernel | 前缀共享粒度细、树操作灵活、page_size 可调 |

### 35.2 为什么 vLLM 选哈希链

1. **块表是 append-only 的**。`BlockTable.append_row` 只往行尾写，`compute_slot_mapping` 只读。GPU 侧不需要同步「块号被改了」。
2. **哈希链查找是 O(块数) 的字典查询**，不需要维护树、不需要分裂、不需要堆。
3. **块级哈希天然编码了因果依赖**：父哈希进哈希，所以「第 i 块命中」自动意味着「前 i−1 块全部命中」。
4. **代价**：只能整块共享；不能去重；尾块浪费上界 `B−1`；命中长度被 `(num_tokens−1)//B` 砍。

### 35.3 为什么 SGLang 选基数树

1. **`req_to_token` 是可变的**。共享就是把槽号复制进另一行，不需要保持块号稳定，所以可以 split / 去重。
2. **树天然表达「最长前缀」**。`match_prefix` 沿树走，命中多长就是多长；`page_size = 1` 时可以精确到 token。
3. **叶子 LRU 堆 + `lock_ref` 路径锁**，让淘汰和活跃请求的保护都落在同一棵树上，不需要额外的块引用计数。
4. **代价**：每个节点是 Python 对象；split / heap / 路径加锁都是 CPU 开销；`page_size > 1` 时仍然有页对齐损失。

### 35.4 五条跨实现的共同不变量

不管用哪套实现，下面五条都成立。它们才是「PagedAttention + Prefix Caching」的本质，而不是某个具体数据结构：

1. **只有满单位才缓存**：vLLM 是满块，SGLang 是满页。不满一块/一页的尾巴永远不共享。
2. **最后一个 token 必须重算**：vLLM `max_cache_hit_length = num_tokens − 1`；SGLang `adjust_max_prefix_ids` 里 `max_prefix_len = input_len − 1`。原因相同：要拿 logits 就必须有一次前向。
3. **共享的前缀是只读的**：没有任何一方会去改已经共享的 KV，所以不需要 COW。vLLM 靠「块号不变」，SGLang 靠「槽号只读复制」。
4. **淘汰延迟到显存压力出现时**：vLLM 在 `get_new_blocks` 里惰性淘汰，SGLang 在 `alloc_token_slots` 前按需淘汰。请求结束时都不立刻清数据。
5. **命中只省 prefill 算力，不省显存**：命中的块/页仍然占着池子；decode 仍然要读全部可见 KV；计费 token 照常算。

### 35.5 怎么选

这不是「谁更好」的问题，而是「你的后端和负载长什么样」：

| 场景 | 更自然的选择 | 原因 |
|---|---|---|
| 用 vLLM 的 paged attention kernel，块大小固定 | vLLM 式块表 | kernel 直接吃 block table + slot mapping |
| 前缀重复粒度细（多轮对话、共享系统提示词但长度不齐） | SGLang 式基数树 | `page_size=1` 时可以按 token 共享 |
| 需要 KV 分层缓存（GPU/CPU/远端） | SGLang 式树 | 节点带 `host_value`，天然支持层级 |
| 要求块表增量更新、GPU 无锁 | vLLM 式块表 | append-only + 块号稳定 |
| 需要 sliding window / hybrid KV group | vLLM 式块表 | null block 占位 + 多 group coordinator |

**但不要把这张表当成「二选一」的产品对比。** 两套引擎都在演进：vLLM 也有 hybrid KV cache、KV connector；SGLang 也有 paged allocator、chunk cache。真正稳定的是 §35.4 的五条不变量和 §29.4 的「映射是否可变」这个根约束。

# 附录 A 可执行验证代码

本文所有数字由下面这个脚本验证，**79 项断言全部通过**，纯标准库，不需要 torch 或 GPU。

保存为 `verify_paged.py`，`python3 verify_paged.py` 即可运行。

## A.1 字节账与三本账

```python
def cdiv(a, b): return -(-a // b)

H, HQ, HKV, D, L, DT = 8, 4, 2, 2, 2, 2
assert H // HQ == D                      # head_dim = 2
assert HQ // HKV == 2                    # GQA group_size = 2
kv_per_token = L * (2 * HKV * D * DT)    # 2 × 16 = 32 B
assert kv_per_token == 32

POOL, B = 1024, 4
assert B * kv_per_token == 128           # 每块 128 B
assert POOL // 128 == 8                  # 8 块
assert 8 * B == 32                       # 32 token 容量
assert (8 - 1) * B == 28                 # 减 null block 后 28

def three_books(lengths, B, n_blocks):
    eff = sum(lengths)
    alloc = sum(cdiv(x, B) for x in lengths)
    cap = alloc * B
    return dict(effective=eff, allocated_blocks=alloc, allocated_cap=cap,
                tail_waste=cap - eff, free_blocks=n_blocks - alloc,
                util_of_allocated=eff / cap, util_of_pool=eff / (n_blocks * B))

s = three_books([3, 3, 3, 3], 4, 8)
assert s["allocated_cap"] == 16
assert s["util_of_allocated"] == 0.75    # 12/16
assert s["util_of_pool"] == 0.375        # 12/32  ← 不同分母

# 尾块浪费上界，对 L=0..39 全部成立
for Ln in range(40):
    assert 0 <= cdiv(Ln, B) * B - Ln <= B - 1
```

## A.2 BlockPool 的忠实移植

这是 vLLM v0.11.0 `BlockPool` + `FreeKVCacheBlockQueue` 的语义等价移植，用于验证 §14、§22 的每一步。

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class Blk:
    bid: int
    ref: int = 0
    hash: Optional[str] = None
    prev: Optional["Blk"] = None
    next: Optional["Blk"] = None
    is_null: bool = False

class FreeQueue:
    """双向链表，支持 O(1) 中间删除。对应 FreeKVCacheBlockQueue。"""
    def __init__(self, blocks):
        self.n = 0
        self.head = Blk(-1); self.tail = Blk(-1)      # 哨兵
        self.head.next = self.tail; self.tail.prev = self.head
        for b in blocks: self.append(b)
    def append(self, b):
        last = self.tail.prev
        last.next = b; b.prev = last
        b.next = self.tail; self.tail.prev = b
        self.n += 1
    def append_n(self, bs):
        for b in bs: self.append(b)
    def popleft_n(self, k):
        assert self.n >= k
        out = []
        for _ in range(k):
            b = self.head.next
            self.head.next = b.next; b.next.prev = self.head
            b.prev = b.next = None
            self.n -= 1
            out.append(b)
        return out
    def remove(self, b):                              # O(1) 中间摘除
        b.prev.next = b.next; b.next.prev = b.prev
        b.prev = b.next = None
        self.n -= 1
    def ids(self):
        out, c = [], self.head.next
        while c is not self.tail:
            out.append(c.bid); c = c.next
        return out

class Pool:
    def __init__(self, n, caching=True):
        self.blocks = [Blk(i) for i in range(n)]
        self.q = FreeQueue(self.blocks)
        self.cache = {}                               # hash -> {bid: blk}
        self.caching = caching
        self.null = self.q.popleft_n(1)[0]             # ← block_id 0 被占掉
        self.null.is_null = True
    def free_count(self): return self.q.n
    def usage(self):
        return 1.0 - self.free_count() / (len(self.blocks) - 1)
    def get_new_blocks(self, k):
        if k > self.free_count(): raise ValueError("pool exhausted")
        ret = self.q.popleft_n(k)
        for b in ret:
            if self.caching: self._maybe_evict(b)      # ← 惰性淘汰
            assert b.ref == 0
            b.ref += 1
        return ret
    def _maybe_evict(self, b):
        if b.hash is None: return False
        d = self.cache.get(b.hash)
        if d is None or b.bid not in d: return False
        d.pop(b.bid)
        if not d: self.cache.pop(b.hash)
        b.hash = None
        return True
    def touch(self, blocks):
        for b in blocks:
            if b.ref == 0 and not b.is_null: self.q.remove(b)
            b.ref += 1
    def free_blocks(self, ordered):
        bs = list(ordered)
        for b in bs: b.ref -= 1
        self.q.append_n([b for b in bs if b.ref == 0 and not b.is_null])
    def cache_full(self, blocks, hashes, ncached, nfull):
        for b, h in zip(blocks[ncached:nfull], hashes[ncached:nfull]):
            assert b.hash is None
            b.hash = h
            self.cache.setdefault(h, {})[b.bid] = b
    def get_cached(self, h):
        d = self.cache.get(h)
        return next(iter(d.values())) if d else None
```

## A.3 §14 生命周期追踪

```python
p = Pool(8)
assert p.free_count() == 7                    # null block 占掉 1
assert p.usage() == 0.0

A = p.get_new_blocks(1)
assert A[0].bid == 1
assert p.free_count() == 6
p.cache_full(A, ["hA0"], 0, 0)                # 长 3，无满块
assert len(p.cache) == 0                      # ← 不满一块不进缓存

p.cache_full(A, ["hA0"], 0, 1)                # 长 4，块满了
assert len(p.cache) == 1

assert cdiv(5, B) - len(A) == 1               # 长 5，追加 1 块
A += p.get_new_blocks(1)
assert [b.bid for b in A] == [1, 2]
assert p.free_count() == 5

bt = [b.bid for b in A]
assert [bt[t // B] * B + t % B for t in range(5)] == [4, 5, 6, 7, 8]

hit = p.get_cached("hA0")                     # B 命中 A 的首块
assert hit.bid == 1 and hit.ref == 1
p.touch([hit])
assert hit.ref == 2
Bb = [hit] + p.get_new_blocks(1)
assert [b.bid for b in Bb] == [1, 3]          # 共享物理块 1
assert p.free_count() == 4
assert 8 - 1 - p.free_count() == 3            # 只用了 3 块装 A+B

p.free_blocks(reversed(A))                    # ← 逆序释放
assert p.blocks[1].ref == 1                   # B 还在用
assert p.blocks[2].ref == 0
assert p.q.ids() == [4, 5, 6, 7, 2]           # 尾块追加到队尾
```

## A.4 §15 null block 的 off-by-one

```python
p = Pool(8)
p.get_new_blocks(7)
assert p.free_count() == 0
try:
    p.get_new_blocks(1)
    raise AssertionError("should have raised")
except ValueError:
    pass                                      # 第 8 块拿不到
```

## A.5 §22 僵尸块 / 复活 / 惰性淘汰

```python
p2 = Pool(8)
X = p2.get_new_blocks(7)
p2.cache_full(X, [f"hX{i}" for i in range(7)], 0, 7)
assert p2.free_count() == 0
p2.free_blocks(reversed(X))
assert [b.ref for b in X] == [0] * 7
assert [b.hash for b in X] == [f"hX{i}" for i in range(7)]   # hash 全保留
assert p2.free_count() == 7
assert p2.usage() == 0.0                                     # ← usage 为 0 但缓存是满的
assert p2.get_cached("hX0").bid == 1                         # 仍可命中
assert p2.q.ids() == [7, 6, 5, 4, 3, 2, 1]                   # 尾块先淘汰

z = p2.get_cached("hX0")
p2.touch([z])
assert z.ref == 1
assert 1 not in p2.q.ids()                                   # 从队列中间摘走
assert p2.free_count() == 6

ev = p2.get_new_blocks(1)
assert ev[0].bid == 7                                        # 征用队首
assert ev[0].hash is None
assert p2.get_cached("hX6") is None                          # 被淘汰
assert p2.get_cached("hX0").bid == 1                         # 前缀保住
assert len(p2.cache) == 6

# 从未用过的块比缓存块先淘汰
p5 = Pool(8)
Y = p5.get_new_blocks(2)
p5.cache_full(Y, ["hY0", "hY1"], 0, 2)
assert p5.q.ids() == [3, 4, 5, 6, 7]
p5.free_blocks(reversed(Y))
assert p5.q.ids() == [3, 4, 5, 6, 7, 2, 1]
assert p5.get_new_blocks(1)[0].bid == 3                      # 不是块 2
assert len(p5.cache) == 2                                    # 缓存完好
```

## A.6 §23 僵尸块占准入名额

```python
def num_blocks_to_allocate(num_tokens, B, new_computed, owned):
    new = cdiv(num_tokens, B) - len(new_computed) - len(owned)
    evictable = sum(b.ref == 0 and not b.is_null for b in new_computed)
    return new + evictable

p4 = Pool(8)
W = p4.get_new_blocks(2)
p4.cache_full(W, ["hW0", "hW1"], 0, 2)
p4.free_blocks(reversed(W))
computed = [p4.get_cached("hW0"), p4.get_cached("hW1")]
assert [b.ref for b in computed] == [0, 0]
assert num_blocks_to_allocate(8, 4, computed, []) == 2        # 僵尸：占 2
assert num_blocks_to_allocate(8, 4, [Blk(90, ref=1), Blk(91, ref=1)], []) == 0
```

## A.7 §20 命中长度的对齐损失

```python
def hit_len(num_tokens, B, cached_full_blocks):
    max_cache_hit_length = num_tokens - 1        # kv_cache_manager.py:180
    max_num_blocks = max_cache_hit_length // B   # single_type_...:271
    return min(max_num_blocks, cached_full_blocks) * B

assert hit_len(8, 4, 2) == 4          # 全缓存却只命中 1 块
assert 8 - hit_len(8, 4, 2) == 4      # 重算整整一块
assert hit_len(9, 4, 2) == 8
assert 9 - hit_len(9, 4, 2) == 1      # 多 1 个 token，重算降到 1
assert hit_len(4, 4, 1) == 0          # 全命中也要全部重算
assert hit_len(2048, 16, 128) == 2032 # 真实模型：重算 16
assert hit_len(2049, 16, 128) == 2048 # 重算 1
```

## A.8 §17 hash 链

```python
def hash_chain(token_ids, B, extra=None):
    out, parent = [], "NONE"
    for i in range(0, len(token_ids) - B + 1, B):    # 只 hash 满块
        h = f"H({parent},{tuple(token_ids[i:i+B])},{extra})"
        out.append(h); parent = h
    return out

hc = hash_chain(list(range(10)), 4)
assert len(hc) == 2
assert hc[0].startswith("H(NONE,")
assert hc[1].startswith(f"H({hc[0]},")               # 链式嵌套

a = hash_chain([1, 2, 3, 4, 9, 9, 9, 9], 4)
b = hash_chain([5, 6, 7, 8, 9, 9, 9, 9], 4)
assert a[1] != b[1]                                  # 尾块相同、前缀不同 → hash 不同
assert hash_chain([1,2,3,4], 4)[0] != hash_chain([1,2,3,4], 4, ("lora-7",))[0]
```

## A.9 §8 / §10 attention 不变性与反例

```python
import math

def softmax(xs):
    m = max(xs); e = [math.exp(x - m) for x in xs]; s = sum(e)
    return [x / s for x in e]

D = 2
K = [[0.5,1.0], [1.0,0.0], [0.0,0.5], [1.0,1.0], [0.5,0.5]]
V = [[1,2], [3,4], [5,6], [7,8], [9,10]]
q, n = [1.0, 0.5], 5

def scores_of(idx):
    return [sum(q[i] * K[t][i] for i in range(D)) / math.sqrt(D) for t in idx]

# 连续
s = scores_of(range(n)); w = softmax(s)
o_cont = [sum(w[t] * V[t][d] for t in range(n)) for d in range(D)]

# 分页：block_table = [5, 1]，物理块 5 装 t=0..3，物理块 1 装 t=4
phys = {5: [0,1,2,3], 1: [4]}
block_table, Bs = [5, 1], 4
s2 = []
for lb in range(cdiv(n, Bs)):
    for off in range(Bs):
        t = lb * Bs + off
        if t >= n: continue
        assert phys[block_table[lb]][off] == t       # 取到的是同一个 token
        s2.append(sum(q[i] * K[t][i] for i in range(D)) / math.sqrt(D))
w2 = softmax(s2)
o_page = [sum(w2[t] * V[t][d] for t in range(n)) for d in range(D)]

assert o_cont == o_page                              # ← 逐位完全相同
# [5.041254658459493, 6.041254658459493]

# 反例：按块独立 softmax 再平均
outs = []
for lb in range(cdiv(n, Bs)):
    idx = list(range(lb * Bs, min((lb + 1) * Bs, n)))
    wl = softmax(scores_of(idx))
    outs.append([sum(wl[j] * V[t][d] for j, t in enumerate(idx)) for d in range(D)])
o_wrong = [sum(o[d] for o in outs) / len(outs) for d in range(D)]
assert o_wrong != o_cont
assert abs(o_wrong[0] - o_cont[0]) > 1.5             # 误差 1.566，量级错误
```

## A.10 §6.2 slot 编解码往返

```python
for pb in range(8):
    for off in range(4):
        slot = pb * 4 + off
        assert (slot // 4, slot % 4) == (pb, off)     # kernel 侧解码恒等
```

## A.11 §26 真实模型

```python
real_kv = 2 * 80 * 8 * 128 * 2
assert real_kv == 327680 and real_kv // 1024 == 320
RPOOL, RB = 50 * 1024**3, 16
assert RB * real_kv == 5 * 1024**2                    # 5 MiB/块
assert RPOOL // (RB * real_kv) == 10240
assert 100 * 2048 * real_kv / 1024**3 == 62.5         # > 50 GiB
assert 100 * 2048 * real_kv > RPOOL                   # 装不下
assert 80 * cdiv(2048, RB) == 10240                   # 刚好填满
assert 80 * cdiv(2049, RB) == 10320 > 10240           # 全长 1 个 token 就超
```

---

# 附录 B 源码索引（vLLM v0.11.0）

按本文出现顺序。路径相对 vLLM 仓库根。

| 主题 | 文件 | 位置 |
|---|---|---|
| `KVCacheBlock` 字段 | `vllm/v1/core/kv_cache_utils.py` | `:170` |
| `block_hash` setter 断言 | 同上 | `:192` |
| `FreeKVCacheBlockQueue` | 同上 | `:216` |
| 为什么不用 `deque` | 同上 | `:219` 注释 |
| LRU 顺序规则 | 同上 | `:225` 注释 |
| `popleft` / `remove` / `append_n` | 同上 | `:268` / `:334` / `:376` |
| `extra_keys` 三来源 | 同上 | `:517` |
| `hash_block_tokens` 三元组 | 同上 | `:547` |
| `get_request_block_hasher` 增量链 | 同上 | `:576` |
| 只 hash 满块的 `break` | 同上 | `:601` |
| `BlockHashToBlockMap` / 不去重 | `vllm/v1/core/block_pool.py` | `:21` / `:35` NOTE#1 |
| null block 占位 | 同上 | `:154` |
| `get_cached_block` | 同上 | `:163` |
| `cache_full_blocks` | 同上 | `:188` |
| `get_new_blocks` + 惰性淘汰 | 同上 | `:257` |
| `_maybe_evict_cached_block` | 同上 | `:286` |
| `touch`（O(1) 摘除） | 同上 | `:322` |
| `free_blocks`（只减 ref + 入队） | 同上 | `:338` |
| `reset_prefix_cache` | 同上 | `:355` |
| `get_usage` 减 null block | 同上 | `:400` |
| `get_computed_blocks` | `vllm/v1/core/kv_cache_manager.py` | `:154` |
| `prompt_logprobs` 跳过缓存 | 同上 | `:169` |
| `max_cache_hit_length = num_tokens - 1` | 同上 | `:180` |
| `allocate_slots` 四段布局图 | 同上 | `:221` |
| 准入闸门 `return None` | 同上 | `:271` |
| `num_tokens_to_cache` 上限 | 同上 | `:300` |
| `get_num_blocks_to_allocate` | `vllm/v1/core/single_type_kv_cache_manager.py` | `:59` |
| `num_evictable_computed_blocks` | 同上 | `:83` |
| `allocate_new_blocks`（append-only） | 同上 | `:109` |
| `cache_blocks` 向下取整 | 同上 | `:133` |
| `free` 逆序释放 | 同上 | `:156`（`:168` 是 `reversed`） |
| `find_longest_cache_hit` | 同上 | `:252` |
| `max_length // block_size` | 同上 | `:271` |
| `use_eagle` 丢最后一块 | 同上 | `:282` |
| `BlockTable` 二维定长矩阵 | `vllm/v1/worker/block_table.py` | `:16` |
| `append_row`（只写行尾） | 同上 | `:49` |
| `compute_slot_mapping` | 同上 | `:76` |
| 不能用 `token_indices // B` 的警告 | 同上 | `:81` 注释 |
| DCP 分支 | 同上 | `:84` |
| 抢占循环 | `vllm/v1/core/sched/scheduler.py` | `:254` |
| `running.pop()` + `num_computed_tokens = 0` | 同上 | `:271`–`:276` |
| 恢复时查缓存的条件 `num_computed_tokens == 0` | 同上 | `:380` |
| 新请求的 `allocate_slots` | 同上 | `:471` |
| `reshape_and_cache_kernel` 布局注释 | `csrc/cache_kernels.cu` | `:217` |
| `reshape_and_cache_flash_kernel` | 同上 | `:282` |
| slot 拆回 (块, 偏移) | 同上 | `:298` |
| NHD / HND 判定 | 同上 | `:312` |
| `paged_attention_kernel` | `csrc/attention/attention_kernels.cuh` | `:85` |
| `x = 16 / sizeof(cache_t)` | 同上 | `:195` |
| 取本序列表头 | 同上 | `:202` |
| warp 按块跨步循环 | 同上 | `:222` |
| `int64` 转换与溢出注释 | 同上 | `:224` / `:252` |
| K 指针 stride 计算 | 同上 | `:268` |
| 逻辑 `token_idx` 与 mask | 同上 | `:263` / `:297` |
| 跨 warp softmax 归约 | 同上 | `:309`–`:323` |

---

# 附录 C 本教程相对主线文档的增量

## C.1 新增的机制（主线文档完全没有）

| # | 机制 | 位置 |
|---|---|---|
| 1 | null block 导致可用块 = `num_gpu_blocks − 1` | §15 |
| 2 | 僵尸块：`ref_cnt=0` 但 hash 保留、仍算空闲 | §22.1 |
| 3 | 惰性淘汰：淘汰发生在**领块时**，不是释放时 | §22.4 |
| 4 | 释放的块追加队尾 → 从未用过的块先被淘汰 | §22.5 |
| 5 | `reversed` 释放 → 尾块先淘汰，保住前缀 | §14.6 |
| 6 | 手写双向链表的唯一理由是 O(1) 中间摘除 | §12.2 |
| 7 | `(num_tokens−1) // B` 导致整块重算 | §20 |
| 8 | `num_evictable_computed_blocks`：命中僵尸块占名额 | §23 |
| 9 | hash 三元组 + `extra_keys`（LoRA / 多模态 / `cache_salt`） | §17.3 |
| 10 | 不去重的理由是 `block_table` append-only | §24 |
| 11 | `prompt_logprobs` 直接禁用 prefix caching | §19 |
| 12 | `use_eagle` 丢最后一个命中块 | §19 |
| 13 | K/V 两套物理布局与 `x = 16/sizeof` 拆分 | §7 |
| 14 | slot 编解码的 Python↔CUDA 对称性 | §6.2 |
| 15 | `int32 → int64` 防溢出 | §9.2 |
| 16 | 抢占是 `running.pop()` + `num_computed_tokens = 0` | §16.2 |
| 17 | 抢占后可能命中自己的僵尸块 | §16.2 |
| 18 | 按块 softmax 的数值反例（误差 1.566） | §10 |
| 19 | OS 分页类比的**不成立**部分（无缺页 / 无 swap / 无 COW / 无 TLB） | §3.3 |
| 20 | `pre-allocated` 段与投机解码不缓存草稿 token | §13.1 |

## C.2 明确纠正的说法

| 主线文档的说法 | 本教程的纠正 | 位置 |
|---|---|---|
| 「8 块 → 8 并发」 | 真实 vLLM 是 **7**（null block） | §15 |
| 「利用率 37.5% → 75%」 | 两个比值分母不同，不构成同一指标的提升 | §0.3 |
| 「只多一次间址」 | 每块一次查表摊到 `B` 个 token，且布局相关 | §7.3 |
| 「分页后并发提高」 | 需限定「同时持有块」，不等于「都能长到 `max_len`」 | §4.3 |
| 「cached token 零成本」 | 只有 prefill FLOPs 归零 | §21 |
| §3.7 未提命中长度损失 | 长 2048 全命中仍重算 16 个 token | §20 |

## C.3 本教程刻意不覆盖的内容

诚实声明边界：

- **性能实测**：没有任何 benchmark 数字。分页 / prefix caching 的实际收益必须实测。
- **hybrid KV cache**：多 KV cache group（sliding window + full attention 混合）的 coordinator 逻辑、`unify_hybrid_kv_cache_specs`、page size 对齐，全部未展开。
- **sliding window / Mamba**：`SlidingWindowManager` 的 `find_longest_cache_hit` 和 `remove_skipped_blocks` 未展开，只提了 null block 的用途。
- **P/D 分离**：`delay_cache_blocks`、`num_external_computed_tokens`、KV 传输未展开。
- **FlashAttention / FlashInfer 的 paged 实现**：只给了布局差异，未读它们的 kernel。
- **CPU offload / KV connector**：未涉及。
- **`caching_hash_fn` 的具体选择**（builtin hash / sha256）与碰撞概率：未涉及。
- **hash 碰撞的后果**：vLLM 用 `bytes` 哈希，理论上存在碰撞导致错误命中的风险，本文未讨论其概率与防护。

---

# 附录 D SGLang v0.5.2 源码索引

路径相对 SGLang 仓库根。

| 主题 | 文件 | 位置 |
|---|---|---|
| `TreeNode` 字段 | `python/sglang/srt/mem_cache/radix_cache.py` | `:43` |
| `__lt__`（LRU 排序键） | 同上 | `:92` |
| child key / page match | 同上 | `:138` / `:96` / `:105` |
| `match_prefix` | 同上 | `:158` |
| `insert` | 同上 | `:195` |
| `cache_finished_req` | 同上 | `:203` |
| `cache_unfinished_req` | 同上 | `:240` |
| `evict`（叶子最小堆） | 同上 | `:294` |
| `inc_lock_ref` / `dec_lock_ref` | 同上 | `:319` / `:333` |
| `_match_prefix_helper` | 同上 | `:367` |
| `_split_node` | 同上 | `:392` |
| `_insert_helper` | 同上 | `:411` |
| `_delete_leaf` | 同上 | `:462` |
| `ReqToTokenPool` | `python/sglang/srt/mem_cache/memory_pool.py` | `:55` |
| `req_to_token` 形状 | 同上 | `:74` |
| `write` / `alloc` / `free` | 同上 | `:80` / `:86` / `:95` |
| KV buffer 形状与 padding slot 0 | 同上 | `:449` |
| `TokenToKVPoolAllocator` | `python/sglang/srt/mem_cache/allocator.py` | `:118` |
| `clear` / `alloc` / `free` | 同上 | `:132` / `:145` / `:156` |
| `PagedTokenToKVPoolAllocator` | 同上 | `:419` |
| `page_size` 默认值 1 | `python/sglang/srt/server_args.py` | `:623` |
| `adjust_max_prefix_ids`（`input_len − 1`） | `python/sglang/srt/managers/schedule_batch.py` | `:660` |
| 命中槽号写入 `req_to_token` | 同上 | `:1193` |
| `alloc_token_slots`（先淘汰后分配） | 同上 | `:979` |
| `_evict_tree_cache_if_needed` | 同上 | `:1785` |

---

# 附录 E SGLang RadixCache 的可执行对照

这是 `RadixCache` + `ReqToTokenPool` + allocator 的**语义等价模型**（不 import torch），用来验证 §34 的每一步状态转移。它和附录 A 的 `Pool` 是同一层意思：不是引擎本体，但每个方法都能和附录 D 的行号对上。

保存为 `verify_sglang_radix.py`，`python3 verify_sglang_radix.py` 运行；33 项断言全部通过。

```python
from collections import defaultdict
import heapq, time, itertools

class TreeNode:
    _id = itertools.count()
    def __init__(self, key=None, value=None):
        self.children = {}
        self.parent = None
        self.key = [] if key is None else key
        self.value = [] if value is None else value
        self.lock_ref = 0
        self.last_access_time = time.monotonic()
        self.id = next(TreeNode._id)
    def __lt__(self, other):
        return self.last_access_time < other.last_access_time

class Allocator:
    def __init__(self, size):
        self.size = size
        self.free = list(range(1, size + 1))     # slot 0 保留给 padding

    def available(self):
        return len(self.free)

    def alloc(self, n):
        assert n <= len(self.free)
        out, self.free = self.free[:n], self.free[n:]
        return out

    def free_slots(self, xs):
        self.free.extend(xs)

class RadixCache:
    def __init__(self, alloc, page_size=4):
        self.alloc = alloc
        self.page_size = page_size
        self.root = TreeNode()
        self.root.lock_ref = 1
        self.evictable_size_ = 0
        self.protected_size_ = 0
        self.clock = itertools.count()

    def _now(self):
        return next(self.clock)

    def _child_key(self, key):
        return tuple(key[:self.page_size])

    def _match_len(self, a, b):
        i = 0
        while i + self.page_size <= min(len(a), len(b)):
            if a[i:i + self.page_size] != b[i:i + self.page_size]:
                break
            i += self.page_size
        return i

    def match_prefix(self, key):
        key = key[:len(key) // self.page_size * self.page_size]
        value, node = self._match_prefix_helper(self.root, key)
        return [x for chunk in value for x in chunk], node

    def _match_prefix_helper(self, node, key):
        node.last_access_time = self._now()
        value = []
        while key:
            ck = self._child_key(key)
            if ck not in node.children:
                break
            child = node.children[ck]
            child.last_access_time = self._now()
            plen = self._match_len(child.key, key)
            if plen < len(child.key):
                new_node = self._split_node(child.key, child, plen)
                value.append(new_node.value)
                node = new_node
                break
            value.append(child.value)
            node = child
            key = key[plen:]
        return value, node

    def _split_node(self, key, child, split_len):
        new_node = TreeNode(key[:split_len], child.value[:split_len])
        new_node.children = {self._child_key(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:]
        new_node.parent.children[self._child_key(key)] = new_node
        return new_node

    def insert(self, key, value=None):
        if value is None:
            value = list(key)
        key = key[:len(key) // self.page_size * self.page_size]
        value = value[:len(key)]
        return self._insert_helper(self.root, key, value)

    def _insert_helper(self, node, key, value):
        node.last_access_time = self._now()
        if not key:
            return 0
        total = 0
        while key:
            ck = self._child_key(key)
            if ck not in node.children:
                break
            node = node.children[ck]
            node.last_access_time = self._now()
            plen = self._match_len(node.key, key)
            total += plen
            key, value = key[plen:], value[plen:]
            if plen < len(node.key):
                node = self._split_node(node.key, node, plen)
        if key:
            new_node = TreeNode(key, value)
            new_node.parent = node
            node.children[self._child_key(key)] = new_node
            self.evictable_size_ += len(value)
        return total

    def inc_lock_ref(self, node):
        while node is not self.root:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.value)
                self.protected_size_ += len(node.value)
            node.lock_ref += 1
            node = node.parent

    def dec_lock_ref(self, node):
        while node is not self.root:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.value)
                self.protected_size_ -= len(node.value)
            node.lock_ref -= 1
            node = node.parent

    def _collect_leaves(self):
        out, stack = [], [self.root]
        while stack:
            n = stack.pop()
            if not n.children:
                out.append(n)
            else:
                stack.extend(n.children.values())
        return out

    def evict(self, num_tokens):
        leaves = self._collect_leaves()
        heapq.heapify(leaves)
        evicted = 0
        while evicted < num_tokens and leaves:
            x = heapq.heappop(leaves)
            if x is self.root:
                break
            if x.lock_ref > 0:
                continue
            self.alloc.free_slots(x.value)
            evicted += len(x.value)
            del x.parent.children[self._child_key(x.key)]
            self.evictable_size_ -= len(x.key)
            if not x.parent.children:
                heapq.heappush(leaves, x.parent)
        return evicted

    def total_size(self):
        out, stack = 0, [self.root]
        while stack:
            n = stack.pop()
            out += len(n.value)
            stack.extend(n.children.values())
        return out


# ---------- §34 的完整追踪 ----------
a = Allocator(8)
c = RadixCache(a, page_size=4)

# A: prompt 4；prefill 结束就插入整页
A = a.alloc(4)
assert c.insert([1, 2, 3, 4], A) == 0
prefix, last = c.match_prefix([1, 2, 3, 4])
c.inc_lock_ref(last)
assert prefix == [1, 2, 3, 4]
assert last.lock_ref == 1
assert c.evictable_size_ == 0
assert c.protected_size_ == 4
assert a.available() == 4

# A: decode 到 5；decode 不更新树
A += a.alloc(1)
assert c.insert([1, 2, 3, 4, 5], A) == 4
assert c.total_size() == 4          # 第 5 个 token 不在树里
assert a.available() == 3

# B: 命中 4，只领 1 个新槽
B = a.alloc(1)
prefix, bl = c.match_prefix([1, 2, 3, 4, 9])
c.inc_lock_ref(bl)
assert prefix == [1, 2, 3, 4]
assert B == [6]
assert prefix + B == [1, 2, 3, 4, 6]
assert bl.lock_ref == 2
assert c.protected_size_ == 4
assert a.available() == 2

# A 结束
c.insert([1, 2, 3, 4], A[:4])
a.free_slots(A[4:])
c.dec_lock_ref(last)
assert a.available() == 3
assert c.protected_size_ == 4
assert c.evict(4) == 0              # B 还锁着，淘汰不了
assert a.available() == 3
assert c.total_size() == 4

# B 结束
a.free_slots(B)
c.dec_lock_ref(bl)
assert c.evictable_size_ == 4
assert c.protected_size_ == 0
assert c.evict(8) == 4
assert a.available() == 8
assert c.total_size() == 0
assert len(c.root.children) == 0

# ---------- page_size = 1：token 粒度共享 ----------
a1 = Allocator(8)
c1 = RadixCache(a1, page_size=1)
v = a1.alloc(2)
c1.insert([1, 2], v)
assert c1.total_size() == 2
p1, n1 = c1.match_prefix([1, 2, 3])
assert p1 == [1, 2]                 # page_size=1 时可精确到 token

# ---------- split 不复制 KV ----------
a2 = Allocator(16)
c2 = RadixCache(a2, page_size=4)
v8 = a2.alloc(8)
c2.insert(list(range(1, 9)), v8)
assert c2.total_size() == 8
p2, n2 = c2.match_prefix([1, 2, 3, 4, 9])
assert p2 == [1, 2, 3, 4]
assert n2.key == [1, 2, 3, 4]
assert c2.total_size() == 8         # 分裂前后总长度不变
```

---

## 一句话总结

**PagedAttention 用「固定块 + 查表」把「随 `max_len` 放大的预留浪费」换成「上界 `B−1` 的尾块浪费」，并消除外碎片、让跨请求共享成为可能；它一个字节的 KV 读取都没省，吞吐提升来自 batch 变大。Prefix caching 用「父哈希链」把因果注意力的前缀依赖编码成可查表的 key，用「引用计数 + 僵尸块 + 惰性淘汰」把复用窗口开到最大；它只省 prefill FLOPs，而且因为 `(num_tokens−1) // B` 的对齐约束，命中率再高也总要重算一点。**

换到 SGLang，块表变成逐 token 槽号、哈希链变成基数树、双向空闲链表变成叶子 LRU 堆；但 §35.4 的五条不变量一条没变：**满单位才缓存、最后一个 token 重算、共享前缀只读、淘汰延迟到压力出现、命中只省算力不省显存。**