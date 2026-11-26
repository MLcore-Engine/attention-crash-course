# GPU 硬件与 Roofline：从 torch 到 CUDA 到 SM

> 配套主线文档：[LLM推理全链路-从玩具模型到GPU水位.md](./LLM推理全链路-从玩具模型到GPU水位.md) 第 4 章。
> 主线 §4.1–§4.12 是**结论版**（只给数字和公式）；这一份是**机理版**：把 `y = x @ W` 这一行 Python 一路拆到 HBM 的一个 burst。
> 基准硬件 **H200 SXM**（与主线、附录 B 完全同一套数字）。所有硬数字都标注三种来源之一：**[spec]** 官方 datasheet、**[calc]** 本文由公式反算、**[meas]** 公开实测区间。
> 与主线一致的约定：**绝对时间只验证量级与比例，不作为性能承诺**。

---

## 0 导读

### 0.1 这份文档回答的六个问题

主线第 4 章已经告诉你要记住 `412` 和 `206`。但真到了线上排障，你会遇到的是另一类问题：

| # | 问题 | 本文位置 |
|---|---|---|
| 1 | 同样是 GEMM，为什么 cuBLAS 能跑 800 TFLOPS，我手写的 kernel 只有 12 TFLOPS？差在哪一层？ | 第二部分（§11–§22）、§19 |
| 2 | `nvidia-smi` 显示 GPU-Util 100%，Tensor Core 却只有 1.8% —— 这两个数分别是谁在说谎？ | §9、§36 |
| 3 | 一个 kernel 的 `dram__throughput` 只有理论带宽的 55%，怎么判断是「访存没合并」「bank conflict」「并行度不够」还是「算法本身 AI 不够」？ | §15、§16、§39 |
| 4 | 为什么 Tensor Core 比 CUDA Core 快 32 倍？这 32 倍是「一条指令干更多活」还是「时钟更快」？ | §24、§25、§26 |
| 5 | HBM 的 4.8 TB/s 是怎么算出来的？为什么随机访问只能拿到一半？延迟 500 ns 这个数字在 kernel 里表现成什么？ | §28、§29、§14 |
| 6 | 从 `torch.empty` 到显存真正被占用，中间经过了几层？为什么 `memory_allocated` 和 `nvidia-smi` 差 3 GB？ | §4、§27 |

### 0.2 五层抽象：一行 Python 的完整旅程

```
①  Python / torch           x @ W                      ← 你写的
        ↓  Dispatcher（按 dtype/device/layout 分派，~1 µs）
②  ATen / 算子库            at::mm → cublasLtMatmul     ← PyTorch 仓库
        ↓  kernel launch（~3–5 µs CPU 侧）
③  CUDA Runtime / Driver    cudaLaunchKernel → 提交到硬件队列
        ↓  JIT / 预编译
④  PTX → SASS               ptxas → cubin → SM 的机器码  ← nvcc 产物
        ↓  warp 调度
⑤  SM + 内存子系统          warp → LSU → L1 → L2 → HBM   ← 硅片
```

每一层都能「拦住」性能，而且**症状完全不同**：

| 层 | 谁负责 | 你能改什么 | 这一层出问题的症状 |
|---|---|---|---|
| ① torch | Python 解释器、autograd | 张量布局、`contiguous()`、融合、`torch.compile` | 每步几千次 dispatch，GPU 空转等 CPU |
| ② ATen / cuBLAS | PyTorch + NVIDIA 库 | 换后端（cuBLAS / CUTLASS / Triton / FlashAttention） | 形状不对齐导致选到慢 kernel（如 `N` 不是 8 的倍数） |
| ③ Runtime | CUDA driver | stream、CUDA Graph、`cudaMallocAsync` | launch 开销吃掉小 kernel；隐式同步打断流水 |
| ④ PTX/SASS | 编译器 | 寄存器上限、`__launch_bounds__`、指令选择 | 寄存器 spill 到 local memory（= HBM），occupancy 塌陷 |
| ⑤ 硬件 | SM / HBM | 分块、swizzle、合并访存、batch 形状 | 带宽或算力只用了一半，`ncu` 里 stall 原因明确 |

> **排障顺序永远是从下往上确认瓶颈，再从上往下改。** 先看 `ncu` 说这条 kernel 是带宽受限还是延迟受限（⑤），再决定是改算法（①）还是改 tile（②）。

### 0.3 基准硬件：三列对照

| 项 | **H200 SXM**（本文基准） | H100 SXM | B200（对照） |
|---|---|---|---|
| HBM | 141 GB HBM3e | 80 GB HBM3 | 192 GB HBM3e |
| HBM 带宽 | **4.8 TB/s** [spec] | 3.35 TB/s | 8 TB/s |
| HBM 延迟 | ~500 ns [meas] | ~500 ns | ~600 ns（代际估计） |
| SM 数 | **132** [spec] | 132 | 双 die 封装，公开口径分散 |
| 每 SM | 4 TC + 128 FP32 core + 4 warp scheduler + 256 KB 寄存器 + 228 KB SMEM | 同 | 4 TC（tcgen05）+ 256 KB TMEM |
| L2 | 50 MB [spec] | 50 MB | 126 MB（双 die 合计，公开口径） |
| FP8 dense | **1979 TFLOPS** [spec] | 1979 | ~4500 |
| BF16 dense | **989 TFLOPS** [spec] | 989 | ~2250 |
| FP32（CUDA Core） | 60.8 TFLOPS [calc] | 60.8 | ~80 |
| **Roofline 拐点（FP8）** | **412 FLOP/B** [calc] | 590 | 562 |
| **GEMM 的 M 阈值** | **≈ 206** [calc] | ≈ 295 | ≈ 281 |
| NVLink | 900 GB/s 双向 | 900 GB/s | 1.8 TB/s |
| PCIe | Gen5 ×16，64 GB/s/向 | 同 | Gen6 |
| TDP | 700 W | 700 W | 1000 W |

两个容易忽略的推论：

1. **M 阈值 = BF16 峰值 ÷ 带宽**。H200 带宽涨了 43%，阈值反而从 295 降到 206 ——**带宽越大，越难把 Tensor Core 喂饱**。
2. B200 的 `8 TB/s` 让阈值回到 281。**HBM 代际升级会同时抬高「算力峰值」和「喂饱难度」**，两者不抵消。

### 0.4 与主线第 4 章的对应关系

| 主线 | 主线给的结论 | 本文的机理版 |
|---|---|---|
| §4.1 HBM 三约束 | 141 GB / 4.8 TB/s / 500 ns 互相独立 | §27 内存层级、§28 HBM 物理 |
| §4.2 Little's Law | 2.4 MB 在途、18 KB/SM | §14 延迟隐藏的严格算法（含这个估算的问题） |
| §4.3 SM | 132 SM、每 SM 4 TC + 128 core | §23 die 平面、§24 SM 微架构 |
| §4.4 Core vs TC | 32 倍 | §25 Tensor Core 指令形态、§26 峰值推导 |
| §4.5 内存层级 | SMEM ~30 TB/s 聚合 | §29 一个 sector 的旅程 |
| §4.6–4.7 Roofline | 412、AI≈2M、M≈206 | §34 口径陷阱、§35 完整推导 |
| §4.8 TinyToy AI | 3.04 / 32.3 / 1.95 | §40 原表复算 |
| §4.9–4.10 生命周期与下界 | (权重+KV)/带宽 | §37 算子位置表、§38 四个突破口 |
| §4.11 vLLM 显存账本 | 50 GiB | §4 allocator 三数关系、§27 每层能装多少 KV |
| §4.12 小结 | 12 行 | §43 速查卡 |

### 0.5 口径约定（这一节决定后面所有数字能不能对比）

| 项 | 本文用法 | 常见错误 |
|---|---|---|
| 带宽单位 | `TB/s` = $10^{12}$ B/s（十进制）；显存容量用 GiB = $2^{30}$ B | 用 1024 进制算带宽，结果偏大 7% |
| 算力单位 | `TFLOPS` = $10^{12}$ FLOP/s；**dense**，不含 2:4 稀疏的 2× | 拿 sparse 峰值算 MFU，MFU 直接虚高一倍 |
| 时钟 | 名义 1.8 GHz（与主线一致）；官方 1979 TFLOPS 对应等效 ~1.83 GHz [calc] | 用 boost 2.0 GHz 复算，对不上官方数字就以为公式错 |
| 「带宽」 | 未标注时指 **HBM 带宽**；L2、SMEM、NVLink 会显式标注 | 把 L2 带宽当 HBM 用 |
| 「延迟」 | 指 **HBM 访问延迟**（发出请求到数据回到寄存器） | 与 kernel 端到端延迟混用 |
| AI 的分母 | 只算 **HBM 往返字节**，不算 L2 命中、不算写回（写回另列） | 分母算上 L2 命中，AI 虚高 5 倍 |
| 反算 | 标 **[calc]** 的数字是本文推的，不是官方值 | 把反算值当 datasheet 引用 |

### 0.6 三条阅读路径

- **只想搞懂为什么 decode 慢**：§27 → §28 → §37 → §38，四节。
- **要写/改 kernel**：§11 → §15 → §16 → §19 → §39，五节。
- **线上排障**：§9 → §36 → §39 → §43，四节 + 速查卡。

---

# 第一部分 torch 层：一行 Python 到底发生了什么

这一部分只回答一个问题：**从你写 `y = x @ W` 到 GPU 上真的有指令在跑，中间有几步，每一步能吃掉多少时间。**

结论先给：**torch 层不决定峰值，但决定你能拿到峰值的百分之几。** 同一块 H200、同一个 GEMM，eager 模式和 CUDA Graph 模式的端到端差距可以到 2 倍以上，原因全在这一层。

---

## 1 Tensor 不是「一块显存」，是六个字段

### 1.1 六个字段

一个 `torch.Tensor` 在 C++ 侧是 `TensorImpl`，它**不拥有数据**，只描述「怎么解释一块 Storage」：

| 字段 | 含义 | 你能读到 |
|---|---|---|
| `storage` | 真正的显存块（引用计数） | `t.untyped_storage().data_ptr()` |
| `storage_offset` | 从 Storage 第几个元素开始 | `t.storage_offset()` |
| `size` | 每个维度的长度 | `t.shape` |
| `stride` | 每个维度走一步跨几个元素 | `t.stride()` |
| `dtype` | 元素类型（决定字节数） | `t.dtype` |
| `device` | 在哪张卡 / CPU | `t.device` |

```python
import torch
x = torch.arange(24).reshape(4, 6)      # size (4,6) stride (6,1)
y = x.t()                               # size (6,4) stride (1,6)  ← 零拷贝
z = y.contiguous()                      # size (6,4) stride (4,1)  ← 真拷贝
print(x.stride(), y.stride(), z.stride())
print(y.storage_offset(), y.data_ptr() == x.data_ptr())   # 0 True
```

### 1.2 stride 决定了三件昂贵的事

**① 拷贝**：`y.contiguous()` 要新分配 `6×4×2 = 48 B`（BF16）并把 24 个元素重排。真实模型里一次 `[B, L, 8192]` 的 `.contiguous()` 是 `B×L×16 KB` 的读写。

**② 访存模式**：`stride=(1,6)` 的转置张量，按最后一维遍历时地址跳 6 个元素 —— **§15 会证明这种访问在 warp 层面会被放大 8 倍**。

**③ 能不能走 fast path**：cuBLAS 支持 `op(A)=T` 的标志位，所以

```python
attn = q @ k.transpose(-1, -2)    # k 是 (B, H, L, d) 的连续张量
```

**不会**触发 `.contiguous()`，而是把 `transpose` 作为 `cuBLAS` 的 `CUBLAS_OP_T` 传下去。这是注意力实现里最常见的「零成本转置」。

> 判断标准只有一个：`t.is_contiguous()`。为 `False` 时，先问「这个算子能不能吃 stride」，再问「值不值得拷」。

### 1.3 为什么这条要在硬件文档里讲

因为**所有「合并访存」的分析（§15）都以 stride 为输入**。你在 Python 里随手一个 `permute`、`expand`、`narrow`，到硬件层就是一个 warp 的 32 个 lane 打在 32 条不同的 128 B cache line 上，带宽直接掉 8 倍。**这不是 kernel 写得不好，是布局选错了。**

---

## 2 `x @ W` 的完整调用链

### 2.1 五跳

以 `torch.nn.functional.linear(x, W)`（`x: [M,K]`，`W: [N,K]`）为例：

```
① Python:      F.linear(x, W)
② 绑定层:      torch._C._nn.linear            ← C++ pybind，参数解析
③ 分派层:      at::linear → at::matmul → at::mm
               Dispatcher 按 (dtype, device, layout, requires_grad) 选 kernel
④ 后端层:      at::cuda::blas::gemm<BF16>
⑤ 库层:        cublasLtMatmul(heuristic 选出的 tile 配置)
               → 生成/调用 SASS kernel
```

### 2.2 每一跳的成本量级

| 跳 | 成本 | 是否随规模变化 | 观测手段 |
|---|---|---|---|
| ① Python 函数调用 + 参数解析 | ~0.5–1 µs | 否 | `torch.profiler` 的 CPU 段 |
| ② pybind 转换 | ~0.2 µs | 否 | 同上 |
| ③ Dispatcher 查表 | ~0.3–1 µs | 否 | `TORCH_SHOW_DISPATCH` |
| ④ ATen → cuBLAS 参数组装 | ~0.5 µs | 否 | nsys 里 CPU 侧的 `cublasLtMatmul` |
| ⑤ kernel launch（提交到队列） | **~3–5 µs** | 否 | nsys 的 launch 行 |
| ⑤' 首次调用的 heuristic / JIT | 10 µs–10 ms | 否 | 只在第一次 |

**关键结论：每一跳都与矩阵大小无关。** 所以大 GEMM 里这些开销可以忽略（0.001%），小 kernel 里它们是全部（§3）。

### 2.3 cuBLAS 选 kernel 时看什么

`cublasLtMatmul` 的启发式主要看：

| 输入 | 影响 |
|---|---|
| `M`（token 数） | 决定是走「大 M 的 tile kernel」还是「小 M 的 split-K / GEMV 型 kernel」 |
| `N % 8`、`K % 16`（BF16）/ `K % 32`（FP8） | 不对齐会退化到 edge kernel，吞吐掉 20–50% |
| 是否有 epilogue（bias、GELU） | 能融合就省一次 HBM 往返 |
| 是否转置 | 影响 SMEM 加载的 swizzle 模式 |

**这就是「词表大小 128256 不是 128 的倍数也无所谓，但 `intermediate_size` 最好是 128 的倍数」这类工程经验的来源。** 它不在数学里，在 kernel 选择里。

> 主线 §4.7 的 `AI ≈ 2M` 是从「权重读一遍」推的，隐含假设是 **cuBLAS 真的只读了一遍权重**。M 太小或 N 不对齐时，kernel 会为了并行度把 K 切开（split-K），**权重被读多遍**，实际 AI 比 2M 更低 —— 这是 §39 里「AI 够但带宽还是不够」的一个来源。

---

## 3 eager 模式的三笔额外成本

### 3.1 三笔账

| 成本 | 量级 | 什么时候致命 |
|---|---|---|
| Python + Dispatcher | ~1–2 µs / op | 每步几千个 op（decode） |
| kernel launch | ~3–5 µs / kernel | 每个 kernel 的 GPU 工作时间 < 10 µs |
| allocator | ~0.5–1 µs / 分配 | 每步反复分配临时张量 |

### 3.2 用 Llama-3-70B decode 算一遍

一层 transformer（decode，B=1）大约发出这些 kernel：

```
input RMSNorm          1
QKV 投影               1（融合）或 3（分离）
RoPE（q、k）           2
KV 写入                1
attention              1
O 投影                 1
残差加                 1
post RMSNorm           1
gate / up 投影         2
SiLU × mul             1
down 投影              1
残差加                 1
                     ───
每层                  ~14 个 kernel
80 层              ~1120 个 kernel
```

GPU 侧真正干活的时间（decode 一步）约：

```
68.5 GB / 3.84 TB/s（80% 效率）≈ 17.8 ms
```

而 1120 次 launch 的 CPU 侧开销：

```
1120 × 5 µs = 5.6 ms
```

**占 17.8 ms 的 31%。** 更糟的是那些 12 ns 就能算完的 RMSNorm：**GPU 干 12 ns，CPU 花 5 µs 提交** —— 这时整个 decode 是 **CPU-bound**，GPU 利用率看起来 100%，实际每个 kernel 之间都是空隙。

> 这就是「GPU-Util 100% 但很慢」的第一种成因：**不是算力满，是 launch 把时间填满了。**

### 3.3 三个解法，代价不同

| 解法 | 省掉什么 | 代价 |
|---|---|---|
| 算子融合（RMSNorm+residual、SiLU+mul） | 减少 kernel 数量，顺带减少 HBM 往返 | 要写自定义 kernel 或被 `torch.compile` 正确融合 |
| CUDA Graph（§6） | 把 N 次 launch 压成 1 次 replay | 形状必须静态、地址必须固定 |
| 更大 batch | 每个 kernel 的活变多，开销占比下降 | KV 容量、SLO |

---

## 4 caching allocator：三个数字，三种含义

### 4.1 三个数不是一回事

```
torch.cuda.memory_allocated()   ← 活着的 tensor 占的字节
torch.cuda.memory_reserved()    ← PyTorch 向驱动要来的段（含空闲块）
nvidia-smi 显示的 used          ← reserved + CUDA context + 其他进程 + NCCL/cuBLAS workspace
```

包含关系：

```
nvidia-smi used
├── CUDA context（驱动、module、句柄）      ~300–600 MB
├── 其他进程 / 其他 MIG 实例
├── NCCL / cuBLAS / cuDNN workspace         几十 MB ~ 数 GB
└── PyTorch reserved
    ├── allocated（活 tensor）
    └── 缓存空闲块（可复用，但不还给驱动）
```

### 4.2 为什么 `empty_cache()` 看起来没用

`torch.cuda.empty_cache()` 只是**把缓存空闲块还给驱动**：

- 活着的 tensor 一个字节都不会释放；
- 碎片化时，空闲块不连续，最大的连续段可能只有几百 MB，**新的 2 GB 分配仍然失败**；
- `nvidia-smi` 的数字可能一点不动（context 和 workspace 还在）。

真正要解决碎片，用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，让分配器用虚拟地址映射而不是固定段。

### 4.3 与主线 §4.11 的账本对齐

主线那个式子：

```
KV池 = 总显存 × gpu_memory_utilization − 权重 − 峰值激活 − non_torch
```

这里的 `non_torch` 就是上表里 context + workspace + NCCL 的那几项。vLLM 的做法是**先跑一次 profile forward 实测峰值激活**，再算剩余 —— 因为激活峰值随 `max_num_batched_tokens` 变化，没法静态算准。

> 注意 `gpu_memory_utilization=0.9` 表示**主动留 10% 不用**（H200 上是 14 GB）。这是保守设计，也是「显存没满却开始排队」的一个来源。

---

## 5 stream、event 与「为什么你的计时是错的」

### 5.1 异步语义

```python
y = x @ W          # 只是提交，CPU 立刻返回
print(y[0, 0])     # 触发同步（.item() / .cpu() / print 都要读值）
torch.cuda.synchronize()   # 显式等所有 stream 完成
```

**GPU 是异步执行的协处理器。** 任何「CPU 侧测得的时间」如果不 `synchronize()`，测到的是**提交时间**，不是执行时间。

### 5.2 正确的计时模板

```python
import torch, time

def bench(fn, warmup=10, iters=50):
    for _ in range(warmup):          # 必须 warmup：首次调用有 heuristic / JIT / 页表
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()          # 关键：等 end 真的发生
    return start.elapsed_time(end) / iters   # ms
```

### 5.3 五个常见错误

| 错误 | 后果 |
|---|---|
| 不 warmup | 把 cuBLAS heuristic（几十 µs~ms）算进平均 |
| 不 `synchronize()` | 测到提交时间，结果偏小几十倍 |
| 用 `time.time()` 包住单次调用 | 同上，且受 CPU 调度噪声影响 |
| 只测一次 | 受 clock 波动、其他进程干扰 |
| 在默认 stream 上计时但 kernel 跑在别的 stream | 计时区间与执行区间不对应 |

---

## 6 CUDA Graph：把 1120 次 launch 压成 1 次

### 6.1 原理

CUDA Graph 把「一串 kernel 及其依赖」录制成一张**静态图**，replay 时由驱动一次性提交，CPU 侧只剩一次 `cudaGraphLaunch`（~2–5 µs），GPU 侧 kernel 之间的空隙也消失。

```
eager:   launch(1) → kernel(1) → launch(2) → kernel(2) → ...
         5 µs        12 ns       5 µs        12 ns
         └── CPU-bound，GPU 大量空转 ──┘

graph:   launch(整图) → kernel(1) → kernel(2) → ... → kernel(1120)
         3 µs           连续执行，无 CPU 介入
```

### 6.2 capture 的四个硬约束

| 约束 | 原因 | 违反的症状 |
|---|---|---|
| **形状静态** | 图里记录的是具体 grid/block 维度 | 换 batch 就要重新 capture |
| **地址静态** | kernel 参数里存的是指针值 | 重放时指针必须仍然有效 → 用固定输入/输出 buffer |
| **不能有 CPU 同步** | capture 期间不允许 `cudaMemcpy` 同步、`item()` | capture 报错 |
| **不能动态分配** | 需要 graph-private memory pool | 用 `torch.cuda.graph(g)` 上下文 |

### 6.3 vLLM 怎么用

- **decode 用 full CUDA graph**：按一批固定的 batch size（如 1,2,4,8,...,256）各 capture 一份，实际 batch 向上取整到最近的档位，多出来的槽位用 padding token。
- **prefill 不用**：输入长度动态，capture 不划算。
- 因此有 `enforce_eager=True` 这个开关 —— 打开就全走 eager，**方便调试，但 decode 会慢 20–40%**（因为回到 §3.2 的 launch 账）。

> 副作用：padding 到固定档位意味着 **batch=65 和 batch=128 走的是同一张图**，KV 和算力却差一倍。这是「水位」计算里一个常被忽略的台阶。

---

## 7 `torch.compile`：融合到底省了什么

### 7.1 链路

```
Python 函数
  → TorchDynamo 抓 FX graph（字节码级，不执行）
  → AOTAutograd（前向+反向图）
  → Inductor（调度、融合、内存规划）
  → Triton kernel（或 C++/CUDA）
  → PTX → SASS
```

### 7.2 融合的收益 = 少了几次 HBM 往返

以 `RMSNorm + residual add` 为例，`h = [B,L,8192]` BF16，单 token 16 KB：

```
不融合：
  read x        16 KB
  write norm    16 KB
  read norm     16 KB
  read residual 16 KB
  write out     16 KB
              ───────
               80 KB / token

融合：
  read x        16 KB
  read residual 16 KB
  write out     16 KB
              ───────
               48 KB / token

省 40%，等价于带宽提升 1.67 倍
```

**这就是为什么「算子融合」在 decode 阶段值钱**：decode 是带宽受限（AI≈2M，M 很小），省字节直接等于省时间；而 prefill 是算力受限，省字节几乎没用。

### 7.3 什么时候别用

| 场景 | 原因 |
|---|---|
| 形状频繁变化 | Dynamo 触发 guard 失败 → 反复重编译（每次几秒） |
| 已经用了手写 kernel（FlashAttention、paged attention） | 编译器的调度打不过手写，还可能破坏 layout 假设 |
| 需要确定性数值 | 融合会改变浮点累加顺序 |

---

## 8 在 torch 层就能测的三个数

这三个数不需要 `ncu`，纯 torch 就能测，而且**足以判断一台机器是不是被喂饱了**。

### 8.1 实测带宽（copy）

```python
def bench_copy(nbytes=1 << 30):        # 1 GiB
    n = nbytes // 2                    # BF16
    a = torch.empty(n, dtype=torch.bfloat16, device="cuda")
    b = torch.empty_like(a)
    ms = bench(lambda: b.copy_(a))
    gbs = 2 * nbytes / (ms * 1e-3) / 1e9   # 读 + 写
    print(f"copy: {ms:.3f} ms, {gbs:.1f} GB/s")
    return gbs
```

判据：H200 上 `copy` 的**有效带宽**（读+写都计入）通常落在 **3.4–4.0 TB/s**（理论 4.8 的 70–83%）。低于 3 TB/s 说明这台机器/这个容器有问题（MIG 切片、PCIe 卡、被别的进程占用）。

### 8.2 实测算力（GEMM 扫 M）

```python
def bench_gemm(M, N=8192, K=8192, dtype=torch.bfloat16):
    a = torch.randn(M, K, dtype=dtype, device="cuda")
    w = torch.randn(N, K, dtype=dtype, device="cuda")
    ms = bench(lambda: a @ w.t())
    tflops = 2 * M * N * K / (ms * 1e-3) / 1e12
    ai = 2 * M * N * K / (K * N * 2 + M * K * 2)   # 沿用主线口径
    print(f"M={M:5d}  {ms:8.3f} ms  {tflops:7.1f} TFLOPS  AI={ai:7.1f}")
    return tflops, ai
```

把 `M` 从 1 扫到 4096，你会看到**两个平台**：

- `M < 206`：吞吐随 M 近似线性上升（带宽受限）；
- `M > 206`：吞吐封顶在 `989 × MFU`（算力受限）。

**这条曲线就是主线 §4.7 的 `M ≈ 206` 的实验形态。** 附录 C 给了完整脚本。

### 8.3 launch 开销

```python
def bench_launch(iters=10000):
    x = torch.empty(8, device="cuda")
    ms = bench(lambda: x.add_(1.0), warmup=100, iters=iters)
    print(f"per-op: {ms*1e3:.2f} µs")   # 含 dispatch + launch
```

H200 上这个数一般在 **3–8 µs**。它就是你 §3.2 那笔账的单价。

---

## 9 误读清单：`nvidia-smi` 到底在说什么

### 9.1 `GPU-Util` 是时间占比，不是算力占比

NVML 的 `utilization.gpu` 定义为：**采样周期内，至少有一个 kernel 在 GPU 上驻留的时间占比**。

```
GPU-Util = 100%
⟺ 这一秒里每个采样点都有 kernel 在跑
⟹ 完全不代表 Tensor Core 忙

GPU-Util = 100% 且 TC = 1.8% 完全自洽：
   有一个跑 32 ms 的 attention kernel，它几乎全在等 HBM
```

**这就是主线 §8.4「排队排死了，Tensor Core 用了 1.8%」的机制。**

### 9.2 三套指标的口径

| 指标 | 定义 | 高值意味着 |
|---|---|---|
| `nvidia-smi` `GPU-Util` | kernel 驻留时间占比 | 有活干（可能与算力无关） |
| `nvidia-smi` `Mem-Util` | 显存控制器忙的时间占比 | 有访存请求（不代表打满带宽） |
| DCGM `DCGM_FI_PROF_SM_ACTIVE` | 有 warp 驻留的 SM 比例 | 并行度还行 |
| DCGM `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` | Tensor Core 活跃周期占比 | **这才是「算力利用率」** |
| DCGM `DCGM_FI_PROF_DRAM_ACTIVE` | HBM 活跃周期占比 | **这才是「带宽利用率」** |

### 9.3 换算关系

```
算力利用率（主线口径的 MFU）≈ PIPE_TENSOR_ACTIVE × 指令效率
带宽利用率                    ≈ DRAM_ACTIVE × 每次访问的有效字节比例
```

两个「×」都不能省。`PIPE_TENSOR_ACTIVE=100%` 但每次 MMA 都因为寄存器依赖停顿，MFU 仍然可能只有 30%。

---

## 10 第一部分小结

```
Tensor = storage + offset + size + stride + dtype + device
stride 决定拷贝代价、访存模式和能不能走 cuBLAS fast path
x @ W 的链路：Python → Dispatcher → ATen → cuBLASLt → kernel
每跳 ~1 µs，launch ~3–5 µs，都与矩阵大小无关
1120 个 kernel/step × 5 µs = 5.6 ms，是 decode 17.8 ms 的 31%
allocator 三个数：allocated ⊂ reserved ⊂ nvidia-smi used
计时三件套：warmup + cuda.Event + synchronize
CUDA Graph 把 N 次 launch 压成 1 次，代价是形状/地址静态
融合省的是 HBM 往返：RMSNorm+residual 80 KB → 48 KB（1.67×）
torch 层能测：copy 带宽、GEMM 的 M 扫、launch 单价
GPU-Util 是时间占比，Tensor Core 利用率要看 PIPE_TENSOR_ACTIVE
```

| torch 层能回答 | torch 层不能回答 |
|---|---|
| 端到端是不是 CPU-bound | 为什么这条 kernel 只跑到 55% 带宽 |
| 融合/图化能省多少 | 是 bank conflict 还是没合并 |
| 显存账本差在哪 | SM 里哪条流水线闲着 |
| GEMM 拐点在哪 | Tensor Core 的指令怎么排 |

后两列，就是第二、三部分的内容。

---

# 第二部分 CUDA 层：线程、warp 与内存

第一部分说明了「提交了多少工作」，这一部分说明「这些工作怎么落到 132 个 SM 上，以及为什么落不满」。

这一部分的每个小节都对应 `ncu` 里的一个指标。**看不懂指标的时候回来查对应的节。**

---

## 11 执行模型：从 grid 到 lane

### 11.1 四个层级

```
grid            ← 一次 kernel launch 的全部工作
 ├── block      ← 调度到某一个 SM 上，永不跨 SM
 │    ├── warp  ← 32 个线程，硬件调度的最小单位
 │    │    └── lane ← 单个线程（32 个 lane 共享一条指令流）
 │    └── ...
 └── block ...
```

| 层级 | 谁决定 | 典型值 |
|---|---|---|
| grid | 你（或 cuBLAS 的启发式） | 数百 ~ 数十万 block |
| block | 你 | 128 / 256 / 1024 线程 |
| warp | 硬件：`threadIdx.x / 32` | 32 线程 |
| lane | 硬件 | 1 线程 |

**warp 的划分规则**：`threadIdx.x` 连续的 32 个线程构成一个 warp。所以 `blockDim.x = 128` 就是 4 个 warp。

### 11.2 SM 的资源上限

一个 block 落到 SM 上之后，SM 能同时驻留多少 block 由四个资源决定：

| 资源 | H100/H200 每 SM 上限 | 说明 |
|---|---|---|
| warp 槽位 | **64**（= 2048 线程） | 硬上限 |
| block 槽位 | 32 | 硬上限 |
| 寄存器 | 65536 × 32 bit | 64K 个寄存器 |
| shared memory | 228 KB | 可配置为 L1 |

> **一个 SM 最多 2048 线程，132 个 SM 全机最多 270,336 线程。** 这个数字后面在 §21 会用到：decode 时你可能只用了其中 1%。

### 11.3 「一个 block 只能在一个 SM 上」

这条约束有两个后果：

1. **共享内存不能跨 SM**。两个 block 之间要交换数据，只能走 L2 或 HBM（或 Hopper 的 distributed shared memory / cluster）。
2. **grid 小于 132 时，有空闲 SM**。`grid=64` 的 kernel 最多用 64 个 SM ——**48% 的机器在睡觉**。这是 §21 的核心。

---

## 12 SIMT 与 warp divergence

### 12.1 一个 warp 一条指令流

32 个 lane **共享 program counter**。遇到分支时：

```cuda
if (tid % 2 == 0) { A } else { B }
```

硬件不是「一半走 A、一半走 B 并行」，而是：

```
执行 A（lane 0,2,4,... 有效，奇数 lane 被 mask 掉）
执行 B（偶数 lane 被 mask 掉）
```

**两条路径串行，代价是两者之和。** 这叫 divergence。

### 12.2 什么时候代价可接受

| 情况 | 代价 | 原因 |
|---|---|---|
| 分支在整个 warp 内一致（uniform） | 0 | 硬件有 uniform datapath，只走一条 |
| 短分支，编译器用 predication | 0（但要两条都算） | `selp` 指令，无跳转 |
| 分支边界与 warp 对齐 | 0 | 不同 warp 走不同路径，互不影响 |
| 分支边界在 warp 内部 | **路径数 × 长度** | 串行化 |

### 12.3 在 LLM 里的三个实例

| 实例 | 现象 | 规避方式 |
|---|---|---|
| **causal mask** | 下三角外的元素要跳过。若按元素 `if (j > i) continue`，warp 内一半 lane 空转 | 按 tile 跳过整块（FlashAttention 的 `if (tile_j > tile_i) break`） |
| **变长序列 padding** | 同一 batch 里短序列的 padding 位置全在空转 | vLLM 用 varlen + `cu_seqlens`，按真实长度切 |
| **MoE 路由** | 不同 token 去不同专家，同一 warp 的 32 个 token 可能分叉到 8 个专家 | grouped GEMM / 按专家重排 token |

> **`if` 本身不贵，warp 内不一致才贵。** 优化分支的正确做法不是删 `if`，是让 `if` 的边界对齐 warp 或 tile。

---

## 13 Occupancy：定义、四个限制因子、以及它不是目标

### 13.1 定义

$$
\text{occupancy} = \frac{\text{SM 上实际驻留的 warp 数}}{64}
$$

注意是**驻留**（resident），不是「活跃执行」。驻留但被 stall 的 warp 也算。

### 13.2 四个限制因子

```
warps_resident = min(
    64,                                   ← 硬件上限
    floor(65536 / (regs_per_thread × 32)),← 寄存器
    floor(smem_per_sm / smem_per_block) × warps_per_block,  ← 共享内存
    32 × warps_per_block                 ← block 槽位
)
```

**寄存器是第一限制因子**，因为它最容易被忽视：

| `regs/thread` | 每 SM 可容纳 warp 数（理论上限） | occupancy | 256 线程 block 取整后 |
|---|---|---|---|
| 32 | 64 | 100% | 100% |
| 40 | 51 | 80% | 75% |
| 64 | 32 | 50% | 50% |
| 96 | 21 | 33% | 25% |
| 128 | 16 | 25% | 25% |
| 168 | 12 | 19% | 12.5% |
| 255 | 8 | 12.5% | 12.5% |

**最后一列才是实际值。** block 是分配粒度：40 regs 理论上能放 51 个 warp，但按 256 线程的 block 只能放 6 个 block = 48 个 warp。**block 越大，取整损失越大。**

### 13.3 occupancy 不是性能目标

两个反例都来自真实 kernel：

| kernel | occupancy | 实测吞吐 | 原因 |
|---|---|---|---|
| 大 GEMM（128×128 tile，8×8 累加器） | 25–50% | 峰值的 80–90% | 每个线程有 64 个独立累加器，**ILP 极高**，不需要靠 warp 数藏延迟 |
| 纯 copy kernel | 接近 100% | 峰值的 80% | 每条指令都在等内存，**必须靠 warp 数** |

**判据**：

```
如果瓶颈是「指令延迟」→ ILP 可以替代 occupancy
如果瓶颈是「内存延迟」→ 只能靠 occupancy × 每 warp 的未完成请求数
```

后者的定量形式就是 §14。

---

## 14 延迟隐藏的严格算法（主线 §4.2 的展开）

### 14.1 主线那个估算

```
在途字节 = 带宽 × 延迟 = 4.8e12 × 500e-9 = 2.4 MB
每 SM = 2.4 MB / 132 = 18 KB
```

这个数的物理含义是 **bandwidth-delay product（BDP）**：为了让 4.8 TB/s 的管道不空，整条链路上任意时刻必须有 2.4 MB 数据在飞。

### 14.2 换算成 kernel 里能操作的单位

```
每 SM 需要：18.2 KB = 142 条 128 B 的 cache line 同时在飞
```

一个 warp 发一条 128 B 的合并 load（32 lane × 4 B），只贡献 **1 条**。所以要 142 个 warp 同时有待完成的 load —— **而一个 SM 最多只有 64 个 warp**。

**结论：每个 warp 必须有 2–3 个未完成的 load 同时在飞。**

```
64 warps × 2.5 outstanding loads = 160 ≥ 142 ✓
```

### 14.3 这个结论怎么落地

| 手段 | 作用 | 典型做法 |
|---|---|---|
| 循环展开 | 一个 warp 一次发多条 load | `#pragma unroll 4`，一次读 4 个 tile |
| 向量化 | 一条 load 搬 16 B 而不是 4 B | `float4` / `int4`，32 lane × 16 B = 512 B = 4 条 line |
| `cp.async` / TMA | 不经过寄存器，直接 global→SMEM，且可排队多个 | §18 |
| 提高 occupancy | 更多 warp 分担 | 降寄存器、降 SMEM |
| 减少依赖 | 让 load 后面不紧跟使用它的指令 | 软件流水（§18） |

### 14.4 主线那个估算的三个不严谨处

1. **500 ns 是「随机访问」的延迟**。顺序流在 row buffer 命中时，有效延迟会降到 100–200 ns（§28.3），所需在途量也随之降到 0.5–1 MB。BDP 是保守上界。
2. **L2 命中会进一步缩短有效延迟**。命中 L2 约 200 ns，对应 0.96 MB。
3. **单 SM 能挂起的未完成请求（MSHR / LSU 队列）有硬上限**，公开文档不给具体数值；实测单 SM 的 copy 带宽通常只有整机 1/132 份额的 30–60%。所以「每 SM 142 条 line」是 **132 个 SM 全部活跃时的摊派值**，不是单个 SM 的目标。

> 但量级是对的，而且**方向极其重要**：decode 阶段 batch=1、每层只有几十个 block、每个 block 几条 load —— **在途请求数比 142 少一个数量级**，所以带宽只能跑到 30–50%。这是主线 §4.2 说的「batch=1 时连一半都不到」的真正机制。

### 14.5 算力的 Little's Law（对照）

延迟 4–20 cycles，每 SM 每 cycle 要 256 FLOP（FP32）→ 在途约 `20 × 256 = 5120 FLOP`，只需 2–3 个 warp 的独立指令。**所以算力比带宽容易喂饱得多。** 这也解释了为什么「算力富余、带宽紧缺」是 LLM 推理的常态。

---

## 15 合并访存：一个 warp 的 32 个地址怎么变成 HBM 事务

### 15.1 三级粒度

| 粒度 | 大小 | 谁决定 |
|---|---|---|
| sector | **32 B** | L1↔L2 的最小传输单位 |
| cache line | **128 B**（= 4 sectors） | L1/L2 的行 |
| DRAM burst | 32–64 B | HBM 代际 |

**关键：即使你只用 4 B，HBM 也要搬 32 B。** 所有带宽浪费都从这条规则来。

### 15.2 五种访问模式的账

`float` 数组，warp 的 32 个 lane 同时读：

| 模式 | 地址跨度 | sector 数 | 搬运字节 | 有效字节 | 效率 |
|---|---|---|---|---|---|
| 连续（`a[tid]`） | 128 B | 4 | 128 B | 128 B | **100%** |
| 步长 2（`a[2*tid]`） | 256 B | 8 | 256 B | 128 B | 50% |
| 步长 4（`a[4*tid]`） | 512 B | 16 | 512 B | 128 B | 25% |
| 步长 8（`a[8*tid]`） | 1024 B | 32 | 1024 B | 128 B | **12.5%** |
| 全部同址（广播） | 4 B | 1 | 32 B | 128 B | 400% |

两个反直觉点：

- **最坏浪费是 8 倍，不是 32 倍**。因为 32 B 的 sector 粒度兜住了：步长 8 个 float（32 B）时，每个 lane 独占一个 sector，正好 8 倍。
- **广播是免费的**。32 个 lane 读同一个地址，硬件只发一个请求。

### 15.3 一个真实例子：读矩阵的一列

```cuda
// A 是 [1024][1024] 的 float，行主序
// 目标：warp 读 A[0..31][0]
float v = A[tid][0];        // 地址 = tid * 1024 * 4 = tid * 4096 B
```

32 个 lane 的地址相差 4096 B → **每个 lane 落在不同的 line 上** → 32 个 sector，1024 B 搬运，128 B 有用 → **12.5%**。

**这就是「转置的代价」在硬件层的真身。** 解决办法不是写更快的 kernel，是改布局（§1.3）：

- 用 `A^T` 存（列主序读 → 连续）；
- 或者按 tile 转置进 SMEM，在 SMEM 里做转置（SMEM 没有 sector 概念，只有 bank，见 §16）；
- 或者让 cuBLAS 的 `CUBLAS_OP_T` 处理。

### 15.4 向量化：一次搬 16 B

```cuda
float4 v = *reinterpret_cast<const float4*>(&a[4 * tid]);
```

32 lane × 16 B = 512 B = 4 条 line = 16 sectors，效率 100%，**而且指令数只有标量版的 1/4**。所有高性能 kernel 的 load 都是 128-bit 的。

**对齐要求**：`float4` 要求 16 B 对齐；PyTorch 的分配器保证 512 B 对齐，所以 `x[1:]` 这种切片会破坏对齐 → 每次多搬一个 sector。

### 15.5 判断有没有合并好

```bash
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum ./app
```

```
sectors / request
  4   → 完美（128 B / 32 B）
  8   → 效率 50%
  32  → 效率 12.5%
```

理想值 = 4（每次请求 4 个 sector）。任何大于 4 的值都是浪费。

---

## 16 共享内存：32 个 bank 和三种冲突

### 16.1 物理结构

```
SMEM 被切成 32 个 bank，每个 bank 每 cycle 服务 1 个 4 B 字
bank 编号 = (字节地址 / 4) % 32
```

一个 warp 的 32 个 lane 访问 SMEM 时：

| 情况 | 事务数 | 名称 |
|---|---|---|
| 32 个 lane 落在 32 个不同 bank | 1 | 无冲突 |
| k 个 lane 落在同一个 bank 的不同地址 | k | **k-way bank conflict** |
| 多个 lane 落在同一 bank 的同一地址 | 1 | 广播，不算冲突 |

### 16.2 经典冲突：按列访问

```cuda
__shared__ float tile[32][32];
float v = tile[tid][0];          // 地址 = tid*32*4
// bank = (tid * 32) % 32 = 0  ← 32 个 lane 全撞 bank 0
```

**32-way conflict → 一次访问退化成 32 次。** 修法：

```cuda
__shared__ float tile[32][33];   // padding 一列
// bank = (tid * 33) % 32 = tid % 32  ← 无冲突
```

### 16.3 Swizzle：不 padding 也能避开

padding 的代价是 SMEM 用量增加 3%，且对 `ldmatrix` 的固定布局不友好。生产 kernel 用 **XOR swizzle**：

```
物理列 = 逻辑列 XOR (行号 & mask)
```

CUTLASS 的 `Swizzle<3,3,3>` 就是这一族。效果：同一行内相邻的元素被打散到不同 bank，且仍能用一条 `ldmatrix` 指令批量加载。

### 16.4 SMEM 的带宽（顺手验证主线 §4.5）

```
每 SM 每 cycle 128 B
× 1.8e9 = 230 GB/s / SM
× 132 SM = 30.4 TB/s 聚合
```

主线写的「~30 TB/s 聚合」就是这个数。**它是 HBM 的 6.3 倍**，所以「把数据搬进 SMEM 反复用」永远比「反复读 HBM」划算。

### 16.5 什么时候该用 SMEM

| 场景 | 该用吗 | 理由 |
|---|---|---|
| GEMM 的 tile 复用 | 必须 | 一个 tile 被 T 个输出复用，见 §19 |
| attention 的 K/V 分块 | 必须 | FlashAttention 的全部前提 |
| 归约（sum/max） | 必须 | 跨 warp 通信只能走 SMEM |
| 纯 elementwise | 不需要 | 没有复用，直接 global→register |
| 转置 | 必须 | SMEM 没有 sector 惩罚 |

---

## 17 寄存器：最贵也最快的存储

### 17.1 事实

| 项 | 值 |
|---|---|
| 每 SM 寄存器 | 65536 × 32 bit = 256 KB |
| 每线程上限 | 255 个 |
| 访问延迟 | 1 cycle（无冲突时） |
| 带宽 | 数十 TB/s 量级 [calc] |

### 17.2 spill：寄存器不够时会发生什么

编译器把放不下的变量放到 **local memory** —— 注意这个名字是骗人的，**它在 HBM 上**（有 L1/L2 缓存）：

```
寄存器 255 个不够
  → spill 到 local memory
  → 每次访问变成一次 global load（走 L1）
  → 延迟从 1 cycle 变成 30~500 cycles
  → 带宽账上凭空多出几十 MB
```

`ncu` 里看 `launch__registers_per_thread` 和 `smsp__sass_inst_executed_op_local_ld`。**后者不为 0 就是 spill 了。**

### 17.3 GEMM 的取舍：tile 越大，寄存器越紧张

128×128 tile、256 线程、每线程 8×8 输出：

```
累加器：8×8 = 64 个寄存器（FP32）
操作数 A/B fragment：~16–32 个
地址/循环变量：~10 个
                        ─────
                        ~100–110 个 / 线程
```

对应 occupancy：

```
65536 / (110 × 32) ≈ 18 warps → 28%
```

**GEMM 就是故意跑在 28% occupancy 上的**：它用 64 个独立累加器换 ILP，用 ILP 换延迟隐藏（§13.3）。想强行拉高 occupancy（`__launch_bounds__(256, 6)`），编译器只能减 tile，反而变慢。

### 17.4 一条实用规则

```
memory-bound kernel：优先 occupancy（少用寄存器）
compute-bound kernel：优先 ILP（多用寄存器，容忍低 occupancy）
```

判断自己是哪一种，用 §39 的决策树。

---

## 18 同步与流水：把「等」变成「重叠」

### 18.1 `__syncthreads()` 的真实成本

它不只是「一条指令」，而是：

```
① 所有 warp 到达 barrier
② 最慢的那个 warp 决定等待时间   ← 真正的成本
③ 内存屏障（保证 SMEM 写入可见）
④ 重新调度
```

**成本 = 最快与最慢 warp 的差 + 固定开销（~20–30 cycles）。** 所以 load 不均（imbalanced）比 barrier 本身贵得多。

### 18.2 双缓冲：GEMM 的标准做法

```
时间 →
SMEM[0]: [load k=0] [compute k=0] [load k=2] ...
SMEM[1]:            [load k=1]    [compute k=1] ...
```

计算第 k 块的同时，DMA 在搬第 k+1 块。**没有双缓冲，SMEM 加载时间和计算时间相加；有了双缓冲，取 max。**

对 4096³ GEMM：

```
不流水： 0.447 ms（内存）+ 0.139 ms（算力）= 0.586 ms
流水：   max(0.447, 0.139) = 0.447 ms
```

省 24%，而且这是理论上限 —— 实际流水能藏掉 70–90% 的加载时间。

### 18.3 三代异步拷贝

| 代际 | 指令 | 特点 |
|---|---|---|
| Ampere | `cp.async.ca.shared.global` | global→SMEM，**不经过寄存器**，可挂起多个，`commit_group` / `wait_group` 控制 |
| Hopper | `cp.async.bulk.tensor`（TMA） | **一个线程发起整个 tile 的拷贝**，地址生成由硬件做，支持多维、支持 multicast 到多个 SM 的 SMEM |
| Blackwell | 同上 + tcgen05 配套 | MMA 由单线程发起，操作数从 SMEM/TMEM 直取 |

### 18.4 TMA 改变了什么

```
Ampere：  32 个 lane 各自算地址 → 发 load → 写 SMEM（要 __syncthreads）
Hopper：  1 个线程填一张 descriptor → 发 TMA → 硬件搬整块 → mbarrier 通知
```

三个后果：

1. **地址计算的指令全部消失**（对 AI 没有影响，但对 issue 带宽有大影响）；
2. **可以多播**：一个 tile 搬到 cluster 里多个 SM，省 L2 带宽；
3. **必须用 mbarrier 做生产者-消费者同步**，代码结构从「所有 warp 干所有事」变成 **warp specialization**。

### 18.5 Warp specialization（Hopper/Blackwell 的主流结构）

```
producer warps（1–2 个）：只发 TMA，不计算
consumer warps（4–8 个）：只发 wgmma / tcgen05，不算地址
epilogue warps：把累加器写回 HBM
```

CUTLASS 3.x 的 GEMM 就是这个结构。**它的本质是把 §14 的「在途请求」交给专门的 warp 去堆，计算 warp 完全不被打断。**

---

## 19 一个 GEMM 的五级演进：从 1 TFLOPS 到 850 TFLOPS

这是全文最重要的一节。它把「代码怎么写」和「硬件怎么跑」放在同一张表里。

**固定问题**：`C[4096,4096] = A[4096,4096] @ B[4096,4096]`，BF16。

```
FLOPs        = 2 × 4096³            = 137.4 GFLOP
最少字节      = 2 × 4096² × 2 B      = 67.1 MB（A、B 各读一遍）
理想 AI       = 137.4e9 / 67.1e6     = 2048 FLOP/B
算力时间      = 137.4e9 / 989e12     = 0.139 ms
带宽时间      = 67.1e6 / 4.8e12      = 0.014 ms
```

理想情况下算力时间是带宽时间的 **10 倍** → 这个 GEMM **应该**是算力受限的。下面看五级实现为什么离这个理想十万八千里。

### 19.0 两个天花板，取 min

任何 kernel 的吞吐都受两个天花板约束：

$$
\text{achieved TFLOPS} \le \min\Big(\underbrace{AI \times 4.8\,\text{TB/s}}_{\text{内存侧}},\ \underbrace{\text{算力峰值}}_{\text{计算侧}}\Big)
$$

五级演进的本质：**前三级在抬内存侧天花板，第四级才抬计算侧天花板。**

| 阶段 | 关键改动 | AI (FLOP/B) | 内存侧上限 | 计算侧上限 | 实测 [meas 区间] |
|---|---|---|---|---|---|
| 0 | 一线程一输出 | 0.5 | **2.4 TFLOPS** | 60.8 | 1–3 |
| 1 | SMEM 32×32 tile | 16 | 76.8 | 60.8 | 15–35 |
| 2 | 128×128 tile + 8×8 寄存器 | 64 | 307 | **60.8** | 45–58 |
| 3 | cp.async 流水 + float4 | 64（+L2 复用） | ~300–1000 | **60.8** | 55–60 |
| 4 | Tensor Core + TMA + 流水线（含 L2 友好排序） | 64 → **~690**（L2 复用后） | ~3300 | **989** | 700–850 |
| — | cuBLAS 参考 | — | — | 989 | 750–850 |

**读法**：阶段 0→2 每级都有效；阶段 2 之后内存侧已经 307，而 **FP32 算力只有 60.8，先到顶了** —— 继续优化访存是白干的。阶段 4 把算力天花板抬了 16 倍，才有 850。

### 19.1 阶段 0：一线程一输出

```cuda
__global__ void gemm_naive(const __nv_bfloat16* A, const __nv_bfloat16* B,
                           __nv_bfloat16* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.f;
    for (int k = 0; k < K; ++k)
        acc += float(A[row * K + k]) * float(B[k * N + col]);   // 每次 2 次 global load
    C[row * N + col] = __float2bfloat16(acc);
}
```

每个输出元素要 `2K` 次 global load，AI：

$$
AI = \frac{2K\ \text{FLOPs}}{2K \times 2\,\text{B}} = 0.5\ \text{FLOP/B}
$$

内存侧上限 `0.5 × 4.8 TB/s = 2.4 TFLOPS`。**这是 989 的 0.24%。**

实测能到 1–3 TFLOPS，比 2.4 略高 —— 因为部分数据被 L1/L2 接住了（尤其 `B[k][col]` 在相邻 lane 之间是连续地址，合并得不错）。

### 19.2 阶段 1：SMEM tiling

**观察**：一个 tile 的 A 和 B 会被 tile 内所有输出复用。

```cuda
#define T 32
__global__ void gemm_smem(const __nv_bfloat16* A, const __nv_bfloat16* B,
                          __nv_bfloat16* C, int M, int N, int K) {
    __shared__ __nv_bfloat16 As[T][T];
    __shared__ __nv_bfloat16 Bs[T][T];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * T + ty, col = blockIdx.x * T + tx;
    float acc = 0.f;

    for (int k0 = 0; k0 < K; k0 += T) {
        As[ty][tx] = A[row * K + k0 + tx];      // 合并访存 ✓
        Bs[ty][tx] = B[(k0 + ty) * N + col];    // 合并访存 ✓
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < T; ++k)
            acc += float(As[ty][k]) * float(Bs[k][tx]);
        __syncthreads();
    }
    C[row * N + col] = __float2bfloat16(acc);
}
```

每个 block 搬 `2T²` 个元素，算 `2T³` 次乘加：

$$
AI = \frac{2T^3}{2T^2 \times 2\,\text{B}} = \frac{T}{2} = 16\ \text{FLOP/B}
$$

内存侧上限 `76.8 TFLOPS`，**超过 FP32 的 60.8** → 变成算力受限。

实测 15–35 TFLOPS：到不了 60.8，因为 `__syncthreads` 每 T 次乘加就来一次，以及 `As[ty][k]` 是广播、`Bs[k][tx]` 无冲突但仍有 SMEM 延迟。

### 19.3 阶段 2：128×128 tile + 8×8 寄存器分块

```
每 block：128×128 输出，256 线程
每线程：8×8 = 64 个输出，64 个累加器
SMEM：As[128][8]、Bs[8][128]（K 方向每次进 8）
```

AI 变成 `T/2 = 64` → 内存侧 307 TFLOPS。**但实测只有 45–58，正好压在 FP32 的 60.8 天花板上。**

这一级是全文最重要的工程结论：

> **当 AI 已经超过拐点（206 是 FP8 口径，BF16 是 206 也是同一数）之后，再优化访存没有收益 —— 因为在同一张 roofline 图上你已经站在「计算受限」的平坦区。**

BF16 的拐点是 `989/4.8 = 206`。阶段 2 的 AI 只有 64，还在左边 —— 但因为用 FP32 FMA（60.8 TFLOPS）计算，实际拐点退化成 `60.8/4.8 = 12.7`，**AI=64 早就越过了**。所以：

```
用 FP32 core 算 GEMM：拐点 12.7，AI 64 → 算力受限，天花板 60.8
用 Tensor Core 算 GEMM：拐点 206，AI 64 → 内存受限，天花板 307
```

**同一份代码，换执行单元，瓶颈从算力变成内存。** 这是 §34 讲「AI 必须和峰值配套」的第一个实例。

### 19.4 阶段 3：异步流水

阶段 2 有一个明显的洞：`As/Bs` 的加载和计算是**串行**的（`__syncthreads` 卡在中间）。

```cuda
// 双缓冲 + cp.async
cp.async.cg.shared.global [As[next]], [A + ...], 16;
cp.async.commit_group;
// ... 计算当前 buffer ...
cp.async.wait_group 0;
__syncthreads();
```

效果（用 §19 的 4096³ 数据）：

```
不流水： 0.447 ms（内存）+ 2.26 ms（FP32 算力）= 2.71 ms
流水：   max(0.447, 2.26) = 2.26 ms          ← 完全藏住访存
```

**但天花板还是 60.8 TFLOPS。** 实测 55–60 已经贴顶，说明这一级做对了，也说明**再往下必须换执行单元**。

### 19.5 阶段 4：Tensor Core

先回答一个必然会被问的问题：**阶段 2 的 AI 只有 64，为什么阶段 4 能跑到 850 TFLOPS（拐点 206 的 4 倍）？**

因为 64 是「每个 block 只看一遍自己的 tile」的 AI。**L2 会把 A/B 的 slab 缓存住**：

```
不加 L2 友好排序：A、B 各被读 N/T = 32 遍 → 2.1 GB → 有效 AI ≈ 64
加 threadblock swizzle：DRAM 流量降到 ~200 MB → 有效 AI ≈ 690 → 越过 206
```

**这就是 CUTLASS 里 threadblock swizzle 的全部意义**：不改一行数学，只改 block 的执行顺序，把 HBM 流量降一个量级。它也是 §34「L2 命中算不算」那个口径陷阱的物理来源。

```cuda
// 一个 warp 计算 16×8 的输出块，K 方向一次 16
wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
wmma::load_matrix_sync(a, As + ..., lda);    // 实际是 ldmatrix
wmma::load_matrix_sync(b, Bs + ..., ldb);
wmma::mma_sync(c, a, b, c);                  // 一条指令 = 2048 次 MAC
```

计算侧天花板从 60.8 抬到 **989**（16 倍）。Hopper 上再叠三层：

| 叠加 | 收益 | 对应机制 |
|---|---|---|
| `wgmma`（warpgroup MMA，m64nNk16） | 单指令覆盖 4 个 warp 的输出 | §25 |
| TMA | 地址计算从 32 lane 降到 1 个线程 | §18.4 |
| warp specialization | 计算 warp 完全不被访存打断 | §18.5 |

最终 700–850 TFLOPS，即 **MFU 71–86%**。剩下的 15–30% 去哪了：

| 损失项 | 量级 |
|---|---|
| 边界 tile（4096 不是 128 的整数倍时） | 0–5% |
| epilogue（写回 + bias） | 3–8% |
| K 循环的流水启动/排空 | 2–5% |
| clock 未跑满 boost | 3–10% |
| 显存 refresh / 其他进程干扰 | 1–3% |

### 19.6 这一节的四句话

```
① AI 不够时，优化访存有线性收益（阶段 0→1→2）
② AI 够了但执行单元慢，优化访存零收益（阶段 2→3）
③ 换执行单元（FP32 → TC）是唯一能抬计算侧天花板的手段（阶段 3→4）
④ 判断自己在哪个阶段，只需要两个数：实测 AI、实测 TFLOPS
```

---

## 20 softmax、归约与 FlashAttention：为什么 attention 必须融合

### 20.1 朴素 attention 的字节账

prefill，L=4096，H_q=64，d=128，BF16，**单层**：

```
S = Q @ Kᵀ        [64, 4096, 4096] × 2 B = 2.15 GB
softmax(S) → P    读 2.15 GB + 写 2.15 GB
O = P @ V         读 2.15 GB
                          ─────────────
单层往返          ≈ 8.6 GB
80 层             ≈ 688 GB
时间（4.8 TB/s）  ≈ 143 ms
```

**S 矩阵本身就有 2.15 GB / 层**，根本装不进 50 MB 的 L2，每次都要落 HBM。

### 20.2 FlashAttention 的字节账

分块之后，**S 的一个 tile 算完立刻用掉，永不落 HBM**：

```
读 K、V：2 × 4096 × 128 × 2 B = 2.1 MB / head
× 64 head = 134 MB / 层
× 80 层   = 10.7 GB
时间      = 2.2 ms
```

**143 ms → 2.2 ms，65 倍。** 而 FLOPs 一个都没少 —— 这就是主线 §4.5 说的「省的是带宽，不是算力」。

### 20.3 同一个算子，两个 AI

| 阶段 | FLOPs（单层，causal） | HBM 字节 | AI | 瓶颈 |
|---|---|---|---|---|
| prefill（FA） | ~275 GFLOP | 134 MB + Q/O | **~2000** | 算力 |
| decode（B=1） | 4·L·d·H_q ≈ 134 MFLOP | KV 2·L·d·H_kv·2 B = 2.1 MB | **~8**（主线 §1.14） | 带宽 |

**AI 差 250 倍。** 所以「attention 是计算密集还是带宽密集」这个问题本身没有答案 —— 必须问「哪个阶段」。

### 20.4 online softmax：融合的数学前提

朴素 softmax 要三趟：求 max、求 exp 和、归一化。融合的前提是**一遍扫完**：

$$
m_i = \max(m_{i-1}, \tilde m_i),\qquad
l_i = l_{i-1} e^{m_{i-1}-m_i} + \tilde l_i e^{\tilde m_i - m_i}
$$

新的 max 变大时，**把旧的累加和按 $e^{\Delta}$ 缩放**。这保证了：无论分几块，结果与全局 softmax 完全一致（主线 §3 已给证明路径）。

没有这个恒等式，FlashAttention 不可能只读一遍 KV。

---

## 21 launch 成本、grid 规模与小 batch 的并行度账

### 21.1 全机的并行度预算

```
132 SM × 64 warp = 8448 warp = 270,336 线程
```

要让机器「看起来忙」，至少需要 30–50% 的 warp 槽位被有效工作占据：

```
≈ 2500–4000 warp ≈ 8–13 万线程
```

### 21.2 decode 的实际并行度

**GEMM（M=1，N=8192，K=8192），tile 128×128：**

```
grid = ceil(8192/128) × ceil(1/128) = 64 × 1 = 64 block
每 block 256 线程 = 8 warp
总 warp = 512 → 全机的 6%
SM 占用 = 64 / 132 = 48%（每个只放 1 个 block，occupancy 8/64 = 12.5%）
```

**attention（B=1，H_q=64）：**

```
grid ≈ 64 block（一 head 一 block）
warp 数取决于 block 大小，通常 4–8
总 warp ≈ 256–512 → 全机的 3–6%
```

**这就是主线 §8.4「Tensor Core 只用 1.8%」的机制**：不是 kernel 写得差，是**工作量本身没有足够的并行度**。132 个 SM 里有一半空闲，剩下的每个只放了 1/8 的 warp。

### 21.3 四种补并行度的手段

| 手段 | 做法 | 代价 |
|---|---|---|
| **split-K** | 把 K 方向切段，各段部分和再归约 | 多一遍 partial sum 的读写 |
| **FlashDecoding** | 把 L 方向切段，各段算 partial attention 再合并 | 多一遍 O 和 m/l 的归约 |
| **更大 batch** | B 变大，M 变大，grid 线性增长 | KV 容量、SLO |
| **更多 head 并行** | 每 head 拆多个 block | 归约开销 |

主线 §5 提到 FlashDecoding 解决「并行度不足」，但**一个字节都省不了** —— 正是因为它补的是 §21.2 这个洞，不是带宽。

### 21.4 tail effect：为什么 grid 最好是 132 的倍数

```
grid = 132 → 1 波，完美
grid = 133 → 2 波，第二波只有 1 个 block → 利用率 50.4%
grid = 264 → 2 波，完美
```

对固定形状的 kernel，**grid 数应该接近 132 的整数倍**。这就是为什么「batch 从 63 涨到 64 有时反而更快」——它刚好跨过一个波次边界。

---

## 22 第二部分小结

```
warp = 32 lane 共享 PC；SM 最多 64 warp / 32 block / 64K 寄存器 / 228 KB SMEM
divergence 的代价 = 路径数 × 长度；让 if 边界对齐 warp 或 tile
occupancy 四个限制因子：寄存器、SMEM、线程数、block 数
寄存器是第一限制因子：64 regs/thread → 50% occupancy
带宽需要的在途量 = 带宽 × 延迟 = 2.4 MB = 142 条 line/SM
所以每 warp 必须有 2–3 个未完成 load（展开 / 向量化 / cp.async）
合并访存：sector 32 B，line 128 B；步长 8 float → 效率 12.5%
SMEM：32 bank，按列访问 32-way conflict；padding 或 swizzle
GEMM 五级：AI 0.5 → 16 → 64；FP32 天花板 60.8；TC 天花板 989
attention 融合：143 ms → 2.2 ms（65×），FLOPs 不变
decode 的并行度只有全机的 3–6%
grid 最好接近 132 的倍数
```

### 症状 → 该查哪一节

| `ncu` 里的症状 | 最可能的原因 | 去看 |
|---|---|---|
| `dram__throughput` 低，`sectors/request` = 32 | 没合并访存 | §15 |
| `dram__throughput` 低，`sectors/request` = 4 | 在途请求不够（occupancy 或展开不足） | §14、§13 |
| `smsp__warp_issue_stalled_mio_throttle` 高 | SMEM bank conflict 或 LSU 排队 | §16 |
| `smsp__warp_issue_stalled_barrier` 高 | load 不均 / 流水没做好 | §18 |
| `launch__registers_per_thread` 高 + `local_ld` > 0 | 寄存器 spill | §17 |
| `sm__pipe_tensor_cycles_active` 高但 TFLOPS 低 | MMA 之间有空泡（依赖/发射不足） | §25、§19.5 |
| `sm__throughput` 低、`dram` 低、`warps_active` 低 | 并行度不足（grid 太小） | §21 |
| 时间线上 kernel 之间有长空隙 | launch 开销 / CPU-bound | §3、§6 |

---

# 第三部分 硬件层：SM 内部与内存子系统

前两部分讲的是「怎么用」。这一部分讲「用的是什么东西」。

目标：让你看到 `dram__throughput = 55%` 时，脑子里浮现的不是一个百分比，而是 **6 个 HBM 堆栈、上千个 bank、一个正在等 row 打开的请求**。

---

## 23 die 平面：从 GPC 到 SM 到 HBM

### 23.1 层级与数量

| 层级 | 数量（GH100，H200 同 die） | 说明 |
|---|---|---|
| die | 1（814 mm²，800 亿晶体管）[spec] | 单 die，不是 chiplet |
| GPC（Graphics Processing Cluster） | 8 | 顶层分区 |
| TPC（Texture Processing Cluster） | 9 / GPC → 72 | 每 TPC 2 个 SM |
| **SM** | 8 × 9 × 2 = **144**（H100/H200 启用 **132**） | 屏蔽 12 个做良率/功耗 |
| Tensor Core | 4 / SM → 528 | |
| FP32 CUDA Core | 128 / SM → 16,896 | |
| L2 slice | 多个，绑定 memory controller | 总计 **50 MB** |
| HBM 堆栈 | **6** | 通过 silicon interposer 连接 |
| memory controller | 2 / 堆栈 → **12** | |

### 23.2 三个「不是」 

1. **GPC 不是调度单位**。调度单位是 SM 里的 warp scheduler。GPC 只影响 L2/显存的物理就近性（GPC 到 L2 的走线长度）。
2. **SM 不是「一个大核」**。它是 4 个几乎独立的 sub-partition，共享 L1/SMEM。
3. **L2 不是「一块缓存」**。它是按地址 hash 分布到多个 slice 的，每个 slice 绑定一组 memory controller —— **这决定了「连续地址」为什么比「随机地址」快**（§28）。

### 23.3 为什么 interposer 重要

HBM 的 6144 bit 总线**走不了 PCB**，必须在硅 interposer 上做（CoWoS）。这也是为什么显存容量和带宽是焊死的 —— 不存在「加一条内存条」。

> 推论：**H200 的 141 GB 不是「可扩展」的容量，换卡就是换机器。** 主线 §8 把「换卡」列为突破 decode 下界的第 ④ 条路，代价最高。

---

## 24 SM 微架构：一条指令的一生

### 24.1 四个 sub-partition

```
SM
├── sub-partition 0: 1 warp scheduler + 32 FP32 lane + 1 Tensor Core + 16K 寄存器
├── sub-partition 1: 同
├── sub-partition 2: 同
├── sub-partition 3: 同
└── 共享：L1TEX（256 KB 统一，含最多 228 KB SMEM）、4 个 SFU、LSU
```

| 资源 | 每 sub-partition | 每 SM |
|---|---|---|
| warp scheduler | 1 | 4 |
| warp 槽位 | 16（512 线程） | 64（2048 线程） |
| FP32 lane | 32 | 128 |
| Tensor Core | 1 | 4 |
| 寄存器 | 16K × 32 bit = 64 KB | 64K × 32 bit = 256 KB |
| SFU | 4 | 16 |

### 24.2 发射：每 cycle 一条 warp 指令

每个 scheduler 每 cycle 从自己的 16 个 warp 里挑 **一个 ready 的**，发射**一条指令**：

```
fetch → decode → issue → operand collector → execute → writeback
```

三个直接后果：

1. **每 SM 每 cycle 最多 4 条 warp 指令**（4 个 scheduler）。这就是「指令发射带宽」的硬上限。
2. **一条指令服务 32 个 lane**。所以「每 cycle 4 条指令」= 最多 128 个 lane-指令/cycle/SM。
3. **warp 之间零成本切换**。这就是为什么 GPU 能靠「多 warp」藏延迟，而 CPU 要靠乱序执行。

### 24.3 各级延迟（1.8 GHz，1 cycle = 0.556 ns）

| 操作 | 周期 | 时间 | 备注 |
|---|---|---|---|
| 寄存器读 | 1 | 0.6 ns | |
| FFMA（FP32 乘加） | 4 | 2.2 ns | 依赖链上必须等 |
| HMMA / MMA（TC 指令） | 8–16 | 4–9 ns | 结果延迟，不是吞吐 |
| SMEM 访问 | 20–30 | 11–17 ns | 无冲突时 |
| L1 命中 | 30–40 | 17–22 ns | |
| L2 命中 | ~200 | ~110 ns | 跨 SM 通信也走这里 |
| HBM | **~900** | **~500 ns** | 主线用的 500 ns 就是这一行 |

**关键比例**：HBM 延迟 = 寄存器延迟的 **900 倍**。这就是为什么 §14 的「在途请求数」是带宽能否喂饱的唯一决定因素。

### 24.4 指令 mix 决定实际吞吐

主线说「GPU 利用率 90% 毫无意义」，在微架构层的解释是：

| 指令类型 | 走哪条流水线 | 峰值 |
|---|---|---|
| FFMA（FP32） | FP32 lane | 128/cycle/SM |
| HMMA（Tensor Core） | TC | 4 条/4 cycle |
| LDS/STS（SMEM） | LSU | 128 B/cycle/SM |
| LDG（global） | LSU → L1TEX | 128 B/cycle/SM（命中 L1）；miss 时受 HBM 限制 |
| MUFU（exp、rcp、sqrt） | SFU | 16/cycle/SM |
| 整数、分支 | INT/ALU | 64/cycle/SM |

**softmax 是 MUFU 密集型**：`exp` 每个元素一条 MUFU，SFU 只有 16 个/cycle/SM。算一下：

```
L=4096，H_q=64，B=1 的一次 decode attention：
exp 次数 = 64 head × 4096 = 262,144 次
SFU 吞吐 = 132 SM × 16 × 1.8e9 = 3.8e12/s
时间 = 262144 / 3.8e12 = 0.07 µs   ← 不是瓶颈
```

但如果把 `softmax` 写成「先算 max、再 exp、再 sum、再除」四趟，HBM 往返会先炸。**结论：exp 不贵，往返贵。**

---

## 25 Tensor Core 到底是什么

### 25.1 三代形态

| 代际 | 指令 | 一个 warp/warpgroup 算多大 | 特点 |
|---|---|---|---|
| Volta | `mma.sync.m8n8k4` | 8×8 输出，K=4 | 1 TC/SM，FP16 |
| Ampere | `mma.sync.m16n8k16` | 16×8 输出，K=16 | 4 TC/SM，BF16/FP16/TF32/INT8 |
| Hopper | `wgmma.mma_async.m64nNk16` | **64×N** 输出，K=16，**4 个 warp 协作** | 异步，A 可来自 SMEM，新增 FP8（`m16n8k32`） |
| Blackwell | `tcgen05.mma` | 单线程发起，结果进 **TMEM** | 2-SM 协作，256 KB TMEM/SM |

### 25.2 `m16n8k16` 的 fragment 布局（必须背下来）

一条 `mma.sync.aligned.m16n8k16` 由 **32 个 lane 共同完成**：

```
计算：D[16,8] += A[16,16] × B[16,8]
MAC 数：16 × 8 × 16 = 2048
每 lane 持有：A 8 个元素、B 4 个元素、D 4 个元素
验算：32×8 = 256 = 16×16 ✓   32×4 = 128 = 16×8 ✓
```

具体的「哪个 lane 拿哪个元素」是固定的：

| 矩阵 | lane `t` 持有的元素 |
|---|---|
| A（行主序） | 行 `t/4` 和 `t/4+8`；列 `(t%4)*2`、`+1`、`+8`、`+9` |
| B（列主序） | 列 `t/4`；行 `(t%4)*2`、`+1`、`+8`、`+9` |
| D | 行 `t/4` 和 `t/4+8`；列 `(t%4)*2`、`+1` |

**这就是「为什么 Tensor Core 要求特定数据布局」的答案**：不是硬件挑剔，是 **32 个 lane 的寄存器是唯一的输入通道**，必须按固定规则把矩阵切片塞进去。

### 25.3 `ldmatrix`：把 SMEM 里的 tile 灌进 fragment

手写上面的布局会累死，所以有一条专门的指令：

```cuda
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {r0,r1,r2,r3}, [addr];
```

**一个 warp、一条指令、搬 4 个 8×8 的 BF16 子块，自动按 fragment 布局分配到 32 个 lane 的寄存器。** 配合 XOR swizzle 的 SMEM 布局时，这次访问无 bank 冲突。

> CUTLASS 里所有的「SMEM → 寄存器」都走 `ldmatrix`。你自己写 `for` 循环逐个搬，光地址计算就吃掉 30% 的指令带宽。

### 25.4 Hopper `wgmma`：4 个 warp 拼一个 64 行

```
warpgroup = 4 warp = 128 lane
wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16
  ├── A：64×16，可来自 SMEM（descriptor）
  ├── B：16×256，来自 SMEM
  ├── D：64×256，累加在 128 个 lane 的寄存器里
  └── 异步：发起后可以立刻去干别的，用 commit/wait group 收
```

三个变化：

1. **A 可以直接从 SMEM 读**，不再需要 `ldmatrix` → 寄存器压力减半；
2. **异步** → 计算和数据搬运真正重叠（§18.4）；
3. **一条指令覆盖 64×256×16 = 262,144 次 MAC** → 指令开销可忽略。

### 25.5 Blackwell `tcgen05`：把累加器搬出寄存器

| 项 | Hopper | Blackwell |
|---|---|---|
| MMA 发起者 | 1 个 warpgroup（128 线程） | **1 个线程** |
| 累加器位置 | 寄存器（每 lane 若干） | **TMEM（256 KB / SM）** |
| 操作数 | SMEM / 寄存器 | SMEM / TMEM |
| 跨 SM | cluster 内 DSMEM | 2-SM 协作 MMA |

对 LLM 推理的实际影响：**累加器不再占寄存器 → tile 可以更大 → 同样的 SMEM 能喂更大的 MMA → 对 §19 的五级演进来说，「阶段 2 的寄存器天花板」被抬高了。**

### 25.6 32 倍从哪来

拆成两步：

```
① 每 cycle 的 MAC 数
   FP32 CUDA Core：128 lane × 1 FMA        = 128 MAC/cycle/SM
   BF16 Tensor Core：4 TC × 512 MAC/cycle    = 2048 MAC/cycle/SM   → 16×
   FP8  Tensor Core：4 TC × 1024 MAC/cycle   = 4096 MAC/cycle/SM   → 32×

② 时钟、SM 数相同 → 比例不变
```

**所以主线那个「32 倍」是 FP8 TC 对 FP32 CUDA Core；BF16 对 FP32 是 16 倍。** 这个区分在算 MFU 时很重要 —— 拿 BF16 的吞吐去除 FP8 的峰值，MFU 会莫名其妙只有 50%。

### 25.7 2:4 稀疏：为什么本文一律用 dense

结构化稀疏（每 4 个权重保留 2 个）能让 TC 吞吐翻倍（官方标称的 3958 TFLOPS 就是稀疏值）。但：

- 需要**训练时**就约束稀疏模式，或事后用带精度损失的剪枝；
- 需要专门的 sparse MMA 指令和元数据布局；
- 实际 LLM 部署里，**主流的「省字节」手段是量化（FP8/FP4），不是稀疏**。

所以本文与主线一律用 **dense 峰值**：989（BF16）/ 1979（FP8）。**看到别人报 MFU 60%，先问他是 dense 还是 sparse 口径。**

---

## 26 峰值算力的完整推导

### 26.1 一张表

| 精度 | 执行单元 | 每 cycle 每 SM | 全机 TFLOPS | 推导 |
|---|---|---|---|---|
| FP64 | CUDA Core | 64 FLOP | 34 | 官方 |
| FP32 | CUDA Core | 128 FMA = 256 FLOP | **60.8** | 132×128×2×1.8e9 |
| TF32 | TC | 2048 FLOP | 495 | 官方 |
| BF16/FP16 | TC | 4096 FLOP | **989** | 528×512×2×1.8e9 |
| FP8 | TC | 8192 FLOP | **1979** | 528×1024×2×1.83e9 |
| INT8 | TC | 8192 OP | 3958 TOPS | 官方 |

### 26.2 那个对不上的 1.8 GHz

主线用 1.8 GHz 算出 1946，官方标称 1979。差 1.7%，原因是：

```
1979e12 / (528 TC × 1024 MAC × 2) = 1.830 GHz
```

**官方数字对应的等效主频是 1.83 GHz，不是名义的 1.8。** 这不影响任何结论（412 的误差 1.7%），但**你如果拿 2.0 GHz boost 去复算，会得到 2163，然后以为公式错了**。

### 26.3 MFU 的定义与那个 40%

$$
\text{MFU} = \frac{\text{实测 FLOP/s}}{\text{峰值 FLOP/s（同精度、dense）}}
$$

主线取 MFU = 40% → 有效算力 792 TFLOPS。这个 40% 是**端到端**（含 attention、norm、softmax、通信、气泡），不是单个 GEMM 的：

| 层次 | 典型值 | 说明 |
|---|---|---|
| 单个大 GEMM kernel | 70–85% | §19 阶段 4 |
| 一层 transformer（含所有算子） | 45–60% | 小算子拉低 |
| 端到端（含调度、通信、气泡） | 35–50% | 主线取 40% |

**把单 kernel 的 80% 当成端到端 MFU，是容量估算最常见的 2 倍高估。**

---

## 27 内存层级：容量、带宽、延迟，以及「能装多少 KV」

### 27.1 五层

| 层级 | 容量 | 带宽 | 延迟 | 谁管理 |
|---|---|---|---|---|
| 寄存器 | 256 KB / SM | 数十 TB/s [calc] | 1 cycle | 编译器 |
| SMEM | 228 KB / SM | 128 B/cycle → 230 GB/s/SM → **30 TB/s 聚合** | 20–30 cycle | 你（显式分配） |
| L1 | 与 SMEM 共享 256 KB | ~128 B/cycle | 30–40 cycle | 硬件 |
| L2 | **50 MB** | **~7 TB/s** [calc] | ~200 cycle | 硬件 |
| HBM | **141 GB** | **4.8 TB/s** | ~900 cycle | 你（分配）+ 硬件 |

### 27.2 每层能装多少 KV

Llama-3-70B，KV = 320 KiB/token：

| 层级 | 容量 | 能装多少 token 的 KV |
|---|---|---|
| SMEM（228 KB） | 228 KB | **0.71** |
| L2（50 MB） | 50 MB | **160** |
| HBM（141 GB） | 141 GB | 430,000 |
| KV 池（50 GiB，主线口径） | 50 GiB | **163,840** |

**这张表解释了 FlashAttention 为什么必须分块**：一个 SM 的 SMEM 连 1 个 token 的 KV 都装不下（0.71），所以只能装「一个 head 的一段 d」—— 而 d=128 的 BF16 K 向量是 256 B，一段 L=64 的 K tile 是 16 KB。**分块尺寸是被 SMEM 逼出来的，不是算法偏好。**

### 27.3 L2 的两个角色

1. **缓存**：临时接住复用数据；
2. **跨 SM 通信总线**：SM 之间不能直接读对方的 SMEM（除 Hopper cluster），必须走 L2。

**L2 带宽只有 HBM 的 1.46 倍** —— 这个比例是 §19 阶段 2→3 的关键：**L2 能补的复用有限，不能靠它把 AI 从 64 抬到 2048。** 真正的复用必须发生在 SMEM/寄存器里。

---

## 28 HBM 物理：4.8 TB/s 是怎么来的

### 28.1 堆栈与位宽

```
6 个 HBM3e 堆栈
每堆栈 1024 bit 数据总线（16 个 pseudo-channel × 64 bit）
总位宽 = 6 × 1024 = 6144 bit = 768 B / 次传输
```

反算 pin 速率：

$$
\text{pin rate} = \frac{4.8 \times 10^{12}\ \text{B/s} \times 8}{6144\ \text{bit}} = 6.25\ \text{Gbps}
$$

**口径提醒**：HBM3e 的公开 pin 速率上限是 9.6 Gbps。4.8 TB/s 与 141 GB 是确定的，6.25 Gbps 是**按 6×1024 bit 反算的结果** —— 说明厂商的堆栈/位宽假设与这个简化模型不同（或部分堆栈用了不同配置）。**结论不变：带宽 = 位宽 × 速率，两者必须匹配才能得到 4.8 TB/s。**

### 28.2 六级结构

```
GPU
 └── memory controller（12 个）
      └── HBM 堆栈（6 个）
           └── pseudo-channel（16 个 / 堆栈，64 bit）
                └── bank group（4 个）
                     └── bank（4 个 / group → 16 / pseudo-channel）
                          └── row（约 1 KB，即 row buffer）
                               └── column（32–64 B 的 burst）
```

全机 bank 数：

```
6 堆栈 × 16 pseudo-channel × 16 bank = 1536 个 bank
（16 bank/pseudo-channel 是 HBM2/HBM3 的公开结构；若某代际为 32 bank，则总数翻倍——
  下面所有「并行 bank 数」的结论按比例缩放，不改变量级）
```

**1536 个 bank 同时可以各有一个 row 打开。** 这是 HBM 能藏住 500 ns 延迟的物理基础。

### 28.3 一次访问的四段时间

| 阶段 | 符号 | 量级 | 说明 |
|---|---|---|---|
| 行激活 | tRCD | ~14 ns | 把 row 搬进 row buffer |
| 列选 + 传输 | CL | ~14 ns | row buffer → 总线 |
| 预充电 | tRP | ~14 ns | 关闭 row，准备下一个 |
| 行周期 | tRC = tRCD+CL+tRP | **~42 ns** | 同一 bank 两次开行最小间隔 |

**row buffer 命中时，只要 CL（14 ns）；未命中要 tRC（42 ns）。** 所以：

```
顺序流（同一 row 内连续列）：每条 32–64 B 只付 CL → 接近峰值
随机流（每次换 row）：每条付 42 ns → 同一 bank 的吞吐掉 3 倍
```

### 28.4 为什么随机访问还能拿到 50%，而不是 1/3

因为上千个 bank 可以**并行**（下面按 1536 计）：

```
每个 bank 每 42 ns 提供一次 64 B
总吞吐 = 1536 × 64 B / 42 ns = 2.3 TB/s   ← 单看行周期
再叠加 6 个堆栈的总线限制（4.8 TB/s）
实际随机访问 ≈ 30–50% 峰值
```

这就是「随机访存只能拿到一半」的物理来源，也是 §15 的 sector 分析为什么重要 —— **一个没合并的 warp 访问，不只是浪费字节，还制造了 32 个随机 row 访问。**

### 28.5 refresh 与 ECC

| 项 | 影响 | 量级 |
|---|---|---|
| refresh（每行每 32–64 ms 刷一次） | 抢占总线 | 2–5% |
| 读写切换（turnaround） | 总线方向切换的空泡 | 3–10%（读写混合时） |
| ECC | HBM 由控制器内联处理 | 不占可见带宽 [spec 口径] |
| 温度升高 | refresh 频率提高 | 高温时再掉 3–8% |

主线取「带宽效率 80%」就是这个 2.5 万字的物理现实压缩成一个数：`4.8 × 0.8 = 3.84 TB/s`。

---

## 29 一个 sector 的旅程：从 HBM 到寄存器

### 29.1 每一跳的带宽（按每 SM 折算）

| 跳 | 聚合带宽 | 每 SM | 每 cycle（1.8 GHz） |
|---|---|---|---|
| HBM → L2 | 4.8 TB/s | **36.4 GB/s** | **20.2 B** |
| L2 → L1 | ~7 TB/s | 53 GB/s | 29.5 B |
| L1 ↔ 寄存器/SMEM | 128 B/cycle/SM | 230 GB/s | 128 B |
| 寄存器 → FP32 执行 | 256 FLOP/cycle/SM | — | 256 FLOP |
| 寄存器 → TC 执行 | 8192 FLOP/cycle/SM（FP8） | — | 8192 FLOP |

**最窄的一跳是 HBM：每 SM 每 cycle 只有 20 B。** 一个 FP8 Tensor Core 每 cycle 要吃 8192 FLOP —— 按 AI=412 算，需要 `8192/412 = 19.9 B`。**刚好相等，这就是拐点的物理含义：HBM 的字节刚好够喂满 TC，一点余量都没有。**

### 29.2 五跳的延迟累加

```
warp 发 LDG.128（32 lane × 16 B = 512 B = 16 sector）
  ↓
L1TEX 查 tag（命中 30–40 cycle）
  ↓ miss
按地址 hash 选 L2 slice → 跨 SM 交叉开关（+ 几十 cycle）
  ↓ miss
L2 查 tag（~200 cycle）
  ↓ miss
memory controller 排队 → 选 bank → 等 row（tRCD 14 ns）→ burst
  ↓
数据回 L2 → 回 L1 → 回寄存器（累计 ~900 cycle ≈ 500 ns）
```

### 29.3 L2 的地址 hash：一个反直觉的后果

L2 slice 按**物理地址 hash** 分布。所以：

| 访问模式 | L2 slice 分布 | 后果 |
|---|---|---|
| 连续 128 B | 落在 **1 个** slice | 单 slice 带宽成为上限 |
| 大范围连续 | 均匀分布到所有 slice | 带宽最大化 |
| 跨步（如 4 KB 步长） | 可能全撞同一个 slice | **slice camping**，带宽掉到 1/N |

**这解释了一个常见现象**：同样是 100% 合并的访存，改一下 tensor 的 stride（比如 padding 到 4096 B 的倍数）吞吐会变 —— 不是 sector 的问题，是 L2 slice 撞车。

### 29.4 为什么单 SM 打不满自己的份额

L1TEX 用 **MSHR（miss status holding register）** 记录未完成的 miss，配合同一路径上的 LSU 队列，**一个 SM 能同时挂起的请求数有硬上限**。公开文档不给具体数值，但可观测的事实是：

```
单 SM 的 copy 带宽 ≈ 整机 1/132 份额的 30–60%
```

所以：

- §14 的「每 SM 142 条 line」是 **132 个 SM 全部活跃时的摊派值**，不是单个 SM 能独立达到的目标；
- **活跃 SM 越少，总带宽越低。** §21 的并行度问题会直接变成带宽问题。

> 这也解释了 decode 为什么拿不到满带宽：不是请求没合并好，是**发请求的 SM 太少**。反过来，prefill 里 4096 个 block 铺满 132 个 SM，带宽就能到 80%。

MSHR 满时的直接后果：

```
后续 load 直接 stall（ncu: smsp__warp_issue_stalled_long_scoreboard 飙升）
```

### 29.5 一个 sector 的字节账

```
你请求：4 B（一个 float）
HBM 搬：32 B（一个 sector）
L2 行：128 B（4 个 sector）
到寄存器：4 B
```

**放大倍数 = 8。** 这就是 §15 里「最坏 12.5% 效率」的物理本体 —— 不是「浪费了缓存」，是**HBM 的 bank 被一个 4 B 的请求占用了一次完整的列选**。

---

## 30 片间接口：NVLink 与 PCIe

### 30.1 带宽对比

| 接口 | 单向带宽 | 双向 | 相对 HBM |
|---|---|---|---|
| HBM3e（卡内） | 4.8 TB/s | — | 1× |
| NVLink 4（H100/H200） | ~450 GB/s | 900 GB/s | **1/10.7** |
| PCIe Gen5 ×16 | 63 GB/s | 126 GB/s | **1/76** |
| NVLink 5（B200） | ~900 GB/s | 1.8 TB/s | 1/4.4 |

**结论：卡间通信比卡内慢一个数量级。** 所有「把数据搬到别的卡」的设计都要先算这笔账。

### 30.2 TP 的 all-reduce 字节账

张量并行（TP=8）每层要做 **2 次 all-reduce**（attention 后、MLP 后），每次的数据量：

$$
\text{bytes} = 2 \times B \times T \times H \times 2\ \text{B}
$$

**decode（B=64，T=1，H=8192，BF16）：**

```
单次 = 2 × 64 × 8192 × 2 = 2.1 MB
每层 2 次 = 4.2 MB
80 层    = 336 MB
时间（450 GB/s）= 0.75 ms
占 decode 一步（17.8 ms）的 4%
```

**prefill（B=1，T=4096）：**

```
单次 = 2 × 4096 × 8192 × 2 = 134 MB
每层 2 次 = 268 MB
80 层    = 21.5 GB
时间 = 48 ms     ← 与 prefill 本身的计算时间同量级
```

**所以：decode 的瓶颈在 HBM，prefill 的瓶颈可能变成 NVLink。** 这解释了为什么 TP 更适合 decode、而不适合长 prefill 的吞吐场景 —— 也是主线 §8 水位模型里 TP 需要单独一维的原因。

### 30.3 为什么 TP 不能超过 8（一般是）

```
TP=8：每卡存 1/8 权重（8.6 GB），all-reduce 8 卡
TP=16：每卡 4.3 GB，但 all-reduce 的延迟和跳数上升，且每卡算的活更少
```

TP 的收益是「权重字节 ÷ TP」，代价是「all-reduce 字节 × (TP−1)/TP」。当 `权重/TP` 还远大于 `all-reduce` 时，收益为正；当模型小到权重装得下时，TP 纯亏。

---

## 31 功率、时钟、温度：为什么实测会漂

| 项 | H200 | 影响 |
|---|---|---|
| TDP | 700 W | 功耗墙 |
| 名义 boost | ~1.98 GHz | 官方峰值按 ~1.83 GHz 计 |
| 满载持续主频 | 1.6–1.8 GHz [meas] | TC 满载时降频 |
| HBM 温度上限 | ~95 °C | 超过后 refresh 加密 |
| 温度对带宽 | −3~8% | 高温时 refresh 更频繁 |

三个实际后果：

1. **同一份代码，跑 10 秒和跑 10 分钟测出的 MFU 不一样**（前者还在 boost，后者已经降频）。基准测试必须报「持续」数字。
2. **主线水位模型里没有时钟维度，但实测会漂 5–15%。** 这是「压测结果 vs 生产结果」差异的一个固定来源。
3. **多卡机器里，中间卡的进风温度最高**，同一批卡的实际带宽可能差 5%。做容量规划时按**最慢的那张**算。

---

## 32 硬件接口总表：想调 X，该动哪一层

### 32.1 资源 → 接口 → 观测

| 硬件资源 | 容量/速率 | 分配者 | torch / CUDA 接口 | 观测指标 |
|---|---|---|---|---|
| 寄存器 | 256 KB/SM | 编译器 | `__launch_bounds__`、`-maxrregcount` | `launch__registers_per_thread` |
| SMEM | 228 KB/SM | 你 | `__shared__`、`cudaFuncSetAttribute` | `launch__shared_mem_per_block` |
| L1 | 与 SMEM 共享 256 KB | 硬件（可配） | `cudaFuncAttributePreferredSharedMemoryCarveout` | `l1tex__t_sector_hit_rate` |
| L2 | 50 MB | 硬件 | `cudaLimitPersistingL2CacheSize`（持久化 L2） | `lts__t_sector_hit_rate` |
| HBM | 141 GB / 4.8 TB/s | 你 + allocator | `torch.empty`、`cudaMalloc` | `dram__bytes`、`dram__throughput` |
| SM | 132 | 硬件 | grid 大小 | `sm__throughput`、`sm__ctas_active` |
| Tensor Core | 1979 TFLOPS（FP8） | 编译器 | 用不用 MMA 指令 | `sm__pipe_tensor_cycles_active` |
| 卡间 | 900 GB/s（NVLink） | 你 | NCCL、`torch.distributed` | `nvlink__throughput` |
| 功耗/温度 | 700 W / 95 °C | 硬件 | — | `nvidia-smi -q -d POWER,TEMPERATURE` |

### 32.2 症状 → 该动哪一层

| 症状 | 先动这一层 | 具体动作 |
|---|---|---|
| kernel 之间有空隙 | ③ Runtime | CUDA Graph、融合 |
| 每个 kernel 只有几 µs | ① torch | 融合、增大 batch |
| 带宽 55%，sector/req > 4 | ⑤ 硬件（布局） | 改 stride、`contiguous()`、向量化 |
| 带宽 55%，sector/req = 4 | ⑤ 硬件（并行度） | 提高 occupancy、展开、`cp.async` |
| 带宽 90%，还是慢 | ④ 算法 | **这是物理下界，只能减字节（量化/MoE/GQA）** |
| TC 20%，带宽 30% | ⑤ 硬件（并行度） | 增大 batch、split-K、FlashDecoding |
| TC 90%，吞吐仍低 | ④ 算法 | 检查是不是走了 FP32 fallback、精度口径对不对 |
| 显存不够 | ① torch | 量化 KV、prefix caching、降 `gpu_memory_utilization` 之外的手段 |

### 32.3 一句话对应表

```
寄存器不够   → 减 tile
SMEM 不够    → 减 tile 或加 swizzle
带宽不够     → 先查合并，再查在途，最后才想「换算法」
算力不够     → 先查是不是没用 TC（FP32 fallback / 精度不对齐）
并行度不够   → 加 batch 或切 K/L
延迟太高     → 加在途（展开、cp.async、TMA）
```

---

## 33 第三部分小结

```
GH100：8 GPC × 9 TPC × 2 SM = 144，启用 132；6 HBM 堆栈；50 MB L2
SM = 4 个 sub-partition，各 1 scheduler + 32 FP32 + 1 TC + 64 KB 寄存器
每 scheduler 每 cycle 发 1 条 warp 指令 → 每 SM 4 条
延迟：寄存器 1 cycle、SMEM 30、L2 200、HBM 900 cycle（500 ns）
MMA m16n8k16：2048 MAC，32 lane 分持 A 8 / B 4 / D 4 个元素
ldmatrix 一条指令完成 SMEM → fragment 的布局转换
32 倍 = FP8 TC vs FP32 CUDA；BF16 TC vs FP32 是 16 倍
2:4 稀疏 2×，但本文一律用 dense 口径
峰值：60.8（FP32）/ 989（BF16）/ 1979（FP8）；等效主频 1.83 GHz
每层能装 KV：SMEM 0.71 token、L2 160 token、HBM 43 万 token
HBM：6 堆栈 × 1024 bit = 6144 bit，反算 6.25 Gbps/pin
HBM 六级：堆栈 → pseudo-channel → bank group → bank → row → column
约 1536 个 bank（16 bank/pseudo-channel 口径）；row 命中 14 ns，未命中 42 ns
每 SM 每 cycle 只有 20 B 来自 HBM —— 与 TC 每 cycle 需要 19.9 B 恰好相等
L2 slice 按地址 hash，跨步访问会 slice camping
NVLink 900 GB/s = HBM 的 1/10.7；PCIe Gen5 = 1/76
TP 的 all-reduce：decode 占 4%，prefill 占 100%（同量级）
降频与高温让实测 MFU 漂 5–15%
```

---

# 第四部分 量化：Roofline 与「利用率」

前三部分讲机理，这一部分讲**怎么把机理变成一个能算、能测、能对比的数**。

---

## 34 算术强度的严格定义与五个口径陷阱

### 34.1 定义

$$
AI = \frac{\text{kernel 完成的 FLOPs}}{\text{它从 HBM 搬的字节数}}\qquad[\text{FLOP/B}]
$$

Roofline 上界：

$$
\text{achieved FLOPS} \le \min\big(\text{峰值 FLOPS},\ AI \times \text{HBM 带宽}\big)
$$

**这个式子里每一个词都必须是同一个口径**，否则算出来的 AI 可以差 5 倍。五个陷阱：

### 34.2 陷阱 1：写回算不算

```
只算读：GEMM 的 AI = 2MNK / (K(M+N)·b)
算读+写：分母再加 M·N·b
```

对小 M 的 GEMM，输出字节不可忽略：

```
M=64, N=8192, K=8192, FP8 权重、BF16 输出：
  只读：2·64·8192·8192 / (8192·8256·1) = 128
  读写：分母 + 64·8192·2 = 1.0 MB → AI ≈ 126
```

差别不大，但对 elementwise 算子（`RMSNorm`）差别是 2 倍。**本文与主线一律「只算读」，写回单独列。**

### 34.3 陷阱 2：L2 命中算不算

`ncu` 同时给你两个字节数：

| 指标 | 含义 |
|---|---|
| `dram__bytes.sum` | 真正到 HBM 的字节 |
| `lts__t_sectors.sum × 32` | L1↔L2 的字节 |

**用前者算 AI，得到的是「对 HBM 的算术强度」；用后者得到的是「对 L2 的算术强度」，后者可能高 3–5 倍。** Roofline 只用前者 —— 因为瓶颈是 HBM。

### 34.4 陷阱 3：split-K 让权重被读多遍

§2.3 提过：M 很小时 cuBLAS 会把 K 切开换取并行度。切 4 段 → **权重读 4 遍** → 实际 AI 只有名义的 1/4。

```
名义：AI = 2M = 128（M=64）
split-K=4 后：AI_实际 = 32
```

**所以「AI 算出来是 128，实测带宽却打满」不是矛盾，是 split-K。** 用 `ncu` 的 `dram__bytes.sum` 反算实际 AI，就能发现。

### 34.5 陷阱 4：精度必须和峰值配套

```
FP8 权重：AI = 2M，峰值 1979，拐点 412
BF16 权重：AI = M，峰值 989，拐点 206
```

两套数字**不能混用**。用 BF16 的 AI 去比 FP8 的拐点，会得出「M=103 就够」的错误结论（实际差 2 倍）。

### 34.6 陷阱 5：KV 读了几遍

| 实现 | KV 读的次数 | AI |
|---|---|---|
| 朴素 attention（先算 S 再算 PV） | 2 遍（K 一遍、V 一遍；S/P 额外落 HBM） | 远低于理论 |
| FlashAttention | 1 遍（tile 内算完 softmax 立刻乘 V） | 8（主线 §1.14 的理想值） |

**「AI=8」是 FlashAttention 的口径，是上界。** 任何多读一遍 KV 的实现，实际 AI 只有 4。

---

## 35 拐点 412 与 M ≈ 206：完整推导 + 精度不变性

### 35.1 拐点

$$
\text{ridge} = \frac{\text{峰值 FLOPS}}{\text{HBM 带宽}} = \frac{1979 \times 10^{12}}{4.8 \times 10^{12}} = 412\ \text{FLOP/B}
$$

### 35.2 GEMM 的 AI

`[M,K] × [K,N]`，权重 $b$ 字节：

$$
\text{FLOPs} = 2MKN,\qquad \text{Bytes} = KN\cdot b\ (\text{M 小时激活可忽略})
$$

$$
AI = \frac{2MKN}{KNb} = \frac{2M}{b}
$$

### 35.3 令 AI = ridge

$$
\frac{2M}{b} = \frac{P}{BW}
\quad\Longrightarrow\quad
\boxed{M^* = \frac{P \cdot b}{2\,BW}}
$$

| 精度 | $b$ | $P$ (TFLOPS) | $M^*$ |
|---|---|---|---|
| FP4 | 0.5 | 3958 | **206** |
| FP8 | 1 | 1979 | **206** |
| BF16 | 2 | 989 | **206** |
| FP32（TF32） | 4 | 495 | **206** |

> FP4 从 Blackwell 才支持，这一行只用来说明不变性；H200 上实际可用的是 FP8/BF16。

**精度不变性**：只要「权重字节翻倍 ⇔ 峰值减半」，$M^*$ 就不变。因为 $P \propto 1/b$，代入 $M^* = Pb/(2BW)$ 后 $b$ 消掉。

> 这条不变性是主线 §4.7 那句「BF16 权重也一样，M = 206 不变」的严格证明。它成立的前提是**峰值与字节成反比**——这个前提在所有 dense TC 精度上都成立，在稀疏/混合精度上不成立。

### 35.4 换硬件，阈值怎么变

$$
M^* = \frac{P_{bf16}}{BW}
$$

| 卡 | BF16 峰值 | 带宽 | $M^*$ |
|---|---|---|---|
| H100 SXM | 989 | 3.35 TB/s | **295** |
| H200 SXM | 989 | 4.8 TB/s | **206** |
| B200 | ~2250 | 8 TB/s | **281** |

**带宽涨得比算力快，阈值就降（H100→H200）；两者同步涨，阈值基本不变（H200→B200 略升）。**

### 35.5 用「有效值」算，阈值会漂

主线 §0.4 的有效值：MFU 40%（792 TFLOPS）、带宽效率 80%（3.84 TB/s）。

```
拐点_有效 = 792 / 3.84 = 206 FLOP/B
AI = 2M = 206 → M* = 103
```

**同样的机器，一套口径说 206，另一套说 103。** 两者都对，因为：

- 206 是「打满理论峰值」的阈值；
- 103 是「打满你能拿到的有效算力」的阈值。

**做容量规划用 206（保守），做 kernel 调优用 103（现实）。** 关键是不许混用 —— 这就是 §34.5 那个陷阱的实际后果。

---

## 36 三种「利用率」：SM、Tensor Core、带宽

### 36.1 三个数各是什么

| 名称 | 定义 | DCGM 指标 | 高值意味着 |
|---|---|---|---|
| **SM 占用** | 有 warp 驻留的 SM 比例 | `DCGM_FI_PROF_SM_ACTIVE` | 有活干 |
| **TC 占用** | Tensor Core 活跃的周期比例 | `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` | 算力在用 |
| **带宽占用** | HBM 活跃的周期比例 | `DCGM_FI_PROF_DRAM_ACTIVE` | 带宽在用 |

### 36.2 三个数可以同时「很怪」

| 场景 | SM | TC | 带宽 | 解释 |
|---|---|---|---|---|
| 主线 §8.4 场景 C | 100% | **1.8%** | 90% | KV 满了，attention 在等 HBM，TC 无事可做 |
| 场景 D | 100% | **72%** | 80% | M=256 越过拐点，健康满载 |
| 小 batch decode | 20% | 5% | 40% | 并行度不足（§21） |
| CPU-bound | **30%** | 10% | 10% | launch 把时间填满（§3） |

**「满载」必须指明是哪一维**（主线 §8.4 的结论）。这张表就是那四个维度的硬件侧证据。

### 36.3 三个换算公式

$$
\text{MFU} = \frac{\text{实测 TFLOPS}}{\text{峰值 TFLOPS}} \approx \text{PIPE\_TENSOR\_ACTIVE} \times \eta_{\text{instr}}
$$

$$
\text{MBU} = \frac{\text{实测 GB/s}}{\text{峰值 GB/s}} \approx \text{DRAM\_ACTIVE} \times \eta_{\text{sector}}
$$

$$
\eta_{\text{sector}} = \frac{\text{有效字节}}{\text{搬运字节}} = \frac{4}{\text{sectors per request}}
$$

两个 $\eta$ 都不能省：

- `PIPE_TENSOR_ACTIVE = 100%` 但每个 MMA 之间等依赖 → $\eta_{\text{instr}} < 1$，MFU 可能只有 30%；
- `DRAM_ACTIVE = 100%` 但每次只用一个 sector 的 4 B → $\eta_{\text{sector}} = 12.5\%$，MBU 只有 12.5%。

### 36.4 「1.8%」是怎么算出来的

主线场景 C（B=5，ctx=32768，KV 打满）：

```
每步 FLOPs = 1.11 TF        （主线场景 C 的算力需求）
每步时间   = 32 ms          （被 KV 读取撑住）
实测算力   = 1.11e12 / 0.032 = 35 TFLOPS
MFU        = 35 / 1979 = 1.8%
```

**它的分子（FLOPs）本来就很小，分母（时间）被 KV 读取撑大。** 这不是 kernel 效率问题，是 §38 的物理下界。

---

## 37 LLM 各算子的 roofline 位置

### 37.1 一张表（Llama-3-70B，FP8 权重，BF16 KV，H200）

| 算子 | 形状 | FLOPs | HBM 字节 | AI | 位置 | 实际瓶颈 |
|---|---|---|---|---|---|---|
| prefill QKV GEMM | M=4096, K=8192, N=1024×3 | 2.1e11 | 25 MB | **8192** | 算力（20× 拐点） | TC |
| prefill MLP gate/up | M=4096, N=28672, K=8192 | 1.9e12 | 235 MB | **8192** | 算力 | TC |
| prefill MLP down | M=4096, N=8192, K=28672 | 1.9e12 | 235 MB | **8192** | 算力 | TC |
| prefill attention | L=4096, 64 head | ~2.8e11 | 134 MB | **~2000** | 算力（FA） | TC |
| **decode QKV GEMM** | **M=64** | 3.2e9 | 25 MB | **128** | 带宽（拐点 31%） | HBM |
| **decode MLP** | **M=64** | 3.0e10 | 235 MB | **128** | 带宽 | HBM |
| **decode attention**（单层） | L=32768, B=1 | 1.1e9 | 134 MB(KV) | **~8** | 带宽（拐点 2%） | HBM |
| softmax（词表） | 128256 | 6.4e5 | 513 KB | **1.2** | 带宽 | HBM |
| RMSNorm | 8192 | 3.3e4 | 32 KB | **1.0** | 带宽 | HBM |
| KV 写入 | — | 0 | 320 KiB/token | **0** | 纯带宽 | HBM |
| lm_head | M=64, N=128256 | 1.3e11 | 1.05 GB | **128** | 带宽 | HBM |

### 37.2 三个立刻可用的结论

1. **prefill 的所有大算子都在拐点右侧 5–20 倍 → 算力受限。** 优化 prefill 只有两条路：提 MFU、减 FLOPs。
2. **decode 的所有算子都在拐点左侧 → 带宽受限。** 优化 decode 只有两条路：减字节、提带宽。
3. **`RMSNorm`/`softmax`/`KV 写入` 的 AI ≈ 0–1.2。** 它们对 FLOPs 贡献接近 0，但对字节贡献不小 —— **融合它们省的是带宽，而且是 decode 阶段最值钱的带宽**（§7.2）。

### 37.3 为什么「decode 优化」和「prefill 优化」不能互相借鉴

| | prefill | decode |
|---|---|---|
| M | 4096 | 1–256 |
| AI | 2000–8192 | 8–128 |
| 瓶颈 | TC | HBM |
| 有效手段 | 更大 tile、更好 MMA、稀疏、蒸馏 | 量化、KV 压缩、融合、更大 batch、投机 |
| 无效手段 | 量化权重（字节不是瓶颈） | 更大 tile（M 撑不起来） |

**这条表解释了主线第 5 章为什么要把两个阶段彻底分开算。**

---

## 38 decode 的物理下界与四个突破口

### 38.1 下界

$$
t_{\text{step}} \ge \frac{W + KV_{\text{total}}}{BW}
$$

- $W$ = 权重字节（Llama-3-70B FP8 = 68.5 GB）
- $KV_{\text{total}}$ = 当前 batch 全部 KV 字节（320 KiB × B × L）

### 38.2 四个方向（与主线 §8.14 一致）

| 方向 | 动哪一项 | 手段 | B=1 时的理论收益 |
|---|---|---|---|
| **① 减权重字节** | $W$ | FP8（已用）、FP4、MoE | FP4：2×；MoE（激活 3%）：最高 30× |
| **② 减 KV 字节** | $KV$ | GQA→MQA、MLA、FP8 KV、sliding window | MQA：$H_q/H_{kv}$ 从 8 → 64，KV 降 8× |
| **③ 摊薄 $W$** | 每 token 摊到的 $W$ | 大 batch、投机解码、MTP | batch 64：$W$ 摊到 1/64 |
| **④ 增带宽** | $BW$ | 换卡（HBM4）、更多卡（TP） | HBM3e→HBM4：1.67× |

### 38.3 ④ 的一个陷阱：TP 不等于「带宽 ×TP」

TP=8 让每张卡只读 1/8 权重 —— **看起来带宽翻了 8 倍**。但 §30.2 的账：

```
decode：all-reduce 0.75 ms vs 计算 17.8 ms → 净收益接近 8×，成立
prefill：all-reduce 48 ms ≈ 计算时间 → 净收益大打折扣
```

**TP 对 decode 是「真·加带宽」，对 prefill 是「加通信」。**

### 38.4 不属于这四个方向的优化

| 手段 | 作用 | 为什么不在四个方向里 |
|---|---|---|
| CUDA Graph | 消除 launch 开销 | 只逼近下界，不改分子分母 |
| 算子融合 | 减少 HBM 往返 | 减少的是激活/中间量的字节，不是权重/KV |
| 更好的 attention kernel | 让带宽效率从 60% → 80% | 同上 |
| 更大 tile | 提高 TC 利用率 | decode 的 M 撑不起大 tile |

**这正是主线 §8.14 那句话：任何不在这四类里的优化，都突破不了下界。**

### 38.5 投机解码：把 decode 推到拐点右侧

```
普通 decode：M = B
投机解码：   M = B × (γ+1)

B=64, γ=3 → M = 256
AI = 2M = 512 > 412 ✓
```

**它用的是「④ 之外的第五种」：改变 M，把带宽受限的 decode 变成算力受限的 prefill 形态。** 主线把它归在方向 ③（摊薄 W），本质相同：一次读权重，产出更多候选 token。

代价：猜错白算。收益条件：

```
decode 阶段 TC 有大量空闲（主线场景 D 也有 28% 空闲）
用空闲算力换紧缺带宽 → 正收益
batch 已经很大（M > 412）→ 无收益
```

---

## 39 测量：从 `ncu` 到判据决策树

### 39.1 三条命令

```bash
# ① 时间线：kernel 之间有没有空隙、launch 花了多久
nsys profile -t cuda,nvtx,osrt -o prof python serve.py
nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum prof.nsys-rep

# ② 单 kernel 深挖：一条命令拿全指标
ncu --set full --kernel-name regex:gemm --launch-count 1 -o gemm python bench.py

# ③ 只抓关键几个数（开销最小）
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,\
smsp__warps_active.avg.pct_of_peak_sustained_active,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum ./app
```

### 39.2 指标清单（按用途分组）

| 用途 | 指标 | 判据 |
|---|---|---|
| **总时间** | `gpu__time_duration.sum` | 分母 |
| **带宽** | `dram__throughput.avg.pct_of_peak_sustained_elapsed` | ≥80% → 带宽受限 |
| **实际字节** | `dram__bytes.sum` | 反算真实 AI |
| **L2** | `lts__t_sector_hit_rate.pct` | 命中率低 → 复用没做出来 |
| **算力** | `sm__throughput.avg.pct_of_peak_sustained_elapsed` | 综合流水线占用 |
| **Tensor Core** | `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | ≥60% → 算力受限 |
| **并行度** | `smsp__warps_active.avg.pct_of_peak_sustained_active`（achieved occupancy） | <30% → 并行度不足 |
| **波次** | `launch__waves_per_multiprocessor` | 非整数 → tail effect |
| **合并度** | `sectors ÷ requests`（都是 `l1tex__t_*_pipe_lsu_mem_global_op_ld.*`） | =4 完美，>4 浪费 |
| **stall 归因** | `smsp__warp_issue_stalled_*_per_warp_active.pct` | 见 §39.4 表 |
| **寄存器** | `launch__registers_per_thread` | >128 → occupancy ≤25% |
| **spill** | `smsp__sass_inst_executed_op_local_ld.sum` | >0 → 有 spill |
| **occupancy 限制因子** | `launch__occupancy_limit_registers / _shared_mem / _blocks / _warps` | 直接告诉你是谁限制的 |

### 39.3 判据决策树

```
                ┌─────────────────────────┐
                │ dram__throughput ≥ 80%? │
                └───────────┬─────────────┘
                     是 ────┤──── 否
                      ↓              ↓
              【带宽受限】    ┌──────────────────────┐
                      │       │ TC ≥ 60%?            │
        sectors/req > 4?      └──────────┬───────────┘
         ├─ 是 → 合并访存（§15）  是 ────┤──── 否
         │       改 layout/向量化   ↓              ↓
         └─ 否 → 在途不足?    【算力受限】  ┌──────────────────┐
                  long_scoreboard 高?      │ warps_active<30%?│
                   ├─ 是 → 展开/cp.async/  └────────┬─────────┘
                   │        TMA（§14/§18）   是 ────┤──── 否
                   └─ 否 → 物理下界，      ↓              ↓
                            减字节（§38）  【并行度不足】  【延迟受限】
                                            grid<132?     看 top stall
                                             ├─ 是 → split-K /  └→ barrier: 流水不均
                                            │   FlashDecoding      short_scoreboard:
                                            └─ 否 → 寄存器/        SMEM 冲突
                                                    SMEM 限制      wait: 依赖链
```

**最后一条分支**：三个都不满足（带宽 <80%、TC <60%、parallelism 正常）→ 去看 `nsys` 时间线，多半是 **kernel 之间的空隙**（§3、§6）。

### 39.4 stall 原因对照

| `smsp__warp_issue_stalled_*` | 含义 | 对应节 |
|---|---|---|
| `long_scoreboard` | 等 global/HBM 数据 | §14、§15 |
| `short_scoreboard` | 等 SMEM 数据 | §16 |
| `mio_throttle` | LSU/SFU 队列满（SMEM 冲突、指令太多） | §16、§24.4 |
| `barrier` | 等 `__syncthreads` | §18.1 |
| `wait` | 等固定延迟的依赖（FFMA 链、MMA 链） | §17.3 |
| `not_selected` | 有 warp 就绪但没被选中（warp 太多） | 健康 |
| `math_pipe_throttle` | 数学流水线排满 | 健康（接近峰值） |
| `imc_miss` | 常量缓存 miss | 少见 |

### 39.5 torch.profiler：不开 ncu 的轻量版

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             schedule=torch.profiler.schedule(wait=1, warmup=2, active=5)) as prof:
    for _ in range(8):
        step()
        prof.step()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

能看到：每个 op 的 CPU 时间、GPU 时间、调用次数、kernel 名。**判断「是不是 CPU-bound」用它就够了，不需要 ncu。**

---

## 40 TinyToy 的 AI 验算（可手算的落地）

主线 §4.8 的完整复算。TinyToy：H=8、H_q=4、H_kv=2、d=2、2 层、BF16。

### 40.1 Decode（B=2 序列，各 L=3）

| 项 | 计算 | 值 |
|---|---|---|
| 每序列每层线性 | 2 × 576 | 1152 FLOPs |
| 每序列每层 attention | 4 × 3 × 2 × 4 | 96 FLOPs |
| 每步总 FLOPs | (1152 + 96) × 2 层 × 2 序列 | **4992** |
| 权重读取 | 1152 × 2 B（整批只读一次） | 2304 B |
| KV 读取 | 2 × 3 × 32 B | 192 B |
| KV 写入 | 2 × 32 B | 64 B |
| 总字节 | | **2560 B** |
| **整体 AI** | 4992 / 2560 | **1.95** |
| 线性部分 AI | 4608 / 2304 | **2.0 = M** ✓ |

### 40.2 Prefill（T=3，causal，2 层）

| 项 | 计算 | 值 |
|---|---|---|
| 线性 FLOPs | 3 × 2304 | 6912 |
| attention FLOPs | 4 × 2 × 4 × (1+2+3) × 2 层 | 384 |
| 总 FLOPs | | **7296** |
| 字节 | 权重 2304 + KV 写 3×32 = 96 | **2400** |
| **整体 AI** | 7296 / 2400 | **3.04** |
| 线性部分 AI | 6912 / 2304 | **3.0 = M** ✓ |

### 40.3 放大 T 验证 AI 随 M 线性增长

```
T=32:
  线性 FLOPs = 32 × 2304 = 73,728
  attn FLOPs  = 4 × 2 × 4 × (32×33/2) × 2 = 32 × 528 × 2 = 33,792
  总 FLOPs    = 107,520
  字节        = 2304 + 32 × 32 = 3328
  AI          = 107,520 / 3328 = 32.3
```

**3.04（T=3）→ 32.3（T=32），近似线性。** 这就是「prefill 越长越计算密集」。

### 40.4 为什么 TinyToy 的绝对值与对照表差 2 倍

```
TinyToy 权重字节 = 2304 B，M=32 时的 FLOPs = 107,520
真实模型：权重 68.5 GB，M=32 时 FLOPs = 32 × 137 GFLOP = 4.4 TFLOP
```

先澄清一个容易搞错的点：**线性层的 AI 严格等于 `2M/b`，与 `H` 无关**。TinyToy 用 BF16（`b=2`），所以 `AI_linear = M`；主线对照表用 FP8（`b=1`），所以 `AI_linear = 2M`。两个数在 T=32 时分别是 32 与 64，**差的 2 倍纯粹是精度因子**。

| 量 | TinyToy（BF16） | Llama-70B（FP8 权重） | 比例 |
|---|---|---|---|
| 线性 AI（T=32） | 32 | 64 | 1/2 |
| 整体 AI（T=32） | 32.3 | ~64 | 1/2 |

`H=8` 只影响「小 T 时 KV 写入等次要项的相对占比」，不影响主项：

```
TinyToy T=3：权重 2304 B，KV 写 96 B → 次要项占 4%
Llama  T=3：权重 68.5 GB，KV 写 0.96 MB → 次要项占 0.001%
```

所以 TinyToy 的整体 AI 会略微更靠近线性 AI（32.3 vs 32），这个偏差是「次要项占比大」造成的，不是规模效应。

**结论：用 TinyToy 验证公式的形状（AI 随 M 线性增长），用真实模型定阈值（206）。** 绝对值之差先换算精度口径，再比较。

---

## 41 口径修正：主线结论需要打折扣的五处

### 41.1 MoE：M* 要重算

MoE 每个 token 只激活 $k/E$ 比例的专家：

```
FLOPs = 2N_active·token  →  分子变小
权重字节 = 全部专家      →  分母在 batch 大时不变
```

所以 **AI 比 dense 更低**：

```
dense：AI = 2M
MoE（E=8, k=2）：AI ≈ 2M × (2/8) = 0.5M
→ M* 从 206 涨到 824
```

**MoE 在 decode 阶段的带宽优势随 batch 增大而消失**（主线 §8.13），在 roofline 上就是「AI 随 batch 增长的斜率只有 dense 的 1/4」。

### 41.2 attention 的 AI 依赖 KV 精度

```
BF16 KV：AI_read = 8（主线 §1.14）
FP8  KV：AI_read = 16
```

而分母（拐点 412）是 **FP8 权重**的口径。混着比会得出「attention 离拐点更近」的错觉 —— 实际上 attention 从来没接近过拐点：

```
8 / 412 = 1.9%
16 / 412 = 3.9%
```

**结论不变：attention 永远是带宽受限。**

### 41.3 「GPU 利用率」有三个意思

| 说法 | 实际指标 | 典型值（同一时刻） |
|---|---|---|
| 运维说「利用率 90%」 | `nvidia-smi` GPU-Util（时间占比） | 90% |
| 平台说「算力水位 72%」 | `PIPE_TENSOR_ACTIVE` 或 MFU | 72% |
| 硬件说「DRAM 活跃 80%」 | `DRAM_ACTIVE` | 80% |

**三个数都对，且互不矛盾。** 任何不说明口径的「利用率」都不能用于容量决策（主线 §8.2 的三种口径在硬件层的对应）。

### 41.4 GQA 不改 FLOPs

```
attention FLOPs = 4·L·d·H_q      ← 只看 H_q，与 H_kv 无关
attention 字节  = 2·L·d·H_kv·b   ← 只看 H_kv
```

GQA 把 `H_kv` 从 64 降到 8，**FLOPs 一个没少，字节少 8 倍** → AI 提升 8 倍。这就是主线「GQA 省的是字节，不是计算」的公式形式。

### 41.5 TP 的有效带宽不是线性叠加

```
TP=8：每卡权重 1/8，但 all-reduce 让每卡多搬 21.5 GB（prefill）
```

有效带宽的账：

$$
BW_{\text{有效}} = \frac{TP \cdot BW}{1 + \dfrac{\text{all-reduce 字节}}{\text{权重字节}}}
$$

**decode 时第二项约 4%，接近线性；prefill 时第二项 >100%，TP 反而可能变慢。**

---

# 第五部分 接回主线

## 42 结论 → 主线用在哪

| 本文结论 | 主线位置 | 主线拿它干什么 |
|---|---|---|
| 拐点 412、M* = 206（§35） | §4.6、§4.7、§6、§7、§8 | 判定「算力是否够」、解释场景 D 的 72% |
| AI 1.95 / 3.04 / 32.3（§40） | §5、§8 场景 B | 玩具模型的自洽性检查 |
| decode 下界 (W+KV)/BW（§38） | §8.14 | 四个突破口的推导起点 |
| HBM 三独立约束（§27/§28） | §8.4 | 四维水位为什么必须分开算 |
| 并行度只有 3–6%（§21） | §8.4 的 1.8% | 解释「排队排死了 TC 只用了 1.8%」 |
| launch 5 µs × 1120 kernel（§3） | §5、§7 | 解释小 batch decode 为什么达不到理论值 |
| allocator 三个数（§4） | §4.11 | KV 池 = 总显存×0.9 − 权重 − 激活 − non_torch |
| 融合省 40% 字节（§7.2） | §6 | cached token 的算术账 |
| NVLink 1/10.7 HBM（§30） | §7、§8 TP 维 | 为什么 TP 需要单独一维 |
| grid 波次（§21.4） | §7 | 为什么 batch 跨过某个数会突然变快 |
| KV 每层能装多少（§27.2） | §3、§4.11 | PagedAttention 的块大小选择依据 |

## 43 一页速查卡

```
═══ 硬件常数（H200 SXM）═══════════════════════════════
容量   141 GB HBM3e        带宽 4.8 TB/s        延迟 ~500 ns
SM     132                 warp 64/SM           线程 2048/SM
寄存器 256 KB/SM            SMEM 228 KB/SM        L1 256 KB（共享）
L2     50 MB（~7 TB/s）     NVLink 900 GB/s       PCIe 64 GB/s/向
峰值   60.8 FP32 / 989 BF16 / 1979 FP8（dense）
拐点   412（FP8）/ 206（BF16）      M* = P·b/(2·BW) = 206

═══ 三个必背的除法 ═══════════════════════════════════
AI      = FLOPs / HBM 字节            只算读，不算写
拐点    = 峰值 / 带宽                 412 = 1979/4.8
M*      = 峰值 · 权重字节 / (2·带宽)   206

═══ 延迟与并行度 ═════════════════════════════════════
在途    = 4.8e12 × 500e-9 = 2.4 MB
每 SM   20 B/cycle（HBM 份额）vs TC 需 19.9 B/cycle → 零余量
MSHR    单 SM 打不满自己 1/132 的份额 → 需要多数 SM 同时活跃
decode  并行度只有全机 3–6% → TC 空闲是结构性的

═══ 五级演进的天花板 ═════════════════════════════════
naive      AI 0.5   → 2.4 TFLOPS
SMEM 32    AI 16    → 76.8
128 tile   AI 64    → 307（但 FP32 只有 60.8）
+流水       AI 64    → 60.8（贴顶）
+TC         AI 64+   → 989 → 实测 700–850

═══ 各层能装多少 KV（320 KiB/token）══════════════════
SMEM 228 KB → 0.71 token      L2 50 MB → 160 token
HBM 141 GB → 43 万 token      KV 池 50 GiB → 163,840 token

═══ decode 下界与四个方向 ════════════════════════════
t ≥ (W + KV) / BW
① 减 W：FP4/MoE    ② 减 KV：MQA/MLA/FP8 KV
③ 摊薄 W：batch/投机  ④ 加 BW：换卡/TP
CUDA Graph / 融合 / 更好的 kernel → 只逼近下界

═══ 排障顺序 ════════════════════════════════════════
DRAM≥80%? → 合并度 >4? → 改布局；否则在途不足 → 展开/cp.async
TC≥60%?   → 算力受限 → 查 FP32 fallback / 精度口径
都不是     → 并行度 → grid<132? → split-K/FlashDecoding
都不是     → 延迟 → 看 stall：barrier/short_scoreboard/wait
kernel 间有空隙 → launch → CUDA Graph / 融合
```

---

# 附录 A 硬件向术语表

| 术语 | 全称/中文 | 一句话定义 | 常见误用 |
|---|---|---|---|
| HBM | High Bandwidth Memory | 堆叠式 DRAM，通过 interposer 连 GPU | 与「显存容量」混为一谈（带宽/容量是两件事） |
| SM | Streaming Multiprocessor | 独立调度单元，含 4 个 sub-partition | 当成「一个核」 |
| GPC | Graphics Processing Cluster | SM 的上一级分组 | 当成调度单位（它不是） |
| warp | — | 32 个线程，共享 PC | 与 block 混用 |
| lane | — | warp 内的单个线程通道 | — |
| SIMT | Single Instruction Multiple Thread | 一条指令 32 lane | 与 SIMD 混用（SIMT 有独立 PC 掩码） |
| divergence | 分支发散 | warp 内走不同路径 → 串行 | 以为「分支不花钱」 |
| occupancy | 占用率 | 驻留 warp / 64 | 当成性能指标 |
| MSHR | miss status holding register | 记录未完成 miss 的表 | 忽略它会高估可挂在途量 |
| sector | 扇区 | 32 B 传输粒度 | 与 cache line（128 B）混用 |
| coalescing | 合并访存 | warp 的 32 地址合成最少 sector | 以为「连续就行」（还要对齐） |
| bank conflict | 存储体冲突 | 多个 lane 撞同一 bank | 与 L2 slice camping 混用 |
| swizzle | 异或打散 | 用 XOR 重排地址避免冲突 | 与 padding 混用 |
| MMA | Matrix Multiply-Accumulate | 矩阵乘加指令 | 与 FMA 混用 |
| fragment | 片段 | 一个 lane 持有的矩阵分片 | 以为「任意布局都行」 |
| TMEM | Tensor Memory | Blackwell 的片上累加器存储 | 与 SMEM 混用 |
| TMA | Tensor Memory Accelerator | Hopper 的整块异步拷贝 | 与 `cp.async` 混用 |
| wgmma | warpgroup MMA | 4 warp 协作的 MMA | 与 `mma.sync` 混用 |
| MFU | Model FLOPs Utilization | 实测 / 峰值（同精度 dense） | 用 sparse 峰值当分母 |
| MBU | Memory Bandwidth Utilization | 实测带宽 / 峰值 | 用「DRAM 活跃时间」当带宽 |
| ridge point | 拐点 | 峰值/带宽 | 拿不同精度的拐点互比 |
| roofline | 屋顶线 | min(峰值, AI×带宽) | 忽略「有效值」与「理论值」的区别 |
| spill | 寄存器溢出 | 变量落到 local memory（在 HBM） | 以为 local 是「本地快速存储」 |
| tail effect | 尾部效应 | grid 不是 SM 整数倍 → 最后一波很空 | 忽略 batch 跨越波次边界的台阶 |
| split-K | — | 沿 K 切分换并行度 | 忘了它会让权重读多遍 |
| slice camping | — | 地址集中到少数 L2 slice | 与 bank conflict 混用 |

---

# 附录 B 常数与公式速查

## B.1 三卡对照

| 项 | H100 SXM | **H200 SXM** | B200 |
|---|---|---|---|
| HBM 容量 | 80 GB | **141 GB** | 192 GB |
| HBM 带宽 | 3.35 TB/s | **4.8 TB/s** | 8 TB/s |
| SM | 132 | **132** | — |
| L2 | 50 MB | **50 MB** | 126 MB |
| FP32 | 60.8 | **60.8** | ~80 |
| BF16 | 989 | **989** | ~2250 |
| FP8 | 1979 | **1979** | ~4500 |
| FP8 拐点 | 590 | **412** | 562 |
| M* | 295 | **206** | 281 |
| NVLink | 900 GB/s | **900 GB/s** | 1.8 TB/s |
| TDP | 700 W | **700 W** | 1000 W |

## B.2 公式

$$
\begin{aligned}
\text{FLOPs}_{\text{GEMM}} &= 2MKN \\
\text{Bytes}_{\text{weights}} &= KN \cdot b \\
AI &= \frac{2M}{b} \\
\text{ridge} &= \frac{P}{BW} \\
M^* &= \frac{P\,b}{2\,BW} = \frac{P_{bf16}}{BW} \\
t_{\text{decode}} &\ge \frac{W + KV}{BW} \\
\text{occupancy} &= \frac{\min(64,\ \lfloor 65536/(r\cdot 32)\rfloor,\ \ldots)}{64} \\
\text{在途字节} &= BW \times \tau_{\text{HBM}} \\
\text{sectors/request} &= \frac{\texttt{l1tex\_\_t\_sectors}}{\texttt{l1tex\_\_t\_requests}} \\
\end{aligned}
$$

## B.3 数字速查

```
1 cycle @1.8 GHz           = 0.556 ns
HBM 延迟 500 ns            = 900 cycle
L2 延迟 200 ns             = 360 cycle
2.4 MB 在途                = 142 条 128 B line
sector 32 B / line 128 B   = 4 sector/line
SMEM 128 B/cycle × 132     = 30.4 TB/s
HBM 4.8 TB/s ÷ 132 SM      = 20.2 B/cycle/SM
TC FP8 8192 FLOP/cycle/SM ÷ 412 = 19.9 B/cycle ← 零余量
```

---

# 附录 C 可运行验证脚本

## C.1 纯 CPU：常数、拐点、占用率、TinyToy（不需要 GPU）

```python
#!/usr/bin/env python3
"""GPU roofline toolkit —— CPU 可跑部分。
运行： python3 gpu_roofline_toolkit.py
所有断言失败会抛 AssertionError。"""

from dataclasses import dataclass

@dataclass(frozen=True)
class Gpu:
    name: str
    sm: int
    hbm_gb: float
    bw_tb_s: float
    fp32_tflops: float
    bf16_tflops: float
    fp8_tflops: float

H100 = Gpu("H100 SXM", 132, 80.0, 3.35, 60.8, 989.0, 1979.0)
H200 = Gpu("H200 SXM", 132, 141.0, 4.8, 60.8, 989.0, 1979.0)
# B200 是双 die 封装，SM 数公开口径分散（148 仅作占位）；本文只用带宽与峰值做拐点对照
B200 = Gpu("B200", 148, 192.0, 8.0, 80.0, 2250.0, 4500.0)

def ridge(g, prec="fp8"):
    peak = {"fp32": g.fp32_tflops, "bf16": g.bf16_tflops, "fp8": g.fp8_tflops}[prec]
    return peak * 1e12 / (g.bw_tb_s * 1e12)

def m_star(g, w_bytes=2):
    """M* = P*b/(2*BW)；w_bytes 是权重字节数。"""
    peak = g.fp8_tflops if w_bytes == 1 else g.bf16_tflops
    return peak * 1e12 * w_bytes / (2 * g.bw_tb_s * 1e12)

def gemm_ai(m, n, k, w_bytes=1, out_bytes=0):
    flops = 2 * m * n * k
    byts = k * n * w_bytes + m * k * w_bytes + m * n * out_bytes
    return flops / byts

def bytes_in_flight(bw_tb_s, latency_ns):
    return bw_tb_s * 1e12 * latency_ns * 1e-9

def occupancy(regs_per_thread, smem_kb_per_block, block, regs_per_sm=65536,
              smem_kb_per_sm=228, max_warps=64, max_blocks=32):
    """block 是分配粒度：先算能放几个 block，再折算 warp。"""
    warps_per_block = -(-block // 32)                 # ceil
    by_regs  = regs_per_sm // (regs_per_thread * block)
    by_smem  = int(smem_kb_per_sm // smem_kb_per_block) if smem_kb_per_block else max_blocks
    blocks = max(0, min(by_regs, by_smem, max_blocks))
    warps = min(max_warps, blocks * warps_per_block)
    return warps / max_warps, {"regs": by_regs, "smem": by_smem, "blocks": blocks}

def tiny_toy():
    """主线 §4.8 / 本文 §40 的三组数。"""
    out = {}
    # decode: B=2, L=3, 2 层
    lin = 2 * 576
    attn = 4 * 3 * 2 * 4
    flops = (lin + attn) * 2 * 2
    byts = 1152 * 2 + 2 * 3 * 32 + 2 * 32
    out["decode"] = (flops, byts, flops / byts)
    # prefill: T=3
    flops = 3 * 2304 + 4 * 2 * 4 * 6 * 2
    byts = 2304 + 3 * 32
    out["prefill_T3"] = (flops, byts, flops / byts)
    # prefill: T=32
    flops = 32 * 2304 + 4 * 2 * 4 * (32 * 33 // 2) * 2
    byts = 2304 + 32 * 32
    out["prefill_T32"] = (flops, byts, flops / byts)
    return out

def main():
    print(f"{'GPU':12s} {'ridge_fp8':>10s} {'ridge_bf16':>11s} {'M*':>6s}")
    for g in (H100, H200, B200):
        print(f"{g.name:12s} {ridge(g):10.1f} {ridge(g,'bf16'):11.1f} {m_star(g):6.0f}")

    # —— 断言：与主线完全一致 ——
    assert abs(ridge(H200) - 412.3) < 0.5, ridge(H200)
    assert abs(ridge(H200, "bf16") - 206.0) < 0.5
    assert abs(m_star(H200) - 206.0) < 0.5
    assert abs(ridge(H100) - 590.7) < 0.5
    assert abs(m_star(H100) - 295.2) < 0.5
    assert abs(ridge(B200) - 562.5) < 0.5

    # AI = 2M（FP8）与 AI = M（BF16）
    assert abs(gemm_ai(64, 8192, 8192, 1) - 128) < 1.0
    assert abs(gemm_ai(64, 8192, 8192, 2) - 64) < 1.0

    # 在途字节 2.4 MB
    inflight = bytes_in_flight(4.8, 500)
    assert abs(inflight - 2.4e6) < 1e4
    print(f"\nin-flight = {inflight/1e6:.2f} MB, "
          f"per SM = {inflight/132/1024:.1f} KB = {inflight/132/128:.0f} 条 128 B line")

    # TinyToy
    tt = tiny_toy()
    for k, (f, b, ai) in tt.items():
        print(f"{k:12s} FLOPs={f:7d}  bytes={b:5d}  AI={ai:6.2f}")
    assert abs(tt["decode"][2] - 1.95) < 0.01
    assert abs(tt["prefill_T3"][2] - 3.04) < 0.01
    assert abs(tt["prefill_T32"][2] - 32.3) < 0.1
    assert tt["decode"][0] == 4992 and tt["decode"][1] == 2560
    assert tt["prefill_T3"][0] == 7296 and tt["prefill_T3"][1] == 2400

    # occupancy
    print("\nregs/thread  occupancy  limits")
    for r in (32, 40, 64, 96, 128, 168, 255):
        occ, why = occupancy(r, 0, 256)
        print(f"{r:11d}  {occ:8.1%}  {why}")
    assert abs(occupancy(32, 0, 256)[0] - 1.0) < 1e-9
    assert abs(occupancy(128, 0, 256)[0] - 0.25) < 1e-9
    assert abs(occupancy(255, 0, 256)[0] - 0.125) < 1e-9

    print("\nall assertions passed ✓")

if __name__ == "__main__":
    main()
```

## C.2 需要 GPU：带宽、GEMM 的 M 扫描、launch 开销

```python
#!/usr/bin/env python3
"""需 torch + CUDA。运行： python3 bench_gpu.py"""
import torch

def bench(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters          # ms

def bench_copy(gib=1):
    n = gib * (1 << 30) // 2
    a = torch.empty(n, dtype=torch.bfloat16, device="cuda")
    b = torch.empty_like(a)
    ms = bench(lambda: b.copy_(a))
    print(f"copy  {ms:7.3f} ms  {2*gib*(1<<30)/(ms*1e-3)/1e9:7.1f} GB/s")

def bench_gemm(N=8192, K=8192):
    print(f"\n{'M':>6} {'ms':>9} {'TFLOPS':>8} {'AI':>8}")
    for M in (1, 8, 16, 32, 64, 128, 206, 256, 512, 1024, 4096):
        a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
        ms = bench(lambda: a @ w.t(), warmup=5, iters=20)
        tflops = 2 * M * N * K / (ms * 1e-3) / 1e12
        print(f"{M:6d} {ms:9.3f} {tflops:8.1f} {2*M/(2):8.1f}")

def bench_launch(iters=5000):
    x = torch.empty(8, device="cuda")
    ms = bench(lambda: x.add_(1.0), warmup=100, iters=iters)
    print(f"\nper-op (dispatch+launch): {ms*1e3:.2f} µs")

if __name__ == "__main__":
    assert torch.cuda.is_available(), "需要 GPU"
    print(torch.cuda.get_device_name(0))
    bench_copy()
    bench_gemm()
    bench_launch()
```

**预期读数**（H200，BF16）：

| 项 | 期望 |
|---|---|
| copy | 3.4–4.0 TB/s |
| GEMM M=64 | ~120–180 TFLOPS |
| GEMM M=206 | ~400–500 TFLOPS（开始拐弯） |
| GEMM M=4096 | ~700–850 TFLOPS |
| per-op | 3–8 µs |

---

# 附录 D 相对主线第 4 章的增量

| 主线原来给的 | 本文补充的机理 | 位置 |
|---|---|---|
| HBM 三约束的结论 | 堆栈→bank→row 的物理结构、pin 速率反算、row 命中/未命中 | §28 |
| 「18 KB/SM 在途」 | 换算成 142 条 line、MSHR 上限、每 warp 需 2–3 个未完成 load | §14 |
| 「4 TC + 128 core」 | sub-partition 结构、每 cycle 4 条 warp 指令、延迟表 | §24 |
| 「32 倍」 | 拆成 FP8 32× / BF16 16×、每 cycle MAC 数 | §25.6 |
| 「FlashAttention 省带宽」 | 143 ms → 2.2 ms 的字节账、online softmax 恒等式 | §20 |
| 「AI ≈ 2M」 | 五个口径陷阱、split-K 的修正、M* 的通用公式 | §34、§35 |
| 「拐点 412」 | 精度不变性证明、有效值口径下的 103、三卡对照 | §35 |
| 「decode 下界」 | 四个方向的定量收益、TP 的两面性 | §38 |
| 「GPU 利用率无意义」 | 三个利用率指标与两个效率因子 | §36 |
| 「Fusion 有用」 | RMSNorm+residual 80 KB → 48 KB | §7.2 |
| 「batch 的理由」 | 并行度账（decode 只用到 3–6%）、波次边界 | §21 |
| 「TinyToy AI 远小于真实模型（因为 H=8 太小）」 | **纠正**：线性 AI 与 H 无关，2 倍差源于 BF16 vs FP8 的精度因子 | §40.4 |
| — | **全新**：torch 调用链与 launch 账、CUDA Graph、allocator 三数、GEMM 五级演进、合并访存/sector 账、bank conflict、nvcc/ncu 判据树 | §1–§19、§39 |

---

# 附录 E 口径来源与不确定处

## E.1 三类来源

| 标记 | 含义 | 本文中的例子 |
|---|---|---|
| **[spec]** | 官方 datasheet / 白皮书 | 141 GB、4.8 TB/s、132 SM、989/1979 TFLOPS、700 W |
| **[calc]** | 本文由公式反算 | 60.8 TFLOPS、6.25 Gbps/pin、30.4 TB/s SMEM、20.2 B/cycle/SM、M*=206 |
| **[meas]** | 公开实测/经验的区间 | ~500 ns 延迟、GEMM 各级实测吞吐、1.6–1.8 GHz 持续主频、随机访存 30–50% |

## E.2 明确的不确定处

| 项 | 不确定点 | 本文处理 |
|---|---|---|
| HBM 堆栈数 | 6×1024 bit 反算得 6.25 Gbps/pin，低于 HBM3e 公开上限 9.6 Gbps | 标注 [calc]，并保留 4.8 TB/s / 141 GB 两个确定值 |
| L2 带宽 | 实测区间 5.5–7 TB/s，随访问模式变化 | 取 ~7 TB/s 量级，标注 [calc] |
| 各级 GEMM 实测 | 依赖编译器、clock、驱动版本 | 给区间，不给单点 |
| Blackwell 细节 | tcgen05/TMEM 的公开资料少于 Hopper | 只讲形态差异，不编造指令级参数 |
| B200 SM 数 | 公开口径分散（双 die 封装） | 只用于拐点对照，不列 SM 数 |
| 2:4 稀疏 | 官方 2× 是标称 | 全文用 dense 口径，并在 §25.7 说明 |
| 温度对带宽 | 3–8% 是经验区间 | 标注 [meas] |

## E.3 与主线数字的对应检查

```
141 GB      ✓ 主线 §4.1 / 附录 B.1
4.8 TB/s    ✓ 同上
500 ns      ✓ 同上
132 SM      ✓ 同上
1979 / 989  ✓ 同上
412         ✓ 1979/4.8 = 412.3
206         ✓ 989/4.8 = 206.0
2.4 MB      ✓ 4.8e12 × 500e-9
30 TB/s     ✓ 128 B × 1.8e9 × 132 = 30.4 TB/s
4992/2560/1.95   ✓ §40.1
7296/2400/3.04   ✓ §40.2
107520/3328/32.3 ✓ §40.3
50 GiB / 163,840 ✓ 主线 §4.11（320 KiB/token）
```

**全部一致。** 本文没有引入任何与主线冲突的硬数字；不一致只可能出现在「本文标了 [calc]/[meas] 的地方」与「读者手上的实测」之间，附录 E.2 已逐条说明。

---

## 一句话总结

> **GPU 的性能不是「利用率」一个数，而是「AI 与拐点的比值」加上「并行度能不能喂满」。**
> 前者由算法和精度决定，后者由 batch 和 grid 决定。
> torch 层决定你提交了多少工作，CUDA 层决定这些工作怎么落到 132 个 SM 上，硬件层决定每一个字节要花多少 ns。
> **三层都对齐，才能解释为什么同一张 H200 上，一个场景 TC 用 72%，另一个场景只用 1.8%。**

