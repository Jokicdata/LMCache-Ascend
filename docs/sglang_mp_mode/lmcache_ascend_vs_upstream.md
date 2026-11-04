# LMCache-Ascend 相对上游补了什么（概念澄清 + 代码证据）

> 面向场景：在给 SGLang 做 MP 适配之前，先厘清「上游 LMCache 已经有什么」「本仓 LMCache-Ascend 额外补了什么」「为什么新模型出现就要针对性适配」。
>
> 结论先行：**上游提供“缓存怎么组织 + 昇腾平台抽象”，本仓提供“KV 在昇腾上怎么搬 + 布局怎么表达”。没有插件不是“跑不了”，而是“普通模型慢约 8 倍，MLA/DSA/混合 Mamba 等新布局直接不认识”。**

---

## 1. 定位与分工

| 项目 | 定位 | 负责 | 不负责 |
|---|---|---|---|
| **上游 LMCache** | 引擎无关、厂商无关的 KV 缓存引擎 | `lmcache server`、ZMQ/gRPC 控制面、缓存索引/淘汰、L1/L2 存储、各引擎通用 connector、**NPU 平台抽象层** | 昇腾原生 kernel、CANN 相关代码 |
| **LMCache-Ascend** | 上游的昇腾硬件插件（community plugin） | NPU 原生 `c_ops`、块传输 kernel、HCCL/HiXL、plane 几何、KV 分组、NPU 连接器、插件激活 | LMCache 主体、缓存协议、引擎本身 |

一句话：**上游负责“缓存怎么管”，本仓负责“在昇腾上怎么搬”。两者是叠加关系，不是替代关系。**

---

## 2. MP mode 支持矩阵

| 问题 | 上游 LMCache | LMCache-Ascend（本仓） |
|---|---|---|
| 有 MP mode 吗 | ✅ 上游主推架构 | ✅ 有，但**只验证了 vLLM 路径** |
| MP 控制面谁提供 | 上游：`lmcache server`、`LMCacheMPConnector`、ZMQ/gRPC | 直接复用 |
| MP 数据面谁提供 | 通用后端 + v0.5.6 起 NPU 平台抽象 | **昇腾原生实现**（kernel + HCCL/HiXL + NpuIPCWrapper） |
| vLLM + MP | ✅ | ✅ 主线，有文档 + E2E 测试 |
| SGLang + MP | ✅ v0.5.6 起（仅 MP，无 in-process） | ❌ 未适配（本仓 SGLang 仍是旧 in-process） |
| TensorRT-LLM + MP | ✅ | ❌ 未适配 |

上游 v0.5.6 与 NPU 相关的关键 PR：`#5341`（NPU platform backends for LMCache-driven transfer）、`#5138`（`NL_X_NP_X_NB_BS_ONE_HS` 格式）、`#5611`（移除 unified MP connector 的硬编码 CUDA 检查）。

---

## 3. 有没有插件 —— 能力差异不只在性能

差异分两类：

### 3.1 功能性差异（没插件 = 布局不认识，会报错/走错路径）

| 代码位置 | 补的功能 | 为什么是功能性（不补就错） |
|---|---|---|
| [`lmcache_ascend/v1/kv_format.py#L13-L89`](../../lmcache_ascend/v1/kv_format.py#L13-L89) | `KVCacheFormat` 枚举：MERGED / SEPARATE / **MLA / DSA** | 上游只按“单张量”或“K=V 两张量”假设；MLA 是 K/V 维度不同，DSA 是 `(k, v, dsa_k)` 三元组 |
| [`lmcache_ascend/v1/kv_format.py#L91-L163`](../../lmcache_ascend/v1/kv_format.py#L91-L163) | `detect()` 自动识别 | 不认识 DSA 三元组就直接走错路径 |
| [`lmcache_ascend/v1/shape_desc.py#L132-L180`](../../lmcache_ascend/v1/shape_desc.py#L132-L180) | 补 `num_planes` / `plane_slot_bytes` / `plane_block_stride_bytes` | 上游 `PageBufferShapeDesc` 是标量字段，表达不了 MLA/DSA 多 plane 的逐 plane 步长，kernel 无法寻址 |
| [`lmcache_ascend/v1/kv_layer_groups.py#L13-L80`](../../lmcache_ascend/v1/kv_layer_groups.py#L13-L80) | 重写 `build_kv_layer_groups` / `hidden_dim_size` | 上游按单张量分组，K/V 分离、tuple MLA 会分错组 |

### 3.2 性能差异（没插件 = 慢约 8 倍，但能跑通）

| 代码位置 | 补的性能 |
|---|---|
| [`lmcache_ascend/__init__.py#L41-L49`](../../lmcache_ascend/__init__.py#L41-L49) | 注释明说：找不到 native 就退到 torch 基线，“约慢 8 倍” |
| `csrc/kernels/multi_layer/*`、`single_layer/*` | AscendC 原生搬运核（代替 torch 逐元素 copy） |
| `csrc/kernels/fused_rope/*`、`csrc/pos_kernels.cpp` | 融合 RoPE（代替分开算） |
| `csrc/hccl/*`、`csrc/hixl/*`、`v1/transfer_channel/*` | HCCL/HiXL 设备直连（代替 CPU 中转） |

**区分结论**：A 类（`kv_format.py` / `shape_desc.py` / `kv_layer_groups.py`）是“布局不认识”的功能性缺失；B 类（`csrc/*` kernel）才是“慢 8 倍”的性能缺失。

---

## 4. 为什么新模型要针对性适配

**根源：K/V Cache 没有统一形状，每个架构在显存里的排列都不一样。**

### 4.1 四种典型格式差异

见 [`kv_format.py`](../../lmcache_ascend/v1/kv_format.py#L25-L61) 的 shape 注释：

```text
MERGED_KV   : [num_blocks, block_size, num_heads, head_dim]   标准 MHA
SEPARATE_KV : (K_tensor, V_tensor)，K、V 分开                  GQA（Qwen3）
MLA_KV      : (k, v)，且 K/V 维度不同   ← DeepSeek V2/V3 latent 压缩
DSA_KV      : (k, v, dsa_k) 三元组      ← DeepSeek V3.2 稀疏注意力
```

### 4.2 `EngineKVFormat` 已有 18 种

见 [`csrc/mp_mem_kernels.h#L26-L45`](../../csrc/mp_mem_kernels.h#L26-L45)，命名缩写对应的差异来源：

| 缩写 | 含义 | 谁引入的差异 |
|---|---|---|
| `TWO` / `ONE` | K、V 分不分开 | GQA/MHA vs 合并 |
| `NP`（num_planes） | 多平面 | MLA/DSA 的 latent + 多个 plane |
| `CS`（cache state） | recurrent state | Mamba/GDN 混合层 |
| `BSV_BSS` | 变长 block | 稀疏/变长序列 |

每种格式就是一个“层数 × 块数 × 头数 × 头维 × 是否分 K/V × 是否多 plane”的排列组合。

### 4.3 为什么这导致“每出新模型就要适配”

1. **KV 物理布局随架构而定，不是模型无关的。** MHA→GQA→MLA→DSA→混合 Mamba，每一步都改变 K/V 的维度、是否合并、是否有额外 plane、是否带 recurrent state。上游只能抽象到“标量 + 分类”，具体布局必须由昇腾侧补。

2. **AscendC kernel 是编译期强类型。** [`csrc/mp_mem_kernels.h#L47-L99`](../../csrc/mp_mem_kernels.h#L47-L99) 的 `PageBufferShapeDesc` 把 `nl/nb/bs/nh/hs/element_size` 编译期定死，kernel 按这个 layout 做 AIV 高效 data copy，不能像 torch 那样运行时动态处理任意 shape。出一个新布局，就要补一个新格式分支。

3. **引擎侧 block 语义不一样。** 同一个模型 Qwen3.8，见 [`docs/mp_mode/qwen3_5.md`](../mp_mode/qwen3_5.md#L24-L28)：

   | 模型 | 引擎版本 | chunk-size |
   |---|---|---|
   | Qwen3.8-27B | vLLM-Ascend **v0.30** | **896** |
   | Qwen3.8-27B | vLLM-Ascend **v0.23** | **1536** |

   同一个模型换一个引擎版本，attention block size 就从 896 变 1536，LMCache 的 `chunk_size` 必须跟着从启动日志读 `N` 对齐。

---

## 5. 总结

```text
模型架构（决定 KV 布局）
   × 引擎 + 版本（决定 block_size / slot 语义）
   × 硬件后端（决定 kernel 是 CUDA / AscendC）
   = 需要适配的组合数
```

| 你的疑问 | 答案 |
|---|---|
| 有没有插件的差异都在性能上吗 | 不是。功能性（布局不认识）**和**性能（慢 8 倍）两种都有 |
| 有插件能跑，没插件也能跑吗 | 普通 GQA/MHA 能跑但慢；MLA/DSA/混合 Mamba 直接支持不了 |
| 为什么新模型要针对性适配 | KV 布局无统一标准，`EngineKVFormat` 已有 18 种且还在涨；AscendC kernel 编译期写死布局 |
| 适配的核心在哪 | “K/V 内存物理排布”不是模型无关的，必须按“模型 × 引擎 × 硬件”逐个枚举补充 |

---

## 与 SGLang MP 适配的关系

本仓 MP 的 NPU 数据面（kernel / HCCL/HiXL / plane 几何 / KV 分组）是 SGLang 适配可**直接复用**的部分；需要新增的是 SGLang 专属的：

1. 插件激活从 vllm-only 改为按引擎分派（[`lmcache_ascend/__init__.py`](../../lmcache_ascend/__init__.py#L22-L28) 目前只认 vLLM）；
2. SGLang MP connector 生命周期（scheduler / TP worker 调用点）；
3. SGLang 的 K/V pool + RadixAttention + unified FULL/SWA/MAMBA component 映射；
4. chunk_size 与 SGLang attention block size 对齐；
5. Qwen3.8 混合 Mamba 的 recurrent state 生命周期。

详见 [`adaptation_walkthrough.md`](./adaptation_walkthrough.md)。