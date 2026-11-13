# Hybrid 模型适配蓝图：Qwen3.8 / GLM-5.3-Flash / DeepSeek-V4.1（SGLang MP mode）

> 前置：Qwen2.5-7B（标准 GQA）已首亮成功（旧 connector + 堆叠 KV 兼容补丁）。
> 本文回答：三个新模型各是什么架构、比 Qwen3.8 多多少工作、统一适配战略是什么、
> 现在需要哪些侦察信息。
>
> 状态：**调研完成，待容器侦察**（第 5 节命令清单）。

---

## 1. 三个模型是什么（外部资料调研结论）

### 1.1 Qwen3.8-27B（已在手，27B）

```text
60+ 层，交替排布：
  GDN 线性注意力层（"白板"状态） + 全注意力层（K/V 卡片）
SGLang 内存池：HybridLinearKVPool（无 k_buffer，构造即崩——已实测）
```

### 1.2 GLM-5.3-Flash（320B 总参 / 18B 激活，2026-08 发布）

| 项 | 内容 | 来源 |
|---|---|---|
| 架构 | **45 层 = 34 KDA 线性注意力层 + 11 DSA 稀疏注意力层**（每 4 层一个） | NVIDIA NIM 模型卡（架构字段） |
| 其它 | mHC 超连接、288 专家 MoE top-8、**vision encoder（原生多模态）**、1 MTP 层 | 同上 |
| 上下文 | 1M tokens | 模型卡 |
| 定位 | GLM-5 系列首个混合架构：**稀疏 + 线性注意力混合**，长上下文成本大降 | 智谱官方 |

与 Qwen3.8 的架构对比：

| | Qwen3.8 | GLM-5.3-Flash |
|---|---|---|
| 线性层 | GDN | KDA（同类"白板"状态） |
| 注意力层 | **全注意力**（存全量 K/V） | **DSA 稀疏注意力**（K/V + 稀疏索引） |
| 额外件 | — | vision encoder（多模态）、MTP |

### 1.3 DeepSeek-V4.1-Flash（552B 总参 / 8~16B 激活，2026-09 发布）

注意力压缩四级跳的终点站：

```text
V2  MLA    潜在缓存（KV 低秩压缩，~15x）
V3.2 DSA   稀疏计算
V4  CSA    连续 4 token 的 KV 聚合成 1 单元（"近处细看"）
    HCA    远距离 token 128x 二次聚合（"远处宏观"）
V4.1-Flash CSA2 + 跨层缓存复用 + FP4 存储 → 全局 890 B/token
```

| 项 | 内容 |
|---|---|
| KV 缓存形态 | **MLA 潜在向量 + 多 plane（latent / rope / DSA 索引相关）+ CSA/HCA 压缩状态** |
| SGLang 侧 | 专用 `deepseek_v4_memory_pool.py` + **NPU 专属** `hardware_backend/npu/dsv4/dsv4_memory_pool.py` |
| 显存 | 1M 上下文仅 5.48 GB KV（对比 Llama3.1 需 274GB） |

### 1.4 上游 LMCache v0.5.6 对这三个模型的支持声明（release notes）

| PR | 内容 | 分类 |
|---|---|---|
| #5042 | **Support GLM-5.3-Flash and Qwen3.8-Flash-Next** | MP Server & Connector |
| #5379 | **Support DeepSeek-V4.1-Flash** | **SGLang Integration** |
| #4828/#5041/#5419 | Unified LMCache radix cache for SGLang in MP mode, with docs | SGLang Integration |
| #5138 | NL_X_NP_X_NB_BS_ONE_HS KV 格式（**vLLM-Ascend MLA/DSA plane 元组**） | Ascend 平台 |

**解读**：LMCache 的 server 侧和 SGLang 集成侧（unified 路径）已经认识这三个模型的
KV 格式——包括昇腾的 MLA/DSA plane 格式。缺的是 **SGLang 引擎侧的接线激活**
（这正是我们在做的事）。

### 1.5 本仓（LMCache-Ascend）现有支持盘点

`docs/recommended_deployment_guide/support_models.md`：

| 模型 | 分支 | 模式 | 引擎 |
|---|---|---|---|
| GLM-5.3 | `main` | **In-process** | vLLM |
| DeepSeek-V4-Flash / V4-Pro | `dsv4_support_045`（绑 LMCache 0.4.5） | **In-process** | vLLM |

**关键结论**：这三个模型在 **vLLM + in-process** 路径上已有不同程度的支持
（GLM-5.3 在 main；DSV4 在专用分支）——意味着 NPU 数据面的 kernel/格式基本就绪。
但 **SGLang + MP mode** 路径上全是空白，这就是本次适配的范围。

---

## 2. 核心判断：三个模型必须一起切到 unified 路径

### 2.1 为什么旧 connector 路径（Qwen2.5-7B 用的）走不通

旧 `LMCacheMPConnector` 的协议方言是"K 列表 + V 列表 + 无 mamba"：

| 模型 | 旧路径的死因 |
|---|---|
| Qwen3.8 | `HybridLinearKVPool` 无 `k_buffer`（构造崩）+ **GDN 状态无法表达**（协议里没有 state 槽位） |
| GLM-5.3-Flash | 同为 hybrid 线性池家族 + DSA 索引数据无处安放 |
| DeepSeek-V4.1 | MLA 是**单一潜在向量/多 plane**，根本没有 K/V 分离——`k_pool`/`v_pool` 语义不存在 |

### 2.2 unified 路径就是为它们设计的

新 `UnifiedLMCacheMPConnector`（LMCache v0.5.6）的构造签名（已从容器源码确认）：

```python
UnifiedLMCacheMPConnector(
    config_file, model_config, tp_size, tp_rank, tp_group, ...,
    page_size,
    token_to_kv_pool_allocator,     # ← 直接拿 allocator，不再要 k_pool/v_pool
    req_to_token_pool,
    tree_components,                # ← FULL / SWA / MAMBA / C128 组件元组
    mamba_component,                # ← 线性层状态一等公民
    sliding_window_size,
)
```

SGLang 侧的组件体系也已存在（`registry.py` 会按模型拼 `tree_components`：
hybrid_ssm → 加 `ComponentType.MAMBA`；我们已确认 SGLang dev476 有完整实现）。

**所以战略不是"每个模型单独适配旧路径"，而是"一次切换到 unified 路径，
三个模型同时受益"**——这正是当初决策树里"300~600 行中等适配"的那个分支。

### 2.3 各模型相对工作量表

| 工作项 | Qwen3.8 | GLM-5.3-Flash | DeepSeek-V4.1 |
|---|---|---|---|
| 切换到 unified 路径（接线 + LMCRadixCache↔新 connector API 映射） | ✅ 必做（打头阵） | 免费继承 | 免费继承 |
| hybrid 线性池组件提取（HybridLinearKVPool → FULL+MAMBA） | ✅ 必做 | 大部分继承（KDA≈GDN 同族） | 不适用 |
| DSA 稀疏注意力层组件映射 | 不适用 | ✅ 增量（DSA 层在 SGLang 的池类型待侦察） | 部分（DSA 索引） |
| MLA plane 提取（DSV4 内存池 → LMCache plane 格式） | 不适用 | 不适用 | ✅ 增量（本仓 shape_desc 的 plane 几何正好是为此造的） |
| CSA/HCA 压缩状态处理 | 不适用 | 不适用 | ⚠️ **最大未知**（V4.1 特有，可能连 unified 路径都要扩展） |
| 多模态（vision encoder 的 KV 排除） | 不适用 | ✅ 小增量（只缓存 LLM 部分） | V4.1-Flash 也是多模态，同款 |
| NPU 数据面 kernel | 已就绪（e2e 3 passed） | 大概率就绪（vLLM in-process 已跑通 GLM-5.3） | **部分在 dsv4_support_045 分支**，需向 main/dev 移植评估 |
| 模型规模现实性（8×910B4=256GB HBM） | 27B ✅ 单机 | 320B 需 FP8+多机或更小变体 | 552B 多机；建议找同架构小模型代理验证 |

**一句话**：Qwen3.8 是主战场（unified 切换 + hybrid 组件）；GLM-5.3-Flash 在其
基础上加 DSA 层映射和多模态排除；DeepSeek-V4.1 的 MLA plane 有本仓 shape_desc
的现成积累，但 CSA/HCA 压缩状态是全新未知数，且 vLLM 侧支持还锁在旧分支。

---

## 3. 适配路线图（分四阶段）

```text
阶段 0：侦察（本文第 5 节命令，零代码）
  └─ 确定 unified 路径在 SGLang dev476 的激活方式
     候选 A：--radix-cache-backend=<lmcache 注册名>（SGLang 有插件注册表机制）
     候选 B：--enable-unified-cache-external-linker（registry 里排在 enable_lmcache 之前）
     候选 C：容器 LMCache 的 examples/sgl_integration/README.md 官方文档给答案

阶段 1：Qwen3.8 打通（核心交付）
  ├─ unified 路径激活（阶段 0 的答案）
  ├─ HybridLinearKVPool → tree_components(FULL+MAMBA) 提取
  │   （大概率本仓 runtime patch：从 allocator 拆出全注意力层 KV + GDN state，
  │     组装 unified_kv_adapter 需要的组件结构）
  ├─ 验收：Stored/Retrieved + 输出正确 + GDN state 回灌后多轮对话不崩
  └─ 预估：300~600 行 Python（当初决策表的中等档）

阶段 2：GLM-5.3-Flash
  ├─ 复用阶段 1 的 hybrid 线路
  ├─ 增量：DSA 层的池类型识别与组件映射（侦察后定）
  ├─ 增量：多模态——vision encoder 的 KV 不进 LMCache（只注册 LLM 层）
  └─ 现实性：320B 单机放不下 → 用同架构小变体或量化版验证

阶段 3：DeepSeek-V4.1
  ├─ MLA plane 提取：SGLang DSV4 池 → LMCache plane 格式
  │   （本仓 v1/shape_desc.py 的 plane 几何 + 上游 #5138 的
  │     NL_X_NP_X_NB_BS_ONE_HS 就是为此准备的）
  ├─ CSA/HCA 压缩状态：先侦察 SGLang dsv4_memory_pool 结构再评估
  ├─ 本仓 dsv4_support_045 分支的 vLLM 积累向 main 移植的可行性评估
  └─ 现实性：552B 多机；建议先用 DSV4-Flash 的较小同架构模型验证
```

---

## 4. 已确认的架构事实（写代码前的地基）

| # | 事实 | 来源 |
|---|---|---|
| 1 | SGLang dev476 有完整组件体系：`ComponentType.FULL/SWA/MAMBA/C128`，`registry.py` 按模型拼 `tree_components` | 容器 grep（本会话早期） |
| 2 | `CacheInitParams` 带 `tree_components`/`component_registry_override` 字段 | 容器源码 |
| 3 | LMCache `UnifiedLMCacheMPConnector` + `SGLangUnifiedKVAdapter` 存在且签名已知 | 容器源码 + 官方测试 `test_unified_sglang_mp_connector.py:138` |
| 4 | SGLang `create_tree_cache` 有插件注册表：`--radix-cache-backend` → `register_radix_cache_backend(...)` | 容器源码（registry.py:220-245） |
| 5 | SGLang registry 有 `enable_unified_cache_external_linker` 分支（排在 enable_lmcache 之前） | 容器源码 |
| 6 | 旧 `lmc_radix_cache.py` 10 个 connector 调用点全是同步 API，与新类异步 API（submit/poll/complete）完全不匹配 | 容器 grep（方法面对比） |
| 7 | LMCache v0.5.6 server 侧已认识三个模型的 KV 格式（#5042/#5379） | release notes |
| 8 | 本仓 vLLM in-process 路径：GLM-5.3 main 已支持；DSV4 在 dsv4_support_045 | support_models.md |

---

## 4.1 侦察结果（2026-11-13，R1~R6 已回收）

| 命令 | 结果 | 判定 |
|---|---|---|
| R1 官方 README | 示例仍是**旧路径**（`--enable-lmcache` + Qwen2.5-14B 标准模型）；注明 hybrid 多组模型应配 `mp_transfer_mode: lmcache_driven`（我们 server 已是） | unified 的官方文档尚未跟上；不能靠 README |
| R2 radix backend 注册 | **无输出** | ❌ 候选 A 死：LMCache 没有注册 SGLang radix backend |
| R3 unified_kv_adapter 前 120 行 | **金矿**：adapter 用 `getattr(kv_pool, "full_kv_pool", kv_pool)` 拿 hybrid 池的全注意力子池；`_resolve_dsa_indexer_tensors` 追加 DSA 索引张量；MAMBA 从 `req_to_token_pool.mamba_pool` 取；**DSV4 有专用路径** `_resolve_dsv4_registered_groups`；`is_mla_enabled` 覆盖 MLA + is_deepseek_v4 | ✅ **上游 adapter 已同时设计好三个模型家族的组件提取** |
| R4 external linker | `enable_unified_cache_external_linker` 在 registry:103 **优先于** enable_lmcache 返回 `_create_unified_radix_cache`；:191-192 有 **backend 选择器** `unified_cache_external_linker_backend`；scheduler/kv_cache_configurator/base_spec_worker 多处引用 | ✅ **候选 B 活且是正解方向**：flag + backend 选择器 |
| R5 模型与池 | `glm5_next.py`、`deepseek_v4.py`（+vit/nextn/dspark）、`DeepSeekV4TokenToKVPool` 等 8 个池类**全部存在**于 dev476 | ✅ 阶段 2/3 的 SGLang 侧地基齐全 |
| R6 HybridLinearKVPool | 有 `self.full_kv_pool`（NPU 上用 `NPUMHATokenToKVPool`）+ `self.mamba_pool`；**支持 use_mla/use_dsa 开关**——意味着 GLM-5.3（KDA+DSA）大概率也用这个池，与 Qwen3.8 同一家族 | ✅ 阶段 1 与阶段 2 共享同一 hybrid 线路，印证"一次适配多模型受益" |

**修正后的激活假设**（待 R7~R11 确认）：

```bash
--enable-unified-cache-external-linker \
--unified-cache-external-linker-backend <lmcache? 具体值待查>
```

## 4.2 侦察结果（第二轮 R7~R11，2026-11-13）

| 命令 | 结果 | 判定 |
|---|---|---|
| R7 adapter 剩余部分 | MAMBA（checkpoint grid 对齐、`recurrent_state=True`、`slots_per_block=1`）、`device_indices_by_group`（FULL/SWA/MAMBA 三种物理地址翻译）、DSA 索引（2-D page-native 校验）、DSV4（C4/C128/C1/C2 压缩比折行）**全部实现完整**；但 `_resolve_pool_tensors` 对 MHA 池仍用 `if not k_buffer` 列表假设——**堆叠张量布尔歧义在 unified 路径同样存在** | ✅ 提取逻辑上游写完；⚠️ 需扩展本仓 compat 补丁覆盖 adapter |
| R8 工厂全貌 | external linker backend **仅 mooncake / mori**，无 lmcache；但 `create_tree_cache` 顶部有**注册表分支**：`--radix-cache-backend=<name>` → `register_radix_cache_backend(...)`（官方插件机制，注释明示 "External backends must call ... at import time"） | ❌ 候选 B 死；✅ **正解 = 候选 D：本仓注册 radix backend factory** |
| R9 参数定义 | `fields/memory.py:213/217` 定义两个 flag；`hicache_hook.py`/`deepseek_v4_hook.py` 有联动校验 | 辅助信息 |
| R10 全树找构造点 | **零输出**——SGLang dev476 无任何 `UnifiedLMCacheMPConnector` 构造点 | unified 路径在 SGLang 侧完全无接线，必须由本仓接 |
| R11 scheduler 段 | linker 与 hicache 共用 `hicache_consumer_index` 加载通路 | linker 生命周期与 hicache 同构，可参照 |

**修正后的架构（候选 D，最终方案雏形）**：

```text
lmcache_ascend 插件激活时（sglang 运行时）
  └─ register_radix_cache_backend("lmcache-ascend", factory)   ★本仓新增
用户启动参数
  └─ --radix-cache-backend=lmcache-ascend
SGLang create_tree_cache
  └─ factory(ctx)                                    ★本仓实现
       ├─ 复刻 _create_unified_radix_cache 组件拼装（FULL+MAMBA/…）
       ├─ cache = UnifiedRadixCache(params)
       └─ cache.init_cache_linker(LMCacheDirectLinker(...))  ★本仓新写（参照 MooncakeDirectLinker）
            └─ 持有 UnifiedLMCacheMPConnector（上游）
                 └─ SGLangUnifiedKVAdapter（上游：三模型组件提取）
                      └─ [扩展] 堆叠张量拆分（复用 mp_connector_compat 的 _split_stacked_pool）
```

**未决问题（R12/R17 目标）**：注册表要求 import time 注册，但插件靠 `import lmcache`
激活——radix backend 分支不经过 `lmc_radix_cache.py` 时，scheduler 进程里谁负责
import lmcache？（候选答案：两个 flag 同时带 / entry_points 自动发现 / scheduler
早期 import 点）

## 4.3 侦察结果（第三轮 R12~R17 + 本地源码通读，2026-11-13，架构锁定）

> 本轮起源码自查：SGLang 已 clone 至 `d:\proj\sglang`（commit `4e60d70ba`，与容器一致），
> LMCache clone 至 `d:\proj\LMCache-src`（commit `3e11e108`，与容器一致）。
> 容器 SGLang 仅 pyproject 构建配置有本地改动，runtime 源码与 commit 一致。

| 发现 | 出处 | 结论 |
|---|---|---|
| 注册表是纯 dict，无 entry_points、无按名导入 | registry.py:51-76 | radix backend 注册路径需要"有人先 import 我们"——鸡生蛋坐实 |
| flexkv 是唯一注册者，且只在 `--enable-flexkv` 分支（默认链内部）被 import | storage/flexkv/\_\_init\_\_.py:77 | 纯 `--radix-cache-backend` 路径对外部包**没有**自动 import 机制 |
| **`LMCacheUnifiedRadixCache` 全世界不存在**（两仓库均只在报错文案里出现） | grep 双仓库 | **这就是本次适配的核心交付物** |
| linker ABC（`UnifiedCacheLinker`）11 个抽象方法 + `UnifiedCacheLinkerWrapper` 全文已读 | unified_cache_linker.py | 接口面完全掌握：lookup/load/start_layer_wise_loading/cancel_queued_load/num+pop_completed_loads/offload/num+pop_completed_offloads/reset/close + layer_done_counter |
| **wrapper 守卫拒绝 MAMBA**（`_EXTERNAL_LINKER_SUPPORTED_COMPONENTS={FULL,SWA}`，构造即 raise）——但 flow 代码里 MAMBA 处理齐全（match/load_back 都有 PoolName.MAMBA 分支） | unified_cache_linker.py:53-58, 152-160 vs 230-231, 344-373 | 守卫是保守过期项；本仓 patch 该 frozenset 加入 MAMBA 即可放行 hybrid |
| `RadixCache` 未定义 `__init_subclass__` | radix_cache.py:300 | 本仓可安全挂 `__init_subclass__` 钩子实现类定义时拦截 |
| connector 完整 API（1166 行）：`submit_lookup(rid, token_ids, *, local_hit_tokens, cache_salt)` / `poll_lookup(op)` / `submit_load(op, device_indices, *, local_hit_tokens, ...)` / `prepare_load_on_stream(op, stream)` / `complete_load(op)` / `submit_store(rid, token_ids, ...)` / `complete_store` / `free_lookup_locks(rid, start, end)` / `end_session/finish_request/end_all_sessions` / `clear/close` | unified_lmcache_mp_connector.py | linker 桥接的目标 API 全部确认 |
| PoolTransfer 结构：name/host_indices/device_indices/keys(hit-policy)/nodes_to_load/indices_from_pool/anchor_index_parts；PoolName 枚举含 KV/MAMBA/SWA/INDEXER + DSV4 全家 | hicache_storage.py:51-118 | linker 流通数据结构确认 |

### 4.3.1 最终架构（锁定，编码蓝本）

```text
激活链（UX 与 Qwen2.5 路径完全一致，零新 flag）：
  用户: python -m sglang.launch_server ... --enable-lmcache --lmcache-config-file x.yaml
    └─ registry.default_radix_cache_factory → enable_lmcache 分支
        └─ import lmc_radix_cache
            └─ (模块顶部) from lmcache... import → import lmcache
                └─ 平台初始化 → import lmcache_ascend → 插件激活 ★
                    └─ 新增: 挂 RadixCache.__init_subclass__ 钩子
        └─ (模块体继续) class LMCRadixCache(RadixCache): ← 钩子触发!
            └─ 钩子捕获 original_init，替换为 dispatcher __init__
        └─ return LMCRadixCache(params, model_config, tp_size, rank, tp_group)
            └─ dispatcher 检查 params.token_to_kv_pool_allocator.get_kvcache():
                 ├─ 标准池(有健康 k_buffer/v_buffer 列表) → original_init
                 │    （旧 connector 路径，Qwen2.5-7B 已验证，compat 补丁继续生效）
                 └─ hybrid/DSV4/MLA → ★unified 路径:
                      ├─ 从池结构推导 tree_components（FULL + MAMBA/SWA）
                      ├─ UnifiedRadixCache.__init__(self, params); self.__class__ 切换
                      ├─ patch _EXTERNAL_LINKER_SUPPORTED_COMPONENTS += MAMBA
                      └─ self.init_cache_linker(LMCacheDirectLinker(...)) ★本仓新写

LMCacheDirectLinker（本仓新文件，实现 UnifiedCacheLinker ABC）：
  lookup(rid, transfers)      → submit_lookup(rid, tokens[rid], ...) + poll 至完成
                                （tokens 由 subclass 的 match_prefix 副通道暂存）
  load(rid, transfers)        → submit_load(lookup_op, device_indices_by_group)
  start_layer_wise_loading()  → prepare_load_on_stream + LayerWiseLoadCounter
  num/pop_completed_loads     → complete_load 结果队列
  offload(transfers)          → submit_store
  num/pop_completed_offload   → complete_store
  cancel_queued_load(rid)     → free_lookup_locks
  reset/close                 → end_all_sessions / close

token_ids 副通道（关键桥接技巧）：
  UnifiedRadixCache.match_prefix 持有 RadixKey(=token_ids)，但 linker.lookup 只收
  transfers(keys=存储哈希)。解法：本仓 subclass（LMCacheAscendUnifiedCache）
  override match_prefix/init_load_back，先暂存 rid→tokens 再调 super()，
  linker 从暂存表取 tokens 喂 submit_lookup。
```

### 4.3.2 交付物清单（阶段 1 编码范围）——✅ 已完成（2026-11-13）

> **编码期间的设计修正**：放弃 linker-ABC 路线（`MambaComponent.build_external_linker_transfer`
> 直接 raise "will support soon" + PoolTransfer 只带哈希不带 tokens，哈希↔token 断层无法桥接），
> 改为 **UnifiedRadixCache 子类直接驱动 connector**——与旧 LMCRadixCache 集成同构
> （阻塞语义），钩子点手握真实 tokens，无需 patch SGLang 任何组件。

| # | 文件（本仓） | 内容 | 状态 |
|---|---|---|---|
| 1 | `integration/sglang/unified_linker.py` | `LMCacheMPDriver`（阻塞式 submit/poll/complete 桥接，全失败开放）+ `LMCacheAscendUnifiedCache(UnifiedRadixCache)`（match_prefix/init_load_back/cache_finished_req/cache_unfinished_req/reset 五钩子 + convert 类切换） | ✅ 642 行 |
| 2 | `integration/sglang/unified_diversion.py` | `RadixCache.__init_subclass__` 钩子（链式保留原语义）+ dispatcher `__init__` + 池族探测（hybrid/MLA/DSV4 → unified，标准 → 旧路径），重导入全部延迟到运行时 | ✅ 172 行 |
| 3 | `__init__.py` 激活段 | sglang 分支改为双补丁循环守卫（compat + diversion，一个失败不拖累另一个） | ✅ |
| 4 | 部署/验证 | 见下方 4.3.3 | ⬜ 待容器执行 |

### 4.3.3 容器部署与 Qwen3.8 首亮步骤

```bash
# 0) 同步代码（git push 后容器 pull，或 docker cp 三个文件：
#    lmcache_ascend/__init__.py
#    lmcache_ascend/integration/sglang/unified_linker.py
#    lmcache_ascend/integration/sglang/unified_diversion.py）

# 1) 清字节码
cd /workspace/src/LMCache-Ascend
find lmcache_ascend -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

# 2) lmcache server 保持运行（chunk 256 不变；hybid 建议 separate-object-groups 可试加）

# 3) 启动 Qwen3.8（与最初基线相同的参数 + lmcache 两参数）
export PYTHONHASHSEED=0
python -m sglang.launch_server \
  --model-path /data/models/Qwen \
  --served-model-name Qwen3.8-27B \
  --host 0.0.0.0 --port 8998 \
  --tp-size 8 --dtype bfloat16 \
  --context-length 8192 --chunked-prefill-size 4096 \
  --mem-fraction-static 0.85 --trust-remote-code \
  --mamba-ssm-dtype bfloat16 \
  --enable-lmcache \
  --lmcache-config-file /workspace/scripts/lmcache_mp.yaml \
  2>&1 | tee /workspace/results/sglang_qwen38_unified.log

# 4) 验证日志（三行关键证据）
grep -E "Installed LMCRadixCache unified-path diversion|unified path engaged|unified driver ready" \
  /workspace/results/sglang_qwen38_unified.log
# 预期：
#   Installed LMCRadixCache unified-path diversion hook   ← 钩子装上
#   LMCRadixCache.__init__ wrapped with unified-path dispatcher ← dispatcher 包装
#   LMCache-Ascend unified path engaged (components=['FULL','MAMBA'], mamba=True) ← 分流成功
#   LMCache-Ascend unified driver ready (components=..., chunk=256) ← 连接器就绪

# 5) 通过后跑 qwen25 指南 7.2 的长前缀三连（Stored/Retrieved 闭环）
#    + 多轮对话语义检查（hybrid state 正确性的验收门槛）
```

**已知 v1 限制（诚实声明）**：① store 用请求完成时的活跃 mamba 状态槽（与 SGLang
自身树插入语义一致，但生成输出较长时状态边界有偏差，多轮验收会暴露）；② SWA/C128
组件未接线（需要它们的模型暂退化为 FULL-only）；③ cache_salt 仅支持全局空盐；
④ MLA/DSV4 分流到 unified 路径但未经 NPU 实测（阶段 2/3 验证）。

## 5. 侦察命令清单（第三轮，R12~R17）

```bash
# ── R12【头号问题】注册表机制：谁在 import 时注册 + 有无 entry_points 自动发现 ──
grep -rn "def register_radix_cache_backend\|def get_radix_cache_factory\|def registered_radix_cache_backends" \
  /sgl-workspace/sglang/python/sglang/srt/ --include="*.py"
grep -rn "entry_points\|importlib.metadata" /sgl-workspace/sglang/python/sglang/srt/mem_cache/ | head -10

# ── R13 linker 接口定义（本仓要实现的窄接口）──
grep -n "def init_cache_linker" -A 25 /sgl-workspace/sglang/python/sglang/srt/mem_cache/unified_radix_cache.py

# ── R14 参照实现：MooncakeDirectLinker（接口怎么满足的）──
wc -l /sgl-workspace/sglang/python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_direct_linker.py
sed -n '1,90p' /sgl-workspace/sglang/python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_direct_linker.py

# ── R15 cache 在哪些时点调 linker（决定异步 API 映射）──
grep -n "self.linker\.\|self.cache_linker\." /sgl-workspace/sglang/python/sglang/srt/mem_cache/unified_radix_cache.py | head -25

# ── R16 组件组数据结构（factory 要喂给 connector 的东西）──
sed -n '1,80p' /workspace/src/LMCache/lmcache/integration/sglang/lmcache_mp_metadata.py

# ── R17 import 链时机：radix backend 分支下 scheduler 进程谁 import lmcache ──
grep -rn "import lmcache\|from lmcache" /sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py \
  /sgl-workspace/sglang/python/sglang/srt/mem_cache/kv_cache_builder.py 2>/dev/null | head -10
```

| 命令 | 回答什么 | 影响哪个决策 |
|---|---|---|
| R12 | 注册表填充机制（手动 import / entry_points） | 注册代码放哪、用户要不要多带 flag |
| R13/R14 | linker 接口方法面 + 参照实现 | 阶段 1 编码蓝本 |
| R15 | cache↔linker 调用时点 | 异步 API（submit/poll）怎么映射 |
| R16 | `SGLangKVComponentGroup` 字段 | factory 组装参数 |
| R17 | 鸡生蛋问题（lmcache 从哪 import） | 激活命令最终形态 |

---

## 6. 风险与依赖（诚实声明）

| 风险 | 影响 | 缓解 |
|---|---|---|
| ~~unified 路径在 SGLang dev476 尚无接线~~ **已证实**（R10 零构造点） | 好于最坏预期：radix backend 注册表（R8）提供官方插件点，阶段 1 = factory + LMCacheDirectLinker（估 300~500 行，无需改 SGLang） | R12~R17 侦察后编码 |
| 堆叠张量布尔歧义在 unified adapter 复现（R7 发现） | `_resolve_pool_tensors` 对 NPU MHA 池会抛 RuntimeError | 扩展现有 mp_connector_compat：包装 adapter 的 `_resolve_pool_tensors`，复用 `_split_stacked_pool` |
| linker 接口与 UnifiedLMCacheMPConnector 异步 API 的映射复杂度 | LMCacheDirectLinker 可能比预估厚 | R13~R15 拿全接口后评估；必要时部分阻塞化 |
| CSA/HCA 压缩状态无先例 | DSV4.1 可能需要扩展 LMCache wire 协议 | 阶段 3 单独评估；先让 MLA plane 部分跑通 |
| GLM-5.3-Flash 320B / DSV4.1 552B 单机放不下 | 无法在当前 8 卡环境端到端验证 | 找同架构小模型（如 Qwen3.8-Flash-Next）做代理验证；或量化版 |
| dsv4 支持锁在 dsv4_support_045 旧分支 | DSV4 的 NPU kernel 可能要向 main 移植 | 阶段 3 做分支 diff 评估 |
| hybrid GDN/KDA state 的 KV 复用正确性 | 回灌错误 → 多轮对话输出错乱 | 验收必须含"多轮对话语义一致"检查（参考 qwen25 指南 7.3） |

## 7. 与既有文档的关系

| 文档 | 分工 |
|---|---|
| 本文 | 三模型适配的**战略蓝图**（调研 + 路线图 + 侦察清单） |
| [progress_and_next_action.md](progress_and_next_action.md) | 总进度与决策树（hybrid 专项行将随阶段 1 推进更新） |
| [qwen25_7b_first_light.md](qwen25_7b_first_light.md) | 已完成的标准模型首亮（旧路径，本文的对照基线） |
| [call_flow_before_after_adaptation.md](call_flow_before_after_adaptation.md) | 旧路径完整调用链（理解"为什么旧路径走不通"用） |
| `docs/recommended_deployment_guide/` | vLLM in-process 路径的三模型部署指南（NPU 数据面参照） |
