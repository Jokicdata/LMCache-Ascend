# 堆叠 KV 兼容补丁详解（mp_connector_compat.py + \_\_init\_\_.py 修改）

> 本文档详解 2026-11-12 落地的两处代码改动——它们解决了 SGLang + LMCache MP 启动时的
> `RuntimeError: Boolean value of Tensor with more than one value is ambiguous` 崩溃，
> 是"SGLang + LMCache-Ascend MP mode + 标准模型"首亮成功的关键补丁。
>
> 阅读前建议先看 [qwen25_7b_first_light.md](qwen25_7b_first_light.md) 的 1.1 节背景。

---

## 1. 问题是什么：一个"柜子形状"的错配

### 1.1 两边的期望

```text
新版 SGLang（0.5.2x dev，mem_cache 重构后）
  MHATokenToKVPool.k_buffer / v_buffer
  = 单张堆叠张量 [num_layers, tokens, kv_heads, head_dim]
  （Qwen2.5-7B 实测: (28, 12203, 128, 1, 128) —— 一整块 10.43 GB 的板子）

LMCache 旧连接器（multi_process_adapter.py）
  LMCacheMPConnector.__init__(k_pool, v_pool)
  = 期望逐层 list[torch.Tensor]（28 张小卡片，每层一张）
```

### 1.2 崩溃点

旧连接器构造函数的第一行校验（`multi_process_adapter.py:73`）：

```python
if not k_pool or not v_pool:        # ← 对"列表"问真假没问题
    raise ValueError(...)            # ← 但对 12203×128×1×128 个值的大张量问，
                                     #   PyTorch 无法回答 → RuntimeError
```

实测报错：

```text
RuntimeError: Boolean value of Tensor with more than one value is ambiguous
```

### 1.3 问题归属（重要）

| 项 | 结论 |
|---|---|
| 谁的 bug | LMCache 旧集成没跟上 SGLang 的内存池重构（**与昇腾、与 NPU 无关，CUDA 上同样会挂**） |
| 为什么社区没修 | SGLang 0.5.2x dev 与 LMCache dev 都处于快速迭代期，此组合恰在两者验证矩阵之外 |
| 为什么由本仓修 | 你的约束是"尽量不改上游"。本仓的插件机制（运行时 patch）天然适合承担兼容成本 |

### 1.4 修复思路（一句话）

**在"连接器构造"和"数据"之间插一层翻译：把堆叠大张量按层拆成 28 个零拷贝视图，
形状立刻变成旧连接器认识的样子，其余逻辑一行不动。**

---

## 2. 全景图：补丁在整条链路中的位置

```text
SGLang scheduler 进程（TP0~3 各一份）
═══════════════════════════════════════════════════════════════════
 import lmcache
   └─ LMCache 平台初始化: NpuDeviceOps.ensure_native
        └─ import lmcache_ascend          ← 插件被自动加载（已验证 auto-imported: True）
             └─ lmcache_ascend/__init__.py 激活段            ★改动二
                  ├─ _seed_partial_lmcache_device_ops()        （原有）
                  ├─ if _is_vllm_runtime(): ...                （原有）
                  ├─ if _is_sglang_runtime():                  ★新增分支
                  │     └─ _patch_sglang_mp_connector_compat()
                  │          └─ install_mp_connector_compat()  ★改动一的主体
                  │               └─ 包装 LMCacheMPConnector.__init__
                  └─ _patch_make_page_buffer_shape_desc()      （原有）
═══════════════════════════════════════════════════════════════════
 Scheduler 构造 → kv_cache_builder → registry(enable_lmcache 分支)
   └─ LMCRadixCache.__init__
        └─ LMCacheMPConnector(k_pool=堆叠张量, v_pool=堆叠张量, ...)
             │
             ▼ 经过包装后的 __init__（★拦截点）
        ┌─────────────────────────────────────────────┐
        │  k_pool: (28, 12203, 128, 1, 128) 大张量     │
        │      ↓ _split_stacked_pool()                │
        │  k_pool: [t0, t1, ..., t27]  28 个视图       │
        │  v_pool: 同上                                │
        └─────────────────────────────────────────────┘
             │
             ▼ 原构造函数（上游代码，零改动）
        _validate_sglang_kv_pools(list, list)     ← 不再歧义，通过
        get_event_ipc_backend(npu)                ← 平台抽象，NPU 分支
        _wrap_sglang_kv_caches(...)
          └─ wrap_one_kv_cache(tensor) × 56       ← 逐张量走平台注册表
               └─ NpuIPCWrapper（本仓 c_ops 生效链）
        register_kv_cache(...) → ZMQ → lmcache server
             │
             ▼ server 侧
        导入 IPC 句柄 + 建每卡 cache context（npu-smi 中 lmcache 进程
        在 NPU 0~3 各持 ~148MB 即此证据）
```

**补丁的本质：只在"数据进连接器"的门上做了一次形状翻译，上下游全部原样。**

---

## 3. 改动一：mp_connector_compat.py 逐段精读

文件路径：`lmcache_ascend/integration/sglang/mp_connector_compat.py`（新增，114 行）

### 3.1 模块级：logger 与幂等标记

```python
logger = init_logger(__name__)

_INIT_PATCHED_FLAG = "_lmcache_ascend_mp_compat"
```

- `init_logger(__name__)` → 日志会以 `lmcache_ascend.integration.sglang.mp_connector_compat`
  的名字出现在 LMCache 统一日志流里（这就是启动日志里那几行 `mp_connector_compat.py:68`
  的来源——logger 名本身就是"本仓代码在跑"的字证）。
- `_INIT_PATCHED_FLAG` → 打在包装函数上的标记属性，防止重复安装（见 3.3）。

### 3.2 核心函数 `_split_stacked_pool`（形状翻译器）

```python
def _split_stacked_pool(pool: object, name: str) -> object:
    if not isinstance(pool, torch.Tensor) or pool.dim() < 2:
        return pool                       # ① 不是堆叠张量 → 原样放行
    layers = [pool[i] for i in range(pool.shape[0])]   # ② 沿第 0 维拆
    logger.info("SGLang stacked %s buffer shape=%s dtype=%s "
                "split into %d per-layer views", name, tuple(pool.shape),
                pool.dtype, len(layers))
    return layers
```

三个设计点：

| # | 代码 | 为什么这么写 |
|---|---|---|
| ① | isinstance + dim < 2 守卫 | **向后兼容**：如果将来 SGLang/LMCache 任一方修好此问题、或换回逐层列表，函数原样放行，补丁自动变为无害空操作。真正的非法输入仍由上游校验器报出干净的错误——补丁不抢上游的"报错职责" |
| ② | `[pool[i] for i in range(shape[0])]` | **零拷贝**：`pool[i]` 返回的是张量**视图**（view），不是副本。28 个视图共享同一块 10.43GB 显存，没有任何数据搬运 |
| ③ | info 级日志 | 形状/dtype/层数全部留痕——首跑即验证了"层在前"假设（28 = Qwen2.5-7B 层数），以后换模型也能从日志直接读出堆叠顺序 |

**零拷贝示意（为什么拆视图是安全的）：**

```text
堆叠张量（一块连续显存）:
┌──────────────────────────────────────────────────────┐
│  第0层 10.43GB/28  │  第1层  │ ... │  第27层          │
└──────────────────────────────────────────────────────┘
        ↑视图0              ↑视图1          ↑视图27
k_pool = [view_0, view_1, ..., view_27]

- 视图 = 同一块存储 + 各自的形状/偏移描述（仅几十字节元数据）
- 写 view_3 的任何元素 = 写原张量对应元素（SGLang 运行时照常更新 KV，
  IPC 共享的正是这同一块存储 → server 侧永远看到最新值）
- NPU IPC 包装的是"存储句柄"，视图与原张量句柄同源 → 注册语义不变
```

### 3.3 安装函数 `install_mp_connector_compat`（包装器）

```python
def install_mp_connector_compat() -> None:
    import lmcache.integration.sglang.multi_process_adapter as mpa

    connector_cls = getattr(mpa, "LMCacheMPConnector", None)
    if connector_cls is None:                                    # ① 类不存在 → 静默退出
        return
    if getattr(connector_cls.__init__, _INIT_PATCHED_FLAG, False):  # ② 已装过 → 幂等退出
        return

    original_init = connector_cls.__init__                       # ③ 保住原函数

    def _compat_init(self, *args, **kwargs):
        if "k_pool" in kwargs:
            kwargs["k_pool"] = _split_stacked_pool(kwargs["k_pool"], "k_pool")
        if "v_pool" in kwargs:
            kwargs["v_pool"] = _split_stacked_pool(kwargs["v_pool"], "v_pool")
        return original_init(self, *args, **kwargs)              # ④ 原构造照常执行

    _compat_init.__name__ = "LMCacheMPConnector.__init__"        # ⑤ traceback 更友好
    setattr(_compat_init, _INIT_PATCHED_FLAG, True)
    connector_cls.__init__ = _compat_init                        # ⑥ 类上替换
    logger.info("Installed SGLang MP connector stacked-KV compat patch")
```

六个步骤的设计意图：

| # | 意图 |
|---|---|
| ① | LMCache 未来若删除旧类，补丁静默退化为 no-op，绝不让插件加载失败 |
| ② | 幂等：多个进程/重复 import 场景下不会套娃包装（套两层虽也能跑，但日志会翻倍且浪费） |
| ③④ | **"翻译完就放行"**：包装器只动 `k_pool`/`v_pool` 两个参数，其余原样转交。原构造里的全部逻辑（校验、事件后端、ZMQ、注册、心跳）一行不改 |
| ⑤ | 若未来又出新错，traceback 显示 `LMCacheMPConnector.__init__` 而不是 `_compat_init`，排查时不迷惑 |
| ⑥ | 替换发生在**类**上——之后无论谁、在哪构造实例，都会经过包装器 |

**为什么拦 kwargs 而不是位置参数？** SGLang 的 `lmc_radix_cache.py` 是唯一构造点，
且 `k_pool`/`v_pool` 通过 `**connector_kwargs` 展开——必然是关键字参数到达，
拦 kwargs 最稳（`*args` 原样透传，兼容任何其它调用方式）。

**调用时序图（为什么补丁必然先于构造生效）：**

```text
T0  Scheduler 进程启动: import sglang        （sglang 进 sys.modules）
T1  Scheduler.__init__ → kv_cache_builder    （KV 池已建好，k_buffer 是堆叠张量）
T2  registry: from ...lmc_radix_cache import LMCRadixCache
     └─ 该文件顶部: import lmcache ...        ← 首次触发 import lmcache
T3      └─ LMCache 平台初始化 → import lmcache_ascend（插件自动加载）
T4          └─ 激活段: _is_sglang_runtime() == True  （T0 已把 sglang 装进 sys.modules）
             └─ install_mp_connector_compat()  ← ★补丁在这时装上
T5  LMCRadixCache.__init__ → LMCacheMPConnector(...)  ← ★构造在补丁之后，必然被拦到
```

---

## 4. 改动二：\_\_init\_\_.py 激活段逐段精读

文件路径：`lmcache_ascend/__init__.py`（修改，新增约 40 行）

### 4.1 新增运行时检测函数

```python
def _is_sglang_runtime():
    """..."""
    return "sglang" in sys.modules or any("sglang" in arg for arg in sys.argv)
```

与紧邻的 `_is_vllm_runtime()` 完全同构（照抄现有模式）：

| 判据 | 覆盖场景 |
|---|---|
| `"sglang" in sys.modules` | 主场景：SGLang scheduler 进程里 `import lmcache` 时，sglang 早已加载（见 3.3 时序图 T0→T4） |
| `sys.argv` 含 "sglang" | 兜底：进程以 `python -m sglang.launch_server` 启动但检测发生在 sglang 完整加载前 |

### 4.2 新增 patch 转发函数（与 vllm 的同款空壳模式）

```python
def _patch_sglang_mp_connector_compat():
    from lmcache_ascend.integration.sglang.mp_connector_compat import (
        install_mp_connector_compat,
    )
    install_mp_connector_compat()
```

为什么是"空壳转发"而不是直接写逻辑？——照抄本仓现有两处先例
（`_patch_logical_block_size` / `_patch_make_page_buffer_shape_desc`）的组织约定：

```text
__init__.py            只做"何时装"（运行时检测 + 激活编排）
integration/xxx/yyy.py 做"装什么"（具体包装逻辑）
```

好处：`__init__.py` 保持薄编排层；具体补丁可独立测试、独立演进。

### 4.3 激活段新增分支（含异常守卫）

```python
if not LMCACHE_ASCEND_PATCHED:
    _seed_partial_lmcache_device_ops()          # 原有：必须最先

    if _is_vllm_runtime():                       # 原有
        _patch_logical_block_size()

    if _is_sglang_runtime():                     # ★新增分支
        try:
            _patch_sglang_mp_connector_compat()
        except Exception as exc:                 # ★异常守卫
            from lmcache.logging import init_logger
            init_logger(__name__).warning(
                "SGLang MP connector compat patch not installed: %s", exc)

    _patch_make_page_buffer_shape_desc()          # 原有：无条件
    LMCACHE_ASCEND_PATCHED = True
```

**异常守卫是本改动最重要的安全设计**，原因写在 `_seed_partial_lmcache_device_ops`
的 docstring 里（本仓用血泪换来的教训）：

> 激活发生在 `import lmcache` **中途**（平台初始化回调我们）。此时任何补丁激活
> 代码抛出的 ImportError/任何异常，都会**中止整个 `import lmcache`** →
> LMCache 把插件软降级为"找不到" → **所有 MP 传输退回 torch 基线（约慢 8 倍）**，
> 且只留下一行不起眼的日志。

所以 sglang 分支必须 try/except 包死：补丁装不上（比如 lmcache 的 sglang 集成模块
因版本变化 import 失败）只损失"这一层兼容"，绝不能连带杀死 NPU 数据面激活。
失败可见、失败不致命。

### 4.4 改动后的激活段全景

```text
lmcache_ascend/__init__.py 底部
│
├─ _seed_partial_lmcache_device_ops()        桥接 device_ops（vLLM/SGLang 通用）
│
├─ if vLLM 运行时:   logical_block_size 补丁   （原有，vLLM 专用）
│
├─ if SGLang 运行时: mp_connector_compat 补丁   ★新增，SGLang 专用
│   └─ try/except 守卫（失败=warning，不中断）
│
├─ shape_desc 补丁                            （原有，无条件——MLA/DSA plane 几何）
│
└─ LMCACHE_ASCEND_PATCHED = True
```

三个补丁互不干扰、各管一段：device_ops 桥接管"数据面入口"，logical_block_size
管 vLLM 的块语义，mp_connector_compat 管 SGLang 的 KV 形状，shape_desc 管格式描述。

---

## 5. 两个文件改动量汇总

| 文件 | 类型 | 行数 | 内容 |
|---|---|---|---|
| `lmcache_ascend/integration/sglang/__init__.py` | 新增 | 1 | 包初始化（照 vllm 目录模式） |
| `lmcache_ascend/integration/sglang/mp_connector_compat.py` | 新增 | 114 | `_split_stacked_pool`（形状翻译）+ `install_mp_connector_compat`（构造包装），双语文档字符串 |
| `lmcache_ascend/__init__.py` | 修改 | +40 | `_is_sglang_runtime` + `_patch_sglang_mp_connector_compat` + 激活段带守卫分支 |

纯 Python，**无需重编译 c_ops**；不修改 SGLang / LMCache 上游任何文件。

---

## 6. 验证证据（补丁已实测生效）

| 证据 | 日志/现象 |
|---|---|
| 补丁安装 | `Installed SGLang MP connector stacked-KV compat patch`（mp_connector_compat.py:114） |
| 形状翻译发生 + 层在前假设成立 | `SGLang stacked k_pool buffer shape=(28, 12203, 128, 1, 128) dtype=torch.bfloat16 split into 28 per-layer views` |
| 上游校验通过 | 无 Boolean ambiguous 报错，构造走到心跳 |
| 连接器注册成功 | `lmcache-heartbeat` 线程 × 4 rank 启动 |
| NPU IPC 链走通 | npu-smi: lmcache 进程在 NPU 0~3 各持 ~148MB |
| 服务就绪 | `Tree cache initialized: impl=LMCRadixCache` × 4，`Application startup complete` |

## 7. 维护注意事项

| 场景 | 影响 | 动作 |
|---|---|---|
| SGLang 修复此问题（改回逐层列表） | `_split_stacked_pool` 守卫放行，补丁自动失效变 no-op | 无需动作 |
| LMCache 删除旧类 `LMCacheMPConnector` | `connector_cls is None` → 静默退出 | 无需动作 |
| 换模型（层数不同） | 日志 shape[0] 直接显示新层数 | 无需动作（拆的是运行时实际形状） |
| LMCache/SGLang 大版本升级 | 激活段 try/except 保证最坏只少一层兼容 | 留意 warning 日志 |
| 上游正式修复后 | 本补丁可整体删除（3 个文件互相独立，无其它依赖） | 删 `mp_connector_compat.py` + `__init__.py` 两个 sglang 分支块 |

## 8. 与既有文档的关系

| 文档 | 关系 |
|---|---|
| [qwen25_7b_first_light.md](qwen25_7b_first_light.md) | 8.1 节是本补丁的部署手册，第 11 节是补丁生效后的问答/benchmark 教程 |
| [progress_and_next_action.md](progress_and_next_action.md) | 第 4 步决策表——本补丁即"改本仓：connector 构造包装"那一格的落地 |
| [control_plane_vs_data_plane.md](control_plane_vs_data_plane.md) | 本补丁位于控制面连接器构造入口，不动数据面任何一行 |
| [lmcache_ascend_vs_upstream.md](lmcache_ascend_vs_upstream.md) | "问题制造者（SGLang/LMCache 交接缝隙）≠ 问题消化者（本仓）"的典型案例 |
