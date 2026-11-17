# 从零开始读懂推理服务——SGLang 加 LMCache-Ascend 启动与一次推理请求的逐函数全程图解

> 这份文档只有一个目的：你敲下命令后，计算机里**哪个文件先被打开、哪个函数先被调用、
> 它返回了什么给谁、下一个是谁**，一环扣一环，直到一个字被吐回你的终端。
> 每一个调用环都标着真实文件路径和行号（全部核对过源码），多进程的地方用【同时】标记。
>
> 环境事实：910B4 × TP8 × Qwen3.8-27B，容器里 SGLang 源码在
> `/sgl-workspace/sglang/python/`，LMCache 在 `/workspace/src/LMCache/`，
> 我们的仓库 LMCache-Ascend 在 `/workspace/src/LMCache-Ascend/`。
> 为了行文简洁，下文一律用相对路径。

---

## 第一部分 启动篇：从敲下命令到服务就绪

### 第 1 环 命令行本身：`python -m sglang.launch_server`

你在终端敲的字符串先经过两件事：

1. `python -m`：Python 解释器启动后，去已安装的包里找 `sglang` 包里的 `launch_server.py` 文件，把它当成主脚本开始执行。
2. 文件自身的执行：`launch_server.py` **从头到尾逐行执行**。关键的行有：

```python
# sglang/launch_server.py
14: suppress_noisy_warnings()          # 模块级调用：压制一堆第三方库的无用警告
...
70: if __name__ == "__main__":         # 判断"我是被直接运行的"（而不是被 import 的）
79:     load_plugins()                 # 加载 SGLang 插件系统（注册第三方钩子）
81:     server_args = prepare_server_args(sys.argv[1:])   # ★把所有 --参数 解析进一个大配置对象
83:     try:
84:         run_server(server_args)     # ★带着配置对象进入真正的启动函数
85:     finally:
86:         kill_process_tree(...)      # 无论成败，进程要退出时清理整棵进程树
```

**`sys.argv[1:]`** 就是命令行里 `python -m sglang.launch_server` 之后的所有参数
（`--model-path`、`--tp-size 8`、`--enable-lmcache`……）。

**返回值**：`prepare_server_args` 返回一个 `ServerArgs` 对象——它是一张"总配置表"，
`--model-path` 存成它的 `model_path` 字段，`--tp-size 8` 存成 `tp_size` 字段。
**之后全世界的组件都从这张表上读配置**，这是理解后面一切的关键。

### 第 2 环 `run_server()` 选路

```python
# sglang/launch_server.py:17-67
def run_server(server_args):
    server_args.resolve_once()        # 对参数做第二轮推导/校验（TP 整除等错误就在这一类环节报）
    cfg = resolving_view(server_args)
    if cfg.encoder_only: ...          # 编码器模式（我们不是）
    elif cfg.smg_grpc_mode: ...       # gRPC 模式（我们不是）
    elif cfg.use_ray: ...             # Ray 模式（我们不是）
    else:
        # 默认 HTTP 模式 —— 我们走这里
        from sglang.srt.entrypoints.http_server import launch_server
        launch_server(server_args)
```

判定全为否后，执行 `import` 拉进 `http_server.py` 里的 `launch_server` 函数并调用。
**返回**：它不返回（会阻塞到服务退出）。

### 第 3 环 主进程分身：`http_server.launch_server` 与 `Engine._launch_subprocesses`

```python
# sglang/srt/entrypoints/http_server.py:2795-2856
def launch_server(server_args, ...):
    (tokenizer_manager, template_manager, port_args,
     scheduler_init_result, subprocess_watchdog, ...) = Engine._launch_subprocesses(
         server_args=server_args,
         init_tokenizer_manager_func=init_tokenizer_manager,
         run_scheduler_process_func=run_scheduler_process,     # ★记住这两个函数
         run_detokenizer_process_func=run_detokenizer_process,
     )
    ...
    _setup_and_run_http_server(server_args, tokenizer_manager, ...)   # FastAPI 网页服务在此装配
```

`Engine._launch_subprocesses` 内部干的事（这是"1 变 10"的瞬间）：

```text
【同时启动三类角色】
  角色一：TokenizerManager —— 留在主进程里的对象（文档注释原话：HTTP 服务、
          引擎、分词器都在主进程里）
  角色二：8 个 scheduler 子进程 —— 用 Python 的 multiprocessing 按
          --tp-size 8 spawn 出来，每个子进程的"入口函数"就是上面传下来的
          run_scheduler_process，并且各绑一张卡：gpu_id 依次 0..7
  角色三：detokenizer 子进程 —— 把生成的数字翻回文字的进程

  角色之间有分工：主进程（tokenizer）= 门房, 8 个 scheduler = 计算车间,
  detokenizer = 翻译, 互相用 ZMQ（跨进程信箱）通信。角色二和三角色三
  是独立进程, 角色一的 tokenization 在独立线程。
```

**返回值**：`_launch_subprocesses` 返回一堆"进程们的手柄和信息"（tokenizer_manager
对象、port_args 端口表、scheduler_init_result……），供主进程用来做最后装配和
等待。

之后主进程进入 `_setup_and_run_http_server`：把 FastAPI（HTTP 服务框架）搭起来，
注册 `/v1/chat/completions` 这类路由，起 uvicorn 监听 8998 端口——**但它故意
等到 8 个 scheduler 都初始化完才对外宣布就绪**（这就是为什么你会看到
`Waiting for application startup` 卡一阵再出现 `startup complete`）。

### 第 4 环 每个 scheduler 子进程的入口：`run_scheduler_process`

8 个子进程【同时】执行同一个函数 `run_scheduler_process`（scheduler.py:6016），
形参里有 `gpu_id`（0..7）和 `tp_rank`（0..7），所以每个进程知道自己是谁家的老几。

```python
# sglang/srt/managers/scheduler.py:6016-6106
def run_scheduler_process(server_args, port_args, gpu_id, tp_rank, ...):
    load_plugins()                                  # 6032
    publish(server_args, role="scheduler", ...)     # 6034 把"我是 scheduler、世界排名第几"登记进全局状态
    configure_scheduler_process(server_args, gpu_id, tp_rank, ...)  # 6043 绑卡、设环境变量、初始化 torch 分布式通信
    ...
    scheduler = Scheduler(server_args, port_args, tp_rank, pp_rank, dp_rank)  # 6075 ★造引擎（见第 5 环）
    pipe_writer.send(scheduler.get_init_info())     # 6084 把自己初始化好的信息通过管道发给父进程
    scheduler.run_event_loop()                      # 6087 ★进入事件循环，从此永远在处理请求
```

两个细节：

- **第 6075 行 `Scheduler(...)` 如果抛任何异常**，走 6089 起的 except：把完整
  traceback 打出来（你看到的 `Scheduler hit an exception: Traceback...` 就是这里），
  然后向父进程发 SIGQUIT 信号——父进程收到后把整棵进程树杀掉（你看到的
  `Received sigquit from a child process` + `kill_process_tree called`）。**这就是
  为什么任何一次构造失败，8 个卡会全部陪葬退出的机制原因。**
- **第 6087 行 `run_event_loop()` 永不返回**（除非有人发停机请求）。它是第三部分
  请求篇的主角。

### 第 5 环 造引擎：`Scheduler.__init__` 的十几步固定流程

`Scheduler(...)` 会触发这个类的构造函数 `__init__`（scheduler.py，从 500 行
左右开始）。**它是整个启动最难啃也最值得啃的一段**。按执行顺序（行号即顺序）：

| 步 | 代码（scheduler.py 行号） | 干什么 | 你的日志/旁证 |
|---|---|---|---|
| 1 | `bootstrap.init_layer_runtime(...)` (526) | 预热运行时 | — |
| 2 | `init_metrics_collector(...)` (529) | 建性能指标收集器 | — |
| 3 | `init_ipc_channels(port_args)` (532) | 打开与门房、翻译家的 ZMQ 信箱 | — |
| 4 | `init_tokenizer()` (540) | 备好分词器 | — |
| 5 | `init_moe_gemm_config()` (543) | 算子库配置 | — |
| 6 | `init_mamba_backend()` (546) | **混合模型的 GDN 层后端起在这定**（你的 `--mamba-ssm-dtype bfloat16` 在此生效） | — |
| 7 | `init_model_worker()` (553) | **加载模型权重**（顺带分配完 KV 显存池） | `Load weight begin/end`、`KV Cache is allocated` |
| 8 | `kv_cache_builder.build_kv_cache(...)` (559-583) | **构建缓存树** ← 第四部分主角 | `Tree cache initialized` |
| 9 | `self.tree_cache = result.tree_cache` (592) | 把造好的树挂到调度器自己身上 | — |

第 7 步内部再展开一层（这就是日志里 `Load weight` 的出处）：

```text
Scheduler.init_model_worker() (scheduler.py:1089)
  → init_tp_model_worker() (scheduler.py:1001)
    → TpModelWorker(...) (tp_worker.py:354)
      → _init_model_runner() (tp_worker.py:518)
        → ModelRunner(...) (model_runner.py:463)
          → self.initialize() (model_runner.py:655)
            → self.load_model() (model_runner.py:1243)
              → 读 config.json → 定模型类 Qwen3_5ForConditionalGeneration
              → 【同时】多线程读 .safetensors 分片，把数字填进 60 多层骨架
              → 返回（权重已在上）
            → 按 --mem-fraction-static 算可用显存 → 造 KV 池（见第 6 环）
```

第 8 步 `build_kv_cache` 的**返回值**是一个 `BuildResult` 结构，里面夹着
`tree_cache`（缓存树）、`req_to_token_pool`（请求↔显存槽映射表）等十来个东西；
第 9 行把其中最重要的 `tree_cache` 当场取出来挂到 `self.tree_cache`。

### 第 6 环 KV 显存池是怎么造出来的（第 5 环里第 7 步的尾巴）

在 `load_model` 之后、`build_kv_cache` 里：

1. 调度器算出"还能拿出多少显存放缓存"；
2. **看模型型别自动选池**（没有开关、用户不可选）：
   - 普通模型 → `MHATokenToKVPool`：每层两块大板 `k_buffer`/`v_buffer`（"卡片柜"）；
   - Qwen3.8 混合模型 → `HybridLinearKVPool`（memory_pool.py:3860）：内部再包一个
     只管全注意力层的子池（NPU 上是 `NPUMHATokenToKVPool`，memory_pool_npu.py:85）
     + 一块 mamba 白板区。
3. 在卡上划连续显存。实测 Qwen3.8：把 16 个全注意力层叠成 `(16, 4757, 128, 1, 256)`
   ——16 层 × 4757 页 × 每页 128 槽 × 1 个 KV 头 × 256 维（TF 里那些数字的来路）。

日志对应（每个 scheduler 各打一条）：

```text
[TP0] KV Cache is allocated. dtype: torch.bfloat16, #tokens: 1561984, K size: 10.43 GB, V size: 10.43 GB
```

（`1561984 = 12203 页 × 128 槽`——等下你会发现这个数和 connector 注册日志对得上。）

### 第 7 环 build_kv_cache 内部：怎么走到 LMCache 门口

`build_kv_cache`（kv_cache_builder.py:394 附近）的核心是这一句：

```python
tree_cache = create_tree_cache(tree_context)
```

`create_tree_cache`（registry.py:218 起）先看"有没有人指定了自定义 backend 名"
（`--radix-cache-backend`，你没有）→ 没有 → 调 `default_radix_cache_factory(ctx)`
（registry.py:79），它是**一棵 if-else 分支树**，按你的配置走其中一支：

```text
default_radix_cache_factory(ctx)
  ├─ 你带了 --enable-lmcache 吗？
  │     ├─ 没有 → 用 SGLang 自带 UnifiedRadixCache（裸模式，本节结束）
  │     └─ 有 → 执行下面两行（registry.py:111-120）：
  │           from sglang.srt.mem_cache.storage.lmcache.lmc_radix_cache import LMCRadixCache
  │           return LMCRadixCache(params, model_config, tp_size, rank, tp_group)
```

**注意**：`--enable-lmcache` 只是一个"选哪条分支"的开关。它不拉起 LMCache，
真正拉起 LMCache 的是下一环——`import lmc_radix_cache` 这句。

### 第 8 环 魔法入口：`import lmc_radix_cache` 到底发生了什么

Python 的 `import 某模块` = **把那个 .py 文件从头到尾执行一遍**。于是：

```text
执行 import lmc_radix_cache
  → 开始执行 storage/lmcache/lmc_radix_cache.py
    → 文件顶部第 23-27 行写着：
        from lmcache.integration.sglang.multi_process_adapter import LMCacheMPConnector
        from lmcache.integration.sglang.sglang_adapter import LMCacheLayerwiseConnector, ...
        from lmcache.integration.sglang.utils import lmcache_get_config
    → 执行第一行 import，就必须先执行 import lmcache（LMCache 整个包）
      → lmcache/__init__.py 从头执行
        → 它的平台初始化发现当前硬件是 NPU，去找昇腾插件：
          → import lmcache_ascend        ★★★ 我们的仓库在这一瞬间被自动拉起！
            → lmcache_ascend/__init__.py 从头执行
              → 激活段（文件底部）：依次装三件武器 + 两个 SGLang 专用补丁（见第 9 环）
              → 返回（我们的仓库执行完毕，回到 lmcache 继续）
        → lmcache 导入完成，返回
    → 继续执行 lmc_radix_cache.py 的剩余部分
      → 执行到 class LMCRadixCache(RadixCache): 这一行
        → ★★★ 我们第 9 环装的分诊台钩子在这一瞬间触发！（见第 10 环）
      → 模块导入完成
  → import 语句结束，LMCRadixCache 这个"类"可用
```

**记住这个"我们被夹在 import 中途插入"的时序**——整个适配方案的地基就是
"谁 import lmcache，谁就自动import我们；import 先于一切构造"。

### 第 9 环 三件武器 + 两个补丁：我们的激活段干了什么

`lmcache_ascend/__init__.py` 底部（154-187 行）在 SGLang 运行时依次执行：

```text
① _seed_partial_lmcache_device_ops()        ← 老成员：给 lmcache 打一个 device_ops 桥（防半初始化崩）
② 判断是 vllm 运行时？不是，跳过
③ 判断是 sglang 运行时？是（因为前面注册的全局状态里填了 role=scheduler）
   → 逐个安装（一个失败不影响另一个）：
     武器A _patch_sglang_mp_connector_compat()   ← 堆叠 KV 拆分 + 页折叠 + adapter 包装
     武器B _patch_sglang_unified_diversion()     ← 分诊台钩子（下面单独讲）
④ _patch_make_page_buffer_shape_desc()      ← 老成员：MLA plane 几何描述
⑤ 标 LMCACHE_ASCEND_PATCHED = True         ← 防止重复安装
```

**武器A（`integration/sglang/mp_connector_compat.py`）三连**：

```text
A1. 包装 LMCacheMPConnector.__init__：
    谁构造它，先拦下 k_pool/v_pool 两块大张量，按层目视拆散成列表（只是
    "视图"，同一块显存的另一种数法，零拷贝），再交给原构造。
    —— 打死布尔歧义第一案发现场。

A2. 包装 SGLangUnifiedKVAdapter._resolve_pool_tensors：
    同一个坑在 unified adapter 深处再犯一次，同样拦下拆散。
    —— 打死布尔歧义第二案发现场。

A3. _fold_page_native_pool：把"页在前"布局重数成"槽在前"布局
    （(4757,128,1,256) → (608896,1,256)），行数被 128 整除，
    connector 的几何校验才放行。
    —— 打死第三案发现场(整除校验)。
```

**武器B（`integration/sglang/unified_diversion.py`）**：往 SGLang 的 `RadixCache`
基类挂一个 `__init_subclass__` 钩子。这是 Python 的内建机制：**任何新类的定义
语句一旦继承 RadixCache，这个钩子立刻被自动调用**，Python 会把这个新类本身
当作参数传进来。

### 第 10 环 类定义瞬间：分诊台是怎样装到 LMCRadixCache 身上的

回到第 8 环的下半场。`lmc_radix_cache.py` 执行到：

```python
class LMCRadixCache(RadixCache):
    ...
```

这一行触发了武器B的钩子。钩子内部（unified_diversion.py:83-91）：

```text
钩子收到新类 LMCRadixCache
  → 确认名字对（就是它）
  → 保存它的原 __init__（备用）
  → 造一个"分诊台"函数 _dispatch_init
  → 把 LMCRadixCache.__init__ 替换成这个分诊台
```

从这一刻起，第 7 环的 `return LMCRadixCache(...)` 一旦执行，"构造缓存树"就会
先经过分诊台。

### 第 11 环 正式构造：分诊台怎么判、统一路径怎么跑

`LMCRadixCache(params, model_config, tp_size, rank, tp_group)` 执行 →
自动调用被换上的分诊台 `_dispatch_init`（unified_diversion.py:102-134）：

```text
分诊台：
  ① 摸一摸 params 里那个内存池（token_to_kv_pool_allocator.get_kvcache()）
  ② 型别判定（只看属性、类名，不 import 池类）：
     ├─ 有 full_kv_pool / mamba_pool 属性 → hybrid（Qwen3.8、GLM-5.3）★
     ├─ 有 kv_buffer 属性               → MLA（DeepSeek V2/V3 系）
     ├─ 类名前缀 DeepSeekV4              → DSV4
     └─ 都没有                          → 标准池（Qwen2.5-7B）
  ③ 分流：
     ├─ 标准池 → 交给保存的原 __init__（老路：武器A生效的路径，Qwen2.5 已验证）
     └─ 其他   → 执行 LMCacheAscendUnifiedCache.convert(self, params, ...)
                  （unified_linker.py:377）
```

`convert` 按顺序干四件事（**顺序是血泪教训**，第 3 件必须先于第 2 件）：

```text
① 拼组件清单：components = [FULL]；若 req_to_token_pool 有 mamba_pool →
   追加 MAMBA；写回 params.tree_components
② instance.__class__ = LMCacheAscendUnifiedCache     ← 先换类！（原因见④）
③ UnifiedRadixCache.__init__(instance, params)       ← 按统一树重新初始化
④ 造 LMCacheMPDriver（见第 12 环）
```

**为什么②必须在③前**：`UnifiedRadixCache.__init__` 初始化到一半会调用
`self.reset()` 清场。Python 的方法查找看实例"当下"的类——若还是旧类，
reset 会分派到旧版实现，摸一个从来没初始化过的属性，当场
`AttributeError: evictable_leaves`（这个坑第 4 次调试才逮住）。先换类，
reset 就正确落到统一树的实现上；我们的 reset 覆写又用
`getattr(self, "_lmc_driver", None)` 容忍"此时驱动还没造"的状态——双保险。

### 第 12 环 造驱动、造连接器、向 lmcache server 注册

第④步 `LMCacheMPDriver(cache, config_file=..., model_config=...,
tp_size, tp_rank, tp_group, ...)`（unified_linker.py:98）：

```text
① tp_group 解包：SGLang 传的是自家包装类 GroupCoordinator，连接器要 torch 的
   进程组。取它的 cpu_group（gloo，跑小整数同步最合适）。(第 5 次调试的坑)
② 调 UnifiedLMCacheMPConnector(config_file=..., page_size=cache.page_size,
    token_to_kv_pool_allocator=..., req_to_token_pool=...,
    tree_components=..., mamba_component=..., ...)    ← 上游 LMCache 的类
```

`UnifiedLMCacheMPConnector.__init__`（LMCache 仓库
`lmcache/integration/sglang/unified_lmcache_mp_connector.py:64` 起）内部按序：

```text
① 造 SGLangUnifiedKVAdapter（:96）
② kv_groups = adapter.resolve_registered_groups()    ← 武器A的 A2/A3 在此生效！
   （把 hybrid 池的全注意力层 + mamba 白板拆成"组件组"：这16层K+16层V归
    FULL 组，白板状态归 MAMBA 组，并折叠成可整除的槽行数）
③ 逐组几何校验（:127-170）：行数整除、页槽比、张量个数
   （"tensor rows 4757 are not divisible..."那个错就长在这里）
④ 计算并行几何、谁当 lookup 领头、谁当 KV 写手（:264-273）
⑤ 读你给的 --lmcache-config-file（YAML），取 mp_host/mp_port（:220）
⑥ register_kv_cache()（:528）：把 16K+16V+mamba 的显存句柄打包，
   经 ZMQ 发给 127.0.0.1:5556 的 lmcache server
⑦ 起心跳线程（每 10 秒一次"我还活着"）
```

**对账时刻**：第 6 环日志里 KV 池 156 万 token，到这里变成 16 层 × 每个
`(12203,128,1,128)`（12203×128=1561984）——数全对上了。

日志证据（每个 scheduler 各一组，共 8 组）：

```text
Init Unified Radix Cache. Components: (FULL, MAMBA). Tree Core: UnifiedTreeCore
LMCache-Ascend unified path engaged (components=['FULL', 'MAMBA'], mamba=True)
LMCache-Ascend unified driver ready (components=['FULL', 'MAMBA'], chunk=256)
Tree cache initialized: source=default impl=LMCacheAscendUnifiedCache hybrid_ssm=True
```

### 第 13 环 回程：scheduler 收尾、主进程宣布开业

统一树造好后，沿原路逐层**返回**：

```text
convert 返回（树已造好）
  → 分诊台 _dispatch_init 返回
    → registry 的 return LMCRadixCache(...) 完成 → create_tree_cache 返回树
      → build_kv_cache 返回结果包
        → Scheduler.__init__ 第 9 步把 tree_cache 挂到 self.tree_cache
          → Scheduler 构造完成（其余尾部初始化略）
            → run_scheduler_process: pipe_writer.send(get_init_info()) 向父进程报"我好了"
              → run_event_loop() 进入永动循环，等待请求
```

8 个子进程【同时】都完成后，主进程那头的 FastAPI 收齐 8 份"我好了"，暖机
（warmup，也就是起服务时你自己看到的那条 `Prefill batch, #new-token: 6` 小请求、
以及 122 token 那次的 ghost 来源），然后：

```text
INFO: Application startup complete.
INFO: Uvicorn running on http://0.0.0.0:8998
```

**启动篇到此结束。没适配的世界里，这个故事在第 8 环就断了**（连 k_buffer 都取不到
的崩溃）；适配之后它一路讲到了第 13 环。

---

## 第二部分 对照篇：没适配时，同一个故事死在哪

没装我们仓库（或装了我们没生效）时，第 11 环的分诊台不存在，于是
`LMCRadixCache.__init__` 直接跑**上游原版**（storage/lmcache/lmc_radix_cache.py:113-149）：

```text
① get_kvcache() 拿到池 → ② 想取 k_buffer/v_buffer 属性 → ③ 塞进 kwargs →
④ 构造 LMCacheMPConnector → ⑤ _validate_sglang_kv_pools 里第一句:
    if not k_pool: ...
```

| 模型 | 死法 | 死在哪个文件哪一行 |
|---|---|---|
| Qwen3.8（hybrid） | 连属性都没有：`'HybridLinearKVPool' object has no attribute 'k_buffer'` | lmc_radix_cache.py:125（② 步取属性） |
| 标准模型但走了 connector | `Boolean value of Tensor ... is ambiguous`（大张量被问真假） | multi_process_adapter.py:73（⑤ 步） |
| （若修了拆分还有下一个） | 4757 行不可被 128 整除 | unified connector 几何校验 |
| （若修了整除还有下一个） | `'GroupCoordinator' object has no attribute 'size'` | connector 的 dist.get_world_size |
| （若修了 group 还有下一个） | mamba 传输 122 token 不对齐检查点网格 128 | unified_kv_adapter.py:170 |

**一句话**：上游 LMCache 只管"缓存本事"，SGLang 只管"引擎本事"，两者之间
那条"交接带"在新设备、新模型上有一串缝——没适配时，任何一条缝都能在
启动的最后一步把你整死，且死于 8 卡陪葬（SIGQUIT 机制）。

---

## 第三部分 请求篇：一次问答请求的完整旅程

服务就绪后，你执行：

```bash
curl http://127.0.0.1:8998/v1/chat/completions \
  -d '{"model":"Qwen3.8-27B","messages":[{"role":"user","content":"..."}],"max_tokens":64}'
```

### 第 14 环 门房收件：FastAPI → TokenizerManager

1. uvicorn（主进程里的 HTTP 服务）收到 POST，把 JSON 交给 FastAPI 的路由函数；
2. 路由函数调 `tokenizer_manager.generate_request(...)`（tokenizer_manager.py）：
   - 把 `messages` 按聊天模板拼成一段字符串；
   - 用分词器切成**一长串整数**（token ids）——就是第四部分日志里的
     "前缀 4800 tokens" 的来源；
   - 生成请求编号 `rid`（形如 `3bc1e971...`，你在 ERROR 日志里见过的那个）；
   - 打点 `TTFT 计时开始`；
3. `tokenizer_manager` 用 ZMQ 把"请求包"发给 TP0 scheduler。

### 第 15 环 调度循环取件与查缓存（主角登场）

scheduler 的 `run_event_loop()`（指挥官 `run_scheduler_process` 第 6087 行调用的
那个）在死循环里不断：收 ZMQ 新请求 → 塞进等待队列 → 组 batch。轮到这个请求时：

```text
scheduler 由该请求组装 MatchPrefixParams（含 token 序列、rid）
  → 调 self.tree_cache.match_prefix(params)      ← 此时 self.tree_cache 就是第 13 环
                                                   挂在身上的 LMCacheAscendUnifiedCache！
    → 我们覆写的 match_prefix（unified_linker.py:428）执行：
       ① 先 super().match_prefix(params)：在【本地】树里找最长命中，
          拿到 device_hit（本地已缓存的 token 数，冷启动时是 0）；
       ② 本地不够，我们补一枪远程：driver.lookup(rid, tokens, ...)
          → connector.submit_lookup(rid, tokens, local_hit_tokens=device_hit,
                                    cache_salt=...)          ← 组装 LOOKUP 消息
          → ZMQ 发给 lmcache server
          → server 收到：查自己索引"这些 token 的哪些 chunk 我存过？"
          → server 顺手把磁盘(L2)里的相关块预取进内存(L1)
          → 我们这头 driver._poll_lookup() 每 0.5 毫秒问一次"好了没"
             （8 个 rank 每轮 poll 各跑一次集合通信保持同步——这正是当初旧
               连接器阻塞式 lookup 的等价物）
          → 拿到总数（server 说"我这边还能补给你 N 个 token"）
       ③ 把我们这枪的结果写进返回值的 host_hit_length 字段
    → match_prefix 返回 MatchResult（关键字段：device_indices=本地命中显存地址、
      host_hit_length=远程可补的长度、mamba_host_hit_length=mamba 可补一格）
```

**如果没有 LMCache**，②③ 不存在，`host_hit_length` 永远是 0。有了它，
第二次提问"你好吗"时，①里 device_hit=0（本地树被清或重启），②里能翻出
几千——`Prefill batch` 日志里 `#cached-token: 4608` 就是这么来的。

### 第 16 环 回灌 KV：`init_load_back`

scheduler 决定放行请求时调 `init_load_back`，同样落到我们的覆写
（unified_linker.py:471）：

```text
① 从 pending 表取出第 15 环存的查询结果
② 向本地"全注意力分配器"要 N 个设备槽位（不够就先 evict 淘汰旧前缀腾地方）；
   同时向 mamba 分配器要 1 个状态槽
③ driver.submit_load(pending, [full 槽地址, mamba 槽地址])
   → connector.submit_load(...)：把目标槽换算成组内 block 编号，发 RETRIEVE
④ driver.complete_load(operation)：等搬运结束
   （server 侧执行 multi_layer_block_kv_transfer → 我们仓库的 c_ops C++扩展 →
     AscendC 算子：内存 → 8 张卡的显存, 毫秒级, 日志 Retrieved NNNN tokens in 0.00x）
⑤ 把回灌的这批槽插回本地树（下次别人问同样的前缀, ①步直接命中）
⑥ 返回 (可用的全部显存地址, 树节点)
```

### 第 17 环 前向计算：prefill 与 decode

- 调度器把"已就位 4608 + 新 192"拼成 prefill batch，交给模型做一次前向；
  明显特征是日志 `Prefill batch, #new-seq: 1, #new-token: 192, #cached-token: 4608`。
  **它只真正计算 192 个新 token**（4608 个的 KV 已在显存里摆好了）。
- 之后逐 token 解码：每一步一个 `Decode batch` 行，输出数字；
- 输出的每个数字 → detokenizer 子进程翻回文字 → 经 ZMQ 给主进程 →
  FastAPI 流式写回 curl（所以你能看到字一个个蹦出来）。

### 第 18 环 持久化这次的前缀（cache_finished_req）

请求跑完，scheduler 调 `tree_cache.cache_finished_req(req)` → 我们的覆写
（unified_linker.py:595）→ `_store_via_lmcache`（:655）：

```text
① 取该请求完整 token 序列 + 显存地址 + 它的 mamba 状态槽
② 算对齐长度（第 6 次调试的坑）：len→chunk(256)对齐→再向 mamba 检查点
   网格(128)向下取整；不足一个 chunk → 0（空槽数组合法，server 那边本就空操作）
③ driver.store(rid, tokens, [各组地址], mamba_transfer_tokens=对齐值, ...)
   → device_indices_by_group(...)（配对齐值后不再炸）
   → submit_store → server 反向 DMA：8 卡显存 → server 内存
   → complete_store → end_session(rid) 释放服务端的读锁/会话
④ 最后 super().cache_finished_req(...) 把前缀插进本地树（顺序不能反：
   先存盘、后插树，保证存储失败时树里也没有半截前缀）
```

**观察点**：server 日志出现 `Stored NNNN tokens in 0.00x seconds`（毫秒=原生
NPU 算子路径；若是几十毫秒说明退回 torch 慢速基线）。任何一步出错只打 ERROR、
本请求当没缓存继续跑（fail-open）——你见过的那行
`LMCache unified store failed for 3bc1e971...` 就是这个兜底机制的演出。

### 第 19 环 请求结束

EOS 或 max_tokens 触发结束 → scheduler 释放该请求显存槽（此时本地树已引用这
些槽，不会被真回收）→ 门房把完整回答（流式）发完 → 计时器停表。

---

## 第四部分 日志阅读速成

### 一行日志的解剖

```text
[2026-11-16 08:09:42,501] LMCache INFO: LMCache-Ascend unified driver ready
└───时间(毫秒)──────────┘ └组件┘ └级别┘ └──────内容────────────────────┘
(unified_linker.py:157:lmcache_ascend.integration.sglang.unified_linker)
└──打印位置：文件路径:行号:logger 的模块名────────────────────┘
```

**模块名 = 谁干的**（排错第一眼）：
`lmcache_ascend.*` = 我们；`lmcache.*` = 上游 LMCache；`sglang.*` = SGLang。

### 启动期正向检查表（SGLang 日志按序 grep）

```bash
grep -E "Load weight end|KV Cache is allocated|Init Unified Radix|unified path engaged|unified driver ready|Tree cache initialized|Application startup complete" \
  /workspace/results/sglang_qwen38_unified.log
```

- 出现 `unified path engaged (... mamba=True)` × 8 → 分流成功（数条数）
- 出现 `unified driver ready` × 8 → connector 注册成功
- `Tree cache initialized: impl=LMCacheAscendUnifiedCache` × 8 → 每卡收工
- 最后 `Application startup complete` → 可发请求

### 请求期关键行

| 日志 | 位置 | 意义 |
|---|---|---|
| `Prefill batch, ... #cached-token: 4608` | SGLang | **缓存命中的直接证据**（数字大=LMCache 干活了） |
| `Stored NNNN tokens in 0.00x s` | server | 前缀已持久化 |
| `Retrieved NNNN tokens in 0.00x s` | server | 前缀已回灌 |
| `LMCache unified store failed for <rid>` | SGLang | 某次存储失败（fail-open，服务不崩，需看下坠 traceback） |
| `Decode batch, ...` | SGLang | 每步解码的正常噪音 |

### 验证三连

```bash
# 1) 缓存闭环（Stored 与 Retrieved 都要出现）
grep -E "Stored|Retrieved" /workspace/results/lmcache_server_unified.log | tail -10
# 2) 8 rank 分流完整性
grep -c "unified path engaged" /workspace/results/sglang_qwen38_unified.log
# 3) 失败扫描
grep -B2 -A15 "ERROR" /workspace/results/sglang_qwen38_unified.log | tail -40
```

**极简闭环实验**（用第 5 章的思路）：发一次长前缀请求（>256 token）→
server 日志见 Stored；`curl -X POST http://127.0.0.1:8998/flush_cache` 清本地树 →
发同前缀请求 → 日志见 Retrieved 且 `#cached-token` 变几千。

---

## 第五部分 全景一图收束

```text
【启动】
终端: python -m sglang.launch_server [一堆参数]
  → launch_server.py: 顶层执行 → prepare_server_args 把参数装进 ServerArgs
    → run_server(): 选 HTTP 模式
      → http_server.launch_server → Engine._launch_subprocesses
        ┌─ 主进程里: TokenizerManager 对象（门房）
        ├─【同时】8 个 scheduler 子进程, 入口 run_scheduler_process
        │      → configure_scheduler_process 绑卡
        │      → Scheduler.__init__:
        │           init_mamba_backend（GDN 后端定型, --mamba-ssm-dtype 在此生效）
        │         → init_model_worker → load_model（权重 3.81GB）
        │         → KV 池（HybridLinearKVPool: 16层K/V + mamba 白板区）
        │         → build_kv_cache → create_tree_cache → default_radix_cache_factory
        │             → enable_lmcache==True → import lmc_radix_cache
        │                 → import lmcache → import lmcache_ascend ★三件武器★
        │                 → class LMCRadixCache(RadixCache) ★分诊台装上★
        │             → LMCRadixCache(...) ★分诊★
        │                 标准池 → 老路（武器A保活）
        │                 混合池 → LMCacheAscendUnifiedCache.convert
        │                     → 先换类 → UnifiedRadixCache.__init__（FULL+MAMBA 组件）
        │                     → LMCacheMPDriver → UnifiedLMCacheMPConnector
        │                        → 几何校验（武器A的拆叠在此兑现）
        │                        → ZMQ 注册显存句柄给 lmcache server → 心跳
        │      → 向父进程报"我好了" → run_event_loop() 永动循环
        └─ 8 份"我好了"齐 → FastAPI 暖机 → "Application startup complete"

【一次请求】
curl → FastAPI → TokenizerManager（切分 token、发 rid、起表）
  → ZMQ → scheduler 事件循环 → match_prefix ★我们:
       本地树先查 → 不够则 submit_lookup→server 查索引序预取→poll 轮询
       → host_hit_length=N（远程可补 N 个 token）
  → init_load_back ★我们: 本地要槽(不够就 evict) + mamba 一槽
       → submit_load→server DMA(内存→8卡显存, c_ops+AscendC 算子)→complete_load
       → 插回本地树
  → prefill 只算新增 token（#cached-token 大=省）→ decode 循环逐 token
  → cache_finished_req ★我们: 对齐(256/128)→submit_store→server 反向 DMA→
       complete_store→end_session → 再插本地树
  → detokenizer 翻字 → 流式回 curl
```

**图例**：★我们 = LMCache-Ascend 的适配代码；其余全部是 SGLang / LMCache 上游
原生流程，一行未改。所有失败点都做了 fail-open：出错只降级为"该请求无缓存"，
绝不带走服务。

---

## 附 关系图：本文与其他文档的分工

| 文档 | 分工 |
|---|---|
| 本文 | 逐函数全流程（启动 + 请求 + 日志 + 报错速查） |
| [hybrid_models_runbook.md](hybrid_models_runbook.md) | 三模型操作手册（跑什么命令） |
| [hybrid_models_adaptation_blueprint.md](hybrid_models_adaptation_blueprint.md) | 设计与侦察记录（为什么这么改） |
| [call_flow_before_after_adaptation.md](call_flow_before_after_adaptation.md) | 标准模型（旧路）前后对照 |
| [python_interception_primer.md](python_interception_primer.md) | 拦截技术（分诊台/钩子）的语法课 |