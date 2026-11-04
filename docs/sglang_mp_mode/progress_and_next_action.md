# SGLang MP 适配：当前进度与下一步行动

> 状态：环境搭建完成，NPU 原生 MP 数据面测试已通过（3 passed）。
> 下一步：补做 SGLang 控制面接口审计，然后直接进入真实模型 E2E 启动（不写产品代码）。
>
> 核心判断：**你的测试通过证明的是「NPU 数据面」，不是「SGLang 接口检查」。接口审计（adaptation_walkthrough 第 5.1/5.2 节）是另一件事，还没做。**

---

## 1. 背景速览（30 秒版）

```text
请求 → SGLang（调度/计算）
          │  --enable-lmcache
          ▼
   UnifiedLMCacheMPConnector   ← 上游控制面，SGLang 侧
          │  ZMQ
          ▼
   lmcache server              ← 上游缓存引擎
          │  NPU IPC 句柄
          ▼
   LMCache-Ascend c_ops        ← 本仓数据面，AscendC 原生 kernel
```

- 上游 LMCache v0.5.6 已有 SGLang MP connector（`unified_lmcache_mp_connector.py`）+ NPU 平台抽象；
- 本仓 LMCache-Ascend 已有 NPU 原生块传输 + HCCL/HiXL + plane 布局描述；
- 这次适配的实质 = **把上游 SGLang MP 控制面接到本仓已有的 NPU 数据面上**，而不是重写 connector。

## 2. 当前进度盘点

| 步骤 | 状态 | 依据 |
|---|---|---|
| 环境/依赖安装（LMCache v0.5.6 + LMCache-Ascend + numba 等） | ✅ | [environment_pitfalls_and_fixes.md](./environment_pitfalls_and_fixes.md) |
| SGLang 裸服务基线（Qwen3.8 + TP8） | ✅ | [../sglang_baseline/README.md](../sglang_baseline/README.md) |
| NPU 原生 MP 数据面测试 | ✅ **3 passed** | `/workspace/results/npu_mp_native_e2e.log` |
| **SGLang 控制面接口审计** | ⬜ **未做** | adaptation_walkthrough 第 5.1/5.2 节 |
| 第一次真实模型 E2E 启动 | ⬜ 待做 | [first_model_e2e_validation.md](./first_model_e2e_validation.md) |
| 代码修改 | ⬜ **第一轮不做** | 见第 5 节 |

## 3. 刚跑完的测试到底证明了什么

### 3.1 逐条解读

```
test_forced_mode_selects_lmcache_driven_context          PASSED
```
传输模式解析正确：NPU 上 `lmcache_driven` 被正确选中（是 server 通过设备 IPC 句柄搬 KV，不是 worker 自搬）。

```
test_lmcache_driven_store_and_retrieve_roundtrip[0]      PASSED   (npu:0)
test_lmcache_driven_store_and_retrieve_roundtrip[1]      PASSED   (npu:1)
```
两张卡上都完成了：注册 KV pool → `Stored 256 tokens` → `Retrieved 256 tokens` → 字节级一致性校验。

```
Engine KV Format: EngineKVFormat.NL_X_NP_X_NB_BS_ONE_HS NL x NP x [NB, BS, 1, HS]
```
**这是最关键的积极信号**：MLA/DSA 多 plane 格式被识别，且走的是原生路径——如果插件没生效会退回 torch 基线（约慢 8 倍），日志形态也会不同。`Stored 0.003~0.005s` 的速度也印证走的是本仓 kernel。

### 3.2 两个 warning 不用管

| warning | 说明 |
|---|---|
| `Producer process has been terminated before all shared NPU tensors released` (NPUIPCTypes.cpp:23) | 测试框架 producer 进程退出时的已知提示，不是失败 |
| `PytestUnknownMarkWarning: Unknown pytest.mark.cuda` | 来自上游 `test_custom_types.py` 的未注册 marker，与本测试无关 |

### 3.3 证明边界（重要）

| 已证明 ✅ | 未证明 ⬜ |
|---|---|
| 本仓 `c_ops` 被上游 `NpuDeviceOps` 正确绑定 | SGLang 实际会创建 `UnifiedLMCacheMPConnector` |
| NPU IPC 注册 + store/retrieve 字节一致 | SGLang 侧模块（unified connector / radix cache）在你的版本里齐全 |
| `lmcache_driven` 上下文选择正确（`EngineType.VLLM` 场景） | SGLang scheduler / TP worker 会调用这些接口 |

该测试内部使用 `EngineType.VLLM`，是"零件级"验证；下一步的 E2E 是"整车级"验证。

## 4. 什么不用做（明确排除）

| 不要做 | 原因 |
|---|---|
| 重写或新写 SGLang MP connector | 上游 v0.5.6 已有 `UnifiedLMCacheMPConnector`，重写 = 重复维护上千行协议代码 |
| 把 vLLM 的 JSON connector 配置照搬到 SGLang | SGLang 用 `--enable-lmcache` + YAML，字段与加载机制不同 |
| 把 vLLM 的 chunk 经验值（896/1536）照搬 | 那是 vLLM-Ascend 特定版本组合；SGLang 必须从自身启动日志读 N |
| 切换 hcomm / CANN 版本 | 当前组合已验证可用（v9.1.0 + CANN 9.0.0） |
| 修改产品代码 | 第一轮目标是"用现有上游实现零代码跑通" |
| 仍走本仓旧 in-process SGLang 路径 | 绑定 LMCache 0.4.5，与当前 MP 环境互斥 |

## 5. 下一步行动清单（按顺序）

### 第 1 步：接口审计（你跳过的那一步，必须补）

在容器内执行并保存输出到 `/workspace/results/sglang_mp_api_audit.log`：

```bash
cd /workspace/src/LMCache
python - <<'PY' 2>&1 | tee /workspace/results/sglang_mp_api_audit.log
import importlib.util
import lmcache
import sglang

print("LMCache:", lmcache.__file__)
print("SGLang:", sglang.__version__, sglang.__file__)

modules = [
    "lmcache.integration.sglang.unified_lmcache_mp_connector",
    "lmcache.integration.sglang.unified_kv_adapter",
    "lmcache.v1.multiprocess.transfer_context.worker_transfer",
    "lmcache.v1.platform.devices.npu",
    "lmcache.v1.platform.devices.npu.ipc_wrapper",
    "sglang.srt.mem_cache.storage.lmcache.lmc_radix_cache",
]
for name in modules:
    try:
        found = importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, AttributeError) as exc:
        found = f"ERROR: {exc}"
    print(f"{name}: {found}")
PY
```

**判定标准**：

| 结果 | 含义 | 行动 |
|---|---|---|
| `unified_lmcache_mp_connector` / `unified_kv_adapter` 都 OK | 上游 SGLang MP 控制面齐全 | 进入第 2 步 |
| `npu` / `ipc_wrapper` OK | 上游 NPU 平台抽象齐全 | 与已通过的 3 passed 互相印证 |
| 任一 ERROR / None | 版本或安装路径问题 | 检查 LMCache 是否真的是 v0.5.6 editable 安装，不启动 E2E |

注：`lmc_radix_cache` 的具体模块路径因 SGLang 版本而异，找不到不代表 connector 不可用；以 unified 两个模块的结果为准。

### 第 2 步：确认 SGLang 实际挂接点

```bash
grep -Rns "UnifiedLMCacheMPConnector\|enable_lmcache" \
  "$(python -c 'import sglang, os; print(os.path.dirname(sglang.__file__))')/srt/mem_cache" \
  2>/dev/null | head -20
```

要回答三个问题：`--enable-lmcache` 最终创建哪个 connector？创建发生在 scheduler 进程还是 TP worker？KV pool 注册在哪个函数？

### 第 3 步：E2E 启动（详见 [first_model_e2e_validation.md](./first_model_e2e_validation.md)）

浓缩检查清单（按 execution 顺序）：

```text
□ 确认裸 SGLang 服务已停止（8 卡不能同时跑两个 TP8）
□ 从旧启动日志 grep attention block size N（不要照抄 vLLM 经验值）
□ 写 /workspace/scripts/lmcache_mp.yaml（mp_host/port + chunk_size=N）
□ 启动 lmcache server（--supported-transfer-mode lmcache_driven）
□ 启动 SGLang（原参数 + --enable-lmcache + --lmcache-config-file + --mamba-ssm-dtype bfloat16）
□ 启动日志确认：UnifiedLMCacheMPConnector + device_type=npu、mode=lmcache_driven + 8 个 rank 都注册
□ 冷请求 → lmcache_server.log 出现 Stored
□ flush_cache 清本地 RadixCache → 同前缀请求 → 出现 Retrieved
```

### 第 4 步：按第一个真实错误决定改哪里

#### 4.1 原则：改上游是最后手段，不是默认路径

**本仓的设计哲学就是"尽量不改上游"**。看 `lmcache_ascend/__init__.py`、`ops.py`、`shape_desc.py` 的机制就知道：

- `ops.py`：把本仓 `c_ops` 绑到上游 `NpuDeviceOps.ensure_native` 门面上，靠**原生绑定**，不动上游一行源码；
- `shape_desc.py`：运行时重新绑定上游 `gpu_connector_utils.make_page_buffer_shape_desc`，是**运行时替换**；
- `kv_layer_groups.py`：提供 `patched_hidden_dim_size` / `build_kv_layer_groups` 替代实现，同样**运行时替换**（本仓曾用同类机制把上游 `CudaIPCWrapper` 换成 `AscendIPCWrapper`）。

所以绝大多数差异可以在本仓用 runtime patch 消化，**不 fork、不改上游 LMCache**。真正必须动上游的只有两类情况：

| 必须动上游的情况 | 正确做法 |
|---|---|
| 上游真缺陷/漏 NPU 分支 | 给上游提 PR 合入；本地过渡期可以在本仓做 runtime patch 硬顶 |
| SGLang 版本过旧，没有 unified 调用路径 | **升级 SGLang 版本**（换版本 > 改它的代码），不要强行走本仓旧 in-process adapter |

注意：本仓 `integration/patch/` 里确实有一个"直接改上游源码"的机制（`apply_patch.py` + `sglang_patch.py`），它重写 SGLang `ModelRunner.init_memory_pool`。但这是**旧 in-process 时代的遗留**：版本窗口锁死 0.5.2~0.5.8，且 `setup.py` 里 `run_patches()` 已被注释，`__init__` 不再调用。**MP 适配不要沿用这条路**——它正是"改上游、锁版本"的反面教材。

#### 4.2 错误分级决策表

| 错误归属 | 改谁 | 优先顺序 |
|---|---|---|
| SGLang 没创建 unified connector | SGLang | ① 先查 `--enable-lmcache` 参数与版本；② SGLang 过旧则升级；③ 上游 LMCache `integration/sglang/` 挂接点缺失→PR |
| connector 生命周期/注册不匹配 | 上游 LMCache `integration/sglang/` | ① 先在本仓做 runtime patch 顶；② 通用问题→PR 上游 |
| NPU IPC / device context | 上游 `platform/devices/npu/` + 本仓 `c_ops` | ① 先确认 vendor v0.5.6 已移除 CUDA 硬编码；② 本仓 `c_ops` 补实现 |
| plane 几何 / KV 布局 / 块传输 | **本仓** `shape_desc.py`、`kv_layer_groups.py`、`csrc/` | 几乎全部可本仓解决 |
| Qwen3.8 多轮才异常 | 上游 adapter + 本仓 kernel（专项） | ① 普通全注意力模型先跑通；② MAMBA 单独排期 |
| 不能只在本仓解决的（上游函数内部行为不符） | 上游 | 提 PR + 记录上游 issue |

#### 4.3 如果改本仓：具体文件与预估代码量

| 场景 | 具体文件 | 预估量 | 备注 |
|---|---|---|---|
| 插件激活 vllm-only → vllm/sglang 分派 | `lmcache_ascend/__init__.py`（`_is_vllm_runtime` + patch 安装段） | 30~80 行 Python | 拆成公共初始化 + 引擎专属初始化 |
| SGLang KV 注册/生命周期适配 | 新建 `integration/sglang/sglang_mp_*.py`，包装 `UnifiedLMCacheMPConnector` / `SGLangUnifiedKVAdapter` 的若干方法 | 100~400 行 Python | 可参考现有 `integration/vllm/` 的设计 |
| SGLang 布局分支（FULL/SWA/MAMBA 分组、K/V pool 形状） | `v1/shape_desc.py`、`v1/kv_layer_groups.py` | 20~100 行 Python | 优先在运行时替换，勿动上游源文件 |
| NPU 新格式 kernel（hybrid Mamba recurrent state 与现有 18 种格式不匹配时） | `csrc/mp_mem_kernels.cpp/.h` + `csrc/kernels/multi_layer/` | 50~200 行 C++ | 需重编译 `c_ops` |
| 测试与文档 | `tests/integration/sglang/`、`docs/sglang_mp_mode/` | 100~300 行测试 | 必交付 |

总预估（取决于接口审计）：**零代码跑通 0 行 → 小兼容 patch 约 50~150 行 → 中等适配约 150~400 行 Python（+ 可能 50~200 行 C++）→ Qwen3.8 MAMBA 另行评估。**

#### 4.4 决策流程（遇到报错时按序走）

```text
① 是配置问题吗？  → chunk_size / host / port / flush_cache       → 改配置，不动代码
② 是姿势问题吗？  → 照搬了 vLLM 配置 / 版本不匹配                 → 按审计结果修正
③ 是版本问题吗？  → SGLang/LMCache 过旧                          → 升级版本
④ 是本仓能顶的？ → shape/layout/激活分派                         → 本仓 runtime patch
⑤ 只剩上游了     → 缺 NPU 分支/真缺陷                             → 上游 PR + 本仓过渡 patch
```

## 6. 与现有文档的分工

| 文档 | 定位 |
|---|---|
| [adaptation_walkthrough.md](./adaptation_walkthrough.md) | 全流程走查 + 接口审计脚本原文（第 5 节）+ 报错归属（第 12 节） |
| [environment_pitfalls_and_fixes.md](./environment_pitfalls_and_fixes.md) | 环境搭建踩坑记录（已完成） |
| [first_model_e2e_validation.md](./first_model_e2e_validation.md) | E2E 启动的完整手把手步骤（下一步主文档） |
| [lmcache_ascend_vs_upstream.md](./lmcache_ascend_vs_upstream.md) | 上游 vs 本仓的能力边界背景 |
| **本文档** | **进度盘点 + 下一步清单（做完一步回头看一步）** |