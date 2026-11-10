# 为什么 NPU 原生测试 3 passed 还不够：完整验证分级指南

> 本文解释 `test_npu_lmcache_driven_e2e.py` 到底验证了什么，以及它通过后，如何继续验证 SGLang 控制面、真实模型、TP8、本地 RadixCache、Qwen3.8 MAMBA state 和性能。
>
> 核心原则：**每个测试只证明它实际覆盖的路径。单元/组件测试通过，不等于完整推理服务已经适配成功。**

## 1. 先用一张图理解测试层级

```text
第 0 级：接口审计
  模块和类是否存在、能否 import
          │
          ▼
第 1 级：NPU 原生 MP 数据面测试
  真实 NPU 进程间能否正确搬运一份测试 KV
          │
          ▼
第 2 级：SGLang MP 启动
  SGLang 是否创建 connector，worker 是否注册真实 KV pool
          │
          ▼
第 3 级：真实请求冷写入/暖命中
  是否出现 Stored / Retrieved，模型输出是否正常
          │
          ▼
第 4 级：TP8、本地 RadixCache、Qwen3.8 MAMBA 专项
  多 rank、本地/外部缓存、recurrent state 是否都正确
          │
          ▼
第 5 级：正式性能对照
  在正确性成立后，测 TTFT/TPOT/吞吐是否有收益
```

越往上，测试越接近真实业务；越往下，越容易定位具体技术问题。原生测试 3 passed 位于第 1 级。

## 2. `pytest ... test_npu_lmcache_driven_e2e.py` 究竟是什么

执行命令：

```bash
export LMCACHEPATH=/workspace/src/LMCache
cd /workspace/src/LMCache-Ascend
pytest -q tests/v1/multiprocess/test_npu_lmcache_driven_e2e.py -s
```

它不是只检查模块能否导入的接口审计，而是一个**真实 NPU 数据面测试**：

```text
子进程在 NPU 上分配并填充 KV plane
  → 导出 NPU IPC 张量句柄和进程间事件
  → 父进程创建真实 NPU cache context
  → store：NPU KV gather 到 CPU 暂存对象
  → 校验 CPU 数据
  → CPU 数据整体 +1
  → retrieve：CPU 数据 scatter 回相同 NPU block
  → 校验 NPU 上每个 plane、block 都正确 +1
```

测试使用的主要数据：

```text
4 层
16 个 block
每 block 16 token
合计 256 token
每层 latent 128 维 + rope 16 维
```

这些是人为构造的小型 MLA 布局，不是 Qwen 模型配置。

### 真实实现与测试替身

| 部分 | 是否真实 |
|---|---|
| NPU 设备和 worker 子进程 | 真实 |
| NPU IPC wrapper、事件和 stream | 真实 |
| NPU cache context、KV 布局识别 | 真实 |
| store/retrieve、gather/scatter | 真实 |
| `_FakeStorageManager` | 测试替身，只提供 CPU 暂存对象 |
| 完整 `lmcache server` 和生产存储集群 | 未启动 |
| SGLang/vLLM 推理服务 | 未启动 |

使用 fake storage 是为了隔离变量：先验证数据面，不把调度、存储淘汰、模型执行混在一起。

## 3. 为什么 3 passed 仍然不能证明完整适配

这三个测试分别是：

1. 显式模式能创建 `LMCacheDrivenTransferContext`；
2. `npu:0` 上 store/retrieve 往返正确；
3. `npu:1` 上 store/retrieve 往返正确。

它们证明的是：

> 对一份受控的小型 KV 数据，NPU IPC、布局、块传输和事件同步能完成正确往返。

但下面每件事都不在这条测试路径里。

## 4. SGLang 控制面：执行什么、通过标志、原理

### 要验证什么

SGLang 是否真的因为 `--enable-lmcache` 创建了 Unified MP connector，并在 worker 中注册真实 KV pool。

原生测试是手工创建 context；生产服务必须由 SGLang 自己在正确的进程和时机创建。

### 执行什么

先启动独立 server：

```bash
export PYTHONHASHSEED=0
lmcache server \
  --host 127.0.0.1 --port 5556 \
  --http-port 8080 \
  --chunk-size "$N" \
  --l1-size-gb 20 \
  --eviction-policy LRU \
  --supported-transfer-mode lmcache_driven \
  2>&1 | tee /workspace/results/lmcache_server.log
```

然后按原参数启动 SGLang，并追加：

```bash
--enable-lmcache \
--lmcache-config-file /workspace/scripts/lmcache_mp.yaml
```

完整命令见 `first_model_e2e_validation.md`。

启动后检查：

```bash
grep -nEi \
  'UnifiedLMCacheMPConnector|transfer context|register|lmcache|error' \
  /workspace/results/sglang_mp.log | head -100
```

### 通过标志

- 出现 NPU `lmcache_driven` transfer context；
- connector 和 KV 注册成功；
- 没有回退到旧 `LMCacheLayerwiseConnector`；
- SGLang 最终 ready；
- 没有 IPC、CUDA、shape、block/chunk 错误。

### 原理

这验证的是**控制面**：引擎是否知道何时创建 connector、连接哪个 server、把哪份 KV 注册进去。控制面正确不保证后续请求一定会命中，因此还要继续测试。

## 5. 真实模型 prefill/decode 与 Stored/Retrieved

### 要验证什么

真实请求产生的 KV 是否会在正确时机写入缓存，后续请求是否能取回并继续生成。

### 执行什么

发送长前缀请求 A：

```bash
python - <<'PY' > /tmp/lmcache_req_a.json
import json
prefix = "KV cache reuse requires a sufficiently long shared prefix. " * 300
json.dump({
    "model": "Qwen3.8-27B",
    "prompt": prefix + "\nSummarize the topic in one sentence.",
    "max_tokens": 32,
    "temperature": 0,
}, open("/tmp/lmcache_req_a.json", "w"))
PY

curl -s http://127.0.0.1:8998/v1/completions \
  -H 'Content-Type: application/json' \
  -d @/tmp/lmcache_req_a.json
```

然后清除本地缓存：

```bash
curl -s -X POST http://127.0.0.1:8998/flush_cache
```

再发送相同前缀、不同尾部的请求 B：

```bash
python - <<'PY' > /tmp/lmcache_req_b.json
import json
prefix = "KV cache reuse requires a sufficiently long shared prefix. " * 300
json.dump({
    "model": "Qwen3.8-27B",
    "prompt": prefix + "\nGive one short title for this text.",
    "max_tokens": 32,
    "temperature": 0,
}, open("/tmp/lmcache_req_b.json", "w"))
PY

curl -s http://127.0.0.1:8998/v1/completions \
  -H 'Content-Type: application/json' \
  -d @/tmp/lmcache_req_b.json
```

检查：

```bash
grep -E "Stored|Retrieved" /workspace/results/lmcache_server.log | tail -50
```

### 通过标志

- A 出现 `Stored`；
- B 出现 `Retrieved`；
- B 的输出正常；
- prefill/decode 无异常，服务没有崩溃或死锁；
- 多轮重复后仍稳定。

### 原理

这一步把真实模型的 KV 生命周期接上了：

```text
A：miss → prefill → store
B：lookup 命中 → retrieve → 只计算新尾部 → decode
```

这是端到端正确性的核心标志，比单纯 ready 更有证明力。

## 6. TP8 全 worker 注册

### 为什么原生测试覆盖不了

原生测试只使用 `npu:0`、`npu:1`，并且 cache context 的 `world_size=1`。它没有创建 8 个生产 worker，也没有验证每个 rank 的本地 device 映射。

### 执行什么

使用 `--tp-size 8` 启动后，检查：

```bash
grep -nEi \
  'rank|worker|register|transfer context|error|failed' \
  /workspace/results/sglang_mp.log | head -150
```

必要时观察 NPU：

```bash
npu-smi info
```

### 通过标志

- 日志能看到各个 TP rank/worker 的初始化或注册信息；
- 没有某个 rank 先成功后异常退出；
- 没有 device 仍指向 `cuda`、CPU 或错误的 NPU index；
- 所有 rank 都进入正常服务状态；
- 真实请求在 TP8 下能完成 Stored/Retrieved。

如果当前日志无法明确区分 8 个 rank，不能只凭 SGLang ready 就宣布“8 个都注册成功”；需要提高日志级别或增加临时注册探针。

### 原理

TP 下每个 worker 持有自己的 KV 分片和 NPU context。一个 rank 注册成功不能推出其他 rank 也成功。

## 7. SGLang 本地 RadixCache 与外部 LMCache 是否分清

### 要验证什么

第二次请求到底命中了 SGLang 本地缓存，还是外部 LMCache。两者的性能含义和验证目标不同。

### 执行什么

做两组对照：

```text
对照 1：A → B（不 flush）
  B 可能直接命中本地 RadixCache，server 不一定出现 Retrieved

对照 2：A → flush_cache → B
  B 若要复用前缀，只能从外部 LMCache 回灌，应出现 Retrieved
```

### 通过标志

- 能理解不 flush 时本地命中是正常现象；
- flush 后仍有外部 `Retrieved`；
- 不会把“第二次 TTFT 很低但无 LMCache retrieve”误判为 LMCache 成功。

### 原理

本地缓存和外部缓存是两级不同存储。验证外部缓存时，必须排除本地缓存这个干扰变量。

## 8. Qwen3.8 MAMBA/recurrent state 生命周期

### 为什么普通 KV 往返覆盖不了

Qwen3.8 不只包含全注意力 KV，还包含 GDN/MAMBA 的 recurrent state。普通 attention KV 正确，不代表 MAMBA state 的保存、恢复、对齐和多轮续写也正确。

### 执行什么

第一轮让模型记住一个数字：

```bash
python - <<'PY' > /tmp/mamba_turn1.json
import json
json.dump({
    "model": "Qwen3.8-27B",
    "messages": [
        {"role": "user", "content": "请记住数字 42，只回答：已记住。"}
    ],
    "max_tokens": 32,
    "temperature": 0,
}, open("/tmp/mamba_turn1.json", "w"))
PY

curl -s http://127.0.0.1:8998/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d @/tmp/mamba_turn1.json
```

确认出现缓存写入后，执行：

```bash
curl -s -X POST http://127.0.0.1:8998/flush_cache
```

第二轮携带相同历史并追问：

```bash
python - <<'PY' > /tmp/mamba_turn2.json
import json
json.dump({
    "model": "Qwen3.8-27B",
    "messages": [
        {"role": "user", "content": "请记住数字 42，只回答：已记住。"},
        {"role": "assistant", "content": "已记住。"},
        {"role": "user", "content": "我刚才让你记住的数字是什么？"}
    ],
    "max_tokens": 32,
    "temperature": 0,
}, open("/tmp/mamba_turn2.json", "w"))
PY

curl -s http://127.0.0.1:8998/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d @/tmp/mamba_turn2.json
```

### 通过标志

- 第二轮能正确回答 42；
- 出现与历史前缀对应的 `Retrieved`；
- 没有 MAMBA component、state shape、dtype、block alignment、device 错误；
- 多轮继续追问仍正常；
- 与无 LMCache 的相同历史结果进行语义对照，没有明显状态丢失。

### 原理

这一轮验证的是“历史状态恢复后是否仍能维持对话语义”，不是简单的张量拷贝。对于量化或 BF16 state，不要求每次输出逐 token 完全一致，但不能忘记关键历史或出现状态错乱。

## 9. 最后才做性能验证

### 为什么不在 3 passed 后直接谈提速

原生测试只证明搬得对，不证明：

- 真实负载有足够前缀重复；
- 命中率足够高；
- 回灌时间小于重新 prefill；
- 控制面和调度没有引入更大开销。

### 执行什么

在 Stored/Retrieved 正确后：

1. 使用相同模型、相同启动参数和相同并发；
2. 使用相同长前缀、不同尾部的 A/B 请求；
3. 排除本地 RadixCache 干扰；
4. 分别测量无 LMCache 与启用 MP 的 TTFT；
5. 重复多轮，观察 P50/P95/P99，而不是只看一次。

可以使用 `first_model_e2e_validation.md` 中的 `verify_lmcache_ttft.py` 做冒烟测试，再设计正式 benchmark。

### 通过标志

功能与性能要分开：

| 类型 | 通过标志 |
|---|---|
| 功能正确 | Stored/Retrieved 正常、输出正常、无生命周期错误 |
| 性能有效 | 暖请求 TTFT 可重复下降，且没有不可接受的 TPOT/吞吐退化 |

性能不达标不一定意味着适配错误，也可能是工作集、命中率或 L1 容量不合适。

## 10. 每一级“通过”到底说明什么

| 验证 | 通过能证明 | 通过不能证明 |
|---|---|---|
| 接口 import | 代码和符号存在 | NPU 能正确传输 |
| 原生 3 passed | 受控 KV 的 NPU 跨进程往返正确 | SGLang 已接入、真实模型可用 |
| SGLang MP ready | 控制面初始化成功 | 请求一定命中、输出一定正确 |
| Stored | 真实请求进入缓存写入路径 | 后续请求能取回 |
| Retrieved | 外部缓存回灌路径被执行 | 所有 rank、MAMBA state 都正确 |
| TP8 请求成功 | 当前 TP8 请求链路可运行 | 长稳和所有边界条件都覆盖 |
| MAMBA 多轮正确 | 历史状态恢复具备基本正确性 | 任意长上下文/抢占/高并发都正确 |
| TTFT 下降 | 当前负载下有实际缓存收益 | 所有业务都会提速 |

## 11. 推荐实际执行顺序

```text
1. 保留原生 3 passed 结果
2. 启动 lmcache server + SGLang MP，检查控制面
3. A 冷请求确认 Stored
4. flush 后 B 暖请求确认 Retrieved
5. 检查 TP8 各 rank 注册与稳定输出
6. 用 flush 对照区分本地 RadixCache 和外部 LMCache
7. 执行 Qwen3.8 MAMBA 多轮记忆测试
8. 正确性全部成立后，再做正式 TTFT/TPOT/吞吐对照
```

如果某一级失败，就停在该级定位，不要继续叠加更复杂的负载。这样可以把问题准确归因到接口、数据面、控制面、模型布局或性能配置。
