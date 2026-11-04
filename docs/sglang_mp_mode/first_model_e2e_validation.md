# 第一次 SGLang + LMCache MP 真实模型验证：手把手教程

> 适用环境：8×Ascend 910B4，SGLang 已能启动 `/data/models/Qwen`，LMCache/LMCache-Ascend 已安装，接口审计和 NPU 原生 MP 测试已经通过。
>
> 当前目标：不先修改代码，直接使用现有上游实现做一次真实模型的端到端验证。
>
> 核心原则：**先启动，后看真实错误；先证明 Stored/Retrieved，后测性能；先排除 SGLang 本地缓存，再判断 LMCache 是否命中。**

## 1. 现在为什么可以开始跑真实模型

前面已经分别证明：

```text
接口审计：
SGLang / UnifiedLMCacheMPConnector / NPU IPC 接口存在

NPU 原生测试：
NPU 0、NPU 1 上都完成 Stored 256 tokens / Retrieved 256 tokens
```

这说明底层数据面已经具备基本能力，但还没有证明真实 SGLang 请求会正确走到这条路径。

本次要验证的完整链路是：

```text
客户端请求
  → SGLang scheduler / RadixCache
  → UnifiedLMCacheMPConnector
  → lmcache server
  → NPU IPC + Ascend 原生块传输
  → SGLang 继续 prefill/decode
```

可能仍然存在的问题：

- SGLang 实际没有创建 Unified MP connector；
- block/chunk 大小不一致；
- TP8 只有部分 rank 注册；
- Qwen3.8 的 MAMBA state 生命周期不兼容；
- 第二次请求只命中了 SGLang 本地缓存，而不是 LMCache。

所以这次启动是“用真实模型发现真实差异”，不是提前宣布适配完成。

## 2. 本次验证的完成标准

只有同时满足以下条件才算第一轮成功：

1. `lmcache server` 正常启动；
2. SGLang 带 `--enable-lmcache` 正常启动；
3. TP8 各 worker 完成 KV 注册；
4. 第一个长前缀请求出现 `Stored`；
5. 清理 SGLang 本地 RadixCache 后，第二个相同前缀请求出现 `Retrieved`；
6. 输出正常，无 IPC、shape、dtype、MAMBA 状态异常；
7. 暖请求 TTFT 有可重复的下降趋势。

仅看到 SGLang ready 或仅看到 LMCache server ready，都不算端到端成功。

## 3. 准备工作

### 3.1 保存裸服务基线

保留已有的 SGLang 裸服务 benchmark 结果，后续用于对照：

- TTFT；
- TPOT/ITL；
- 吞吐；
- 启动参数和并发数。

### 3.2 不要同时运行两个 TP8 模型服务

如果裸 SGLang 还在运行，先保存日志并停止它。8 张 910B4 不应同时承载两个 27B TP8 服务。

后续需要三个终端，但都进入**同一个容器、同一个 Python 环境**：

```text
终端 A：lmcache server
终端 B：SGLang
终端 C：验证客户端
```

### 3.3 创建结果目录

```bash
mkdir -p /workspace/scripts /workspace/results
```

## 4. 第一步：找到 attention block size N

在原来的 SGLang 启动日志中执行：

```bash
grep -iE "attention block size|block size|page size" \
  /workspace/scripts/sglang_server.log | tail -30
```

如果日志不在该位置，替换为实际路径。

记下日志中的数字 N。例如，仅作格式说明：

```text
Setting attention block size to ... N ...
```

**不要直接照抄 vLLM 文档中的 896、1536 等数值。** 那些值属于特定 vLLM-Ascend 版本组合，不自动适用于当前 SGLang。

如果日志里找不到 N，先保留完整启动日志并确认，不要盲目把 N 当成 256。

### 原理：block、chunk、batched tokens 不是一回事

| 名称 | 含义 |
|---|---|
| attention block size N | 推理引擎管理 KV 的物理/逻辑块跨度 |
| LMCache chunk_size | LMCache 做 hash、存储和传输时使用的 token 块大小 |
| `--chunked-prefill-size` | 一次调度允许处理的 prefill token 预算 |

LMCache 要按引擎块边界取放 KV，所以 `chunk_size` 必须与 N 对齐。它不需要等于 `--chunked-prefill-size 4096`。

## 5. 第二步：创建 SGLang 使用的 LMCache 配置

在提示中输入上一步确认的 N：

```bash
read -p "请输入 SGLang 日志中的 attention block size N: " N

cat > /workspace/scripts/lmcache_mp.yaml <<EOF
mp_host: tcp://127.0.0.1
mp_port: 5556
chunk_size: ${N}
EOF

cat /workspace/scripts/lmcache_mp.yaml
```

配置含义：

```text
mp_host / mp_port：SGLang 到哪里找 lmcache server
chunk_size：LMCache 按多少 token 一个 chunk 处理 KV
```

server 与客户端配置必须使用相同的 N。

## 6. 第三步：启动 lmcache server（终端 A）

```bash
export PYTHONHASHSEED=0
export LMCACHEPATH=/workspace/src/LMCache

lmcache server \
  --host 127.0.0.1 --port 5556 \
  --http-port 8080 \
  --chunk-size "$N" \
  --l1-size-gb 20 \
  --eviction-policy LRU \
  --supported-transfer-mode lmcache_driven \
  2>&1 | tee /workspace/results/lmcache_server.log
```

注意：如果新开终端，当前 shell 中没有上一步的 `N`，需要先重新设置：

```bash
N=上一步确认的数字
```

### 参数解释

- `--port 5556`：SGLang 控制请求入口；
- `--http-port 8080`：健康检查和管理端口；
- `--l1-size-gb 20`：L1 主机内存缓存容量；
- `--eviction-policy LRU`：容量不足时按 LRU 淘汰；
- `--supported-transfer-mode lmcache_driven`：由缓存服务通过 NPU IPC 执行数据搬运。

### 健康检查

在终端 C 执行：

```bash
curl -s http://127.0.0.1:8080/healthcheck
```

如果该版本端点不同，以启动日志和 `lmcache server --help` 为准。至少应看到 server 正在监听 5556。

## 7. 第四步：启动带 LMCache 的 SGLang（终端 B）

保留原来能跑通模型的全部环境变量和参数，只新增最后两个 LMCache 参数：

```bash
export PYTHONHASHSEED=0

python -m sglang.launch_server \
  --model-path /data/models/Qwen \
  --served-model-name Qwen3.8-27B \
  --host 0.0.0.0 --port 8998 \
  --tp-size 8 \
  --dtype bfloat16 \
  --context-length 8192 \
  --chunked-prefill-size 4096 \
  --mem-fraction-static 0.85 \
  --trust-remote-code \
  --mamba-ssm-dtype bfloat16 \
  --enable-lmcache \
  --lmcache-config-file /workspace/scripts/lmcache_mp.yaml \
  2>&1 | tee /workspace/results/sglang_mp.log
```

`--mamba-ssm-dtype bfloat16` 是当前 910B 跑 Qwen3.8 的必要设置，必须保留。

### 启动时重点观察

```text
是否创建 UnifiedLMCacheMPConnector
是否创建 device_type=npu、mode=lmcache_driven 的 transfer context
是否每个 TP rank 都注册 KV pool
是否出现 CUDA-only / IPC / shape / block size 异常
是否最终进入 ready
```

不同版本日志文字可能不同，不要机械匹配完整句子，重点看语义。

### 如果启动失败

先保存：

```bash
grep -nEi -B 10 -A 25 -m 3 \
  'Traceback|RuntimeError|ValueError|ImportError|fatal|Error' \
  /workspace/results/sglang_mp.log
```

最早异常通常比最后“worker 全部退出”更有价值。

## 8. 第五步：验证 Stored/Retrieved，而不只是服务能启动

### 8.1 请求 A：冷请求

在终端 C 生成一个长前缀请求：

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

观察 `lmcache_server.log`：

```bash
grep -E "Stored|Retrieved" /workspace/results/lmcache_server.log | tail -30
```

冷请求预期看到 `Stored`。第一次没有 Retrieved 是正常的，因为缓存还是空的。

### 8.2 清掉 SGLang 本地缓存

```bash
curl -s -X POST http://127.0.0.1:8998/flush_cache
```

这一步非常重要。SGLang 自带 RadixCache，如果不清除，第二次请求可能直接命中本地 KV，根本无法证明 LMCache 参与了。

### 8.3 请求 B：相同前缀，不同尾部

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

再次观察：

```bash
grep -E "Stored|Retrieved" /workspace/results/lmcache_server.log | tail -40
```

预期：

- 出现 `Retrieved`，说明共享前缀从 LMCache 回灌；
- 新尾部可能继续出现 `Stored`；
- 回答内容正常。

### 原理：为什么尾部可以不同

LMCache 复用的是**连续 token 前缀**，不是要求整道题完全相同：

```text
请求 A = [共享长前缀][问题 A]
请求 B = [共享长前缀][问题 B]
         <--- 可复用 ---> <- 新计算 ->
```

## 9. 第六步：用流式请求粗测 TTFT

下面的脚本会完整读取响应，同时记录第一个非空 token 的到达时间。保存为 `/workspace/scripts/verify_lmcache_ttft.py`：

```python
import json
import time

import requests

URL = "http://127.0.0.1:8998/v1/completions"
FLUSH_URL = "http://127.0.0.1:8998/flush_cache"
MODEL = "Qwen3.8-27B"
PREFIX = "KV cache reuse requires a sufficiently long shared prefix. " * 300


def run_request(prompt):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": 32,
        "temperature": 0,
        "stream": True,
    }
    started = time.perf_counter()
    ttft = None
    pieces = []

    with requests.post(URL, json=payload, stream=True, timeout=(10, 600)) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            text = event["choices"][0].get("text", "")
            if text and ttft is None:
                ttft = time.perf_counter() - started
            if text:
                pieces.append(text)

    return ttft, "".join(pieces)


a_ttft, a_text = run_request(PREFIX + "\nSummarize the topic in one sentence.")
print(f"A cold TTFT: {a_ttft:.4f}s")

requests.post(FLUSH_URL, timeout=30).raise_for_status()

b_ttft, b_text = run_request(PREFIX + "\nGive one short title for this text.")
print(f"B warm TTFT: {b_ttft:.4f}s")

print("A output:", a_text[:200])
print("B output:", b_text[:200])
```

执行：

```bash
python /workspace/scripts/verify_lmcache_ttft.py \
  2>&1 | tee /workspace/results/lmcache_ttft_smoke.log
```

这只是正确性冒烟测试，不是正式 benchmark。单次 TTFT 会受调度、cache 写入、机器状态影响，不要在单次数据上得出精确提速倍数。

## 10. 常见现象怎么处理

| 现象 | 优先判断 |
|---|---|
| `lmcache server` 启动失败 | 查看是否还有缺包或端口占用 |
| SGLang 报 connection refused | server 未启动、host/port 不一致或进程不在同一网络命名空间 |
| 出现 CUDA IPC/CUDA context 错误 | 走到了 CUDA 专用路径，检查 unified connector 和 NPU device 选择 |
| block/chunk/shape 不匹配 | N 的取值或 SGLang KV layout 不符合预期 |
| 只有部分 rank 注册成功 | TP worker 初始化和 local rank/device 映射 |
| 第一次没有 Stored | connector 没启用、请求太短或请求未走到保存路径 |
| 第二次没有 Retrieved | 没 flush、本地缓存抢先命中、前缀未按 chunk 对齐、L1 已淘汰 |
| TTFT 没有下降 | 命中率低、传输/调度开销大、只跑了一次或测量负载不合适 |
| 多轮时才报错 | 优先检查 Qwen3.8 MAMBA recurrent state，而不是普通 attention KV |

## 11. 正式启动与错误反馈需要保留什么

请保存：

```text
/workspace/results/lmcache_server.log
/workspace/results/sglang_mp.log
/workspace/results/lmcache_ttft_smoke.log
/workspace/scripts/lmcache_mp.yaml
```

同时记录：

- attention block size N；
- SGLang 版本；
- LMCache 和 LMCache-Ascend commit；
- 最早出现的完整异常栈；
- A/B 两次请求是否分别 Stored/Retrieved。

这些材料决定下一步是：

1. 直接使用现有实现，进入正式 benchmark；
2. 只改配置；
3. 修改上游 SGLang connector；
4. 修改 LMCache-Ascend 的 NPU layout/块传输；
5. 单独处理 Qwen3.8 混合 MAMBA 状态路径。

## 12. 本阶段的本质

前两步测试的是“零件”：

```text
接口是否存在 + NPU 原生跨进程传输是否正确
```

本阶段测试的是“整车”：

```text
真实 SGLang 请求是否能在正确时间调用缓存，
KV 是否能保存并在后续请求中可靠回灌，
同时不破坏模型输出和 decode 生命周期。
```

因此现在采用“先跑、再按真实错误适配”的方法最稳妥：能不改代码就不改，必须改时也能精确知道应该改哪一层。
