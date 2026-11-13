# A/B 性能对比手册：纯 SGLang vs SGLang + LMCache-Ascend MP

> 目标：同一台 910B4、同一个模型（Qwen2.5-7B-Instruct TP4）、同一路径的请求，
> 测出「带 LMCache」相对「纯 SGLang」的 TTFT 收益，并拿到可写进报告的证据链。
>
> 配套脚本：[sglang_ab_ttft.py](../../benchmark/sglang_mp/sglang_ab_ttft.py)（本仓新增，纯标准库）

---

## 0. 先回答两个关键问题

### 0.1 本仓库 benchmark 目录有直接能用的吗？

| 目录 | 内容 | 结论 |
|---|---|---|
| `benchmark/v1/rag/` | RAG 场景压测（rag.py / online.sh / offline.sh） | ❌ **vLLM-only**——README 写明 "tested against vllm-ascend v0.18.0"，走 vLLM + CacheBlend 路径 |
| `benchmark/cacheblend/` | CacheBlend 挖洞测试 | ❌ 同为 vLLM 路径 |
| `benchmark/sglang_mp/sglang_ab_ttft.py` | 本文新增的 A/B 对比脚本 | ✅ SGLang 专用，本次主工具 |

并发压力测试另有官方工具：容器里 SGLang 自带的 `sglang.benchmark.serving`
（用法见第 5 节）。

### 0.2 两种服务怎么切换？要 pip install 来回装吗？

**不需要。零安装、零卸载。** 两种模式用的是**同一份 SGLang、同一份 LMCache、
同一份 LMCache-Ascend 插件**——区别只在启动命令的最后两个参数：

| | A 组（纯 SGLang） | B 组（LMCache MP） |
|---|---|---|
| 启动参数 | 基础参数 | 基础参数 + `--enable-lmcache --lmcache-config-file ...` |
| 其余一切（TP/模型/端口） | 完全相同 | 完全相同 |
| pip 环境 | 完全不动 | 完全不动 |

**为什么不用重装**（原理，理解了以后换任何开关都有底气）：

```text
A 组启动（不带 --enable-lmcache）：
  registry 的 enable_lmcache 分支不命中
    → 用默认 UnifiedRadixCache（SGLang 自带）
    → lmc_radix_cache.py 根本不会被 import
    → import lmcache 不发生 → lmcache_ascend 插件也不加载
  ⇒ A 组是真·纯净：LMCache 全家连 import 都没有，零开销

B 组启动（带 --enable-lmcache）：
  registry 命中 enable_lmcache 分支
    → import lmc_radix_cache → import lmcache → 插件自动激活
    → 兼容补丁装上 → LMCRadixCache + 连接器 + NPU IPC 注册
```

换句话说：**那两个参数就是总开关，开关后面的整条链路（包括我们的补丁）
都挂在它后面。** 切换 = 重启服务时带不带这两个参数，仅此而已。

lmcache server（终端 A 那个 5556 进程）在 A 组测试期间**可以继续跑着**——
A 组没人连它，不占 NPU 计算资源（每卡那 ~148MB 句柄内存也在 B 组重启时
才会重新注册），留着省事。

---

## 1. 测试前检查（2 分钟）

```bash
# 1) 服务还在、是哪种模式（确认当前状态）
curl -s http://127.0.0.1:8998/get_server_info | python -m json.tool | grep -i lmcache
# 有 "enable_lmcache": true  → 当前是 B 组
# 没这行 / false            → 当前是 A 组

# 2) lmcache server 活着（B 组需要）
curl -s http://127.0.0.1:8080/health 2>/dev/null && echo "server OK" || echo "server 未响应"

# 3) 把脚本放进容器（一次性）
#    Windows 本机：git push 后容器里 git pull；或直接 docker cp：
#    docker cp sglang_ab_ttft.py <容器>:/workspace/scripts/
```

---

## 2. 手把手完整流程

### 第 1 步：起 A 组服务（纯 SGLang）

```bash
# 1) 停掉现有服务（无论现在跑的是哪种）
pkill -f sglang.launch_server
sleep 10
npu-smi info | head -40        # 确认 sglangschedul 进程消失

# 2) 启动 A 组：基础参数原样，【不带】lmcache 两参数
export PYTHONHASHSEED=0
python -m sglang.launch_server \
  --model-path /data/models/Qwen2.5-7B-Instruct \
  --served-model-name Qwen2.5-7B-Instruct \
  --host 0.0.0.0 --port 8998 \
  --tp-size 4 --dtype bfloat16 \
  --context-length 8192 --chunked-prefill-size 4096 \
  --mem-fraction-static 0.85 --trust-remote-code \
  2>&1 | tee /workspace/results/sglang_pure_A.log
```

### 第 2 步：跑 A 组基准

```bash
# 新终端（服务终端要保持运行）
cd /workspace/scripts
python sglang_ab_ttft.py --mode pure --repeat 5 --load-total 32 --load-conc 8
#            ↑ 脚本会自己等就绪；--mode 只是给结果打标签，不影响测试内容
```

### 第 3 步：切 B 组服务（加两个参数重启）

```bash
# 1) 停 A 组
pkill -f sglang.launch_server
sleep 10
npu-smi info | head -40        # 确认清干净

# 2) lmcache server 若没跑，先起（终端 A）
export PYTHONHASHSEED=0
lmcache server --host 127.0.0.1 --port 5556 --http-port 8080 \
  --chunk-size 256 --l1-size-gb 20 --eviction-policy LRU \
  --supported-transfer-mode lmcache_driven \
  2>&1 | tee /workspace/results/lmcache_server_ab.log

# 3) 启动 B 组：与 A 组逐字相同，只在末尾加两行
export PYTHONHASHSEED=0
python -m sglang.launch_server \
  --model-path /data/models/Qwen2.5-7B-Instruct \
  --served-model-name Qwen2.5-7B-Instruct \
  --host 0.0.0.0 --port 8998 \
  --tp-size 4 --dtype bfloat16 \
  --context-length 8192 --chunked-prefill-size 4096 \
  --mem-fraction-static 0.85 --trust-remote-code \
  --enable-lmcache \
  --lmcache-config-file /workspace/scripts/lmcache_mp.yaml \
  2>&1 | tee /workspace/results/sglang_lmcache_B.log
```

### 第 4 步：跑 B 组基准 + 取证

```bash
cd /workspace/scripts
python sglang_ab_ttft.py --mode lmcache --repeat 5 --load-total 32 --load-conc 8

# B 组的加分项：顺手抓 server 侧证据（A/B 报告的完整证据链）
grep -E "Stored|Retrieved" /workspace/results/lmcache_server_ab.log | tail -30
```

### 第 5 步：对比两份 JSON

```bash
ls -t /workspace/results/ab_*.json | head -2
# 两个文件：ab_pure_*.json 和 ab_lmcache_*.json，各字段含义见第 4 节
```

---

## 3. 脚本三阶段在测什么（读数前必看）

| 阶段 | 做法 | A 组预期 | B 组预期 |
|---|---|---|---|
| **1 REFRESH** | 每次先 `flush_cache` 再发同一长前缀请求 ×N | N 个 TTFT **基本持平**（本地缓存被清，每次都全量 prefill） | 第 1 个高（冷+Stored），第 2..N 个**大幅下降**（本地清了但 LMCache 还在 → Retrieved） |
| **2 RADIX** | 不 flush 连发 ×N | 都很快（本地 RadixCache 命中） | 同样快。**若这里反而慢 → 接线有问题，先排查再谈性能** |
| **3 LOAD** | 唯一前缀 ×32 并发 8 | 吞吐基线 | 吞吐与 A 接近（LMCache 收益在 TTFT 不在吞吐） |

**headline 指标 = 阶段 1 里：A 组 median TTFT vs B 组第 2..N 次的 median TTFT。**

为什么必须 flush：SGLang 本地 RadixCache 命中时第二次请求秒回，但那是
进程内存里的缓存，重启即失、容量受限——测它等于没测 LMCache。flush 之后，
A 组"无路可退"，B 组"还有 LMCache"，两者的差距才是 LMCache 的真实价值。

---

## 4. 结果怎么读（字段对照 + 报告模板）

每份 JSON 的关键字段：

```text
phase1_refresh.stats.median / mean / p95   ← 核心：冷路径 TTFT 分布
phase1_refresh.ttft_samples[0]             ← B 组的第 1 个采样（真冷，Stored）
phase1_refresh.ttft_samples[1:]            ← B 组的热采样（Retrieved）
phase2_radix.stats.median                  ← 健全性检查（两组都应 < 1s 量级）
phase3_load.req_per_s                      ← 原始吞吐（两组应接近）
```

报告模板（把数填进去即可）：

```text
┌─────────────────────────────────────────────────────────────┐
│ SGLang + LMCache-Ascend MP 性能对比（910B4 × TP4）           │
│ 模型 Qwen2.5-7B-Instruct，前缀 ~4800 chars，chunk 256       │
├──────────────────────┬──────────────┬───────────────────────┤
│ 指标                 │ A 纯 SGLang  │ B + LMCache MP        │
├──────────────────────┼──────────────┼───────────────────────┤
│ REFRESH TTFT median  │   x.xxx s    │   y.yyy s（第2..N次） │
│ REFRESH TTFT p95     │   x.xxx s    │   y.yyy s             │
│ TTFT 下降幅度        │      —       │   (x-y)/x = NN.N%     │
│ RADIX TTFT median    │   z s        │   z' s（健全性 ✓）    │
│ LOAD 吞吐 req/s      │   r          │   r'（预期 r≈r'）     │
├──────────────────────┴──────────────┴───────────────────────┤
│ server 侧证据：Stored N tokens in 0.00xs / Retrieved ...    │
└─────────────────────────────────────────────────────────────┘
```

**证据链三件套**（缺一不可，防"空转"质疑）：

1. TTFT 差值（客户端测得）
2. server 日志的 `Stored` / `Retrieved`（缓存真的被用了）
3. 两篇日志里 `split into 28 per-layer views`（本仓补丁在工作）

---

## 5. 并发压测（可选，用 SGLang 自带工具再压一轮）

`sglang.benchmark.serving` 新入口（旧 `bench_serving` 已废弃）。语料用
[qwen25_7b_first_light.md 11.2 节](qwen25_7b_first_light.md)的自造 ShareGPT
（`local_corpus.json`），两种模式各跑一次，参数逐字相同：

```bash
python -m sglang.benchmark.serving \
  --backend sglang \
  --host 127.0.0.1 --port 8998 \
  --model-name Qwen2.5-7B-Instruct \
  --dataset-name sharegpt \
  --dataset-path /workspace/scripts/local_corpus.json \
  --num-prompts 32 --request-rate 4 \
  --output-file /workspace/results/bench_{A或B}.json \
  2>&1 | tee /workspace/results/bench_{A或B}.log
```

看 `median TTFT (ms)`、`Output token throughput (tok/s)`、`Request throughput (req/s)`。
离线环境务必带 `--dataset-path`，否则静默回退 HF 下载并抛离线异常。

---

## 6. 公平性检查清单 + 常见坑

| # | 检查项 | 原因 |
|---|---|---|
| 1 | 两组启动命令**逐字 diff**（只差 lmcache 两行） | 任何其它差异（TP/上下文/mem-fraction）都会污染结论 |
| 2 | 每次切组前 `pkill` + npu-smi 确认进程消失 | 上次实验的服务还在跑 → 显存不够 → 数据失真 |
| 3 | 阶段 1 每次采样前 flush + sleep 1s | flush 未落地就发请求 = 测了本地 radix |
| 4 | B 组先空跑一次长前缀再正式采样 | 确认 Stored 已发生（否则第 1 个采样也污染 stats） |
| 5 | 两组同时段连跑（别隔一天） | NPU 温度/其它租户负载会变 |
| 6 | `--repeat` ≥ 5，报 median 不报单次 | 单次抖动太大，median 稳健 |
| 7 | A 组测试时 lmcache server 留着可以，但**别让 B 组忘了确认它活着** | B 组连不上 server 会直接启动失败（不至于静默，但浪费一轮） |
| 8 | 报告同时给 TTFT 和吞吐 | 只报吞吐会掩盖 LMCache 的真实价值（它本来就不提吞吐） |

常见坑：

| 症状 | 原因 | 解法 |
|---|---|---|
| B 组阶段 2 反而比 A 组慢 | 接线异常（lookup 未命中路径开销） | 对照 [qwen25_7b_first_light.md](qwen25_7b_first_light.md) 三层证据检查表排查 |
| B 组阶段 1 全部和 A 组一样 | LMCache 没生效（server 没起 / Stored 没发生） | grep server 日志；确认 `--lmcache-config-file` 路径 |
| TTFT 抖动巨大 | 并发残留 / NPU 4~7 那 2853MB 残留影响调度 | 重启容器清残留后重测 |
| 阶段 1 第 1 次和第 2 次一样快 | flush 没生效（检查 flush_cache 返回） | 手动 curl flush 后 sleep 2 再发 |

---

## 7. 与既有文档的关系

| 文档 | 分工 |
|---|---|
| 本文 | A/B **切换方法 + 对比流程 + 读数方法** |
| [qwen25_7b_first_light.md](qwen25_7b_first_light.md) 11 节 | 单模式下的问答/压测（B 组内部的详测） |
| [call_flow_before_after_adaptation.md](call_flow_before_after_adaptation.md) | 链路原理（读懂阶段 2 为什么能检查接线） |
| `benchmark/v1/rag/` | vLLM 侧的 RAG 基准（与本 A/B 无关，勿混用） |
