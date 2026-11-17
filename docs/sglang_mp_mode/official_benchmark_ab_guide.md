# 官方自带 Benchmark 工具做 A/B 对比测试：TTFT 与吞吐测量手把手教程

> 适用前提：Qwen3.8-27B 已按 [hybrid_models_runbook.md](hybrid_models_runbook.md) 跑通
> （服务日志里有 `unified path engaged (components=['FULL', 'MAMBA'], mamba=True)`）。
>
> 本教程全部使用 **SGLang 官方自带的 benchmark 工具 + 原生参数**，不写任何自定义
> 脚本；所有命令照抄即可。
>
> A/B 的含义：同一台机器、同一份模型、同一组启动参数，**唯一区别是服务端加不加
> LMCache 两个参数**。A = 裸 SGLang（对照组），B = LMCache 适配服务（实验组）。

---

## 一、这套测试要回答的三个问题（先看目标）

| 实验 | 官方数据集 | 模拟的真实场景 | 重点指标 | 回答的问题 |
|---|---|---|---|---|
| 实验一 | `generated-shared-prefix`（gsp） | 多租户/检索系统共享同一段超长系统提示词 | **TTFT P50/P90**、缓存命中率 | LMCache 跨进程缓存让 TTFT 快多少（核心） |
| 实验二 | `random` | 完全随机、互不重复的冷缓存稳态负载 | **吞吐、TPOT** | LMCache 的 store 动作拖慢推理多少（开销体检） |
| 实验三 | gsp 多轮（`--gsp-num-turns`） | 多轮对话 | TTFT / TPOT | mamba 状态复用带来多少收益（hybrid 特有） |

三个实验互补：实验一证明"收益"，实验二证明"开销可控"，实验三证明"hybrid 专属价值"。

---

## 二、用的官方工具是什么

- 入口命令：`python -m sglang.benchmark.serving`，即 SGLang 官方标准 benchmark
  入口（源码在 sglang 仓 [serving.py](file:///d:/proj/sglang/python/sglang/benchmark/serving.py)）。
- **旧入口 `python -m sglang.bench_serving` 已废弃**，报错/无输出时先检查自己是不是
  用了旧名字。本教程全部用新入口。
- 它原生支持我们需要的全部能力：合成共享前缀数据集（**本地生成、不联网**）、随机
  数据集、完整的指标报告（TTFT / TPOT / ITL / 吞吐 / 缓存命中率）、实验前清缓存的
  `--flush-cache`、命中统计 `--cache-report`、逐请求明细 `--output-details`。
- benchmark 客户端在**容器里跑**（sglang 已装好），自身不占 NPU，另开一个终端即可。

---

## 三、实验前准备：两套服务怎么切换

**核心机制（一句话）：两套服务不需要重装任何包，只差启动命令末尾两行参数；
切换 = 停旧起新。** 每次切换务必 `pkill` + 等 10 秒 + 确认显存释放，否则新服务会
因 `avail mem` 不足起不来。

### 3.1 启动 A 侧（裸 SGLang，对照组）

就是 runbook 第 3 步的启动命令，**去掉最后两行 LMCache 参数**：

```bash
pkill -f sglang.launch_server; sleep 10
export PYTHONHASHSEED=0
python -m sglang.launch_server \
  --model-path /data/models/Qwen3.8-27B \
  --served-model-name Qwen3.8-27B \
  --host 0.0.0.0 --port 8998 \
  --tp-size 8 --dtype bfloat16 \
  --context-length 8192 --chunked-prefill-size 4096 \
  --mem-fraction-static 0.85 --trust-remote-code \
  --mamba-ssm-dtype bfloat16 \
  2>&1 | tee /workspace/results/sglang_qwen38_bare_A.log
```

### 3.2 启动 B 侧（LMCache 适配服务，实验组）

同一个命令，**末尾多这两行**：

```bash
  --enable-lmcache \
  --lmcache-config-file /workspace/scripts/lmcache_mp.yaml \
```

日志重定向到 `/workspace/results/sglang_qwen38_lmcache_B.log`，其余完全一样。

### 3.3 lmcache server 保持运行（跑 B 侧前检查）

```bash
ps aux | grep "lmcache server" | grep -v grep
```

没在跑就用 runbook 第 2 步的命令起（`--l1-size-gb 20` 那份）。**跑 A 侧时它开不开
都无所谓**：A 侧没加载 lmcache 模块，两边互不干扰（这也是对照组纯净性的保证）。

### 3.4 就绪判定（30 秒看一眼）

```bash
tail -5 /workspace/results/sglang_qwen38_bare_A.log     # 看到 The server is fired up and ready to roll!
tail -5 /workspace/results/sglang_qwen38_lmcache_B.log  # 同上；B 侧另看两行：
grep -E "unified path engaged|unified driver ready" /workspace/results/sglang_qwen38_lmcache_B.log
```

---

## 四、实验一：共享前缀负载 —— TTFT 核心对比

### 4.1 这个数据集是什么

`--dataset-name generated-shared-prefix`（下文简称 gsp）是官方内置的**合成共享前缀**
数据集：它用 tokenizer 随机拼出 N 组 2048 token 的"系统提示词"，每组下面挂 M 个
128 token 的"问题"，一共 N×M 个请求。

关键特性（这三点保证了 A/B 对比公平）：

1. **纯本地合成、不联网**：随机英文词拼句，走网络下载的担心不存在；
2. **磁盘缓存**：生成一次后缓存到 `~/.cache/sglang/benchmark/gen_shared_prefix_*.pkl`，
   **同一组 `--gsp-*` 参数 + 同一个 `--seed` 的两次运行，加载的是同一份数据**
   （逐字节一致）；
3. **seed 全局生效**：即使缓存被绕过（如多轮），官方代码里 `random.seed(args.seed)` +
   `np.random.seed(args.seed)` 也保证同 seed 两次生成内容一致。

### 4.2 四个 run 的完整命令（照抄；每条只改输出文件名和最后一行的 flush 开关）

四条命令大部分相同，公共部分如下（每次都要带上，别偷懒省略）：

```bash
export HF_HUB_OFFLINE=1
python -m sglang.benchmark.serving \
  --backend sglang \
  --host 127.0.0.1 --port 8998 \
  --model Qwen3.8-27B \
  --tokenizer /data/models/Qwen3.8-27B \
  --dataset-name generated-shared-prefix \
  --gsp-num-groups 16 \
  --gsp-prompts-per-group 8 \
  --gsp-system-prompt-len 2048 \
  --gsp-question-len 128 \
  --gsp-output-len 256 \
  --seed 42 \
  --cache-report \
  --output-details
```

然后按下面的顺序执行 4 次，前 3 次照抄公共部分，第 4 次多一行 `--flush-cache`：

| 步 | 服务 | 追加参数 | 输出文件 | 这一跑在测什么 |
|---|---|---|---|---|
| **A1** | A（裸） | `--output-file /workspace/results/bench_qwen38_A1_gsp_cold.jsonl` | 同上 | 全冷基线：TTFT、吞吐的参照值 |
| **A2** | A（裸） | `--flush-cache` + `--output-file /workspace/results/bench_qwen38_A2_gsp_flushed.jsonl` | | 验证 flush 把 radix 清干净（应和 A1 一样冷） |
| **B1** | B（LMCache） | `--output-file /workspace/results/bench_qwen38_B1_gsp_cold.jsonl` | | 冷跑 + 把前缀写进 lmcache server（日志大量 Stored） |
| **B2** | B（LMCache） | `--flush-cache` + `--output-file /workspace/results/bench_qwen38_B2_gsp_hot.jsonl` | | **核心**：SGLang 树被清，靠 lmcache server 命中（日志大量 Retrieved） |

B2 具体命令示例（注意最后一行 `--flush-cache` 位置）：

```bash
export HF_HUB_OFFLINE=1
python -m sglang.benchmark.serving \
  --backend sglang \
  --host 127.0.0.1 --port 8998 \
  --model Qwen3.8-27B \
  --tokenizer /data/models/Qwen3.8-27B \
  --dataset-name generated-shared-prefix \
  --gsp-num-groups 16 \
  --gsp-prompts-per-group 8 \
  --gsp-system-prompt-len 2048 \
  --gsp-question-len 128 \
  --gsp-output-len 256 \
  --seed 42 \
  --cache-report \
  --output-details \
  --flush-cache \
  --output-file /workspace/results/bench_qwen38_B2_gsp_hot.jsonl
```

### 4.3 为什么这么设计（这套逻辑看懂就毕业了）

- **A2 与 B2 是唯一的核心对比**：两者 SGLang 进程内的缓存都被 `--flush-cache` 清空，
  站在同一起跑线；唯一差别是 B2 能从**独立的 lmcache server 进程**把 KV + mamba 状态
  拿回来。A2 和 B2 的 TTFT 差值 = LMCache 的净收益。
- `--flush-cache` 是官方 benchmark 参数：开跑前自动调一次服务端 `/flush_cache`，
  **只清 SGLang 自己的 radix tree，不动 lmcache server 里的数据**。
  （runbook 5.2 已实测过这个行为：flush 后热请求仍然走 Retrieved。）
- A1/A2 几乎一样 ⇒ 证明裸 SGLang 的 radix 确实被 flush 清掉了、对比是干净的；
  B1/B2 的对比 ⇒ 同一服务 flush 前后，TTFT 下降了多少。

### 4.4 结果读法（终端最后的分块报告）

每条命令跑完，终端会打印分块报告，只需要抄三个数：

```text
============ Serving Benchmark Result ============
...
Total token throughput (tok/s):      2653.21          ← 实验二才重点看
...
------------------Time to First Token------------------
Mean TTFT (ms):     812.34                            ← 抄这三个
Median TTFT (ms):   795.10
P90 TTFT (ms):      934.22                            ← 对比首选
...
------------------Cache Hit Details------------------   ← 只有 B2 这条才有大数
Cache hit rate:     96.2%
```

- **TTFT 块**：拿 `P90 TTFT` 做对比首选指标（比均值更能反映尾延迟）；
- **Cache Hit Details 块**：`--cache-report` 加的，B2 的 hit rate 应显著高于 A2；
- **`--output-details` 的 JSONL 文件**：每请求一行，含各自的 prompt_len / output_len /
  ttft / tpot，事后想画分布图或查某个异常请求都能用，本次实验先不深挖。

### 4.5 与 lmcache server 日志对账（B 侧数据的铁证）

```bash
grep -E "Stored|Retrieved" /workspace/results/lmcache_server_unified.log | tail -30
```

| 时间段 | 应看到 | 含义 |
|---|---|---|
| B1 期间 | 大量 `Stored ... tokens` | 冷请求把前缀 KV/mamba 状态写进 server |
| B2 期间 | 大量 `Retrieved ... tokens` | flush 后靠 server 命中回灌 |
| A1/A2 期间 | **零新增** | 对照组根本不连 lmcache，日志纹丝不动 |

### 4.6 填表（建议抄进你的实验记录）

| run | P50 TTFT (ms) | P90 TTFT (ms) | Cache hit rate | 备注 |
|---|---|---|---|---|
| A1 冷 | | | ~0% | 基线 |
| A2 flush 后 | | | ≈A1（证明 flush 生效） | |
| B1 冷 | | | | store 期间 |
| B2 flush 后 | | | **越高越好** | **核心数** |

**一句话结论模板**：
"在 2048-token 共享前缀场景下，flush 后复测：裸 SGLang 的 P90 TTFT 为 X ms，
LMCache-Ascend 为 Y ms，降低 Z%（TTFT 快 Z%）。"

---

## 五、实验二：随机负载 —— 吞吐基线 + store 开销体检

随机数据集的每个 prompt 都是独立随机的 1024 token，前缀互不重复 ⇒ B 侧永远命中
不了缓存，只会白做 store。A/B 的**吞吐和 TPOT 差值 = LMCache store 的纯开销**。
这是回答"LMCache 会不会拖慢正常推理"的唯一靠谱方法。

A、B 各跑一遍（两边命令一样，改输出文件名即可；现在服务切到哪边就跑哪边）：

```bash
export HF_HUB_OFFLINE=1
python -m sglang.benchmark.serving \
  --backend sglang \
  --host 127.0.0.1 --port 8998 \
  --model Qwen3.8-27B \
  --tokenizer /data/models/Qwen3.8-27B \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 512 \
  --num-prompts 300 \
  --seed 42 \
  --output-file /workspace/results/bench_qwen38_A_random.jsonl
```

B 侧同名把输出文件改成 `bench_qwen38_B_random.jsonl`。

读法：抄 `Total token throughput (tok/s)`、`Output token throughput (tok/s)`、
`Median TPOT (ms)`、`Mean TTFT (ms)` 四个数填表：

| run | 总吞吐 (tok/s) | 输出吞吐 (tok/s) | P50 TPOT (ms) | Mean TTFT (ms) |
|---|---|---|---|---|
| A random | | | | |
| B random | | | | |

判读原则：B 比 A 的吞吐/TPOT 掉 1~3% 属正常（store 是异步搬运）；若掉幅超过
10%，说明 store 通道在抢带宽/串行阻塞，值得排查（贴日志回来）。

可选变体（同样是官方参数，一行一改）：

```bash
  --max-concurrency 200          # 模拟上层限流并发（否则默认全部请求瞬间发出）
  --request-rate 4               # 按泊松过程每秒 4 个请求地送，测低负载 TTFT
```

---

## 六、实验三（可选）：多轮 —— mamba 状态复用收益

hybrid 模型的卖点之一是 mamba 状态也能缓存：多轮对话第 2 轮起，B 侧可以直接
回灌上一轮的 FULL KV + MAMBA 状态，而不是重新 prefill 全部历史。

### 6.1 官方 gsp 多轮版（推荐）

把实验一的公共参数改四处：

```bash
  --gsp-num-groups 16 \
  --gsp-prompts-per-group 4 \
  --gsp-num-turns 4 \
  --gsp-output-len 128 \
```

注意：`--gsp-num-turns > 1` 时官方代码会**绕过磁盘缓存**、每次重新合成数据，但
同 seed 内容仍然一致，A/B 依然公平。A、B 各跑一遍对比 TTFT/TPOT。

### 6.2 hicache 套件（容器里有 sglang 源码目录时才做）

SGLang 仓库 `benchmark/hicache/` 下还有一套官方多轮 benchmark。容器里如果是
源码 clone（不是纯 pip 装），可以：

```bash
cd <sglang源码目录>/benchmark/hicache
python bench_multiturn.py \
  --host 127.0.0.1 --port 8998 \
  --model-path /data/models/Qwen3.8-27B \
  --api-format openai \
  --num-clients 32 --num-rounds 4 \
  --request-length 512 --output-length 64 \
  --log-file /workspace/results/bench_qwen38_B_multiturn.jsonl --tag B
```

找不到该目录、或报 `No module named 'sglang.bench_serving'` 就**直接跳过**——
6.1 已经覆盖了多轮收益的测量，hicache 只是锦上添花。

---

## 七、常见坑清单（出问题先查这里）

| # | 症状 | 原因与解决 |
|---|---|---|
| 1 | `python -m sglang.bench_serving` 无输出/报错 | 旧入口已废弃，必须用 `python -m sglang.benchmark.serving` |
| 2 | 报 `Couldn't find file ... tokenizer` 或想联网下载 tokenizer | 离线环境必须 `--tokenizer /data/models/Qwen3.8-27B` 指本地目录，且 `export HF_HUB_OFFLINE=1` |
| 3 | 请求回 404 / model 不存在 | `--model` 的值必须等于服务端的 `--served-model-name`（即 `Qwen3.8-27B`） |
| 4 | connection refused | `--port 8998 --host 127.0.0.1` 要和当前服务的端口对上；换过服务要等日志 ready 再跑 |
| 5 | B2 命中率明显低于预期（<80%） | 20GB L1 放不下全部前缀被 LRU 淘汰。看 lmcache server 日志的淘汰/evict 信息；对策：减少 `--gsp-num-groups`（如 16→8），或重启 lmcache server 时把 `--l1-size-gb 20` 调大（如 40）后重跑 B1、B2 |
| 6 | 忘了 flush、或没有 `--flush-cache` | 没 flush 时 SGLang 自带 radix 也会命中共享前缀，A2 就不"冷"了，对比失真 |
| 7 | 改了任何一个 `--gsp-*` 或 `--seed` | 磁盘缓存 key 变了，会生成**另一份数据**，与之前 run 的"热"关系失效、命中率掉 |
| 8 | `--tokenize-prompt` 与 gsp 同用直接断言报错 | 官方限制，gsp 数据集禁止该参数，删掉 |
| 9 | 换服务时新服务起不来、`avail mem` 很小 | 旧服务没死干净：重新 `pkill -f sglang.launch_server`，`sleep 10` 后 `npu-smi info` 确认 8 卡显存都回到 ~29GB/卡 |
| 10 | gsp 请求数不是 1024 | gsp 请求总数 = `--gsp-num-groups × --gsp-prompts-per-group`，与 `--num-prompts` 无关（后者只对 random/sharegpt 等生效） |
| 11 | 两个 benchmark 并发跑、数字互相污染 | 一次只跑一个客户端；跑的时候别在服务终端里敲交互命令 |
| 12 | A 侧日志里搜不到 Stored/Retrieved | 正常现象：A 侧没加载 lmcache 模块，正是对照组的纯净性证明，不是故障 |

---

## 八、最终汇总表（写进实验报告）

| 实验 | 指标 | A（裸 SGLang） | B（LMCache-Ascend） | 变化 |
|---|---|---|---|---|
| gsp flush 后（A2 vs B2） | P90 TTFT (ms) | | | |
| gsp flush 后（A2 vs B2） | Cache hit rate | | | |
| random 稳态（A vs B） | 总吞吐 (tok/s) | | | |
| random 稳态（A vs B） | P50 TPOT (ms) | | | |
| 多轮（可选） | P90 TTFT (ms) | | | |

结论三句话套路：

1. 共享前缀场景，LMCache-Ascend 把 P90 TTFT 从 X 降到 Y（-Z%），收益来自跨进程
   lmcache server 命中（日志 Retrieved 佐证）；
2. 冷缓存随机负载下吞吐/TPOT 基本持平（-W%），store 开销可接受；
3. 多轮场景（可选）TTFT 进一步受益于 mamba 状态回灌。

三组数据齐了，"LMCache 值不值"就是一句话的事。