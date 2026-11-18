# 官方自带 Benchmark 工具做 A/B 对比测试：TTFT 与吞吐测量手把手教程

> 适用前提：Qwen3.8-27B 已按 [hybrid_models_runbook.md](hybrid_models_runbook.md) 跑通
> （服务日志里有 `unified path engaged (components=['FULL', 'MAMBA'], mamba=True)`）。
>
> 本教程全部使用 **SGLang 官方自带的 benchmark 工具 + 原生参数**，不写任何自定义
> 脚本；所有命令照抄即可。
>
> A/B 的含义：同一台机器、同一份模型、同一组启动参数，**唯一区别是服务端加不加
> LMCache 两个参数**。A = 裸 SGLang（对照组），B = LMCache 适配服务（实验组）。
>
> **总流程：先功能验证（第四节），全绿之后才做性能对比（第五节起）——
> 功能不对，性能数字全是假的。**

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

## 四、性能测试前的功能验证（必做）

### 4.1 为什么必须先验功能，再测性能

LMCache 回灌有一个恶劣特性：**数据搬错了不会崩，只会让输出胡言乱语**。
KV 或 mamba 状态回灌错误时，模型照样一个 token 一个 token 地生成，benchmark
照样打印漂亮的 TTFT——但生成的已经是垃圾。跳过功能验证直接跑性能，拿到的
是"很快地输出垃圾"的数字，毫无意义。

功能验证回答三个层层递进的问题：

1. **服务活着吗**：问一句话，能吐出人话吗？（见 4.2）
2. **真的走了本仓库和 LMCache 吗**：启动日志和缓存日志里找证据。（见 4.3、4.4）
3. **搬过去的数据是对的吗**：冷热两次同请求，答案必须一致。（见 4.5）

三问全过，才有资格进第五节谈性能。

### 4.2 第 1 步：吐人话测试（A、B 两侧各一次）

服务 ready 后，发一条最普通的聊天请求：

```bash
curl -s http://127.0.0.1:8998/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.8-27B",
    "messages": [{"role": "user", "content": "用一句话介绍你自己"}],
    "max_tokens": 64,
    "temperature": 0
  }'
```

判定：返回 JSON 的 `content` 字段是通顺的人话（不乱码、不无限重复）。

**A 侧起来后同样跑一次**（这是隔离问题归属的关键）：

| A 侧（裸） | B 侧（LMCache） | 结论 |
|---|---|---|
| 人话 | 人话 | 两套服务本身都正常 ✅ |
| 人话 | 乱码/重复/空 | LMCache 适配引入的问题 ❌ 先修它，别测性能 |
| 乱码 | 乱码 | 服务/模型本身的问题，与 LMCache 无关 |

### 4.3 第 2 步：证明"走了本仓库"（B 侧启动日志五连证据）

本仓库的适配方式是运行时给 SGLang 打补丁，补丁装没装上、hybrid 模型有没有
被分诊到统一路径，日志会说话：

```bash
grep -E "Installed SGLang MP connector|Installed unified KV adapter|unified path engaged|unified driver ready|Tree cache initialized" \
  /workspace/results/sglang_qwen38_lmcache_B.log
```

应看到五行（顺序大致如下）：

| 日志行 | 含义 | 谁打的 |
|---|---|---|
| `Installed SGLang MP connector stacked-KV compat patch` | 堆叠张量拆分补丁已生效 | mp_connector_compat.py |
| `Installed unified KV adapter stacked-KV compat patch` | unified adapter 的同款补丁已生效 | mp_connector_compat.py |
| `unified path engaged (components=['FULL', 'MAMBA'], mamba=True)` | hybrid 模型被分诊进统一路径 | unified_diversion.py → unified_linker.py |
| `LMCache-Ascend unified driver ready (components=[...], chunk=256)` | 驱动器构造完成，可以收发缓存 | unified_linker.py |
| `Tree cache initialized: impl=LMCacheAscendUnifiedCache ...` | 树缓存类已换成本仓实现 | SGLang scheduler |

**A 侧对照**：对 A 侧日志跑同一条 grep，**一行都不该有**——这正是对照组纯净性
的证明（坑清单 #12）。

### 4.4 第 3 步：证明"真的走了 LMCache"（Stored/Retrieved 闭环冒烟）

4.2 的短请求**存不进缓存**（长度不足一个 chunk=256，对齐后为 0，这是设计行为，
不是故障），所以验证缓存闭环必须用长前缀。用官方 benchmark 跑一个只有 4 个请求
的迷你 gsp 当探针（参数故意选小：与第五节正式实验的参数不同 → 磁盘缓存 key 不同
→ 冒烟数据不会污染正式实验的"冷/热"状态）：

**run1（冷，存入）**：

```bash
export HF_HUB_OFFLINE=1
python -m sglang.benchmark.serving \
  --backend sglang \
  --host 127.0.0.1 --port 8998 \
  --model Qwen3.8-27B \
  --tokenizer /data/models/Qwen3.8-27B \
  --dataset-name generated-shared-prefix \
  --gsp-num-groups 2 --gsp-prompts-per-group 2 \
  --gsp-system-prompt-len 2048 --gsp-question-len 128 --gsp-output-len 64 \
  --seed 42 \
  --cache-report \
  --output-file /workspace/results/smoke_run1_cold.jsonl
```

判定 run1（两条 grep）：

```bash
grep -c "Stored" /workspace/results/lmcache_server_unified.log   # 跑完比跑前多 ≥2
grep -c "complete physical blocks" /workspace/results/sglang_qwen38_lmcache_B.log  # 应为 0（不涨）
```

- lmcache server 日志新增 **≥2 条 `Stored`**（2 组前缀 × 每组第 1 条请求触发）；
- SGLang 日志**不刷** `ValueError: ... indices do not contain complete physical
  blocks`——若刷屏说明容器里还是旧代码（store 对齐 bug，见坑清单 #13）。

**run2（flush 后，取回）**：同一条命令，只加 `--flush-cache`，输出文件改成
`smoke_run2_hot.jsonl`。判定：

- lmcache server 日志出现 **`Retrieved`**（本次实验真正第一次走回灌路径）；
- 终端报告 `Cache hit rate` > 0（4 条请求共享 2 个 2048-token 前缀，正常 80% 左右）；
- run2 的 `Mean TTFT` 明显低于 run1（前缀不用重新 prefill）。

### 4.5 第 4 步：回灌正确性（最关键，也最容易被跳过）

4.4 只能证明"数据搬过去了"，不能证明"搬的数据是对的"——gsp 的内容是随机 token，
输出本来就无语义，错了也看不出来。这一步用**藏数字的长前缀**做冷热对比，答案
可以机判：

```bash
# 构造 ~2400 token 的前缀：200 句话，每句藏一个不同的数字
P=$(python3 -c "print(' '.join(f'The secret number in round {i} is {i*7+3}.' for i in range(200)))")

# 第 1 次（冷）：完整 prefill；等命令返回后，lmcache server 日志应出现 Stored
curl -s http://127.0.0.1:8998/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"Qwen3.8-27B\", \"messages\": [{\"role\": \"user\", \"content\": \"$P Question: what is the secret number in round 199? Answer with the number only.\"}], \"max_tokens\": 16, \"temperature\": 0}"
```

正确答案是 **1396**（199×7+3）。这个数字只存在于前缀末尾附近，模型必须"读过"
整个前缀才答得出来——回灌的 KV/mamba 有一点错，答案就露馅。

**第 2 次（热，走回灌）**：确认第 1 次的 `Stored` 已出现后，清掉 SGLang 本地缓存，
再发**一模一样**的请求：

```bash
curl -X POST http://127.0.0.1:8998/flush_cache
# 然后原样重复上面那条 curl
```

此时本地 radix 已空，前缀 KV + mamba 状态只能从 lmcache server 回灌（日志应出现
`Retrieved`）。判定：

| 现象 | 结论 |
|---|---|
| 两次都答 1396，且文字一致 | KV + mamba 回灌正确 ✅ **功能验证通过，进第五节测性能** |
| 第 1 次对（1396），第 2 次乱答/乱码/答别的数 | 回灌数据坏了 ❌ 停下，把两次输出 + Retrieved 日志贴回来排查 |
| 第 2 次日志没有 Retrieved | 根本没走缓存（store 可能失败了），回 4.4 检查 |

背后的道理：temperature=0 是贪心解码，相同的前缀状态必然产生相同的输出。冷热两次
输出一致 = 回灌后的状态与重新计算的状态等价 = 缓存数据正确。这比"看着像人话"
严格得多。

### 4.6 功能验证通过清单（全绿才继续）

- [ ] B 侧 curl 吐人话；A 侧 curl 也吐人话
- [ ] B 侧启动日志五连证据齐全；A 侧日志 grep lmcache 零命中
- [ ] smoke run1：有 Stored、无 ValueError 刷屏
- [ ] smoke run2：有 Retrieved、hit rate > 0、TTFT 比 run1 低
- [ ] 藏数字测试：冷热两次都答 1396

任何一项不过：先解决问题再谈性能。全部通过：进第五节。

---

## 五、实验一：共享前缀负载 —— TTFT 核心对比

### 5.1 这个数据集是什么

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

### 5.2 四个 run 的完整命令（照抄；每条只改输出文件名和最后一行的 flush 开关）

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

### 5.3 为什么执行四次、为什么是这个顺序（这套逻辑看懂就毕业了）

这是标准的**控制变量法**：四个 run 两两配对出三组对比，每一组只变一个变量。

| 对比 | 谁和谁比 | 唯一不同的变量 | 结论 |
|---|---|---|---|
| ①flush 有效性 | A1 vs A2 | 开跑前是否 `--flush-cache` | 两者结果几乎一样 ⇒ flush 确实把 SGLang 自己的 radix cache 清干净了，A2 是真"冷" |
| ②同服务 flush 前后 | B1 vs B2 | 同上（B2 能靠 lmcache server 续热） | TTFT 下降幅度 = flush 之后 LMCache 补回来的速度 |
| ③核心收益 | A2 vs B2 | 服务端有没有接 LMCache | **A2 与 B2 的 TTFT 差值 = LMCache 的净收益** |

- **③为什么成立**：A2 和 B2 的 SGLang 进程内缓存都被 `--flush-cache` 清空，站在同一
  起跑线；唯一差别是 B2 能从**独立的 lmcache server 进程**把 KV + mamba 状态拿回来。
- `--flush-cache` 是官方 benchmark 参数：开跑前自动调一次服务端 `/flush_cache`，
  **只清 SGLang 自己的 radix tree，不动 lmcache server 里的数据**（runbook 5.2 已实测
  过这个行为：flush 后热请求仍然走 Retrieved）。没有它，B2 的"热"就分不清是 SGLang
  本地 radix 还是 LMCache 贡献的。
- **顺序依赖只有一个硬约束**：**B1 必须在 B2 之前**——B2 的热数据全部来自 B1 写入
  lmcache server 的内容，跑反了 B2 就是一场空。A1/A2 谁先谁后无所谓（A2 自带 flush，
  不依赖 A1 留下什么），但都必须在 A 侧服务上跑；同样 B1/B2 必须在 B 侧服务上跑。
- **四步的本质分工**：A1、B1 是"生产数据/测冷跑"，A2、B2 是"复试测热"；A1/A2 负责
  证明对照组干净，B1/B2 负责展示实验组收益。少任何一步，结论都会被质疑。

### 5.4 结果读法（终端最后的分块报告）

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

### 5.5 与 lmcache server 日志对账（B 侧数据的铁证）

```bash
grep -E "Stored|Retrieved" /workspace/results/lmcache_server_unified.log | tail -30
```

| 时间段 | 应看到 | 含义 |
|---|---|---|
| B1 期间 | 大量 `Stored ... tokens` | 冷请求把前缀 KV/mamba 状态写进 server |
| B2 期间 | 大量 `Retrieved ... tokens` | flush 后靠 server 命中回灌 |
| A1/A2 期间 | **零新增** | 对照组根本不连 lmcache，日志纹丝不动 |

### 5.6 填表（建议抄进你的实验记录）

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

## 六、实验二：随机负载 —— 吞吐基线 + store 开销体检

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

## 七、实验三（可选）：多轮 —— mamba 状态复用收益

hybrid 模型的卖点之一是 mamba 状态也能缓存：多轮对话第 2 轮起，B 侧可以直接
回灌上一轮的 FULL KV + MAMBA 状态，而不是重新 prefill 全部历史。

### 7.1 官方 gsp 多轮版（推荐）

把实验一的公共参数改四处：

```bash
  --gsp-num-groups 16 \
  --gsp-prompts-per-group 4 \
  --gsp-num-turns 4 \
  --gsp-output-len 128 \
```

注意：`--gsp-num-turns > 1` 时官方代码会**绕过磁盘缓存**、每次重新合成数据，但
同 seed 内容仍然一致，A/B 依然公平。A、B 各跑一遍对比 TTFT/TPOT。

### 7.2 hicache 套件（容器里有 sglang 源码目录时才做）

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
7.1 已经覆盖了多轮收益的测量，hicache 只是锦上添花。

---

## 八、常见坑清单（出问题先查这里）

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
| 13 | SGLang 日志持续刷 `ValueError: LMCache group 'full' indices do not contain complete physical blocks`（服务不崩，但 B 侧存不进缓存） | store 传入的索引未按 chunk/page 对齐——完成请求的序列长度（前缀+输出）几乎不可能是 128/256 的倍数。修复已提交在 unified_linker.py 的 `_store_via_lmcache`（把 tokens 和索引一并截断到 chunk 与 page 的公倍数）。若仍刷屏，说明容器里是旧代码：重新 docker cp 覆盖 + 清 `__pycache__` + 重启服务 |

---

## 九、压测崩溃排查预案（B2 检索路径崩了怎么办）

### 9.1 先分清"谁崩了"

| 现象 | 含义 |
|---|---|
| 服务日志里 `PTA call acl api failed`（Device:N） | **SGLang 服务进程在 NPU 上崩了**：分页注意力算子的 acl 调用失败，通常是算子读到了非法地址 / 页表越界 / 资源耗尽 |
| traceback 停在 `torch.cat` 之类不相关的行 | CANN 异步算子出错后，栈显示的是"错误被回收时 CPU 恰好走到的地方"，**不是真正出错点**（CANN 日志自己会注明 stacktrace may be inaccurate） |
| `py-spy dump failed` / `CUDA coredump` 提示 | 崩溃后自动取证工具的噪音行，可以无视 |
| benchmark 终端同时报错 | 服务死了之后的连锁反应（连接被断开），**根因在服务端** |

### 9.2 排查四步

第 1 步，收集定位材料（三条命令，跑完贴回来）：

```bash
# ① 崩溃时刻 lmcache server 日志尾部（多少条 Retrieved/Stored、有没有 evict/淘汰）
grep -E "Stored|Retrieved|evict|Evict" /workspace/results/lmcache_server_unified.log | tail -50

# ② B 侧 SGLang 日志里的 lmcache 相关行（有没有 fail-open 警告）
grep -E "unified|lmcache|LMCache" /workspace/results/sglang_qwen38_lmcache_B.log | tail -40

# ③ SGLang 日志里最早的那段完整 traceback（真正 Inner error 原文在最上面）
grep -n -B 5 "The Inner error is reported as above" /workspace/results/sglang_qwen38_lmcache_B.log | head -40
```

第 2 步，**降并发二分**（判断是"数据路径坏了"还是"并发压崩的"）：

```bash
# B2 命令原样，只加一行 --max-concurrency，从 1 试起：
  --max-concurrency 1
# 不崩 → 4 → 16 → 全开，找出崩溃阈值
```

| 结果 | 结论 | 下一步 |
|---|---|---|
| `--max-concurrency 1` 也崩 | 回灌数据几何/分片错误（与并发无关，必现） | 服务重启前 `export ASCEND_LAUNCH_BLOCKING=1` 再复跑拿真栈（CANN 官方建议；测完记得 unset） |
| 只有高并发崩 | 并发/资源类问题（evict 与回灌交叉、池容量压力） | 贴出崩溃时的并发阈值 + 第 1 步三份材料 |
| 怎么跑都不崩 | 偶发 | 复跑两遍留档，重点保留第 1 步日志 |

第 3 步，**记忆两个参考点**：runbook 5.2 的单请求 retrieve 是成功过的（说明回灌大方向
没有写反）；B1 的 128 条 store 也是成功的（说明 store 路径无恙）。B2 是四跑里唯一
同时把"回灌 + 高并发 + 长输出"压在一起的组合，崩点必然在这三者的交界处。

第 4 步，把材料贴回来一起分析，重点提供：崩溃发生在压测开始后大约第几个请求、当时
`Retrieved` 日志已经出现了多少条。

---

## 十、本教程全部参数速查表（每个参数什么意思）

### 10.1 服务启动参数（第三节命令里的）

| 参数 | 含义 |
|---|---|
| `PYTHONHASHSEED=0` | 固定 Python 哈希种子；SGLang 多进程（8 rank）要求各进程行为一致，不加可能导致偶发不一致 |
| `--model-path` | 本地模型权重目录；离线环境必须是本地路径（此教程为 `/data/models/Qwen3.8-27B`） |
| `--served-model-name` | 服务对外暴露的模型名；**benchmark 的 `--model` 必须与它完全一致** |
| `--host 0.0.0.0` | 监听地址；0.0.0.0 = 容器内所有网卡都可访问 |
| `--port 8998` | 服务 HTTP 端口；benchmark 的 `--port` 要对应 |
| `--tp-size 8` | 张量并行卡数（此环境 = 8 卡 NPU） |
| `--dtype bfloat16` | 模型权重精度 |
| `--context-length 8192` | 单请求最大上下文长度（前缀 + 输出之和的上限） |
| `--chunked-prefill-size 4096` | prefill 分块大小；长前缀分块预填充，降低单次显存峰值 |
| `--mem-fraction-static 0.85` | 静态内存池占单卡 HBM 的比例；KV 池容量来自剩余部分 |
| `--trust-remote-code` | 允许执行模型目录里的自定义建模代码；Qwen3.8 是自定义架构，不开起不来 |
| `--mamba-ssm-dtype bfloat16` | SSM 状态池精度；Qwen3.8 GDN 层在 910B 上的**硬约束**（FP32 会报仅 Ascend950 支持），必须带 |
| `--enable-lmcache` | 声明启用 LMCache（**只有 B 侧有**，A/B 切换就看它） |
| `--lmcache-config-file` | LMCache 配置文件路径（chunk 大小等），同上只有 B 侧有 |
| `2>&1 \| tee xxx.log` | 输出同时打屏幕和写日志文件，查错留档用 |

### 10.2 benchmark 通用参数（每条命令都带）

| 参数 | 含义 |
|---|---|
| `HF_HUB_OFFLINE=1` | 禁止 HuggingFace 联网；离线环境必须设，否则工具可能尝试在线拉 tokenizer/数据集 |
| `--backend sglang` | 压测后端类型，决定打哪个 API（sglang = 服务原生 `/generate` 接口） |
| `--host` / `--port` | 服务地址与端口（127.0.0.1:8998） |
| `--model` | 请求体里的模型名 = 服务端 `--served-model-name` |
| `--tokenizer` | tokenizer 本地目录；合成数据集要用它做编解码，离线必须指本地 |
| `--dataset-name` | 数据集类型：`generated-shared-prefix`（共享前缀合成）/ `random`（随机）/ `sharegpt` 等 |
| `--seed` | 随机种子；**同 seed + 同参数 = 同一份数据**，A/B 对比公平的关键 |
| `--cache-report` | 跑完额外打印缓存命中统计（命中 token 数、命中率、device/host/storage 分栏） |
| `--output-details` | 把每个请求的明细（ttft/tpot/长度）逐行写进输出 JSONL |
| `--output-file` | 输出 JSONL 路径；不给会自动命名到当前目录 |
| `--flush-cache` | **开跑前**先调服务端 `/flush_cache` 清空 SGLang 进程内缓存；不影响 lmcache server |
| `--warmup-requests` | 正式计数前的预热请求数（默认 1，可不动） |
| `--ready-check-timeout-sec` | 等待服务就绪的最长秒数（默认 60，可不动） |

### 10.3 gsp 数据集参数（实验一）

| 参数 | 含义 |
|---|---|
| `--gsp-num-groups` | 系统提示词组数（多少个互不相同的前缀）；与每组条数相乘 = 总请求数 |
| `--gsp-prompts-per-group` | 每组问题条数（多少个请求共享同一个前缀） |
| `--gsp-system-prompt-len` | 每组系统提示词的目标 token 数（此教程 2048） |
| `--gsp-question-len` | 每个问题的 token 数（128） |
| `--gsp-output-len` | 每个请求生成多少个 token（256） |
| `--gsp-range-ratio` | 长度抖动比例；**保持默认 1.0**（改成别的值会禁用磁盘缓存、每次重新生成数据） |
| `--gsp-num-turns` | 多轮轮数；>1 时一组变一段多轮对话，且绕过磁盘缓存（实验三用） |
| （注意） | gsp 总请求数 = `--gsp-num-groups × --gsp-prompts-per-group`，**`--num-prompts` 对它无效** |

### 10.4 random 数据集参数（实验二）

| 参数 | 含义 |
|---|---|
| `--random-input-len` | 每个随机请求的输入 token 数（1024） |
| `--random-output-len` | 每个随机请求的输出 token 数（512） |
| `--num-prompts` | 总请求数（random/sharegpt 等数据集生效；gsp 不生效） |
| `--request-rate` | 请求到达速率；默认 `inf` = 瞬间全发（饱和压测）；设 4 = 每秒 4 个按泊松到达，测低负载 TTFT |
| `--max-concurrency` | 同时在途请求上限（模拟上层限流）；排查崩溃时用它降并发二分 |

### 10.5 hicache bench_multiturn 参数（7.2 可选节）

| 参数 | 含义 |
|---|---|
| `--num-clients` | 同时对话的客户端（会话）数 |
| `--num-rounds` | 每个客户端来回多少轮 |
| `--request-length` / `--output-length` | 每轮输入 / 输出的 token 数 |
| `--api-format` | `sglang` = 打原生 `/generate`；`openai` = 打 `/v1/chat/completions` |
| `--model-path` | 同时用作 tokenizer 目录和请求模型名（给本地路径 `/data/models/Qwen3.8-27B`） |
| `--log-file` / `--tag` | 结果写哪个 JSONL、给这轮结果打什么标签 |

---

## 十一、最终汇总表（写进实验报告）

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