# Qwen2.5-7B 标准模型首亮教程（SGLang + LMCache-Ascend MP mode）

> 目标读者：第一次接触本适配工作的工程师。
> 目标：在昇腾 910B4 上，用**社区验证矩阵内的标准模型 Qwen2.5-7B-Instruct**，跑通
> `SGLang → LMCache MP server → LMCache-Ascend NPU 数据面` 的完整闭环，
> 拿到第一组 Stored / Retrieved 证据。
>
> 预计耗时：模型下载约 30~60 分钟（视网速），其余步骤约 30 分钟。

---

## 0. 一页结论

### 0.1 三个关键问题的答案

| 问题 | 答案 |
|---|---|
| 社区 LMCache + SGLang 支持昇腾卡吗 | **设计上支持，验证上没有**。LMCache v0.5.6 加了 NPU 平台后端并移除 CUDA 硬编码，但社区 CI 只在 NVIDIA GPU 上跑过 SGLang 集成测试。昇腾上的真实数据面由 LMCache-Ascend 插件提供（本仓 e2e 测试 3 passed 已验证） |
| 能支持什么模型 | ① 标准 GQA/MHA（Qwen2.5 系列、Qwen3-dense、Llama）——旧接线理论可行，**本教程目标**；② MLA（DeepSeek）——上游支持；③ **hybrid（Qwen3.8/GDN/Mamba）——SGLang 旧接线有 bug，全社区无人走通，属后续专项** |
| 需要重新拉镜像 / 改 SGLang / 改 LMCache 版本吗 | **都不需要**。当前 LMCache dev 分支已含 v0.5.6 全部内容（最新），SGLang 0.5.21.dev476（最新），LMCache-Ascend 已编译安装。唯一缺的是本地没有一个标准模型 |

### 0.2 为什么要"降级"到 Qwen2.5-7B

上一轮用 Qwen3.8-27B（hybrid 架构）启动，在 SGLang 调度器初始化阶段就崩溃：

```text
AttributeError: 'HybridLinearKVPool' object has no attribute 'k_buffer'
```

这个错误**与昇腾卡无关**（同代码在 NVIDIA 上跑 Qwen3.8 一样崩），是 SGLang 的 LMCache
接线代码没有跟上它自己的 hybrid 内存池重构。详细分析见
[environment_pitfalls_and_fixes.md](environment_pitfalls_and_fixes.md) 和
[progress_and_next_action.md](progress_and_next_action.md)。

Qwen2.5-7B-Instruct 是 LMCache 官方 Buildkite CI 的默认验证模型（标准 GQA，非 hybrid），
SGLang 会给它分配带 `k_buffer` 的标准内存池——恰好落在社区验证过的路径上。
**先用它拿到"这条管线在昇腾上能通"的基线事实，再回头攻坚 hybrid。**

### 0.3 当前环境版本盘点（证明不需要动任何版本）

| 组件 | 当前版本 | 是否最新 | 本次是否要动 |
|---|---|---|---|
| LMCache | dev 分支（含 v0.5.6，2026-10-15 发布） | ✅ | ❌ 不动 |
| SGLang | 0.5.21.dev476+g4e60d70ba | ✅ | ❌ 不动 |
| LMCache-Ascend | main（已编译，c_ops 已绑定） | ✅ | ❌ 不动 |
| 容器镜像 | 现有 ascend 镜像 | — | ❌ 不重拉 |
| 模型 | 只有 Qwen3.8-27B（hybrid） | — | ➕ 新增 Qwen2.5-7B |

---

## 1. 背景知识（5 分钟版）

### 1.1 hybrid 是什么，为什么它挂而标准模型能过

大模型把"看过的每个字"的处理结果存成 KV Cache。存法分两派：

- **标准注意力（full attention）**：每个字存一张 K 卡 + 一张 V 卡。SGLang 给这类模型
  分配标准内存池，里面有 `k_buffer` / `v_buffer` 两个大格子。
- **线性注意力（GDN/Mamba）**：不存卡片，只维护一块不断覆盖更新的"状态白板"。

Qwen3.8 是 **hybrid（混合）**：一半层存卡片、一半层写白板。SGLang 为它分配
`HybridLinearKVPool` 混合池——**没有统一的 `k_buffer` 格子**，而 SGLang 的 LMCache
搬运代码写死了"先打开 k_buffer"，所以一启动就摔跤。

Qwen2.5-7B 全部层都是标准注意力 → 标准池 → 有 k_buffer → 能过第一关。

### 1.2 社区验证矩阵（你为什么之前会踩坑）

LMCache 官方 CI 测试 SGLang 集成时的固定组合：

```text
✅ 模型：Qwen/Qwen2.5-7B-Instruct（写在 CI 的 common.sh 里）
✅ 硬件：NVIDIA GPU
✅ 路径：SGLang 旧接线（--enable-lmcache）→ LMCacheMPConnector（旧类）
```

你之前的三维组合（Qwen3.8 hybrid × 昇腾 × 新旧混搭）全部出圈。本教程把**模型维度**
拉回圈内，剩下"昇腾"这一维正是本次适配要验证的价值所在。

### 1.3 残存风险（诚实声明）

有一处没人验证过的接缝：SGLang 旧接线调用的旧类 `LMCacheMPConnector`
（LMCache 仓 `multi_process_adapter.py`）与 dev 分支新 server 的协议兼容性。
这正是上一轮决策中"路线 A：先探旧路"要回答的问题。**本教程跑完即得到答案**：
- 跑通 → 旧路在昇腾可用，标准模型需求直接达成；
- 挂在连接/协议处 → 记录报错，按第 8 节决策树处理（可能是小 patch 可救，也可能要等 SGLang 侧换新接线）。

无论哪种结果都不白跑。

---

## 2. 第 0 步：环境盘点（5 分钟）

在容器里执行，逐项确认：

```bash
# 1) 三个组件在位
python -c "import lmcache, sglang, lmcache_ascend; print('lmcache:', lmcache.__file__); print('sglang:', sglang.__version__)"
ls /workspace/src/LMCache-Ascend/lmcache_ascend/c_ops*.so 2>/dev/null || \
  python -c "import lmcache_ascend.c_ops; print('c_ops OK')"

# 2) NPU 卡空闲（确保 Qwen3.8 的服务已停）
npu-smi info | head -30

# 3) 端口干净（5556/8080/8998 没有残留进程）
ss -ltnp 2>/dev/null | grep -E "5556|8080|8998" || echo "ports free"
```

如果 Qwen3.8 的 SGLang 还在跑：`pkill -f sglang.launch_server`，等 10 秒后复查 npu-smi。

---

## 3. 第 1 步：准备 Qwen2.5-7B-Instruct 模型

### 3.1 先查本地有没有可用的标准模型（可能省掉下载）

```bash
ls -la /data/models/
```

| 本地已有 | 能否直接用 | 说明 |
|---|---|---|
| Qwen3-32B / Qwen2.5-14B 等纯 dense 模型 | ✅ 可替代 | 都是标准 GQA，跳过 3.2~3.3，直接进第 4 步（注意改模型路径和 served-model-name，32B 建议保持 TP8） |
| 只有 Qwen3.8-27B | ➕ 需下载 | 它是 hybrid，本教程用不了，走 3.2 |

### 3.2 下载 Qwen2.5-7B-Instruct（在**有网的 Windows 宿主机**上）

模型约 **15.4 GB**（5 个 safetensors 分片 + tokenizer + config）。国内网络必须走
hf-mirror.com 镜像：

```powershell
# Windows PowerShell
$env:HF_ENDPOINT = "https://hf-mirror.com"
pip install -U "huggingface_hub[cli]"

# 下载（约 15.4GB，视网速 20~60 分钟）
huggingface-cli download Qwen/Qwen2.5-7B-Instruct `
  --local-dir "D:\hf-downloads\Qwen2.5-7B-Instruct"
```

校验完整性（5 个 safetensors 分片都在才算完整）：

```powershell
Get-ChildItem "D:\hf-downloads\Qwen2.5-7B-Instruct" | Format-Table Name, Length
# 预期看到: model-00001-of-00005.safetensors ~ model-00005-of-00005.safetensors
#          config.json  tokenizer.json  tokenizer_config.json  ...
```

### 3.3 传入容器（用你之前传 wheelhouse 的同一条路）

先把文件从 Windows 传到昇腾服务器（scp / 共享目录均可），然后在服务器上：

```bash
# 服务器上：拷入容器（容器名按实际替换，docker ps 查看）
docker cp /path/on/server/Qwen2.5-7B-Instruct <容器名>:/data/models/Qwen2.5-7B-Instruct
```

> 注意：若 `/data` 不是挂载卷，`docker cp` 进去的文件在容器重建后会丢。
> 检查方法：`docker inspect <容器名> | grep -A3 '"Destination": "/data"'`
> ——有输出说明是挂载，可直接往宿主机对应路径放文件，更稳妥。

### 3.4 容器内校验模型

```bash
ls -lh /data/models/Qwen2.5-7B-Instruct/
python - <<'PY'
import json
c = json.load(open('/data/models/Qwen2.5-7B-Instruct/config.json'))
print("architectures:", c["architectures"])          # 预期 Qwen2ForCausalLM
print("layers:", c["num_hidden_layers"])              # 预期 28
print("q_heads:", c["num_attention_heads"])           # 预期 28
print("kv_heads:", c["num_key_value_heads"])          # 预期 4（GQA，标准模型特征）
PY
```

`num_key_value_heads < num_attention_heads` 即为标准 GQA —— 正是我们要的。

---

## 4. 第 2 步：写配置 + 启动 lmcache server

### 4.1 SGLang 侧 YAML

标准模型**不需要** `--separate-object-groups`（那是 hybrid 的建议参数）。
chunk_size 取 256（对任何常见 page_size 都是整数倍，见第 6 步核查）：

```bash
mkdir -p /workspace/scripts /workspace/results
cat > /workspace/scripts/lmcache_mp.yaml <<EOF
mp_host: tcp://127.0.0.1
mp_port: 5556
chunk_size: 256
EOF
```

### 4.2 启动 lmcache server（终端 A，新开的 shell 记得重设环境变量）

```bash
export PYTHONHASHSEED=0
lmcache server \
  --host 127.0.0.1 --port 5556 \
  --http-port 8080 \
  --chunk-size 256 \
  --l1-size-gb 20 \
  --eviction-policy LRU \
  --supported-transfer-mode lmcache_driven \
  2>&1 | tee /workspace/results/lmcache_server_qwen25.log
```

保持这个终端开着。看到 server 就绪日志即成功（无 traceback）。

---

## 5. 第 3 步：启动 SGLang（终端 B）

与 Qwen3.8 基线命令相比，**三处变化**：模型路径、名字、**去掉 `--mamba-ssm-dtype`**
（那是 hybrid GDN 层的专属参数，标准模型不需要）：

```bash
export PYTHONHASHSEED=0
python -m sglang.launch_server \
  --model-path /data/models/Qwen2.5-7B-Instruct \
  --served-model-name Qwen2.5-7B-Instruct \
  --host 0.0.0.0 --port 8998 \
  --tp-size 8 --dtype bfloat16 \
  --context-length 8192 --chunked-prefill-size 4096 \
  --mem-fraction-static 0.85 --trust-remote-code \
  --enable-lmcache \
  --lmcache-config-file /workspace/scripts/lmcache_mp.yaml \
  2>&1 | tee /workspace/results/sglang_mp_qwen25.log
```

> 保持 TP8：本教程的隐性目标之一是验证 8 个 rank 全部完成 NPU 上下文注册。

### 启动日志三个观察点（按语义找，不机械匹配）

| # | 找什么 | 含义 |
|---|---|---|
| 1 | **过了 `lmc_radix_cache.py` 的初始化**（不再出现 `k_buffer` AttributeError） | 标准池路线打通第一关 |
| 2 | `Creating transfer context (device_type=npu, mode=lmcache_driven)` × 8 | 每个 TP rank 都连上了 NPU 数据面 |
| 3 | server 日志（终端 A）出现 KV 注册 / engine 相关条目 | 旧 connector 与新 server 完成握手 |

任何一个观察点失败 → 直接跳到第 8 节决策树，把**最早的那条完整 traceback** 留档。

---

## 6. 第 4 步：核查 page_size 与 chunk_size 对齐

服务起来后（日志出现 ready / The server is fired up）：

```bash
curl -s http://127.0.0.1:8998/get_server_info | python -m json.tool | grep -iE "page_size|block" | head -5
```

对齐规则：**chunk_size 必须是 page_size 的整数倍**。

| 查到的 page_size | 处理 |
|---|---|
| 1、2、4、8、…、128、256 | ✅ 256 都满足，什么都不用改 |
| 512 或更大 | 把 YAML 和 server 的 `--chunk-size` 都改成该值，重启 server + SGLang |

> 参考：之前 Qwen3.8 查到 page_size=128（hybrid/NPU 后端自动设的）；标准模型常见为 1。

---

## 7. 第 5 步：验证缓存闭环（本教程的验收标准）

### 7.1 构造长前缀请求（前缀必须超过 chunk_size=256 token）

```bash
python - <<'PY'
import json
prefix = "以下是关于分布式KV缓存系统的一次内部技术评审记录，涉及内存池分层与传输通道选型。" * 120
prompt = prefix + "\n\n请用一句话总结上面的内容。"
payload = {
    "model": "Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 32,
    "temperature": 0,
}
open("/workspace/scripts/probe_request.json", "w").write(json.dumps(payload))
print("prompt chars:", len(prompt))
PY
```

### 7.2 三连验证

```bash
# ① 冷请求（第一次见这个前缀 → 应触发 Stored）
curl -s http://127.0.0.1:8998/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @/workspace/scripts/probe_request.json | head -c 300; echo

# ② 清掉 SGLang 本地 RadixCache（关键！否则第二次只命中本地前缀缓存，测不出 LMCache）
curl -s -X POST http://127.0.0.1:8998/flush_cache; echo

# ③ 热请求（同前缀 → 应触发 Retrieved）
curl -s http://127.0.0.1:8998/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @/workspace/scripts/probe_request.json | head -c 300; echo

# ④ 看证据
grep -E "Stored|Retrieved" /workspace/results/lmcache_server_qwen25.log | tail -20
```

### 7.3 验收清单（全勾才算"首亮"）

- [ ] 启动无 `k_buffer` 报错，8 rank 都有 npu transfer context
- [ ] 冷请求输出正常（不是乱码/空回复）
- [ ] server 日志出现 `Stored ... tokens`
- [ ] `flush_cache` 后热请求输出正常
- [ ] server 日志出现 `Retrieved ... tokens`
- [ ] 两次输出语义一致（temperature=0 下应基本相同）
- [ ] TP8 无死锁（第二次请求能正常返回）

**全部勾上 = LMCache-Ascend MP mode 在 SGLang + 昇腾 + 标准模型上首亮成功。**

---

## 8. 失败决策树

| 症状 | 含义 | 动作 |
|---|---|---|
| 仍然 `k_buffer` AttributeError | 意外（标准模型不该出现） | 检查是不是真的用了 Qwen2.5 路径启动；贴 traceback 排查 |
| 过了 k_buffer，挂在 `LMCacheMPConnector(...)` 构造（TypeError unexpected kwarg 之类） | 旧接线与 LMCache dev 接口错配 | 留档报错；这是已知交接缝隙，按"本仓 runtime patch 包装构造"评估（见 progress_and_next_action.md 第 4 步决策表） |
| connector 构造成功但连不上 server（ZMQ/超时/注册失败） | 旧类与新 server 协议不兼容（第 1.3 节预判的残存风险） | 留档最早 traceback；此为路线 A 探路的结果，转入版本匹配评估 |
| NPU 侧报错（c_ops / IPC / transfer context） | 数据面问题——**这才是 LMCache-Ascend 的管辖范围** | 贴报错，按本仓数据面排查（对照 e2e 测试 test_npu_lmcache_driven_e2e.py 的行为差异） |
| 输出乱码或语义异常 | KV 回灌正确性问题 | 缩小前缀长度复测；对比关闭 LMCache 的裸服务输出 |

通用要求：任何失败都保留**最早一条完整 traceback**（不是最后 worker 退出的部分），
并 `tee` 留档到 `/workspace/results/`。

---

## 9. 成功之后做什么

1. **记录基线**：同前缀冷/热请求各测 5 次，记录 TTFT（响应里 usage 或加时间戳），
   与裸 SGLang（无 LMCache）对比——这是后续 A/B 报告的第一组数据。
2. **回归 Qwen3.8 hybrid 专项**：标准模型跑通证明"管线 + 数据面"没问题后，
   hybrid 的问题就纯粹集中在 SGLang 旧接线对 HybridLinearKVPool 的支持上——
   那是独立的一块适配工作（注入 k_buffer property 或切新接线），见
   [progress_and_next_action.md](progress_and_next_action.md) 第 4 步。
3. **更新文档**：把本教程的实测结果（page_size 实际值、是否踩到 1.3 节风险）
   回填到本文档末尾的"实测记录"小节。

## 10. 与既有文档的关系

| 文档 | 角色 |
|---|---|
| 本文 | 标准模型首亮的手把手入口 |
| [progress_and_next_action.md](progress_and_next_action.md) | 总进度、决策树、改动量预估 |
| [first_model_e2e_validation.md](first_model_e2e_validation.md) | 原 Qwen3.8 E2E 方案（hybrid 专项时再启用） |
| [environment_pitfalls_and_fixes.md](environment_pitfalls_and_fixes.md) | 环境坑记录（k_buffer 崩溃的详细分析在本文档补充） |
| [control_plane_vs_data_plane.md](control_plane_vs_data_plane.md) | 控制面/数据面概念与调用链（理解第 5 步观察点用） |
| [lmcache_ascend_vs_upstream.md](lmcache_ascend_vs_upstream.md) | 本仓与上游的能力边界 |

## 实测记录（跑完后回填）

```text
日期：
page_size 实测值：
chunk_size 最终值：
1.3 节残存风险是否触发（旧 connector ↔ 新 server）：
Stored / Retrieved 日志摘录：
TTFT 对比（裸 vs LMCache）：
遗留问题：
```
