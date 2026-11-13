# Hybrid 模型统一路径：分步操作手册（代码已写完，从这里开始跑）

> 状态：代码已交付（3 个文件，语法已验证），**本手册是你接下来的全部操作**。
> 顺序：第 0~4 步（部署 + Qwen3.8 首亮）→ 第 5 步（闭环验证）→ 第 6 步（GLM-5.3）
> → 第 7 步（DeepSeek-V4.1）。
>
> 设计与代码细节见 [hybrid_models_adaptation_blueprint.md](hybrid_models_adaptation_blueprint.md) 4.3 节。

---

## 第 0 步：把代码同步进容器（5 分钟）

改动的 3 个文件（Windows 本机 `d:\proj\LMCache-Ascend`）：

```text
lmcache_ascend/__init__.py                              （改）
lmcache_ascend/integration/sglang/unified_linker.py     （新）
lmcache_ascend/integration/sglang/unified_diversion.py  （新）
```

**方式 A：git（推荐）**

```powershell
# Windows 本机
git add lmcache_ascend/__init__.py lmcache_ascend/integration/sglang/unified_linker.py lmcache_ascend/integration/sglang/unified_diversion.py
git commit -m "feat: unified-path diversion for hybrid/MLA/DSV4 pools (SGLang MP)"
git push
```

```bash
# 容器里
cd /workspace/src/LMCache-Ascend && git pull
```

**方式 B：docker cp（git 不通时）**

```bash
# 服务器上
docker cp unified_linker.py    <容器>:/workspace/src/LMCache-Ascend/lmcache_ascend/integration/sglang/
docker cp unified_diversion.py <容器>:/workspace/src/LMCache-Ascend/lmcache_ascend/integration/sglang/
docker cp __init__.py          <容器>:/workspace/src/LMCache-Ascend/lmcache_ascend/
```

## 第 1 步：清字节码 + 快速自检（1 分钟，不用起服务）

```bash
cd /workspace/src/LMCache-Ascend
find lmcache_ascend -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

# 分流钩子能装上、探测函数能跑
python - <<'PY'
import lmcache
from lmcache_ascend.integration.sglang.unified_diversion import (
    install_unified_diversion, _needs_unified,
)
install_unified_diversion()
from sglang.srt.mem_cache.radix_cache import RadixCache
print("hook installed:", getattr(RadixCache, "_lmcache_ascend_unified_diversion", False))
# 容器里此时没有真实 params，探测返回 False 是正常的（False=走旧路径不误伤）
print("probe on dummy:", _needs_unified(type("P", (), {"token_to_kv_pool_allocator": None})()))
PY
```

预期输出两行 `True` / `False`。出错就把 traceback 贴回来。

## 第 2 步：确保 lmcache server 在跑（终端 A）

```bash
# 已在跑（Qwen2.5 阶段起的）就不用动；没跑则：
export PYTHONHASHSEED=0
lmcache server --host 127.0.0.1 --port 5556 --http-port 8080 \
  --chunk-size 256 --l1-size-gb 20 --eviction-policy LRU \
  --supported-transfer-mode lmcache_driven \
  2>&1 | tee /workspace/results/lmcache_server_unified.log
```

## 第 3 步：停旧服务 + 启动 Qwen3.8（终端 B）

```bash
# 3.1 停掉现在跑着的 Qwen2.5（或其他）服务
pkill -f sglang.launch_server
sleep 10
npu-smi info | head -40        # 确认 sglangschedul 进程消失、显存回到 ~29GB/卡

# 3.2 启动 Qwen3.8 —— 与你最初基线完全相同的参数 + lmcache 两参数
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
```

> 注意：`--mamba-ssm-dtype bfloat16` 这回**要带上**（它是 Qwen3.8 的 GDN 层专属，
> 当初的硬约束）；`/workspace/scripts/lmcache_mp.yaml` 沿用 Qwen2.5 阶段那份
> （chunk 256 不用改）。

## 第 4 步：启动日志四行证据（分流是否成功，30 秒判生死）

```bash
grep -E "unified-path diversion|unified path engaged|unified driver ready|compat patch" \
  /workspace/results/sglang_qwen38_unified.log
```

按顺序应看到：

| 日志行 | 含义 |
|---|---|
| `Installed SGLang MP connector stacked-KV compat patch` | 旧路径补丁也装上了（无害，标准池才用） |
| `Installed LMCRadixCache unified-path diversion hook` | **分流钩子装上** |
| `LMCRadixCache.__init__ wrapped with unified-path dispatcher` | dispatcher 包装成功 |
| `LMCache-Ascend unified path engaged (components=['FULL', 'MAMBA'], mamba=True)` | **Qwen3.8 的 hybrid 池被正确识别并分流** |
| `LMCache-Ascend unified driver ready (components=..., chunk=256)` | 连接器就绪、注册完成 |

**任何一行缺失或中途 traceback** → 把最早的那条完整 traceback 贴回来。

## 第 5 步：闭环验证（首亮验收，服务起来后跑）

```bash
# 5.1 长前缀冷请求（server 日志应出现 Stored）
python - <<'PY'
import json, urllib.request
prefix = "以下是关于分布式KV缓存系统的一次内部技术评审记录，涉及内存池分层与传输通道选型。" * 120
payload = {
    "model": "Qwen3.8-27B",
    "messages": [{"role": "user", "content": prefix + "\n\n请用一句话总结上面的内容。"}],
    "max_tokens": 64, "temperature": 0,
}
req = urllib.request.Request("http://127.0.0.1:8998/v1/chat/completions",
    data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=600) as r:
    print(json.loads(r.read())["choices"][0]["message"]["content"])
PY

# 5.2 清本地缓存 + 热请求（应出现 Retrieved）
curl -s -X POST http://127.0.0.1:8998/flush_cache; echo
# （重跑 5.1 同一段命令）

# 5.3 看证据
grep -E "Stored|Retrieved" /workspace/results/lmcache_server_unified.log | tail -10

# 5.4 多轮对话语义检查 —— hybrid state 回灌正确性的验收门槛
python /workspace/scripts/chat.py Qwen3.8-27B
# 交互中问：
#   你: 我叫张三，是一名数据库工程师
#   你: 我刚才说我叫什么、做什么工作？   ← 能答对 = mamba state 回灌正确
#   你: new; 你: 我叫张三...（再来一轮长上下文）
```

**验收清单（全勾 = Qwen3.8 unified 首亮成功）**：

- [ ] 第 4 步四行日志齐全（尤其 `mamba=True`）
- [ ] 冷请求输出正常（不是乱码/空回复）
- [ ] server 日志 `Stored ... tokens`
- [ ] flush 后热请求正常 + server 日志 `Retrieved ... tokens`
- [ ] **多轮对话记住前文**（hybrid 正确性门槛，最关键）
- [ ] TP8 无死锁

## 第 6 步：GLM-5.3-Flash（Qwen3.8 通过后）

### 6.1 现实性先想清楚

GLM-5.3-Flash 是 **320B 总参**模型，8×910B4（256GB HBM）单机放不下 BF16 全量
（约需 640GB+）。两条路：

| 路 | 做法 |
|---|---|
| A. 官方量化版 | 查 ModelScope/HF 上有无 FP8 或 W8A8 版（320B FP8 ≈ 320GB，仍偏紧；W4 可能可行） |
| B. 小变体代理 | 同架构（KDA 线性 + DSA 稀疏混合）的较小模型先验证分流正确性 |

### 6.2 模型下载（服务器直下，同 Qwen2.5 的三条通道）

```bash
# ModelScope（国内最稳）
pip install -U modelscope
modelscope download --model ZhipuAI/GLM-5.3-Flash \
  --local_dir /data/models/GLM-5.3-Flash
# 或量化版/小变体按实际仓库名替换
```

### 6.3 启动命令（预期形态，具体参数随模型变体调整）

```bash
pkill -f sglang.launch_server; sleep 10
export PYTHONHASHSEED=0
python -m sglang.launch_server \
  --model-path /data/models/GLM-5.3-Flash \
  --served-model-name GLM-5.3-Flash \
  --host 0.0.0.0 --port 8998 \
  --tp-size 8 --dtype bfloat16 \
  --context-length 32768 --chunked-prefill-size 4096 \
  --mem-fraction-static 0.85 --trust-remote-code \
  --enable-lmcache \
  --lmcache-config-file /workspace/scripts/lmcache_mp.yaml \
  2>&1 | tee /workspace/results/sglang_glm53_unified.log
```

预期日志：`unified path engaged (components=['FULL', 'MAMBA'], mamba=True)`
（GLM 的 KDA 线性层走 MAMBA 组件；DSA 层的索引缓存是 v1 已知限制之一，
若报 DSA/indexer 相关错误，贴回来按第 8 节排）。

注意：GLM-5.3 是多模态（vision encoder），v1 只缓存 LLM 部分 KV——这正是
组件化设计的自然行为，vision 部分不进树。

## 第 7 步：DeepSeek-V4.1（最后，风险最高）

同样有规模问题（552B 多机）。用 DSV4-Flash 系列或同架构小模型代理。
启动命令形态与第 6 步相同（模型路径、名字换成 DSV4），预期分流日志
`mamba=False`（DSV4 不是线性混合，走 MLA plane 路线）。

**v1 预警**：DSV4 的压缩池（C4/C128/C1/C2）在 unified adapter 里有上游实现
（`_resolve_dsv4_registered_groups`），但 NPU 数据面从未实测，且 CSA/HCA
压缩状态可能踩上游盲区——把它当探索性验证，报错信息价值很高。

## 第 8 步：失败排查决策树

| 症状 | 含义 | 动作 |
|---|---|---|
| 第 1 步自检报错 | 导入/钩子问题 | 贴 traceback |
| 第 4 步缺 `unified path engaged` | 分流没触发（探测没识别 hybrid 池） | 贴启动日志中 `Load weight end` 前后 20 行 |
| `unified path engaged` 后崩溃 | convert/连接器构造问题 | 贴最早 traceback |
| 服务正常但 5.3 无 Stored | store 钩子没触发或 token 对齐问题 | 贴 server 日志尾部 + scheduler 日志 |
| Stored 有但 Retrieved 0 | lookup 命中但 load 未恢复 | 检查 flush 后重试是否同前缀；贴两边日志 |
| 多轮对话失忆/胡言 | **mamba state 回灌错误**（v1 已知风险区） | 缩短对话轮次复测，贴 5.4 的完整输出 |
| 任何 `DSA`/`indexer`/`C4`/`C128` 字样的 NotImplementedError | 组件/池族覆盖缺口 | 记录并跳过该模型，报错贴回（决定 v2 范围） |

**通用规则**：任何失败保留最早一条完整 traceback（不是最后 worker 退出的
部分），`tee` 留档 `/workspace/results/`，然后贴回来。

## 里程碑状态

| 模型 | 状态 |
|---|---|
| Qwen2.5-7B（标准，旧路径） | ✅ 已首亮（2026-11-12） |
| Qwen3.8-27B（hybrid，unified） | ⬜ 代码就绪，待第 0~5 步 |
| GLM-5.3-Flash | ⬜ 待 Qwen3.8 通过后第 6 步 |
| DeepSeek-V4.1 | ⬜ 探索性，第 7 步 |
