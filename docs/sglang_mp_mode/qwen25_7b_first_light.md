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

### 3.2 下载 Qwen2.5-7B-Instruct（昇腾服务器可直接下载，推荐）

模型约 **15.4 GB**（5 个 safetensors 分片 + tokenizer + config）。你的服务器此前已确认
可通过 **netentsec 代理**访问外网，所以不必绕道 Windows 宿主机，直接在服务器上下载。

#### 3.2.0 第 0 步：连通性自测（30 秒，决定走哪条下载通道）

```bash
# 测 1：hf-mirror 国内镜像直连（不需要代理，多数内网也能通）
curl -sS -o /dev/null -w "hf-mirror: %{http_code} (%{time_total}s)\n" \
  --max-time 10 https://hf-mirror.com || echo "hf-mirror: 不可达"

# 测 2：代理通道（代理地址按 network_and_offline_install.md 里确认的那个填，
#        通常形如 http://<代理IP>:<端口>）
export https_proxy=http://<代理IP>:<端口> http_proxy=http://<代理IP>:<端口>
curl -sS -o /dev/null -w "proxy->hf: %{http_code} (%{time_total}s)\n" \
  --max-time 10 https://huggingface.co || echo "proxy->hf: 不可达"

# 测 3：ModelScope 魔搭（国内站，一般直连就通，Qwen 官方同步发布）
curl -sS -o /dev/null -w "modelscope: %{http_code} (%{time_total}s)\n" \
  --max-time 10 https://modelscope.cn || echo "modelscope: 不可达"
```

按结果选通道：**测 1 通 → 走 A1；只有测 2 通 → 走 A2；想最稳 → 走 A3**。

#### 3.2.A1 通道一：hf-mirror 直连（服务器上，免代理）

```bash
# 下载到 /data/models/（这是容器挂载卷，容器重建不丢，且容器内直接可见）
export HF_ENDPOINT=https://hf-mirror.com
pip install -U "huggingface_hub[cli]"   # 若 pip 超时，先 export 上面代理再装

huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir /data/models/Qwen2.5-7B-Instruct
```

#### 3.2.A2 通道二：代理 + HF 官方源（服务器上）

```bash
export https_proxy=http://<代理IP>:<端口> http_proxy=http://<代理IP>:<端口>
pip install -U "huggingface_hub[cli]"

huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir /data/models/Qwen2.5-7B-Instruct
# 下载期间代理变量保持导出状态；下载完成后记得 unset，避免影响后续容器内服务通信
```

#### 3.2.A3 通道三：ModelScope 魔搭（国内最快最稳，Qwen 官方同步）

```bash
pip install -U modelscope   # 同理，pip 不通就先挂代理

modelscope download --model Qwen/Qwen2.5-7B-Instruct \
  --local_dir /data/models/Qwen2.5-7B-Instruct
```

#### 3.2.1 下载后校验（无论哪条通道）

```bash
ls -lh /data/models/Qwen2.5-7B-Instruct/
# 预期看到: model-00001-of-00005.safetensors ~ model-00005-of-00005.safetensors
#          config.json  tokenizer.json  tokenizer_config.json  ...

du -sh /data/models/Qwen2.5-7B-Instruct/
# 预期约 15GB；明显偏小说明分片没下全，重跑同一条下载命令会断点续传补齐
```

> 下载目标统一放 `/data/models/`（容器挂载卷）：**如果直接在容器内的 shell 里跑上面
> 命令，文件即刻可见，跳过 3.3；如果在宿主机 shell 里跑，且 /data 的挂载源就是这个
> 目录，容器内同样直接可见，也跳过 3.3。** 只有下载到了非挂载路径才需要 3.3 的搬运。

### 3.3 （备选路线）Windows 宿主机下载后传入容器

> 仅当服务器三条下载通道全断时才走这条老路（与之前传 wheelhouse 同一流程）。

```powershell
# Windows PowerShell
$env:HF_ENDPOINT = "https://hf-mirror.com"
pip install -U "huggingface_hub[cli]"

# 下载（约 15.4GB，视网速 20~60 分钟）
huggingface-cli download Qwen/Qwen2.5-7B-Instruct `
  --local-dir "D:\hf-downloads\Qwen2.5-7B-Instruct"
```

校验（5 个分片都在才算完整）：

```powershell
Get-ChildItem "D:\hf-downloads\Qwen2.5-7B-Instruct" | Format-Table Name, Length
```

然后把文件从 Windows 传到昇腾服务器（scp / 共享目录均可），再在服务器上：

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
import json, math
c = json.load(open('/data/models/Qwen2.5-7B-Instruct/config.json'))
print("architectures:", c["architectures"])          # 预期 Qwen2ForCausalLM
print("layers:", c["num_hidden_layers"])              # 预期 28
print("q_heads:", c["num_attention_heads"])           # 预期 28
print("kv_heads:", c["num_key_value_heads"])          # 预期 4（GQA，标准模型特征）
gcd = math.gcd(c["num_attention_heads"], c["num_key_value_heads"])
valid_tp = [t for t in range(1, gcd + 1) if gcd % t == 0]
print("有效 TP 取值（头数整除约束）:", valid_tp)       # 预期 [1, 2, 4]
PY
```

`num_key_value_heads < num_attention_heads` 即为标准 GQA —— 正是我们要的。

### 3.4.1 TP 整除约束（换模型必查，实测踩坑记录）

**TP 并行要求 `num_attention_heads` 和 `num_key_value_heads` 都能被 tp_size 整除**，
否则启动即崩在 `qwen2.py` 的断言：

```text
assert self.total_num_heads % tp_size == 0
AssertionError
```

| 模型 | Q 头 | KV 头 | 最大有效 TP |
|---|---|---|---|
| Qwen2.5-7B-Instruct | 28 | 4 | **4**（1/2/4 可用，8 不可用） |
| Qwen3.8-27B（hybrid） | 能被 8 整除 | 能被 8 整除 | 8 |

Qwen3.8 基线命令里的 `--tp-size 8` **不能照搬**给 Qwen2.5-7B——2026-11-10 实测
TP8 启动即崩，本文档统一改用 **TP4**（用 4 张卡，其余 4 张空闲）。

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

与 Qwen3.8 基线命令相比，**四处变化**：模型路径、名字、**去掉 `--mamba-ssm-dtype`**
（那是 hybrid GDN 层的专属参数，标准模型不需要）、**TP8 改 TP4**（头数整除约束，
见 3.4.1 节）：

```bash
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
  2>&1 | tee /workspace/results/sglang_mp_qwen25.log
```

> 启动前**必须确认旧服务已停**（第 2 步的 npu-smi 检查）。一个实测教训：
> 2026-11-10 首次启动时日志显示 `Load weight begin. avail mem=2.59 GB`（正常应为
> 约 29 GB）——原因是 Qwen3.8 的旧服务还在跑，占着每卡约 26.5 GB。看到 avail mem
> 异常低就先回去杀进程，别改任何参数。

### 启动日志四个观察点（按语义找，不机械匹配）

| # | 找什么 | 含义 |
|---|---|---|
| 0 | `Load weight begin. avail mem≈29 GB`（不是 2.x GB） | 旧服务已清干净 |
| 1 | **过了 `lmc_radix_cache.py` 的初始化**（不再出现 `k_buffer` AttributeError） | 标准池路线打通第一关 |
| 2 | `Creating transfer context (device_type=npu, mode=lmcache_driven)` × 4 | 每个 TP rank 都连上了 NPU 数据面 |
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

- [ ] 启动无 `k_buffer` 报错，4 rank 都有 npu transfer context
- [ ] 冷请求输出正常（不是乱码/空回复）
- [ ] server 日志出现 `Stored ... tokens`
- [ ] `flush_cache` 后热请求输出正常
- [ ] server 日志出现 `Retrieved ... tokens`
- [ ] 两次输出语义一致（temperature=0 下应基本相同）
- [ ] TP4 无死锁（第二次请求能正常返回）

**全部勾上 = LMCache-Ascend MP mode 在 SGLang + 昇腾 + 标准模型上首亮成功。**

---

## 8. 失败决策树

| 症状 | 含义 | 动作 |
|---|---|---|
| `assert self.total_num_heads % tp_size == 0` AssertionError | TP 数与头数不整除（见 3.4.1） | 按 3.4 节脚本算出有效 TP，改 `--tp-size` 重启 |
| `Load weight begin. avail mem` 只有几 GB / 后续 OOM | 旧服务没停，显存被占 | `pkill -f sglang.launch_server`，npu-smi 确认回到 ~29GB 再启动 |
| 仍然 `k_buffer` AttributeError | 意外（标准模型不该出现） | 检查是不是真的用了 Qwen2.5 路径启动；贴 traceback 排查 |
| 过了 k_buffer，挂在 `LMCacheMPConnector(...)` 构造（TypeError unexpected kwarg 之类） | 旧接线与 LMCache dev 接口错配 | 留档报错；这是已知交接缝隙，按"本仓 runtime patch 包装构造"评估（见 progress_and_next_action.md 第 4 步决策表） |
| `LMCacheMPConnector` 构造中 `RuntimeError: Boolean value of Tensor with more than one value is ambiguous`（`_validate_sglang_kv_pools` 处，2026-11-12 实测） | 新版 SGLang 重构后 `k_buffer`/`v_buffer` 是**单张量**（逐层堆叠成一块），旧连接器期望**逐层 list[Tensor]**——`if not k_pool` 对大张量求布尔值必然歧义。与 NPU 无关，CUDA 上同样会挂 | ✅ **本仓补丁已实现**：`lmcache_ascend/integration/sglang/mp_connector_compat.py`（构造包装，堆叠张量→逐层零拷贝视图），由 `lmcache_ascend/__init__.py` 在 SGLang 运行时自动安装。部署见下方 8.1 |
| connector 构造成功但连不上 server（ZMQ/超时/注册失败） | 旧类与新 server 协议不兼容（第 1.3 节预判的残存风险） | 留档最早 traceback；此为路线 A 探路的结果，转入版本匹配评估 |
| `LMCache chunk size must be a multiple of SGLang page size` | chunk_size 与 page_size 不整除 | 按第 6 步查 page_size，同步改 YAML 和 server 的 `--chunk-size` 后两边重启 |
| NPU 侧报错（c_ops / IPC / transfer context） | 数据面问题——**这才是 LMCache-Ascend 的管辖范围** | 贴报错，按本仓数据面排查（对照 e2e 测试 test_npu_lmcache_driven_e2e.py 的行为差异） |
| 输出乱码或语义异常 | KV 回灌正确性问题 | 缩小前缀长度复测；对比关闭 LMCache 的裸服务输出 |

通用要求：任何失败都保留**最早一条完整 traceback**（不是最后 worker 退出的部分），
并 `tee` 留档到 `/workspace/results/`。

### 8.1 堆叠 KV 兼容补丁的部署（纯 Python，无需重编译 c_ops）

补丁涉及 3 个文件（本仓 `lmcache_ascend/` 下）：

| 文件 | 说明 |
|---|---|
| `integration/sglang/__init__.py` | 新增（包初始化） |
| `integration/sglang/mp_connector_compat.py` | 新增（构造包装：堆叠张量→逐层零拷贝视图） |
| `__init__.py` | 修改（新增 `_is_sglang_runtime` + 激活段 sglang 分支，带异常守卫） |

部署步骤（容器内执行）：

```bash
# 0) 先确认 lmcache_ascend 是哪种安装方式，决定改哪里生效
python -c "import lmcache_ascend; print(lmcache_ascend.__file__)"
#   输出 /workspace/src/LMCache-Ascend/...  → 可编辑安装，改源码即生效
#   输出 site-packages/...                  → 常规安装，需 docker cp 到对应目录（或重装）

# 1) 把 3 个文件同步进容器仓库（git pull / docker cp 均可），以 docker cp 为例：
#    docker cp mp_connector_compat.py <容器>:/workspace/src/LMCache-Ascend/lmcache_ascend/integration/sglang/
#    docker cp __init__.py <容器>:/workspace/src/LMCache-Ascend/lmcache_ascend/integration/sglang/
#    docker cp __init__.py <容器>:/workspace/src/LMCache-Ascend/lmcache_ascend/   # 注意两个 __init__.py 路径不同

# 2) 重启 SGLang（lmcache server 不用动）
```

重启后验证补丁生效的两条日志（SGLang 启动日志里 grep）：

```bash
grep -E "Installed SGLang MP connector|stacked k_pool buffer|stacked v_pool buffer" \
  /workspace/results/sglang_mp_qwen25.log
# 预期：
#   Installed SGLang MP connector stacked-KV compat patch   ← 补丁装上（每个 scheduler 进程一条）
#   SGLang stacked k_pool buffer shape=(28, ...) ... split into 28 per-layer views  ← 拆分发生
#   SGLang stacked v_pool buffer shape=(28, ...) ... split into 28 per-layer views
# shape[0]=28 即层数（Qwen2.5-7B），可同时确认堆叠顺序是"层在前"
```

若出现 `SGLang MP connector compat patch not installed: ...` 的 warning，
说明激活期导入失败（半初始化顺序问题）——把 warning 内容贴出来单独处理。

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
