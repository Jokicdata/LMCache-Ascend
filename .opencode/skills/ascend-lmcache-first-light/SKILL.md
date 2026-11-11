---
name: ascend-lmcache-first-light
description: "在全新 Ascend 910B4 服务器上从零搭建 SGLang + LMCache MP + LMCache-Ascend 完整闭环（容器、模型、依赖、编译、验证、首亮）。当用户拿到新昇腾环境要部署 LMCache-Ascend 栈、要求重建环境、或提到首亮/first light/Stored/Retrieved 验证时调用。"
license: Apache-2.0
compatibility: opencode
metadata:
  hardware: Ascend 910B4 (A2)
  model: Qwen2.5-7B-Instruct
  pipeline: SGLang + lmcache server + LMCache-Ascend MP
---

# Ascend 910B4 全新环境搭建 SGLang + LMCache MP 首亮 Playbook

在任意一台**全新的** 8×Ascend 910B4 服务器上，从零搭出
`SGLang → lmcache server → LMCache-Ascend NPU 数据面` 闭环，
并用标准模型 Qwen2.5-7B-Instruct 拿到 Stored / Retrieved 证据。

**详细原理、失败决策树、每步"为什么"见**（路径相对本 SKILL.md）：

- `../../../docs/sglang_mp_mode/fresh_env_qwen25_first_light.md`（从零搭建总入口）
- `../../../docs/sglang_mp_mode/environment_pitfalls_and_fixes.md`（坑点根因）
- `../../../docs/sglang_mp_mode/qwen25_7b_first_light.md`（首亮原理 + 失败决策树）

本 skill 是可执行浓缩版。**执行时严格按阶段顺序，每阶段核对通过标志后才进入下一阶段；出问题先查上述文档再动手。**

---

## 开工前必须收集的参数（先探测或问用户，不要猜）

| 参数 | 探测方法 | 默认值 |
|---|---|---|
| CANN 版本 | 宿主机 `ls /usr/local/Ascend` | `cann-9.0.0` |
| 容器内 pip 是否能访问 PyPI | 容器内 `curl -I --connect-timeout 10 https://pypi.org/simple/aiofile/` | 假设不通，走 wheelhouse |
| 下载/git 代理地址（如需要） | 问用户，或宿主机 `env \| grep -i proxy` | 无 |
| 容器名 | — | `sglang-npu` |
| 目标标准模型 | 问用户 | `Qwen/Qwen2.5-7B-Instruct` |

**内网注意**：若 GitHub / quay.io 不可直连，优先问用户内网镜像地址
（内部 git 托管 / 内部 registry），或让用户先手工把两个源码仓库和镜像搬运到位，
skill 从"容器内首检"阶段继续。**TP 数必须在下载模型后用 config.json 算出来，
禁止照搬其他模型的 TP**（见阶段 3）。

---

## 铁律（20 个实测坑，违反任何一个必返工）

1. 镜像必须 `910b` 后缀（A2/A3 不通用）。
2. 容器必须 `--shm-size=100g`。
3. `/data` 和 `/workspace` 必须是挂载卷（容器重建不丢）。
4. 标准模型 TP 由头数整除约束决定：Qwen2.5-7B（28Q/4KV 头）**只能 TP1/2/4**。
5. 宿主机下载 wheel 必须 `--python-version 3.11 --abi cp311 --platform manylinux*_aarch64`（匹配容器 Python，先查实际版本）。
6. 容器离线安装依赖用 `--no-index --find-links=<wheelhouse>`。
7. `setuptools-scm` 必须在编译 LMCache-Ascend **之前**装好，否则 `_version.py` 不生成。
8. editable 安装两个仓库统一用 `--no-index --no-deps --no-build-isolation -e .`。
9. 编译 LMCache-Ascend 前必须 `export CPATH=$CANN/aarch64-linux/pkg_inc/profiling:$CPATH`，**每个新 shell 重新 export**。
10. hcomm 子模块**不要切版本**，保持父仓库固定值。
11. 编译失败看**第一处** `fatal error/CMake Error`，不看末尾 setuptools 提示。
12. pip 输出的 `ERROR: requires...` 是全环境报告，成败只看 `Successfully installed` + import 验证。
13. `ms-service-profiler`/`mindstudio-kpp` 等既有工具的依赖冲突**不处理**。
14. `cufile-python`/`nvtx` 是 CUDA 包，NPU 不装。
15. 跑 pytest 前必须 `export LMCACHEPATH=/workspace/src/LMCache`（防联网 clone）。
16. 启动 SGLang 前必须 `npu-smi info` 确认旧服务已停（avail mem ≈ 29GB 才对）。
17. 标准模型**不带** `--mamba-ssm-dtype`（hybrid 专属参数）。
18. 验证 Retrieved 前必须先 `POST /flush_cache`（否则本地 RadixCache 抢走命中）。
19. 长命令存成 `.sh` 执行，反斜杠续行后**不写注释**。
20. 任何失败留档**最早**的完整 traceback 到 `/workspace/results/`。

---

## 阶段 1：宿主机检查

```bash
npu-smi info | head -30          # 8 张 910B4
ls /dev/davinci*                 # davinci0~7 + manager/devmm_svm/hisi_hdc
ls /usr/local/Ascend             # 记下 CANN 版本 → 后续 CPATH 用
```

## 阶段 2：拉镜像 + 起容器

```bash
export SGLANG_IMAGE=quay.io/ascend/sglang:main-cann9.0.0-910b
docker pull "$SGLANG_IMAGE"

docker run -itd \
  --name sglang-npu \
  --shm-size=100g --net=host --ipc=host --privileged=true \
  -v /workspace:/workspace -v /data:/data \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
  -v /usr/local/sbin:/usr/local/sbin \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /var/queue_schedule:/var/queue_schedule \
  --device=/dev/davinci0 --device=/dev/davinci1 \
  --device=/dev/davinci2 --device=/dev/davinci3 \
  --device=/dev/davinci4 --device=/dev/davinci5 \
  --device=/dev/davinci6 --device=/dev/davinci7 \
  --device=/dev/davinci_manager --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  --entrypoint=bash \
  "$SGLANG_IMAGE"

docker exec -it sglang-npu bash
```

容器内首检（记下 Python 版本，阶段 4 的 wheelhouse 按它准备）：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -c "import torch, torch_npu; print(torch_npu.npu.device_count())"   # 8
python -c "import sglang; print(sglang.__version__)"
python --version
```

## 阶段 3：下载标准模型（容器内）

**若容器 pip 通**，先测下载通道再下载（不通则代理变量先 export）：

```bash
# 通道自测：hf-mirror / 代理→HF / ModelScope 三选一
curl -sS -o /dev/null -w "%{http_code}\n" --max-time 10 https://hf-mirror.com

# 首选 hf-mirror：
export HF_ENDPOINT=https://hf-mirror.com
pip install -U "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir /data/models/Qwen2.5-7B-Instruct

# 备选 ModelScope：
# pip install -U modelscope
# modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir /data/models/Qwen2.5-7B-Instruct
```

下载后**必须**校验并算 TP：

```bash
du -sh /data/models/Qwen2.5-7B-Instruct/   # ≈15GB，偏小=分片不全，重跑续传
python - <<'PY'
import json, math
c = json.load(open('/data/models/Qwen2.5-7B-Instruct/config.json'))
print(c["architectures"])   # Qwen2ForCausalLM
gcd = math.gcd(c["num_attention_heads"], c["num_key_value_heads"])
print("valid TP:", [t for t in range(1, gcd+1) if gcd % t == 0])  # [1,2,4] → 用 4
PY
```

## 阶段 4：Python 依赖（容器 pip 通 → 直接装；不通 → 宿主机 wheelhouse）

**容器 pip 通时**（跳过 wheelhouse）：

```bash
python -m pip install aiofile aiofiles awscrt blake3 cachetools \
  google-api-core google-cloud-bigtable grpcio-observability httptools numba \
  opentelemetry-api opentelemetry-exporter-otlp opentelemetry-exporter-prometheus \
  opentelemetry-sdk py-cpuinfo redis setuptools-scm sortedcontainers
```

**不通时**，宿主机执行（注意 cp311/aarch64 按容器实际版本替换）：

```bash
mkdir -p /workspace/wheelhouse_cp311
python -m pip download \
  --dest /workspace/wheelhouse_cp311 \
  --platform manylinux_2_17_aarch64 --platform manylinux2014_aarch64 \
  --python-version 3.11 --implementation cp --abi cp311 \
  --only-binary=:all: \
  aiofile aiofiles awscrt blake3 cachetools \
  google-api-core google-cloud-bigtable grpcio-observability \
  httptools numba \
  opentelemetry-api opentelemetry-exporter-otlp \
  opentelemetry-exporter-prometheus opentelemetry-sdk \
  py-cpuinfo redis setuptools-scm sortedcontainers
# /workspace 非挂载时：docker cp /path/wheelhouse_cp311 sglang-npu:/workspace/
```

容器内安装（先 dry-run 确认不动 torch/torch-npu）：

```bash
python -m pip install --dry-run --no-index \
  --find-links=/workspace/wheelhouse_cp311 \
  aiofile aiofiles awscrt blake3 cachetools google-api-core \
  google-cloud-bigtable grpcio-observability httptools numba \
  opentelemetry-api opentelemetry-exporter-otlp \
  opentelemetry-exporter-prometheus opentelemetry-sdk \
  py-cpuinfo redis setuptools-scm sortedcontainers
# 确认无 torch 替换后去掉 --dry-run 重跑
```

## 阶段 5：安装 LMCache（容器内）

```bash
mkdir -p /workspace/src /workspace/scripts /workspace/results
cd /workspace/src
git clone https://github.com/LMCache/LMCache.git
cd LMCache
git fetch origin --tags
git checkout v0.5.6        # detached HEAD 提示是正常的
git submodule update --init --recursive

python -m pip install --no-index --no-deps --no-build-isolation -e .
```

## 阶段 6：安装 LMCache-Ascend（容器内，CPATH 必设）

```bash
cd /workspace/src
git clone --recurse-submodules https://github.com/LMCache/LMCache-Ascend.git
cd LMCache-Ascend
git submodule update --init --recursive
git submodule status third_party/hcomm    # 确认状态，不要切版本

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CPATH=/usr/local/Ascend/cann-9.0.0/aarch64-linux/pkg_inc/profiling${CPATH:+:$CPATH}
# ★ CANN 版本不同就换路径；每个新 shell 编译前都要重新 export

python -m pip install -v --no-index --no-deps --no-build-isolation -e . \
  2>&1 | tee /workspace/results/lmcache_ascend_build.log
```

编译失败时：

```bash
grep -nEi 'fatal error|error:|CMake Error|FAILED:|Killed' \
  /workspace/results/lmcache_ascend_build.log | head -20
```

- `prof_common.h: No such file` → CPATH 没设
- `Killed` → `export MAX_JOBS=2` 重试
- 找不到 hcomm → 子模块没初始化

## 阶段 7：安装验证（全过才继续）

```bash
python - <<'PY'
import torch, torch_npu, numba, sortedcontainers, aiofile
import lmcache, lmcache_ascend
import lmcache_ascend.c_ops as ops
print("lmcache:", lmcache.__file__)
print("lmcache_ascend:", lmcache_ascend.__file__)
print("c_ops:", ops.__file__)
print("npu_available:", torch.npu.is_available())
print("npu_count:", torch.npu.device_count())
PY
lmcache server --help
```

通过标准：import 全过、无 `CLI-only mode`、无 `lightweight`、
`--help` 显示 `--host/--port/--l1-size-gb/--chunk-size` 完整参数、NPU count=8。

## 阶段 8：接口审计 + NPU 原生测试

```bash
export LMCACHEPATH=/workspace/src/LMCache    # ★ 防联网 clone

# 审计（所有模块 True、imports OK）
python - <<'PY'
import importlib.util, lmcache, sglang
for m in ["lmcache.integration.sglang.unified_lmcache_mp_connector",
          "lmcache.integration.sglang.unified_kv_adapter",
          "lmcache.v1.platform.devices.npu",
          "lmcache.v1.platform.devices.npu.ipc_wrapper",
          "sglang.srt.mem_cache.storage.lmcache.lmc_radix_cache"]:
    print(m, importlib.util.find_spec(m) is not None)
from lmcache.integration.sglang.unified_lmcache_mp_connector import UnifiedLMCacheMPConnector
print("key imports: OK")
PY

# 原生数据面测试（预期 3 passed + Stored/Retrieved 256 tokens × 2 设备）
cd /workspace/src/LMCache-Ascend
pytest -q tests/v1/multiprocess/test_npu_lmcache_driven_e2e.py -s \
  2>&1 | tee /workspace/results/npu_mp_native_e2e.log
```

## 阶段 9：启动 lmcache server（终端 A）

```bash
mkdir -p /workspace/scripts /workspace/results
cat > /workspace/scripts/lmcache_mp.yaml <<EOF
mp_host: tcp://127.0.0.1
mp_port: 5556
chunk_size: 256
EOF

export PYTHONHASHSEED=0
lmcache server \
  --host 127.0.0.1 --port 5556 --http-port 8080 \
  --chunk-size 256 --l1-size-gb 20 --eviction-policy LRU \
  --supported-transfer-mode lmcache_driven \
  2>&1 | tee /workspace/results/lmcache_server_qwen25.log
```

## 阶段 10：启动 SGLang（终端 B）

先 `npu-smi info` 确认无残留进程（avail mem ≈ 29GB）。
TP 用阶段 3 算出的值（Qwen2.5-7B = 4）：

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

日志观察点：① avail mem≈29GB ② 过 lmc_radix_cache 初始化 ③
`Creating transfer context (device_type=npu, mode=lmcache_driven)` × TP 数 ④
server 端出现注册条目。

就绪后核查对齐（chunk 必须是 page 的整数倍，否则两边同步改后重启）：

```bash
curl -s http://127.0.0.1:8998/get_server_info | python -m json.tool | grep -iE "page_size|block" | head -5
```

## 阶段 11：首亮验证（验收标准）

```bash
# 构造长前缀请求（必须超过 chunk_size=256 token）
python - <<'PY'
import json
prefix = "以下是关于分布式KV缓存系统的一次内部技术评审记录，涉及内存池分层与传输通道选型。" * 120
payload = {"model": "Qwen2.5-7B-Instruct",
           "messages": [{"role": "user", "content": prefix + "\n\n请用一句话总结上面的内容。"}],
           "max_tokens": 32, "temperature": 0}
open("/workspace/scripts/probe_request.json", "w").write(json.dumps(payload))
PY

# ① 冷请求 → ② flush 本地缓存 → ③ 热请求 → ④ 看证据
curl -s http://127.0.0.1:8998/v1/chat/completions -H "Content-Type: application/json" \
  -d @/workspace/scripts/probe_request.json | head -c 300; echo
curl -s -X POST http://127.0.0.1:8998/flush_cache; echo
curl -s http://127.0.0.1:8998/v1/chat/completions -H "Content-Type: application/json" \
  -d @/workspace/scripts/probe_request.json | head -c 300; echo
grep -E "Stored|Retrieved" /workspace/results/lmcache_server_qwen25.log | tail -20
```

**验收清单（全勾 = 首亮成功）**：

- [ ] 启动无 k_buffer 报错，TP 数个 rank 都有 npu transfer context
- [ ] 冷请求输出正常，server 日志有 `Stored ... tokens`
- [ ] flush 后热请求输出正常，server 日志有 `Retrieved ... tokens`
- [ ] 两次输出语义一致（temperature=0）
- [ ] 无死锁

## 失败处理

任何阶段失败：留档**最早**的完整 traceback 到 `/workspace/results/`，然后查
`../../../docs/sglang_mp_mode/qwen25_7b_first_light.md` 第 8 节失败决策树
（含 connector 构造错配、ZMQ 协议不兼容、chunk/page 不整除、NPU c_ops 等症状的分流）。

## 完成后

1. 同前缀冷/热各测 5 次记录 TTFT，与裸 SGLang 对比；
2. 回填实测记录到 `docs/sglang_mp_mode/qwen25_7b_first_light.md` 末尾；
3. hybrid 模型（Qwen3.8）是独立专项，不要在本流程混做。
