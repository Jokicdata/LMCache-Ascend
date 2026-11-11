# 全新环境从零搭建到 Qwen2.5-7B 首亮（SGLang + LMCache MP × Ascend 910B4）

> 目标读者：拿到一台**全新的** 8×Ascend 910B4 服务器，从 `docker pull` 开始，
> 一步步搭出 `SGLang → lmcache server → LMCache-Ascend NPU 数据面` 的完整闭环，
> 并用标准模型 Qwen2.5-7B-Instruct 拿到 Stored / Retrieved 证据。
>
> 本文是 [qwen25_7b_first_light.md](qwen25_7b_first_light.md)（首亮流程）与
> [environment_pitfalls_and_fixes.md](environment_pitfalls_and_fixes.md)（踩坑记录）
> 的合体"从零版"：**所有坑都已经预先避开或给出处理办法**，照抄命令即可。
>
> 预计耗时：镜像拉取 20~60 分钟，模型下载 30~60 分钟，编译安装约 30 分钟，
> 启动验证约 15 分钟。

---

## 0. 全景图：一共要做什么

```text
宿主机                          容器内（sglang-npu）
──────                          ──────────────────
① 检查 NPU/驱动/CANN
② docker pull + docker run  ──→ ③ 下载 Qwen2.5-7B 模型（/data/models/）
                                ④ 安装 Python 依赖（wheelhouse 离线装）
                                ⑤ 安装 LMCache（editable）
                                ⑥ 安装 LMCache-Ascend（editable + CPATH 修复）
                                ⑦ 安装验证（import + lmcache server --help）
                                ⑧ 接口审计 + NPU 原生测试
                                ⑨ 启动 lmcache server（终端 A）
                                ⑩ 启动 SGLang TP4（终端 B）
                                ⑪ 冷/热请求验证 Stored / Retrieved
```

**为什么这个顺序**：模型和依赖互不阻塞可以并行；两个仓库必须先装依赖再 editable
安装（否则 LMCache-Ascend 的 `_version.py` 不会生成——这是实测踩过的坑）；
测试必须在启动服务前跑（数据面坏掉时先修数据面，别在完整服务里排错）。

---

## 1. 阶段一：宿主机检查（3 分钟）

```bash
# 1) 8 张卡都在、驱动正常
npu-smi info | head -30

# 2) 设备节点存在（应有 davinci0~7 + 3 个管理节点）
ls /dev/davinci*

# 3) 确认 CANN 版本（决定后面 CPATH 修复的路径）
ls /usr/local/Ascend
# 预期看到 cann-9.0.0（本文所有命令以 9.0.0 为准；
# 如果是 8.5.x，后文 CPATH 路径相应改成 cann-8.5.x）
```

| 检查项 | 通过标志 |
|---|---|
| `npu-smi info` | 8 张 910B4，无异常告警 |
| `/dev/davinci*` | davinci0~7、davinci_manager、devmm_svm、hisi_hdc |
| CANN 目录 | 存在 `cann-9.0.0`（记下实际版本号） |

---

## 2. 阶段二：拉取镜像并启动容器（宿主机上）

### 2.1 拉镜像

910B（A2）与 910C（A3）镜像**不通用**，务必选 `910b` 后缀：

```bash
export SGLANG_IMAGE=quay.io/ascend/sglang:main-cann9.0.0-910b
docker pull "$SGLANG_IMAGE"
```

> 镜像内已集成 SGLang、torch-npu、sgl-kernel-npu、CANN、ascend attention backend，
> **不需要源码安装 SGLang**。追求稳定可换 `v0.5.x-cann9.0.0-910b` 形式的 tag。

### 2.2 启动容器

```bash
docker run -itd \
  --name sglang-npu \
  --shm-size=100g \
  --net=host \
  --ipc=host \
  --privileged=true \
  -v /workspace:/workspace \
  -v /data:/data \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
  -v /usr/local/sbin:/usr/local/sbin \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /var/queue_schedule:/var/queue_schedule \
  --device=/dev/davinci0 --device=/dev/davinci1 \
  --device=/dev/davinci2 --device=/dev/davinci3 \
  --device=/dev/davinci4 --device=/dev/davinci5 \
  --device=/dev/davinci6 --device=/dev/davinci7 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  --entrypoint=bash \
  "$SGLANG_IMAGE"

docker exec -it sglang-npu bash   # 之后所有操作都在容器内
```

每个参数的含义：

| 参数 | 为什么 |
|---|---|
| `--shm-size=100g` | SGLang 多进程走共享内存 IPC，默认 64M 会 OOM |
| `--net=host` | 共享宿主机网络，lmcache server 与 SGLang 同机直连 |
| `--ipc=host` | 共享 IPC 命名空间，进程间通信更稳 |
| `-v /data:/data` | 模型放这里，容器重建不丢 |
| `-v /workspace:/workspace` | 源码、脚本、日志、wheelhouse 都放这里 |
| 8×`davinci` + 3 个管理节点 | 昇腾容器标准挂法，缺一个 NPU 就不可用 |

### 2.3 容器内首检

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # 重复执行无害
python -c "import torch, torch_npu; print('NPU count:', torch_npu.npu.device_count())"
# 期望输出: NPU count: 8
python -c "import sglang; print('SGLang:', sglang.__version__)"
python --version   # 记下 Python 版本（本文环境为 3.11.15，后文 wheelhouse 按它准备）
```

---

## 3. 阶段三：下载 Qwen2.5-7B-Instruct 模型（容器内，可与阶段四~六并行）

### 3.1 为什么是 Qwen2.5-7B 而不是 Qwen3.8

Qwen3.8 是 hybrid（GDN/Mamba）架构，SGLang 的 LMCache 接线代码在它的混合内存池上
启动即崩（`HybridLinearKVPool has no attribute 'k_buffer'`，与昇腾无关）。
Qwen2.5-7B 是标准 GQA 模型，落在社区验证过的路径上。先用它证明"管线在昇腾上能通"，
hybrid 是后续专项。

### 3.2 先测连通性，选下载通道（30 秒）

```bash
# 测 1：hf-mirror 国内镜像直连（多数内网能通，免代理）
curl -sS -o /dev/null -w "hf-mirror: %{http_code}\n" --max-time 10 \
  https://hf-mirror.com || echo "hf-mirror: 不可达"

# 测 2：代理通道（地址按实际网络环境填）
export https_proxy=http://<代理IP>:<端口> http_proxy=http://<代理IP>:<端口>
curl -sS -o /dev/null -w "proxy->hf: %{http_code}\n" --max-time 10 \
  https://huggingface.co || echo "proxy->hf: 不可达"

# 测 3：ModelScope 魔搭（国内站，Qwen 官方同步发布，一般直连就通）
curl -sS -o /dev/null -w "modelscope: %{http_code}\n" --max-time 10 \
  https://modelscope.cn || echo "modelscope: 不可达"
```

### 3.3 下载（三条通道任选其一，约 15.4 GB）

**通道 A1：hf-mirror 直连（推荐）**

```bash
export HF_ENDPOINT=https://hf-mirror.com
pip install -U "huggingface_hub[cli]"    # pip 超时则先 export 上面的代理变量

huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir /data/models/Qwen2.5-7B-Instruct
```

**通道 A2：代理 + HF 官方源**

```bash
export https_proxy=http://<代理IP>:<端口> http_proxy=http://<代理IP>:<端口>
pip install -U "huggingface_hub[cli]"

huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir /data/models/Qwen2.5-7B-Instruct
# 下载完记得 unset https_proxy http_proxy，避免影响容器内服务通信
```

**通道 A3：ModelScope（最稳）**

```bash
pip install -U modelscope

modelscope download --model Qwen/Qwen2.5-7B-Instruct \
  --local_dir /data/models/Qwen2.5-7B-Instruct
```

### 3.4 下载后校验（必须做）

```bash
ls -lh /data/models/Qwen2.5-7B-Instruct/
# 预期: model-00001-of-00005.safetensors ~ model-00005-of-00005.safetensors
#       config.json  tokenizer.json  tokenizer_config.json ...

du -sh /data/models/Qwen2.5-7B-Instruct/
# 预期约 15GB；明显偏小 = 分片没下全，重跑同一条下载命令会断点续传

python - <<'PY'
import json, math
c = json.load(open('/data/models/Qwen2.5-7B-Instruct/config.json'))
print("architectures:", c["architectures"])   # 预期 Qwen2ForCausalLM
print("layers:", c["num_hidden_layers"])      # 预期 28
print("q_heads:", c["num_attention_heads"])   # 预期 28
print("kv_heads:", c["num_key_value_heads"])  # 预期 4（GQA 特征）
gcd = math.gcd(c["num_attention_heads"], c["num_key_value_heads"])
print("有效 TP 取值:", [t for t in range(1, gcd+1) if gcd % t == 0])
# 预期 [1, 2, 4] —— 决定后面用 TP4 而不是 TP8！
PY
```

> **TP 整除约束（实测踩坑）**：TP 要求 Q 头数和 KV 头数都能被 tp_size 整除。
> Qwen2.5-7B 是 28/4，最大 TP=4。**照搬 Qwen3.8 的 `--tp-size 8` 会启动即崩**：
> `assert self.total_num_heads % tp_size == 0`。

---

## 4. 阶段四：准备 Python 依赖 wheelhouse（宿主机上操作）

### 4.1 为什么需要这一步（核心坑）

容器里 `pip install` 访问 PyPI 大概率超时（Docker daemon、Git、pip 的代理配置
**互相独立**，宿主机能上网不代表容器里 pip 能上网）。而且：

- `--no-build-isolation` 只是不用隔离构建环境，**不会跳过依赖下载**；
- 本地有源码 ≠ 本地有依赖；
- 宿主机 Python 3.12 下载的 wheel 容器 Python 3.11 装不上（`blake3` 找不到版本）。

所以标准做法：**宿主机用"跨平台参数"下载一套 cp311 的 wheelhouse，拷进容器离线装**。

### 4.2 先测容器内 pip 是否真的不通（1 分钟，也许能省掉本阶段）

容器内执行：

```bash
curl -I --connect-timeout 10 https://pypi.org/simple/aiofile/ 2>&1 | head -2
```

- 返回 `HTTP/2 200` → 容器能直连 PyPI，跳到 4.4 用 `--index-url` 或直接在线装；
- 超时/不可达 → 走下面的 wheelhouse 流程。

### 4.3 宿主机下载 wheelhouse

在**宿主机**（Python 3.12 也行）执行：

```bash
mkdir -p /workspace/wheelhouse_cp311

python -m pip download \
  --dest /workspace/wheelhouse_cp311 \
  --platform manylinux_2_17_aarch64 \
  --platform manylinux2014_aarch64 \
  --python-version 3.11 \
  --implementation cp \
  --abi cp311 \
  --only-binary=:all: \
  aiofile aiofiles awscrt blake3 cachetools \
  google-api-core google-cloud-bigtable grpcio-observability \
  httptools numba \
  opentelemetry-api opentelemetry-exporter-otlp \
  opentelemetry-exporter-prometheus opentelemetry-sdk \
  py-cpuinfo redis setuptools-scm sortedcontainers
```

参数含义：

| 参数 | 作用 |
|---|---|
| `--python-version 3.11 --abi cp311` | 匹配容器 Python 3.11（最关键，否则装不上） |
| `--platform manylinux*_aarch64` | 匹配服务器 ARM64 Linux |
| `--only-binary=:all:` | 只下 wheel 不下源码，避免在宿主机错误构建 |

> 如果宿主机 pip 也需要代理：`export https_proxy=... http_proxy=...` 后再跑。
> 如果某包报"找不到匹配 wheel"：说明它没有 aarch64 cp311 wheel，此时应在一台
> **能联网的相同 aarch64 Linux 环境**里跑同样的 `pip download`。
> `cufile-python`、`nvtx` 是 CUDA 专属依赖，NPU 环境不用装。

`/workspace` 是挂载卷的话宿主机下载完容器内立即可见；否则 `docker cp`：

```bash
docker cp /path/to/wheelhouse_cp311 sglang-npu:/workspace/
```

### 4.4 容器内离线安装依赖

```bash
# dry-run 先看会动哪些包（重点：不能替换 torch/torch-npu、不能大幅降级 numpy）
python -m pip install \
  --dry-run \
  --no-index \
  --find-links=/workspace/wheelhouse_cp311 \
  aiofile aiofiles awscrt blake3 cachetools \
  google-api-core google-cloud-bigtable grpcio-observability \
  httptools numba \
  opentelemetry-api opentelemetry-exporter-otlp \
  opentelemetry-exporter-prometheus opentelemetry-sdk \
  py-cpuinfo redis setuptools-scm sortedcontainers
```

确认不会动 `torch`/`torch-npu`/SGLang 配套包后，去掉 `--dry-run` 正式执行同一命令。

> pip 末尾可能输出一串 `ERROR: ... requires ...` —— 那是**全环境依赖检查报告**，
> 不是安装失败。判断标准看最后的 `Successfully installed ...`。其中
> `ms-service-profiler`、`mindstudio-kpp` 等是容器既有工具的冲突，**不用管**。
> `prometheus-client`/`protobuf` 的版本冲突先记录不动，以实际运行验证为准。

**为什么这批依赖里必须包含 `setuptools-scm`**：LMCache-Ascend 构建时靠它生成
`_version.py`。实测教训：第一次没装它就编译，c_ops 编译成功了但
`import lmcache_ascend` 报 `No module named 'lmcache_ascend._version'`，只能重装。

---

## 5. 阶段五：安装 LMCache（容器内）

```bash
mkdir -p /workspace/src /workspace/scripts /workspace/results
cd /workspace/src

git clone https://github.com/LMCache/LMCache.git
cd LMCache

# 稳定路线：checkout v0.5.6 tag（本仓 LMCACHE_UPSTREAM_TAG 指向它）
git fetch origin --tags
git checkout v0.5.6
# 看到 "HEAD is now at 3e11e108..." 和 detached HEAD 提示 = 正常，不是报错
# （tag 不是分支，只读安装保持 detached 即可；要改源码再 git switch -c <分支>）

git submodule update --init --recursive

# 离线 editable 安装：只装本项目，不联网，不动依赖
python -m pip install \
  --no-index \
  --no-deps \
  --no-build-isolation \
  -e .
```

> 可选：`git fetch origin` 可能报 `nightly-rbln would clobber existing tag` ——
> 是远程滚动 tag 与本地不一致，Git 拒绝覆盖，**与 v0.5.6 无关，忽略**。

---

## 6. 阶段六：安装 LMCache-Ascend（容器内，有一个必须的 CPATH 修复）

### 6.1 克隆（含子模块）

```bash
cd /workspace/src

git clone --recurse-submodules \
  https://github.com/LMCache/LMCache-Ascend.git
cd LMCache-Ascend
git submodule update --init --recursive

# 确认 hcomm 子模块状态（应显示 v9.1.0，且与 git ls-tree 记录一致）
git submodule status third_party/hcomm
```

> **hcomm 不要动**：CANN 9.0.0 不要求机械搭配 hcomm 9.0.0。父仓库固定的 v9.1.0
> 就是匹配版本。实测 `git checkout v9.0.0` 会被本地修改阻止（Git 保护机制），
> 也**不应该**强行切。

### 6.2 CPATH 修复（CANN 9.0.0 已知坑，不做必挂）

CANN 9.0.0 下 HIXL 编译目标缺一个 include 路径，会报：

```text
prof_api.h:21:10: fatal error: prof_common.h: No such file or directory
```

文件其实存在于 `/usr/local/Ascend/cann-9.0.0/aarch64-linux/pkg_inc/profiling/`，
只是编译器没搜到。**编译前必须 export CPATH**：

```bash
export CPATH=/usr/local/Ascend/cann-9.0.0/aarch64-linux/pkg_inc/profiling${CPATH:+:$CPATH}
```

> CPATH 只对当前 shell 的编译有效。**每次新开 shell 重新编译前都要重新 export**；
> 运行已编译好的插件不需要它。CANN 是 8.5.x 就把路径换成 `cann-8.5.x`。

### 6.3 编译安装

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CPATH=/usr/local/Ascend/cann-9.0.0/aarch64-linux/pkg_inc/profiling${CPATH:+:$CPATH}

cd /workspace/src/LMCache-Ascend
python -m pip install -v \
  --no-index \
  --no-deps \
  --no-build-isolation \
  -e . \
  2>&1 | tee /workspace/results/lmcache_ascend_build.log
```

编译约 10~20 分钟。**判断成败看最后是否 `Successfully installed`**。
如果失败，不要看末尾的 setuptools 长篇提示，直接搜第一处真实错误：

```bash
grep -nEi 'fatal error|error:|CMake Error|FAILED:|undefined reference|Killed' \
  /workspace/results/lmcache_ascend_build.log | head -20
```

| 第一处错误 | 处理 |
|---|---|
| `prof_common.h: No such file` | CPATH 没设（6.2 节），新 shell 忘了重新 export |
| `Killed` / out of memory | 并行编译内存不足，`export MAX_JOBS=2` 后重试 |
| 找不到 `third_party/hcomm` | 子模块没初始化，重跑 6.1 的 submodule 命令 |

> 不要创建空 `prof_common.h`，不要从别的 CANN 版本拷头文件，不要切 hcomm 版本。

---

## 7. 阶段七：安装验证（容器内）

```bash
python - <<'PY'
import torch
import torch_npu
import numba
import sortedcontainers
import aiofile
import lmcache
import lmcache_ascend
import lmcache_ascend.c_ops as ops

print("lmcache:", lmcache.__file__)
print("lmcache_ascend:", lmcache_ascend.__file__)
print("c_ops:", ops.__file__)
print("npu_available:", torch.npu.is_available())
print("npu_count:", torch.npu.device_count())
PY

lmcache server --help
```

**通过标志（全部满足才能往下走）**：

| 检查项 | 通过标志 | 不通过说明 |
|---|---|---|
| import 全部成功 | 无 ModuleNotFoundError | 缺哪个包回 4.4 节补装 |
| 无 `CLI-only mode` 警告 | 日志无 `No compute backend loaded` | numba 没装好 |
| 无 `lightweight` 警告 | 无 `lmcache-cli (lightweight) detected` | sortedcontainers 缺 |
| NPU 识别 | `npu_available: True`，`npu_count: 8` | 容器设备挂载问题 |
| `lmcache server --help` | 显示 `--host --port --l1-size-gb --chunk-size` 等**完整参数** | 只显示 `--format/--output` = lightweight 模式 |
| `lmcache.__file__` | 指向 `/workspace/src/LMCache/...` | 装到了别的 site-packages |

> `lmcache server --help` 的 banner 显示 `LMCache vunknown` 是版本元数据问题，
> **不影响功能**，可忽略。

---

## 8. 阶段八：接口审计 + NPU 原生测试（启动服务前的最后关卡）

### 8.1 接口审计（确认上游 SGLang MP 接口都在）

```bash
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
    print(f"{name}:", importlib.util.find_spec(name) is not None)

from lmcache.integration.sglang.unified_lmcache_mp_connector import (
    UnifiedLMCacheMPConnector,
)
from lmcache.integration.sglang.unified_kv_adapter import SGLangUnifiedKVAdapter
from lmcache.v1.platform.devices.npu import NpuDeviceSpec
from lmcache.v1.platform.devices.npu.ipc_wrapper import NpuIPCWrapper
print("All key imports: OK")
PY
```

**通过标志**：所有模块 `True`，最后 `All key imports: OK`。

### 8.2 NPU 原生 MP 数据面测试

```bash
# 关键：不设 LMCACHEPATH 会去联网 clone 上游源码（离线环境必挂）
export LMCACHEPATH=/workspace/src/LMCache

cd /workspace/src/LMCache-Ascend
pytest -q tests/v1/multiprocess/test_npu_lmcache_driven_e2e.py -s \
  2>&1 | tee /workspace/results/npu_mp_native_e2e.log
```

**通过标志**：`3 passed`，且日志中出现（npu:0 和 npu:1 各一次）：

```text
Stored 256 tokens
Retrieved 256 tokens
```

> 这个测试验证的是"NPU 跨进程搬运 KV 数据正确"（数据面），**不**代表 SGLang
> 控制面已通。日志里的 `NPUIPCTypes.cpp` 警告和 deprecation warnings 可忽略。
> 为什么 3 passed 还不够，详见 [native_test_vs_engine_e2e.md](native_test_vs_engine_e2e.md)。

---

## 9. 阶段九：启动 lmcache server（终端 A）

### 9.1 写 SGLang 侧配置

```bash
mkdir -p /workspace/scripts /workspace/results

cat > /workspace/scripts/lmcache_mp.yaml <<EOF
mp_host: tcp://127.0.0.1
mp_port: 5556
chunk_size: 256
EOF
```

> chunk_size=256 对常见 page_size（1/2/4/.../128/256）都是整数倍。启动后第 11 步
> 会实测 page_size 并核查对齐；若实测是 512+ 再两边同步改。

### 9.2 启动 server

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

**通过标志**：无 traceback，server 就绪日志出现。保持终端开着。

---

## 10. 阶段十：启动 SGLang（终端 B，新 shell 记得重设环境）

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

**与 Qwen3.8 命令的三处不同（都踩过坑）**：

| 项 | Qwen3.8 | Qwen2.5-7B | 原因 |
|---|---|---|---|
| TP | 8 | **4** | 头数 28/4 整除约束，TP8 启动即崩 |
| `--mamba-ssm-dtype bfloat16` | 需要 | **去掉** | hybrid GDN 层专属参数，标准模型没有 |
| chunk 对齐 | 复杂（hybrid） | 256 即可 | 标准 GQA 路径 |

**启动日志四个观察点**：

| # | 找什么 | 含义 |
|---|---|---|
| 0 | `Load weight begin. avail mem≈29 GB` | 旧服务没占显存（若只有几 GB = 有残留进程，回去 `pkill -f sglang.launch_server`） |
| 1 | 顺利过了 `lmc_radix_cache.py` 初始化 | 标准内存池路线通了（不再有 k_buffer 报错） |
| 2 | `Creating transfer context (device_type=npu, mode=lmcache_driven)` × 4 | 每个 TP rank 都连上 NPU 数据面 |
| 3 | 终端 A 的 server 日志出现 KV 注册条目 | connector 与 server 完成握手 |

任何观察点失败 → 保存**最早一条完整 traceback**（不是最后 worker 退出的部分），
按 [qwen25_7b_first_light.md](qwen25_7b_first_light.md) 第 8 节决策树处理。

### 10.1 服务就绪后核查 page_size 对齐

```bash
curl -s http://127.0.0.1:8998/get_server_info | python -m json.tool | grep -iE "page_size|block" | head -5
```

规则：**chunk_size 必须是 page_size 的整数倍**。page_size ≤ 256 → 什么都不用改；
512+ → 把 YAML 的 `chunk_size` 和 server 的 `--chunk-size` 都改成该值，两边重启。

---

## 11. 阶段十一：验证缓存闭环（本教程的验收标准）

### 11.1 构造长前缀请求（前缀必须超过 256 token）

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

### 11.2 三连验证

```bash
# ① 冷请求（第一次见这个前缀 → 应触发 Stored）
curl -s http://127.0.0.1:8998/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @/workspace/scripts/probe_request.json | head -c 300; echo

# ② 清掉 SGLang 本地 RadixCache（关键！否则第二次只命中本地缓存，测不出 LMCache）
curl -s -X POST http://127.0.0.1:8998/flush_cache; echo

# ③ 热请求（同前缀 → 应触发 Retrieved）
curl -s http://127.0.0.1:8998/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @/workspace/scripts/probe_request.json | head -c 300; echo

# ④ 看证据
grep -E "Stored|Retrieved" /workspace/results/lmcache_server_qwen25.log | tail -20
```

### 11.3 验收清单（全勾 = 首亮成功）

- [ ] 启动无 `k_buffer` 报错，4 个 rank 都有 npu transfer context
- [ ] 冷请求输出正常（不是乱码/空回复）
- [ ] server 日志出现 `Stored ... tokens`
- [ ] `flush_cache` 后热请求输出正常
- [ ] server 日志出现 `Retrieved ... tokens`
- [ ] 两次输出语义一致（temperature=0 下应基本相同）
- [ ] TP4 无死锁（第二次请求能正常返回）

**全部勾上 = LMCache-Ascend MP mode 在 SGLang + 昇腾 + 标准模型上首亮成功。**
之后回填 [qwen25_7b_first_light.md](qwen25_7b_first_light.md) 末尾的实测记录小节。

---

## 12. 附录：全流程坑点速查表

| # | 阶段 | 坑 | 一句话解法 |
|---|---|---|---|
| 1 | 容器 | 镜像选错 A3 | 必须选 `910b` 后缀 |
| 2 | 容器 | `--shm-size` 太小 | 100g |
| 3 | 模型 | 照搬 TP8 | Qwen2.5-7B 用 **TP4**（头数 28/4） |
| 4 | 模型 | 分片没下全 | `du -sh` 校验约 15GB，重跑命令断点续传 |
| 5 | pip | 容器访问 PyPI 超时 | 宿主机 wheelhouse + 容器 `--no-index` 离线装 |
| 6 | pip | 宿主机 3.12 下载容器 3.11 装不上 | `--python-version 3.11 --abi cp311 --platform aarch64` |
| 7 | pip | `ERROR: requires ...` 一大串 | 看最后 `Successfully installed`，既有工具冲突不管 |
| 8 | 安装 | `--no-build-isolation` 仍联网 | 它不跳依赖；离线要加 `--no-index --no-deps` |
| 9 | LMCache | checkout tag 后 detached HEAD | 正常现象，不是报错 |
| 10 | LMCache-Ascend | `prof_common.h: No such file` | 编译前 `export CPATH=.../pkg_inc/profiling` |
| 11 | LMCache-Ascend | `_version.py` 不生成 | setuptools-scm 要在编译**前**装好（4.4 节已含） |
| 12 | LMCache-Ascend | hcomm checkout 被拒 | 不要切版本，保持父仓库固定的 v9.1.0 |
| 13 | 验证 | `CLI-only mode` | numba 缺，4.4 节补装 |
| 14 | 验证 | `lmcache-cli (lightweight)` | sortedcontainers 缺，4.4 节补装 |
| 15 | 测试 | pytest 尝试联网 clone | `export LMCACHEPATH=/workspace/src/LMCache` |
| 16 | 启动 | avail mem 只有几 GB | 旧服务没停，`pkill -f sglang.launch_server` |
| 17 | 启动 | 标准模型带 `--mamba-ssm-dtype` | 去掉，那是 hybrid 专属 |
| 18 | 验证 | 第二次没有 Retrieved | 忘了 `flush_cache`（本地缓存抢走了命中） |
| 19 | 通用 | 反斜杠续行后写注释 | 长命令存成 .sh 再跑 |
| 20 | 通用 | 只看最后一条报错 | 永远留档**最早**的完整 traceback |

## 13. 与既有文档的关系

| 文档 | 角色 |
|---|---|
| **本文** | 全新环境从零搭建 + 首亮的**总入口** |
| [qwen25_7b_first_light.md](qwen25_7b_first_light.md) | 首亮流程的原理详解、失败决策树、实测记录 |
| [environment_pitfalls_and_fixes.md](environment_pitfalls_and_fixes.md) | 每个坑的根因深挖 |
| [network_and_offline_install.md](network_and_offline_install.md) | 代理/镜像/Windows 下载的全量方案 |
| [native_test_vs_engine_e2e.md](native_test_vs_engine_e2e.md) | 为什么 3 passed 还不够（测试分级） |
| [progress_and_next_action.md](progress_and_next_action.md) | 总进度、决策树、改动量预估 |
