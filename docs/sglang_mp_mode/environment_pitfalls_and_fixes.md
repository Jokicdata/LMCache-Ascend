# SGLang + LMCache-Ascend 环境搭建踩坑与解决方案记录

> 本文记录本次在 8×Ascend 910B4 上，从 SGLang 启动 Qwen3.8、运行 benchmark，到离线安装 LMCache/LMCache-Ascend 的完整排障过程。
>
> 目的：后续重建环境或遇到同类报错时，可以直接按“报错现象 → 根因 → 处理命令”定位，不必重复踩坑。

## 1. 当前环境与当前状态

本次使用的关键环境：

| 项目 | 当前情况 |
|---|---|
| 硬件 | 8×Ascend 910B4（A2） |
| 模型目录 | `/data/models/Qwen` |
| 推理引擎 | SGLang，Qwen3.8 混合 GDN/Mamba + 全注意力模型 |
| CANN | 9.0.0，路径 `/usr/local/Ascend/cann-9.0.0` |
| hcomm | v9.1.0，子模块 commit 与父仓库一致 |
| 容器 Python | 3.11（实际为 3.11.15） |
| 宿主机 Python | 3.12 |
| LMCache | 本地 editable 源码安装 |
| LMCache-Ascend | 本地 editable 源码安装，包含原生 `c_ops` 编译 |

当前已经做到：

- SGLang 裸推理服务和 benchmark 可以运行；
- LMCache 能识别 `torch_device_type=npu`；
- LMCache-Ascend 的 `_version.py` 问题已通过重新安装解决；
- `lmcache server --help` 能输出完整帮助信息。

这意味着环境搭建基本完成，下一步应做接口审计和 NPU 原生 MP 测试，而不是继续反复安装。

## 2. 问题总表

| 阶段 | 报错/现象 | 最终处理 |
|---|---|---|
| SGLang 启动 | `FP32 initial_state is only supported on Ascend950` | 加 `--mamba-ssm-dtype bfloat16` |
| benchmark | 去 HuggingFace 下载 config/tokenizer/ShareGPT | 使用本地模型路径；离线时选 `random-ids` 或提供本地语料 |
| benchmark 命令 | `--backend: command not found` | 去掉续行符后的注释，改成脚本文件 |
| 模型路径 | `/workspace/models/...` 不存在 | 统一改为 `/data/models/Qwen`，容器挂载 `/data:/data` |
| Git | `nightly-rbln would clobber existing tag` | 与 v0.5.6 无关，忽略该滚动 tag |
| Git | checkout tag 后 `detached HEAD` | 正常现象；开发前 `git switch -c <branch>` |
| pip 网络 | 容器访问 PyPI/镜像均超时 | 配置容器代理，或宿主机下载 wheelhouse 后 `docker cp` |
| pip 参数 | 有源码仍访问 PyPI | `--no-build-isolation` 不跳过运行依赖；离线加 `--no-index --no-deps` |
| wheel 兼容 | 宿主机 3.12 下载，容器 3.11 找不到包 | 用 cp311 跨平台参数重新下载干净 wheelhouse |
| 原生编译 | `Failed to build c_ops.so` | 查看第一处 CMake/编译错误，不看末尾 setuptools 提示 |
| 原生编译 | `prof_common.h: No such file or directory` | 文件实际存在；给 HIXL 编译补 `pkg_inc/profiling` |
| hcomm | `v9.9.0` 不存在、checkout `v9.0.0` 被 dirty 阻止 | 不切换；保留父仓库固定的 v9.1.0 |
| 运行时 | `No module named numba` / CLI-only | 从本地 wheelhouse 安装 numba |
| 运行时 | `sortedcontainers` 缺失 / lightweight CLI | 安装 sortedcontainers |
| 包初始化 | `No module named lmcache_ascend._version` | 安装 setuptools-scm 后重新 editable 安装 LMCache-Ascend |
| pip check | 大量 requires / incompatible | 区分缺包、版本冲突和无关工具；以 import/server 验证为准 |

下面逐项记录。

## 3. SGLang 启动 Qwen3.8：FP32 initial_state 不支持

### 现象

```text
RuntimeError: FP32 initial_state is only supported on Ascend950 (arch35)
```

调用位置在 NPU GDN backend 的 `chunk_gated_delta_rule`。

### 根因

Qwen3.8 包含 GDN/线性注意力层。当前 CANN 算子在 910B（A2）上不接受 FP32 recurrent state；FP32 state 是 Ascend950 对应路径的支持范围。

### 解决

在原有 SGLang 启动参数上追加：

```bash
--mamba-ssm-dtype bfloat16
```

这个参数决定 Mamba/GDN 的 SSM state 精度，与 LMCache 是否启用是两个独立问题。后续带 LMCache 启动时仍需保留。

## 4. Benchmark 阶段的问题

### 4.1 压测客户端访问 HuggingFace 失败

#### 现象

```text
Network is unreachable
huggingface.co/.../config.json
huggingface.co/datasets/.../ShareGPT_V3_unfiltered_cleaned_split.json
```

#### 根因

`sglang.benchmark.serving` 是压测客户端，不是推理服务。它在发请求前需要：

- 加载 tokenizer 来统计 token；
- 准备 prompt 数据集。

把 `--model Qwen3.8-27B` 当成 HF 仓库名时，会尝试联网；random 数据集还可能需要语料文件。

#### 解决

模型和 tokenizer 使用本地路径，并单独指定 served name：

```bash
--model /data/models/Qwen
--served-model-name Qwen3.8-27B
--tokenizer /data/models/Qwen
```

完全离线且不需要真实文本时，优先使用：

```bash
--dataset-name random-ids
```

它不需要 ShareGPT 语料。若使用 `random` 的真实文本采样，则需保证 `--dataset-path` 指向**真实存在且合法**的 JSON 文件。

### 4.2 长命令变成 `--backend: command not found`

#### 根因

shell 续行符 `\` 后又有空格和行尾注释，导致续行被破坏，下一行被当成独立命令。

#### 解决

不要在反斜杠后写注释。将长命令保存为 `bench.sh` 再执行：

```bash
bash /workspace/scripts/bench_random_ids.sh
```

### 4.3 模型目录修正

当前模型目录统一为：

```text
/data/models/Qwen
```

因此：

```bash
--model-path /data/models/Qwen
```

如果模型在宿主机而推理在容器中，还要保证容器启动时挂载：

```bash
-v /data:/data
```

## 5. Git 版本与分支问题

### 5.1 `nightly-rbln would clobber existing tag`

这是远程滚动 tag 与本地同名 tag 指向不同 commit，Git 拒绝自动覆盖。与固定版本 `v0.5.6` 无关，不需要为其删除或强更新 tag。

### 5.2 checkout tag 后进入 detached HEAD

`v0.5.6` 是 tag，不是分支。以下输出表示 checkout 已成功：

```text
HEAD is now at 3e11e108 ...
```

只读安装可以保持；需要修改源码时创建分支：

```bash
git switch -c sglang-mp-adapt
```

## 6. pip 网络：为什么 Docker/Git 可以，pip 不行

### 现象

容器内：

```text
Failed to connect to pypi.org port 443: Connection timed out
Failed to connect to pypi.tuna.tsinghua.edu.cn port 443: Connection timed out
```

宿主机却能访问，并返回：

```text
HTTP/1.1 200 Connection established
Proxy-agent: netentsec
HTTP/2 200
```

### 根因

Docker daemon、Git、pip 的网络和代理配置彼此独立。宿主机使用了 `netentsec` 代理，容器没有自动继承。

### 可选解决方案

1. 有代理：在容器内配置 `HTTP_PROXY`/`HTTPS_PROXY`；
2. 容器是 bridge 网络时，不能直接用宿主机 `127.0.0.1`，需使用宿主机网关或 LAN IP；
3. 最稳妥：宿主机直接下载 wheelhouse，然后：

   ```bash
   docker cp /workspace/wheelhouse_cp311 <容器名>:/workspace/
   ```

本次最终采用了宿主机下载 + 容器离线安装，避免继续修改容器网络。

## 7. 离线安装参数到底什么意思

以下命令只安装本地项目，不访问 PyPI，也不自动变更依赖：

```bash
python -m pip install \
  --no-index \
  --no-deps \
  --no-build-isolation \
  -e .
```

| 参数 | 含义 |
|---|---|
| `--no-index` | 禁止访问 PyPI |
| `--no-deps` | 不自动安装/升级运行依赖 |
| `--no-build-isolation` | 使用当前环境的构建依赖和 CANN 工具链 |
| `-e .` | editable 安装当前源码目录 |

关键教训：**本地有源码不代表本地有全部依赖。** `--no-deps` 成功也不代表运行依赖已经齐全，必须继续做 import 验证。

## 8. 宿主机 Python 3.12、容器 Python 3.11

### 现象

宿主机下载的包拷入容器后：

```text
ERROR: No matching distribution found for blake3
```

而 `aiofile` 等纯 Python 包能识别。

### 根因

宿主机 pip 默认按 Python 3.12 下载 wheel，容器需要 cp311。`py3-none-any` 可以通用，`cp311-abi3` 也能用，但 cp312 二进制 wheel 不行。

### 解决

在宿主机用跨版本参数重新下载一个干净目录：

```bash
python -m pip download \
  --dest /workspace/wheelhouse_cp311 \
  --platform manylinux_2_17_aarch64 \
  --platform manylinux2014_aarch64 \
  --python-version 3.11 \
  --implementation cp \
  --abi cp311 \
  --only-binary=:all: \
  <包名列表>
```

然后把整个 `wheelhouse_cp311` 拷入容器。容器安装时：

```bash
python -m pip install \
  --no-index \
  --find-links=/workspace/wheelhouse_cp311 \
  <包名列表>
```

## 9. 如何理解 pip 的 `ERROR` / `requires`

### `requires X, which is not installed`

表示某包声明的依赖尚未安装。例如：

```text
lmcache requires aiofile, which is not installed.
```

### `requires X, but you have Y which is incompatible`

表示版本不符合声明。例如：

```text
lmcache requires protobuf<7, but you have protobuf 7.36.2.
```

这代表“未满足声明的版本约束”，不等于一定运行失败，也不应该不加判断地全局降级。

### 其他工具的冲突

`ms-service-profiler`、`mindstudio-kpp`、`te` 等报错属于容器既有工具，不是当前 LMCache 任务，不要一并处理。

### 为什么有 ERROR，最后仍 Successfully installed

pip 会安装目标包，然后检查整个环境并报告冲突。判断本次是否安装成功应看：

- 最后的 `Successfully installed ...`；
- import 是否成功；
- `lmcache server --help` 是否完整；
- NPU 是否被识别。

## 10. LMCache-Ascend 原生编译失败

### 10.1 不要只看最后的 RuntimeError

末尾只看到：

```text
Failed to build c_ops.so
cmake ... returned non-zero exit status 2
```

这只是外层包装。正确排查方式是保存完整日志并找第一处真实错误：

```bash
python -m pip install -v \
  --no-index --no-deps --no-build-isolation -e . \
  2>&1 | tee /workspace/results/lmcache_ascend_build.log

grep -nEi \
  'fatal error|error:|CMake Error|FAILED:|undefined reference|Killed' \
  /workspace/results/lmcache_ascend_build.log | head -80
```

不要盲目升级 setuptools。

### 10.2 `prof_common.h` 实际存在，但编译器找不到

真实错误：

```text
prof_api.h:21:10: fatal error: prof_common.h: No such file or directory
```

检查后确认文件存在于：

```text
/usr/local/Ascend/cann-9.0.0/aarch64-linux/pkg_inc/profiling/prof_common.h
```

问题是 HIXL 编译目标没有把这个目录加入 include 搜索路径，不是文件不存在。

本次有效处理：

```bash
export CPATH=/usr/local/Ascend/cann-9.0.0/aarch64-linux/pkg_inc/profiling${CPATH:+:$CPATH}

cd /workspace/src/LMCache-Ascend
python -m pip install -v \
  --no-index --no-deps --no-build-isolation -e .
```

`CPATH` 只对当前 shell 的编译有效。新开 shell 后重新编译，需要再次 export；运行已编译的插件不依赖它。

永久修复可以在 `csrc/hixl/CMakeLists.txt` 的 `target_include_directories` 中增加：

```cmake
${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/pkg_inc/profiling
```

不要创建空 `prof_common.h`，也不要从其他 CANN 版本随意复制头文件。

## 11. hcomm 版本与 checkout 报错

当前实际状态：

```text
git describe: v9.1.0-dirty
submodule commit 与 git ls-tree 记录一致
```

这说明当前 hcomm 正是 LMCache-Ascend 固定的子模块版本。CANN 9.0.0 不要求机械地搭配同名 hcomm 9.0.0。

以下两个操作都没有必要：

```bash
git checkout v9.9.0   # tag 不存在
git checkout v9.0.0   # 本地修改会被覆盖，Git 拒绝切换
```

第二个报错是 Git 的保护机制，不是构建故障。不要为了切版本执行 `reset --hard` 或删除本地文件。既然已经成功编译，保持 v9.1.0 即可。

## 12. 运行时缺包与初始化问题

### 12.1 `No module named numba`

导致：

```text
No compute backend loaded; CLI-only mode
```

处理：

```bash
python -m pip install \
  --no-index --find-links=/workspace/wheelhouse_cp311 \
  numba
```

同时 pip 自动安装匹配依赖 `llvmlite`。

### 12.2 `sortedcontainers` 缺失

导致：

```text
lmcache-cli (lightweight) detected
```

处理：

```bash
python -m pip install \
  --no-index --find-links=/workspace/wheelhouse_cp311 \
  sortedcontainers
```

安装后 `lmcache server --help` 才能显示完整 server 参数。

### 12.3 `No module named lmcache_ascend._version`

原生扩展已编译，但 Python 包初始化失败：

```text
ModuleNotFoundError: No module named 'lmcache_ascend._version'
```

原因是第一次构建时 `setuptools-scm` 尚未安装，版本文件没有生成。后续 `device_ops` 导入失败也是这个初始化错误引起的连锁反应。

处理：安装 `setuptools-scm` 后重新 editable 安装 LMCache-Ascend，并保留编译所需 `CPATH`：

```bash
export CPATH=/usr/local/Ascend/cann-9.0.0/aarch64-linux/pkg_inc/profiling${CPATH:+:$CPATH}
cd /workspace/src/LMCache-Ascend
python -m pip install -v \
  --no-index --no-deps --no-build-isolation -e .
```

## 13. 当前仍需记录但不必盲目处理的版本冲突

安装日志中仍可能看到：

- `grpcio-observability==1.78.0` 与新版本的冲突；
- `opentelemetry-api<=1.40.0` 与新版本的冲突；
- `opentelemetry-exporter-prometheus<=0.61b0` 与新版本的冲突；
- `prometheus-client<=0.24.1` 与当前版本的冲突；
- `protobuf<7` 与当前版本的冲突；
- `cufile-python`、`nvtx` 未安装。

处理原则：

1. 先以 import、NPU 识别和 `lmcache server --help` 验证；
2. 出现真实 API 不兼容时，再针对具体包准备指定版本；
3. 不为消除所有 pip check 输出而破坏 SGLang 环境；
4. CUDA 专属包不为 NPU 强装。

## 14. 最终验证与下一步

当前无报错且 `lmcache server --help` 输出完整后，执行下面的工作，不再继续折腾安装。

### 第一步：执行 SGLang MP 接口审计

运行 `docs/sglang_mp_mode/adaptation_walkthrough.md` 第 5.1 节的完整 Python 脚本，保存：

```text
/workspace/results/sglang_mp_api_audit.log
```

再执行第 5.2 节的 `grep`，确认 SGLang 实际挂接点和 MAMBA component。

### 第二步：执行 NPU 原生 MP 测试

```bash
mkdir -p /workspace/results
cd /workspace/src/LMCache-Ascend
pytest -q tests/v1/multiprocess/test_npu_lmcache_driven_e2e.py -s \
  2>&1 | tee /workspace/results/npu_mp_native_e2e.log
```

该测试通过只证明 NPU MP 数据面可用，不等于 SGLang 控制面已经接通。

### 第三步：根据结果决定下一步

- 接口齐全且原生测试通过：直接使用现有上游实现做第一次 SGLang MP 启动；
- 接口缺失：按具体缺失模块检查版本和源码路径；
- 原生测试失败：先修 NPU IPC/块传输，再启动完整 TP8 服务；
- 第一次启动失败：保留最早完整异常栈，按 SGLang、上游 LMCache、LMCache-Ascend 分层定位。

核心原则：**先审计，后改代码；先验证数据面，再验证控制面；先正确性，后性能。**
