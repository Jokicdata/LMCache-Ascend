# LMCache 离线可编辑安装处理计划

## Summary

当前报错不是本地源码缺失，而是 pip 在安装项目时仍需解析并安装运行依赖。`aiofile` 不在本机可用包中，pip 因此访问 PyPI，在离线机器上失败。

目标是在不破坏现有 SGLang、torch、torch-npu 环境的前提下完成 LMCache 的可编辑安装，并继续后续接口审计。

## Current State Analysis

- 报错发生在 `/workspace/src/LMCache` 的 editable install 阶段。
- `--no-build-isolation` 只表示不创建隔离构建环境、不自动安装构建依赖，**不会跳过运行依赖解析**。
- 本地有 LMCache 源码不等于本地有其全部 Python 依赖。
- 当前最直接缺失的是 `aiofile`；继续安装后可能还发现其他依赖，需通过 `pip check` 和 import 验证，而不是假定只缺一个包。
- 现有 Ascend 科学计算栈不应被公网 PyPI 的通用 torch 包替换。

## Proposed Changes / Execution Steps

### 1. 先采用不访问网络的安装方式

在服务器上检查依赖并安装本地项目：

```bash
cd /workspace/src/LMCache
python -m pip show aiofile || true
python -m pip install --no-index --no-deps --no-build-isolation -e .
python -m pip check
```

参数分工：

- `--no-index`：禁止访问 PyPI；
- `--no-deps`：不自动解析/安装项目运行依赖；
- `--no-build-isolation`：使用当前环境中的构建工具。

如果 `pip check` 和后续 import 均通过，则无需额外操作。

### 2. 若仍缺 `aiofile`，从联网机器准备最小 wheelhouse

优先在**相同容器镜像、相同 Python 版本和相同 CPU 架构**的联网环境中执行：

```bash
mkdir -p wheelhouse
python -m pip download --dest wheelhouse aiofile
```

将 `wheelhouse` 拷贝到离线服务器，例如 `/workspace/wheelhouse`，然后执行：

```bash
python -m pip install --no-index --find-links=/workspace/wheelhouse aiofile
cd /workspace/src/LMCache
python -m pip install --no-index --no-deps --no-build-isolation -e .
```

不要用不匹配的 Windows wheel 或 x86_64 wheel 安装到 aarch64 Linux 服务器。

### 3. 若后续仍有多项依赖缺失，准备完整依赖闭包

在相同运行环境的联网机器上，基于同一份 LMCache 源码执行：

```bash
python -m pip download --dest wheelhouse /path/to/LMCache
```

将完整 wheelhouse 拷贝到离线端，先检查解析是否可离线完成：

```bash
cd /workspace/src/LMCache
python -m pip install --dry-run --no-index \
  --find-links=/workspace/wheelhouse --no-build-isolation -e .
```

解析成功后再去掉 `--dry-run` 安装。如果解析会升级/替换 torch、torch-npu 等 Ascend 配套包，则不要直接全量安装；改为只安装缺失的普通 Python 依赖，再用 `--no-deps` 安装 LMCache。

### 4. 安装完成后验证

```bash
python -m pip check
python - <<'PY'
import aiofile
import lmcache
print("aiofile:", aiofile.__file__)
print("lmcache:", lmcache.__file__)
PY
lmcache server --help
```

随后继续执行 LMCache-Ascend 安装和 SGLang MP 接口审计。

## Assumptions & Decisions

- 优先保护当前已经能运行 Qwen3.8 的 SGLang/NPU 环境，不允许 pip 为解决依赖而任意升级科学计算栈。
- 先走最小依赖修复；只有出现更多缺失时才制作完整 wheelhouse。
- 不通过反复访问 PyPI、更换公网镜像或关闭 SSL 来绕过离线问题。
- 本步骤只解决 Python 打包与依赖问题，不代表 SGLang MP 适配已经验证通过。

## Verification

完成标准：

1. editable install 不再访问网络并成功结束；
2. `pip show lmcache` 指向 `/workspace/src/LMCache`；
3. `aiofile` 和 `lmcache` 可导入；
4. `lmcache server --help` 正常输出；
5. `pip check` 没有 LMCache 相关缺依赖错误；
6. 原有 torch、torch-npu、SGLang 版本未被意外替换。
