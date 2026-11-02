# LMCache/LMCache-Ascend 联网代理与 Windows 离线安装完整指南

> 适用场景：服务器可以 `docker pull`、`git clone`，但 `pip install` 访问 PyPI 报 `Network is unreachable`；或者服务器完全不能访问 PyPI，需要通过 Windows 下载安装包后离线安装。
>
> 目标环境：Ascend 910B4、Linux aarch64、Python 3.11、SGLang 容器。其他环境只需替换架构和 Python 版本。
>
> 核心原则：**优先修复 pip 的网络路径；不能联网时准备与目标服务器匹配的 wheelhouse；不要为了装依赖而替换现有 torch、torch-npu 和 SGLang 科学计算栈。**

## 1. 为什么 Docker/Git 能联网，pip 却不能

三者不共享同一套网络配置：

| 操作 | 实际发起网络请求的位置 | 代理配置来源 |
|---|---|---|
| `docker pull` | 通常是宿主机 Docker daemon | Docker daemon 或 Docker client 配置 |
| `git clone` | 当前 shell 中的 Git 进程 | `http.proxy`、环境变量、系统网络 |
| `pip install` | 当前 shell/容器中的 Python pip | pip 配置、`HTTP_PROXY`/`HTTPS_PROXY` 等环境变量 |

因此可能出现：

```text
Docker daemon 有代理 → docker pull 成功
Git 配置了代理       → git clone 成功
pip 没有代理         → Network is unreachable
```

如果 pip 在容器中运行，还要额外考虑容器网络。容器里的 `127.0.0.1` 默认指向**容器自己**，不是宿主机代理。

## 2. 第一步：先做诊断，不要直接乱改配置

在运行 pip 的同一个 shell、同一个容器中执行：

```bash
uname -m
python --version
python -m pip --version
ldd --version | head -1

env | grep -i proxy || true
python -m pip config list
python -m pip config debug
git config --global --get http.proxy || true

curl -I --connect-timeout 10 https://pypi.org/simple/aiofile/
curl -I --connect-timeout 10 https://pypi.tuna.tsinghua.edu.cn/simple/aiofile/
```

记录以下信息：

- 架构：本指南按 `aarch64`；
- Python：本指南按 3.11；
- glibc：决定 manylinux wheel 是否兼容；
- 当前是否已有代理变量；
- 直接访问 PyPI 是超时、无路由、DNS 失败，还是证书错误。

### 常见报错与方向

| 现象 | 优先处理 |
|---|---|
| `Network is unreachable`、连接超时 | 配置代理或完全离线安装 |
| `Temporary failure in name resolution` | DNS/网络问题，不是 pip 包问题 |
| `407 Proxy Authentication Required` | 代理需要用户名密码 |
| `SSLError / CERTIFICATE_VERIFY_FAILED` | 企业代理 CA 证书问题 |
| `No matching distribution found` | wheel 与 Python/架构/glibc 不匹配，或镜像缺包 |
| 能访问镜像但不能访问 pypi.org | 使用可信 PyPI 镜像 |

## 3. 方案 A：服务器有 HTTP 代理，优先直接联网安装

以下用 `http://127.0.0.1:7890` 做示例，必须替换成你的真实代理地址。端口不是固定要求。

### 3.1 临时配置：只对当前 shell 生效

```bash
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export http_proxy=$HTTP_PROXY
export https_proxy=$HTTPS_PROXY
export NO_PROXY=localhost,127.0.0.1,::1
export no_proxy=$NO_PROXY
```

注意：即使访问的是 HTTPS 网站，代理 URL 通常仍是 `http://代理地址`，因为客户端通过 HTTP CONNECT 建立隧道。

先测试：

```bash
curl -I https://pypi.org/simple/aiofile/
python -m pip install --dry-run aiofile
```

确认能解析后安装：

```bash
python -m pip install aiofile aiofiles
```

然后安装 LMCache：

```bash
cd /workspace/src/LMCache
python -m pip install --no-build-isolation --no-deps -e .
```

这里保留 `--no-deps`，避免 pip 顺便升级或替换现有科学计算栈。缺少的普通 Python 依赖应显式安装。

### 3.2 只对单次 pip 命令设置代理

```bash
python -m pip --proxy http://127.0.0.1:7890 install aiofile
```

如果代理需要认证：

```bash
python -m pip --proxy http://用户名:密码@代理地址:端口 install aiofile
```

密码中的 `@`、`:`、`/`、`#` 等字符需要 URL 编码。密码可能留在 shell history 或进程信息中，不建议长期使用命令行明文方式。

### 3.3 持久化 pip 代理

```bash
python -m pip config set global.proxy http://127.0.0.1:7890
python -m pip config list
```

pip 配置文件可能包含代理密码，应限制权限：

```bash
chmod 600 ~/.config/pip/pip.conf 2>/dev/null || true
```

取消代理：

```bash
python -m pip config unset global.proxy
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
```

### 3.4 `sudo` 后代理失效

`sudo` 可能清空环境变量。优先使用当前用户和 `python -m pip`。如果必须使用 sudo，先确认不会破坏系统 Python，再使用：

```bash
sudo -E python -m pip install aiofile
```

不要盲目对系统 Python 使用 root pip。

## 4. 方案 B：代理在宿主机，但 pip 在 Docker 容器中

### 4.1 容器使用 `--net=host`

容器共享宿主机网络命名空间，宿主机代理通常可以通过下面的地址访问：

```bash
export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=$HTTPS_PROXY
```

仍需确保代理确实监听在宿主机对应端口。

### 4.2 容器使用默认 bridge 网络

容器中的 `127.0.0.1` 不是宿主机。可尝试宿主机网关：

```bash
ip route | awk '/default/ {print $3}'
```

将代理设置为：

```bash
export HTTPS_PROXY=http://上一步得到的网关IP:7890
export HTTP_PROXY=$HTTPS_PROXY
```

或者在创建容器时添加：

```bash
--add-host=host.docker.internal:host-gateway
```

然后容器内使用：

```bash
export HTTPS_PROXY=http://host.docker.internal:7890
export HTTP_PROXY=$HTTPS_PROXY
```

如果连接被拒绝，检查：

- 代理是否只监听宿主机 `127.0.0.1`；
- 代理是否允许来自 Docker 网段的连接；
- 宿主机防火墙是否放行；
- 代理端口是否正确。

已有容器不能直接补 `--add-host`，可以使用网关 IP，或按原挂载参数重建容器。

## 5. 方案 C：不改代理，使用可信 PyPI 镜像

如果只是 `pypi.org` 不可达，但国内镜像可访问：

```bash
python -m pip install \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  aiofile aiofiles
```

其他可测试镜像：

```text
https://mirrors.aliyun.com/pypi/simple/
https://repo.huaweicloud.com/repository/pypi/simple/
```

持久化示例：

```bash
python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

恢复默认：

```bash
python -m pip config unset global.index-url
```

注意：

- 镜像只解决 Python 包下载，不负责 Git 子模块、CANN、hcomm 等系统依赖；
- 不要用镜像中的通用 `torch`/`torch-npu` 覆盖 Ascend 配套版本；
- HTTPS 证书正常时不需要 `trusted-host`。

## 6. 方案 D：企业代理导致证书验证失败

如果报错是：

```text
SSLError
CERTIFICATE_VERIFY_FAILED
self-signed certificate in certificate chain
```

这通常说明企业代理在做 TLS 检查，需要安装企业根 CA，而不是简单关闭证书校验。

获取企业 CA 后，例如保存为 `/workspace/certs/company-ca.pem`：

```bash
python -m pip --cert /workspace/certs/company-ca.pem install aiofile
```

或者对当前 shell 设置：

```bash
export REQUESTS_CA_BUNDLE=/workspace/certs/company-ca.pem
export SSL_CERT_FILE=/workspace/certs/company-ca.pem
```

还应把 CA 加入系统或容器信任链，具体路径取决于发行版。不要把生产环境的 CA 私钥或内部证书提交到 Git。

`--trusted-host` 和关闭 SSL 校验会降低安全性，不建议作为正式解决方案。

## 7. 方案 E：只缺少量包，使用 Windows 浏览器下载

浏览器方式适合 `aiofile` 这类纯 Python 小包。**它不会自动解析完整依赖树**；如果缺很多包，优先使用第 8 节的跨平台 `pip download`。

### 7.1 浏览器打开文件页

例如：

- aiofile：<https://pypi.org/project/aiofile/#files>
- aiofiles：<https://pypi.org/project/aiofiles/#files>
- blake3：<https://pypi.org/project/blake3/#files>
- cachetools：<https://pypi.org/project/cachetools/#files>
- numba：<https://pypi.org/project/numba/#files>
- llvmlite：<https://pypi.org/project/llvmlite/#files>
- opentelemetry-api：<https://pypi.org/project/opentelemetry-api/#files>
- opentelemetry-sdk：<https://pypi.org/project/opentelemetry-sdk/#files>
- py-cpuinfo：<https://pypi.org/project/py-cpuinfo/#files>
- redis：<https://pypi.org/project/redis/#files>
- sortedcontainers：<https://pypi.org/project/sortedcontainers/#files>
- setuptools-scm：<https://pypi.org/project/setuptools-scm/#files>

如果 PyPI 页面无法打开，可尝试镜像站或公司制品库，但要确认包来源可信。

### 7.2 如何选择正确文件

纯 Python 包通常选择：

```text
包名-版本-py3-none-any.whl
```

例如 `aiofile`、`aiofiles`、`cachetools` 等。

包含二进制扩展的包必须同时匹配：

```text
Linux + aarch64 + CPython 3.11 + 服务器 glibc
```

典型文件名包含：

```text
cp311-cp311-manylinux_2_17_aarch64.whl
```

也可能看到：

```text
cp311-cp311-manylinux2014_aarch64.whl
```

以下文件不能用于目标服务器：

```text
win_amd64.whl
macosx_*.whl
manylinux_*_x86_64.whl
cp310 / cp312（服务器是 cp311 时）
```

`.tar.gz` 是源码包，通常需要在服务器安装编译器和头文件后构建，不是最简单的浏览器离线方案。

### 7.3 拷贝到服务器

在 Windows 中新建目录，例如：

```text
D:\wheelhouse
```

把下载的 `.whl` 全部放入。可用 WinSCP 图形界面把目录传到：

```text
/workspace/wheelhouse
```

如果 Windows 已安装 OpenSSH，也可在 PowerShell 执行：

```powershell
scp -r D:\wheelhouse root@服务器IP:/workspace/
```

服务器检查：

```bash
ls -l /workspace/wheelhouse
```

安装单个离线包：

```bash
python -m pip install \
  --no-index \
  --find-links=/workspace/wheelhouse \
  aiofile
```

如果提示缺其他依赖，说明浏览器没有下载完整依赖闭包，需要补下载或改用第 8 节。

## 8. 方案 F：Windows 跨平台下载完整 wheelhouse（推荐离线方案）

Windows 本机 pip 可以“为另一套 Linux 环境下载 wheel”，不必在 Windows 上安装这些包。

前提：

1. Windows 安装 Python 3.11；
2. 能访问 PyPI；
3. 目标包提供匹配的二进制 wheel；
4. 已知服务器是 `aarch64 + CPython 3.11`。

### 8.1 在 Windows PowerShell 执行

```powershell
$packages = @(
  "aiofile",
  "aiofiles",
  "awscrt",
  "blake3",
  "cachetools",
  "google-api-core",
  "google-cloud-bigtable",
  "grpcio-observability",
  "httptools",
  "numba",
  "opentelemetry-api",
  "opentelemetry-exporter-otlp",
  "opentelemetry-exporter-prometheus",
  "opentelemetry-sdk",
  "py-cpuinfo",
  "redis",
  "setuptools-scm",
  "sortedcontainers"
)

py -3.11 -m pip download `
  --dest D:\wheelhouse `
  --platform manylinux_2_17_aarch64 `
  --platform manylinux2014_aarch64 `
  --python-version 3.11 `
  --implementation cp `
  --abi cp311 `
  --only-binary=:all: `
  $packages
```

该命令会同时下载这些包的依赖，比浏览器逐包下载可靠。

如果 Windows pip 本身也需要代理：

```powershell
$env:HTTP_PROXY="http://127.0.0.1:7890"
$env:HTTPS_PROXY=$env:HTTP_PROXY
```

代理端口同样只是示例。

### 8.2 如果出现“找不到匹配 wheel”

可能原因：

- 包没有 aarch64 Linux wheel；
- 没有 CPython 3.11 wheel；
- 包只提供源码；
- 新 wheel 要求更高的 glibc；
- 包本身是 CUDA 专用，不适用于 NPU。

此时不要强行下载 x86_64 包，也不要去掉 `--only-binary` 后在 Windows 构建 Linux 包。正确做法是：

> 在能联网的相同 SGLang Linux aarch64 镜像中执行 `pip download`，或在该 Linux 环境中先构建 wheel，再拷贝到离线服务器。

`cufile-python` 和 `nvtx` 属于 CUDA 相关依赖，不应为了在 NPU 上消除 `pip check` 警告而强行安装。

### 8.3 传输并做离线 dry-run

将 `D:\wheelhouse` 传到服务器 `/workspace/wheelhouse`，先检查：

```bash
python -m pip install \
  --dry-run \
  --no-index \
  --find-links=/workspace/wheelhouse \
  aiofile aiofiles awscrt blake3 cachetools \
  google-api-core google-cloud-bigtable grpcio-observability \
  httptools numba \
  opentelemetry-api opentelemetry-exporter-otlp \
  opentelemetry-exporter-prometheus opentelemetry-sdk \
  py-cpuinfo redis setuptools-scm sortedcontainers
```

重点检查是否准备：

- 替换 `torch`、`torch-npu`；
- 大幅变更 `numpy`；
- 降级 `protobuf`；
- 降级 `prometheus-client`；
- 删除 SGLang 的配套包。

如果出现以上情况，先停止，不要为了消除所有依赖警告而破坏当前可运行的 SGLang 环境。

确认无误后，去掉 `--dry-run` 执行同一命令。

## 9. 最后安装本地 LMCache/LMCache-Ascend

依赖准备好后，安装本地源码时禁止再次联网和自动变更依赖：

```bash
cd /workspace/src/LMCache
python -m pip install \
  --no-index \
  --no-deps \
  --no-build-isolation \
  -e .
```

安装 Ascend 插件：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /workspace/src/LMCache-Ascend
python -m pip install \
  --no-index \
  --no-deps \
  --no-build-isolation \
  -e .
```

`--no-index` 只禁止包索引访问，不妨碍读取本地源码和本地 wheelhouse。

如果修改了 C++/AscendC 源码，需要重新执行 LMCache-Ascend 的安装命令完成编译；只改 Python 则重启相关进程即可。

## 10. 安装后的完整验证

```bash
python - <<'PY'
import torch
import torch_npu
import numba
import aiofile
import lmcache
import lmcache_ascend
import lmcache_ascend.c_ops as ops

print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("npu available:", torch.npu.is_available())
print("npu count:", torch.npu.device_count())
print("lmcache:", lmcache.__file__)
print("lmcache_ascend:", lmcache_ascend.__file__)
print("c_ops:", ops.__file__)
PY

lmcache server --help
python -m pip check
```

成功标准：

- 不再出现 `No module named 'aiofile'`；
- 不再出现 `CLI-only mode`；
- `lmcache` 指向本地 editable 源码；
- `lmcache server --help` 正常输出；
- NPU 数量和之前一致；
- `pip check` 不再有本次需要安装的 LMCache 核心依赖缺失。

`pip check` 是全环境检查。`mindstudio-kpp`、`ms-service-profiler`、`te` 等既有问题与当前任务无关，不要一并处理。

对于 LMCache 声明的 `protobuf<7`、`prometheus_client<=0.24.1` 与现有环境冲突，先记录，不要在未验证影响前直接降级。以实际 import、server 启动和接口审计结果为准。

## 11. 代理/离线安装最常见误区

1. **本地有源码不等于有完整依赖**。
2. `--no-build-isolation` 不等于 `--no-deps`，也不等于 `--no-index`。
3. Docker 能 pull 不代表容器里的 pip 自动继承代理。
4. 容器中的 `127.0.0.1` 不一定是宿主机。
5. Windows 默认下载到的是 Windows wheel，不能直接用于 Linux aarch64。
6. 浏览器下载不会自动补齐完整依赖树。
7. `pip check` 的所有警告不代表都要修复，先区分当前项目与既有工具。
8. 不要用公网通用 torch/torch-npu 覆盖 Ascend 镜像配套版本。
9. 不要把带代理密码的配置、CA 私钥或内部凭证提交到 Git。
10. 不要把 `pip install --no-deps` 成功当成运行依赖已经齐全，必须继续做 import 和 server 验证。

## 12. 推荐实际执行顺序

```text
1. 在 pip 所在容器里跑第 2 节诊断
2. 有代理 → 第 3/4 节，直接联网安装
3. 无代理但镜像可达 → 第 5 节
4. 企业证书错误 → 第 6 节
5. 只缺 aiofile 等小包 → Windows 浏览器，第 7 节
6. 缺完整依赖 → Windows 跨平台 pip download，第 8 节
7. 跨平台 wheel 不存在 → 联网的相同 Linux aarch64 镜像准备 wheelhouse
8. 离线安装依赖 → 本地 editable 安装两个仓库 → 第 10 节验证
9. 验证通过后再继续 SGLang MP 接口审计
```
