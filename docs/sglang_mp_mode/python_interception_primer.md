# "拦截"是什么：Python 语法入门 + 本仓补丁逐行精讲

> 阅读门槛：**不需要精通 Python**。本文从"函数是什么"讲起，每个语法点都配
> 可运行的小例子，最后把 [mp_connector_compat.py](../../lmcache_ascend/integration/sglang/mp_connector_compat.py)
> 的每一行和语法点一一对应。
>
> 前置阅读：[call_flow_before_after_adaptation.md](call_flow_before_after_adaptation.md)
> （知道补丁在整条链路里的位置，本文只讲"这个位置上的技术"）。

---

## 1. 术语澄清：这个技术叫什么

| 说法 | 含义 |
|---|---|
| **拦截**（口语） | 别人调用某函数时，先经过我写的代码 |
| **monkey patch / 猴子补丁**（正式术语） | **程序运行时**，把某个类/模块的属性（通常是方法）替换成自己的版本 |
| **方法包装 / wrapping**（本补丁的做法） | monkey patch 的一种：新方法内部先做点事，再调用**原方法**，不丢弃原逻辑 |

三个常见近亲（都是同一族技术）：

| 技术 | 你见过的例子 | 和本补丁的关系 |
|---|---|---|
| **装饰器** `@xxx` | Python 官方语法糖 | 完全同源：装饰器就是"函数版"的包装，本补丁是"类方法版"的包装 |
| `unittest.mock.patch` | 测试框架里替换函数 | 同族：也是运行时换属性 |
| **本仓既有 patch** | `_seed_partial_lmcache_device_ops`、`shape_desc` 补丁 | 全是 monkey patch，本补丁只是最新一个 |

**为什么用它**：SGLang 和 LMCache 的源码我们一行都不想改（改了以后升级全是麻烦），
但又必须改变 `LMCacheMPConnector` 的行为 → 只能在运行时"偷换"它的方法。

---

## 2. 铺垫一：Python 的函数是"东西"，可以装进变量

这是理解一切的前提。C/Java 里函数和普通数据是两种东西；**Python 里函数是对象
（object），和数字、字符串一样，是个"东西"**，可以被赋值、传递、存放：

```python
def hello(name):
    return f"hello {name}"

greet = hello        # 注意：没有括号！不是调用，是把"函数本身"装进变量
print(greet("Tom"))  # hello Tom —— 变量里的函数照样能调用

import math
m = math.sqrt        # 模块里的函数也能取出来装进变量
print(m(16))         # 4.0
```

关键区分：

```python
hello("Tom")   # 有括号 = 立刻执行函数，拿到"结果"
hello          # 无括号 = 拿到"函数本身"这个对象
```

本补丁的第一行核心代码就是"无括号取函数"：

```python
original_init = connector_cls.__init__   # 把类的构造方法取出来存好（没调用它）
```

## 3. 铺垫二：方法存在"类"身上，实例只是去类里找

```python
class Fridge:
    def __init__(self):      # 方法定义在类的身体里
        self.stock = []

f1 = Fridge()
f1.stock.append("apple")

print(Fridge.__init__)   # 方法真实存放地点：类身上 ✅
print(f1.__init__)       # 实例上"看得到"它，但那只是转交（bound method）
```

实例调用方法的实际过程：

```text
f1.stock.append("x")
   │
   ▼
Python 在 f1 自己身上找 "stock"  → 找到
f1.stock.append
   │
   ▼
在 f1.stock（列表对象）的"类"（list）身上找 "append" → 找到
   │
   ▼
执行 list.append(f1.stock, "x")     ← self 被自动塞进去
```

**由此得出拦截的钥匙**：方法既然是"类身上的一个属性"，而 Python 允许给属性
重新赋值——那么**给类的 `__init__` 重新赋值 = 换掉这个类的构造逻辑**。

## 4. 铺垫三：`MyClass(...)` 这行代码，Python 内部做了什么

你每天都在写 `conn = LMCacheMPConnector(k_pool=..., v_pool=...)`，但 Python
看到这行时实际执行的是三步：

```python
conn = LMCacheMPConnector(k_pool=t1, v_pool=t2)
```

```text
第 1 步  type(LMCacheMPConnector).__call__(LMCacheMPConnector, k_pool=t1, v_pool=t2)
         （调用"类的类"的 __call__，这是 new 一个对象的总入口）
第 2 步  分配一块内存，造出空对象 obj
第 3 步  LMCacheMPConnector.__init__(obj, k_pool=t1, v_pool=t2)
         ↑↑↑ 看这里！Python 拿着【类身上的 __init__ 属性】来初始化 obj
             —— 而这个属性，我们已经在启动早期换成了 _compat_init
             —— 所以第 3 步实际执行的是 _compat_init(obj, k_pool=..., v_pool=...)
```

`self` 是什么：**就是第 3 步里那个刚造出来的空对象 `obj`**。Python 自动把它
塞成第一个参数，所以你在方法里写的 `self.xxx = ...` 其实是在往这个新对象
身上挂属性。

**这就是"拦截"的全部原理**：不改 `MyClass(...)` 的调用方，不改原 `__init__`
的代码，只在两者之间的"类属性查表"这个环节做了一次偷换。

---

## 5. 正片：拦截的三个动作

```python
# ── 动作一：把原方法存好（无括号，取函数本身） ──────────────
original_init = connector_cls.__init__

# ── 动作二：定义新方法（内部先翻译参数，再转交原方法） ──────
def _compat_init(self, *args, **kwargs):
    if "k_pool" in kwargs:
        kwargs["k_pool"] = _split_stacked_pool(kwargs["k_pool"], "k_pool")
    if "v_pool" in kwargs:
        kwargs["v_pool"] = _split_stacked_pool(kwargs["v_pool"], "v_pool")
    return original_init(self, *args, **kwargs)

# ── 动作三：换上去（类属性赋值） ────────────────────────────
connector_cls.__init__ = _compat_init
```

动作三执行后的世界：

```text
                    ┌────────────── LMCacheMPConnector（类）──────────────┐
                    │  __init__ ──────────────► _compat_init   ← 换上的新表项
                    │  lookup_kv / store_kv ...► 原来那些（没动）
                    └────────────────────────────────────────────────────┘
                                            ▲
SGLang 代码（未改一行）：                     │ 类属性查表
LMCacheMPConnector(k_pool=..., v_pool=...) ──┘
```

---

## 6. 涉及的 5 个 Python 语法点逐个讲

### 6.1 `*args, **kwargs` —— "收下一切参数，再原样转交"

`def f(self, *args, **kwargs)` 里的两个星号是**打包**语法：

| 写法 | 含义 |
|---|---|
| `*args` | 把所有**按位置**传的参数收进一个元组 `args` |
| `**kwargs` | 把所有**按名字**传的参数收进一个字典 `kwargs`（kwargs = keyword arguments） |

```python
def show(*args, **kwargs):
    print("位置参数元组:", args)
    print("关键字参数字典:", kwargs)

show(1, 2, k_pool="大张量", v_pool="大张量")
# 位置参数元组: (1, 2)
# 关键字参数字典: {'k_pool': '大张量', 'v_pool': '大张量'}
```

调用时的 `f(*args, **kwargs)` 是**解包**（把元组和字典摊开还原成参数）：

```python
original_init(self, *args, **kwargs)
# 等价于：original_init(self, 第1个位置参数, 第2个..., k_pool=..., v_pool=...)
```

**为什么包装器必须用这套语法**：原构造函数有 10 个参数，将来还可能增减。
写 `*args, **kwargs` 就不用逐一列举——**收下一切，改两个，全部转交**。
这是包装器的万能模板。

配合**字典操作**就能精准改参数：

```python
if "k_pool" in kwargs:            # in 判断：字典里有没有这个键
    kwargs["k_pool"] = 新值       # 有就改写（只改这一个键，其它不动）
```

### 6.2 闭包（closure）—— 内层函数"记住"外层的变量

```python
original_init = connector_cls.__init__      # 外层变量

def _compat_init(self, *args, **kwargs):
    ...
    return original_init(self, *args, ...)  # ← 内层函数引用了外层变量！
```

问题：`install_mp_connector_compat()` 执行完就返回了，它的局部变量
`original_init` 按理应该消失——为什么 `_compat_init` 以后还能用它？

**闭包规则：只要内层函数还引用着外层函数的变量，Python 就把这些变量
"打包"跟着内层函数一起存活**（这个打包体就叫闭包）。

```python
def make_counter():
    count = 0                    # 外层变量
    def inner():
        nonlocal count           # 声明：我要用外层的 count（写它时需要 nonlocal）
        count += 1
        return count
    return inner                 # 把内层函数返回出去

c = make_counter()   # make_counter 已返回，count 却"活"在 c 身上
print(c(), c(), c()) # 1 2 3
```

在本补丁里：`original_init`（原构造函数）被打包进了 `_compat_init` 的闭包里。
**只要 `_compat_init` 活着（它挂在类上，类活多久它活多久），原构造函数就
被安全地保管着**——这就是"翻译完还能转交给原作者"的机制。

### 6.3 `setattr` / `getattr` —— 用字符串名字读写属性

```python
setattr(obj, "name", value)      # 等价于 obj.name = value
getattr(obj, "name")             # 等价于 obj.name
getattr(obj, "name", default)    # 带默认值：属性不存在时返回 default 而不报错
```

为什么有了点号语法还要这两个函数？——**当属性名是变量/在配置里/可能不存在时**：

```python
# 本补丁的幂等检查：
flag = getattr(connector_cls.__init__, "_lmcache_ascend_mp_compat", False)
#                                              ↑ 属性名是常量        ↑ 没有这个属性时
#                                                                    返回 False 而不是崩
if flag:
    return                      # 已经装过 → 不再装第二层
```

`getattr` 带默认值是 Python 里"安全探测"的惯用法（否则要用 try/except）。

### 6.4 函数身上挂属性 —— 幂等标记的实现

函数是对象 → **对象就能挂属性**（就像 `self.stock` 一样）：

```python
def my_func():
    pass

my_func._patched = True          # 给函数本身挂一个属性
print(getattr(my_func, "_patched", False))   # True
```

本补丁用它做**幂等标记**（防止装两层）：

```python
setattr(_compat_init, _INIT_PATCHED_FLAG, True)   # 装之前，在新函数身上盖章
...
if getattr(connector_cls.__init__, _INIT_PATCHED_FLAG, False):
    return                                        # 下次再装时：发现已盖过章 → 退出
```

为什么要防重复：本插件在 8 卡场景下每个 scheduler 进程各自激活一次（进程隔离，
不会重复）；但同一进程里如果 lmcache 被多处触发加载，或用户手动调了两次
install，就会包装两层——功能还能跑，但日志翻倍、白做一层翻译。加个章，
第二次调用立刻退出。

### 6.5 `__name__` 赋值 —— 让报错信息不说谎

```python
_compat_init.__name__ = "LMCacheMPConnector.__init__"
```

Python 的函数有个 `__name__` 属性（就是定义时写的名字）。如果将来原构造
函数里某行又抛了新错，traceback 会打印"出错的函数名"——不改名的话会显示
`_compat_init`（用户不认识），改了显示 `LMCacheMPConnector.__init__`
（一看就知道是连接器构造）。**一行改名，省掉将来排错时的困惑。**

---

## 7. 20 行最小可运行示例（冰箱水果版）

下面这段完整复刻了本补丁的结构，任何装了 Python 的机器（包括你的 Windows）
都能直接跑。**强烈建议亲手跑一遍**：

```python
# fridge_demo.py —— 模拟：上游校验只认列表，新版数据源给的是字符串
class Fridge:
    def __init__(self, fruit):          # ← 模拟 LMCacheMPConnector.__init__
        if not fruit:                   # ← 模拟 _validate_sglang_kv_pools 的
            raise ValueError("empty")   #    "if not k_pool"（只对列表合法）
        self.fruit = fruit
        print(f"原构造执行：收到 {type(fruit).__name__}，长度 {len(fruit)}")

# ====== 下面是补丁（结构和 mp_connector_compat.py 一一对应） ======
original_init = Fridge.__init__                     # 动作一：存好原方法

def _compat_init(self, *args, **kwargs):            # 动作二：定义包装器
    if "fruit" in kwargs and isinstance(kwargs["fruit"], str):
        print("包装器：检测到字符串（=堆叠张量），翻译成列表（=逐层视图）...")
        kwargs["fruit"] = list(kwargs["fruit"])     #   翻译：str → list
    return original_init(self, *args, **kwargs)     #   转交原方法

_compat_init.__name__ = "Fridge.__init__"           # 改名（traceback 友好）
Fridge.__init__ = _compat_init                      # 动作三：换上去

# ====== 验证 ======
print("--- 适配后：可以跑了 ---")
Fridge(fruit="apple")      # 字符串被翻译成 ['a','p','p','l','e']，校验通过

print("--- 幂等：再装一次会退出 ---")
flag = getattr(Fridge.__init__, "_patched", False)
print("已装过？", flag)    # 本 demo 没盖章，正式代码里有，见 6.4
```

运行结果：

```text
--- 适配后：可以跑了 ---
包装器：检测到字符串（=堆叠张量），翻译成列表（=逐层视图）...
原构造执行：收到 list，长度 5
--- 幂等：再装一次会退出 ---
已装过？ False
```

对照表——demo 的每个角色对应真实补丁的什么：

| demo 里 | 真实补丁里 |
|---|---|
| `Fridge` | `LMCacheMPConnector` |
| `fruit` 字符串（一整串） | `k_pool` 堆叠张量 `(28, 12203, 128, 1, 128)` |
| `list("apple")` → `['a','p','p','l','e']` | `_split_stacked_pool` → 28 个逐层视图 |
| `if not fruit` 校验 | `_validate_sglang_kv_pools` 的 `if not k_pool` |
| 调用方 `Fridge(fruit="apple")` 无需改动 | SGLang 的 `lmc_radix_cache.py` 无需改动 |

---

## 8. 真实代码逐行注释版

把 [mp_connector_compat.py](../../lmcache_ascend/integration/sglang/mp_connector_compat.py)
的 `install_mp_connector_compat` 完整抄下来，每行标注用到的语法点：

```python
def install_mp_connector_compat() -> None:
    import lmcache.integration.sglang.multi_process_adapter as mpa
    #       └─ import 语句：把 LMCache 的模块对象拿进来（才能改它）

    connector_cls = getattr(mpa, "LMCacheMPConnector", None)
    #               └─ 6.3 getattr：安全取属性；名字是字符串常量，不存在得 None

    if connector_cls is None:
        return          # 类不存在（LMCache 未来删了旧类）→ 静默退出，不炸

    if getattr(connector_cls.__init__, _INIT_PATCHED_FLAG, False):
        return          # 6.3+6.4：探测幂等标记，已装过 → 退出
    #        └─ 此时 __init__ 还是原函数（没有标记 → False）

    original_init = connector_cls.__init__
    #               └─ 2 节：无括号取"函数本身"，存进变量

    def _compat_init(self, *args, **kwargs):
    #                └─ 6.1 *args/**kwargs：收下一切参数
        if "k_pool" in kwargs:
        #          └─ 字典的 in 判断
            kwargs["k_pool"] = _split_stacked_pool(kwargs["k_pool"], "k_pool")
            #  └─ 字典按 key 改值（只动这一个键）
        if "v_pool" in kwargs:
            kwargs["v_pool"] = _split_stacked_pool(kwargs["v_pool"], "v_pool")
        return original_init(self, *args, **kwargs)
        #       └─ 6.2 闭包：引用外层的 original_init
        #                            └─ 6.1 解包：原样转交全部参数

    _compat_init.__name__ = "LMCacheMPConnector.__init__"
    #             └─ 6.5 改名：traceback 友好

    setattr(_compat_init, _INIT_PATCHED_FLAG, True)
    #      └─ 6.4 给函数挂属性（盖幂等章）

    connector_cls.__init__ = _compat_init
    #                      └─ 3 节：类属性赋值 = 换方法（动作三，最后才执行！）
```

注意一个容易看漏的顺序细节：**换方法（最后一行）之前，所有准备都已就绪**——
存原函数、定义新函数、盖章、改名全部完成后才执行替换。中间任何一行失败，
类保持原样，不会出现"半换"状态。

---

## 9. 拦截前后的调用流程（语法视角）

```text
【拦截前】
LMCacheMPConnector(k_pool=大张量, v_pool=大张量, ...)
  → Python: 在【类】身上查 "__init__" 属性
  → 查到【原函数】（上游写的那个）
  → 原函数(self, k_pool=大张量, ...)
      → 第一行 _validate_sglang_kv_pools(...)
          → if not k_pool   ← 对多元素张量求布尔值 → RuntimeError 💥

【拦截后】
（启动早期，import lmcache 中途已把类身上的 __init__ 换成 _compat_init）

LMCacheMPConnector(k_pool=大张量, v_pool=大张量, ...)   ← 调用方一字未改
  → Python: 在【类】身上查 "__init__" 属性
  → 查到【_compat_init】（我们换上去的）
  → _compat_init(self, k_pool=大张量, v_pool=大张量, ...)
      → "k_pool" in kwargs → True
      → kwargs["k_pool"] = 28 个逐层视图的列表
      → "v_pool" in kwargs → True → 同上
      → return original_init(self, **kwargs)   ← 闭包里保管的原函数登场
          → _validate_sglang_kv_pools(列表, 列表)
              → if not k_pool   ← 列表非空 → False → 通过 ✓
          → get_event_ipc_backend(npu) → NPU IPC → register_kv_cache → 心跳 ✅
```

---

## 10. 这种技术的地位与安全性

### 10.1 它是 Python 生态的正规军，不是黑魔法

| 场景 | 谁在用 |
|---|---|
| 装饰器 `@staticmethod` `@property` `@functools.lru_cache` | Python 标准库——装饰器本身就是"编译期版的方法包装" |
| `unittest.mock.patch` | 官方测试框架——测试里替换任意函数/方法 |
| pytest 的 fixture、wrapt 库 | 整个测试生态 |
| **torch_npu 的 `transfer_to_npu`** | 你启动日志里那一大段 warning 的来源——它 monkey patch 了几十个 torch 函数把 cuda 换成 npu |
| **本仓三个既有补丁** | `device_ops` 懒代理、`logical_block_size`、`shape_desc`，全部同族 |

### 10.2 本补丁的安全设计（为什么它退化路径全部安全）

| 担心 | 代码里的应对 | 结果 |
|---|---|---|
| 装了两次怎么办 | `_INIT_PATCHED_FLAG` 幂等章（6.4） | 第二次立即退出 |
| 装到一半失败 | 顺序设计：替换是最后一行（第 8 节末尾） | 类保持原样，不会半换 |
| 激活时上游模块还没就绪 | `__init__.py` 激活段的 try/except 守卫 | 只打 warning，绝不炸插件 |
| 未来 SGLang 修好了（改回逐层列表） | `_split_stacked_pool` 的 isinstance+dim 守卫 | 不是张量就原样放行，补丁自动变 no-op |
| 未来 LMCache 删了旧类 | `getattr(..., None)` 判空 | 静默退出 |
| 原构造函数将来抛新错 | 只拦两个参数其余透传 + `__name__` 改名 | 错误照常抛出且 traceback 指向正确的"函数名" |

### 10.3 一个诚实的代价说明

monkey patch 的固有代价：**它对"属性查找顺序"敏感**。如果别的代码在我们
替换**之前**就写了 `from xxx import LMCacheMPConnector` 并以某种方式缓存了
原 `__init__` 的引用，替换不影响那份缓存引用。本场景无此风险（SGLang 每次
构造都是 `LMCacheMPConnector(...)` 走类属性查表，且 import lmcache 先于
一切构造），但把这套手法用到别处时需要先确认这一点——本仓既有的
`logical_block_size.py` 里有个 `_rebind` 函数，专门处理这个问题的更复杂版本，
可做进阶参考。

---

## 11. 动手练习（可选，检验是否真懂了）

1. 把第 7 节的 fridge_demo.py 跑一遍，然后**删掉补丁三行**再跑 `Fridge(fruit="apple")`，
   亲眼看看"原构造"直接处理字符串会发生什么（`not "apple"` 对非空字符串是 False，
   校验会通过但行为不对——再体会一下为什么上游那个 `if not k_pool` 碰上张量
   会炸而碰上字符串不炸）。
2. 给 demo 加上幂等章：把 `setattr(_compat_init, "_patched", True)` 加进
   补丁，然后连续调用两次"安装"，在第二次加个 print 验证它直接返回。
3. 进阶：试着不翻译参数，让 `_compat_init` 只打印一句 "intercepted!" 然后
   转交——感受"包装器可以只观察不修改"（这就是 logging 装饰器的原理）。

## 12. 与既有文档的关系

| 文档 | 分工 |
|---|---|
| 本文 | 拦截技术的 **Python 语法原理**（零基础可读） |
| [stacked_kv_compat_patch_explained.md](stacked_kv_compat_patch_explained.md) | 补丁的**设计决策**（为什么这么设计，讲给有基础的人） |
| [call_flow_before_after_adaptation.md](call_flow_before_after_adaptation.md) | 补丁在**整条链路**中的位置（A/B/C 三阶段全流程） |
| [qwen25_7b_first_light.md](qwen25_7b_first_light.md) | 补丁的**部署与验证**手册 |
