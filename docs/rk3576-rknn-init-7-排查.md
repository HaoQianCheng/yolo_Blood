# RK3576 板端 RKNN 推理排查：`rknn_init` 返回 -7

> 场景：YOLO11-OBB 血袋标签检测模型，在 WSL2 上用 rknn-toolkit2 v2.3.2 转换成 `best_rk3576.rknn`，传到 RK3576 板端后，推理初始化失败。
> 目标：定位根因，给出复现命令。

---

## 1. 问题背景

| 项 | 值 |
|----|----|
| 模型 | YOLO11n-OBB（单类 `blood_label`），`best_rk3576.rknn`（13.3 MB） |
| 转换环境 | WSL2 Ubuntu 22.04 + rknn-toolkit2 v2.3.2（PC 端，见 [wsl2-rknn-env](../../.claude/projects/C--Users-qianc-Projects-yolo-Blood/memory/wsl2-rknn-env.md)） |
| 板端 | RK3576 EVB1，192.168.131.200，Ubuntu 22.04.3 aarch64，Python 3.10.12 |
| 现象 | `rknn_init()` 返回 **-7**，无法初始化推理上下文 |

---

## 2. 排查思路链

按时间顺序，每一步都基于上一步的证据：

### 步骤 1：先试官方 `rknn-toolkit-lite2`

板端装 `rknn-toolkit-lite2` v2.3.2（aarch64 wheel），调 `RKNNLite.init_runtime(target='rk3576')` → 抛异常：

```
Exception: Unsupported run platform: Linux aarch64
```

**发现**：lite2 的 Cython 模块 `rknn_platform_utils` 把 aarch64 误判成 `Linux_x64`（`get_host_os_platform()` 返回 `Linux_x64`，`get_librknn_api_require_dll_dir()` 返回 `None`）。这是该 wheel 的平台检测缺陷，monkey-patch `get_os_platform` 也无效（Cython 内部直接走 C 层）。

→ 结论：lite2 这条路在这个 wheel 上走不通，改用 ctypes 直调 `librknnrt.so` 的 C API。

### 步骤 2：ctypes 直调 C API

从 `rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so`（v2.3.2，2025-04-09 build）下载库，放到 `/usr/lib/`，用 ctypes 调 `rknn_init / rknn_query / rknn_inputs_set / rknn_run / rknn_outputs_get`。

→ `rknn_init` 返回 **-7**。

### 步骤 3：查错误码 -7 的含义

查 `rknn_api.h`：

```c
#define RKNN_ERR_CTX_INVALID   -7   /* context is invalid. */
```

`RKNN_ERR_CTX_INVALID` 发生在 `rknn_init` 阶段，且文件 MD5 一致（排除传输损坏），指向**模型与运行环境不兼容**。

### 步骤 4：查模型版本与驱动版本

- `.rknn` 文件头 offset 0 = `52 4b 4e 4e`（`RKNN` magic）✅
- offset 8 = `06 00 00 00` → 模型版本 **6**
- offset 448 = `2.3.2(compiler version: 2.3.2 ...)`
- 板端驱动：`RKNPU driver: v0.9.8`（内核 builtin）

`librknnrt.so` 里的字符串证据：

```
RKNN Model version: %d.%d.%d not match with rknn runtime version: %d.%d.%d
Mismatch driver version, %s requires driver version >= %d.%d.%d, but you have driver version: %d.%d.%d which is incompatible!
```

→ 根因：**板端 NPU 驱动 v0.9.8 太旧，无法加载 toolkit2 v2.3.2 生成、模型版本 6 的 .rknn**。驱动版本（0.9.x）与 runtime 版本（2.3.x）差距过大。

### 步骤 5：确认 RK3576 + OBB 本身是官方支持组合

`rknn_model_zoo` 的 `yolov8_obb` 示例明确支持 RK3576：

```
yolov8_obb | INT8 | RK3562|RK3566|RK3568|RK3576|RK3588|RV1106B
```

→ 排除「RK3576 不支持 OBB」的可能，问题纯粹是驱动版本。

---

## 3. 板端排查命令（照着跑就能复现）

> 前提：已把 `best_rk3576.rknn` 和 `librknnrt.so`（v2.3.2）放到板端。
> SSH 连板：`ssh root@192.168.131.200`

### 3.1 看板端基本环境

```bash
# 系统 / 架构 / 内核
uname -a
cat /etc/os-release | head -3

# Python
python3 --version
which python3
```

预期：`aarch64`、`Ubuntu 22.04.3`、`Python 3.10.12`。

### 3.2 看 NPU 驱动版本（**最关键的一步**）

```bash
cat /sys/kernel/debug/rknpu/version
```

预期输出：`RKNPU driver: v0.9.8`  ← **这就是根因所在**。

补充：

```bash
# 驱动是内核 builtin 还是模块
modinfo rknpu 2>/dev/null | head -5
# 看 NPU 核数与负载
cat /sys/kernel/debug/rknpu/load
# 开机时的 NPU 内核日志
dmesg 2>/dev/null | grep -i rknpu | head -5
```

### 3.3 看 librknnrt.so 版本

```bash
strings /usr/lib/librknnrt.so | grep -iE "librknnrt version|compiler version"
```

预期：`librknnrt version: 2.3.2 (...)`。

对比驱动版本：librknnrt 2.3.2 vs driver 0.9.8 → 不匹配。

### 3.4 看 .rknn 模型版本

```bash
# 文件头 magic 与版本号
xxd -l 16 /root/blood_label/best_rk3576.rknn
# 52 4b 4e 4e = "RKNN", offset 8 的 uint32 = 模型版本号(应为 06)
```

### 3.5 验证文件完整性

```bash
# 板端
md5sum /root/blood_label/best_rk3576.rknn
ls -la /root/blood_label/best_rk3576.rknn
```

与本机对比 MD5，一致则排除传输损坏。

### 3.6 复现 `rknn_init` 返回 -7

最小复现脚本（保存为 `/root/blood_label/init_repro.py`）：

```python
import ctypes
lib = ctypes.CDLL("/usr/lib/librknnrt.so")

lib.rknn_init.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
lib.rknn_init.restype = ctypes.c_int

with open("/root/blood_label/best_rk3576.rknn", "rb") as f:
    data = f.read()
print(f"模型大小: {len(data)} bytes")

ctx = ctypes.c_void_p()
buf = ctypes.create_string_buffer(data, len(data))
ret = lib.rknn_init(ctypes.byref(ctx), buf, len(data), 0, None)
print(f"rknn_init: {ret}")   # 预期 -7
```

运行：

```bash
cd /root/blood_label
python3 init_repro.py
```

预期输出：

```
模型大小: 13932597 bytes
rknn_init: -7
```

### 3.7 查 RKNN 错误码定义（确认 -7 含义）

板端无 `rknn_api.h`，从 GitHub 拉取确认：

```bash
curl -sL https://raw.githubusercontent.com/airockchip/rknn-toolkit2/master/rknpu2/runtime/Linux/librknn_api/include/rknn_api.h \
  | grep -E "RKNN_ERR_CTX_INVALID|RKNN_ERR_MODEL_INVALID|RKNN_ERR_DEVICE_UNMATCH"
```

预期：

```c
#define RKNN_ERR_MODEL_INVALID   -6
#define RKNN_ERR_CTX_INVALID     -7   /* context is invalid. */
#define RKNN_ERR_DEVICE_UNMATCH  -10  /* the device is unmatch, please update rknn sdk */
```

### 3.8 看 librknnrt.so 内嵌的版本兼容提示

```bash
strings /usr/lib/librknnrt.so | grep -iE "driver version|requires driver|not match"
```

预期看到：

```
RKNN Model version: %d.%d.%d not match with rknn runtime version: %d.%d.%d
Mismatch driver version, %s requires driver version >= %d.%d.%d, but you have driver version: %d.%d.%d which is incompatible!
Current driver version: %d.%d.%d, recommend to upgrade the driver to the new version: >= %d.%d.%d
```

→ 证明 runtime 会校验驱动版本，不匹配即拒绝。

---

## 4. 诊断结论

**根因：板端 NPU 驱动 v0.9.8 太旧，与 rknn-toolkit2 v2.3.2 / librknnrt v2.3.2 / 模型版本 6 不兼容，导致 `rknn_init` 返回 -7 (`RKNN_ERR_CTX_INVALID`)。**

证据：

1. 板端 `cat /sys/kernel/debug/rknpu/version` → `v0.9.8`
2. `librknnrt.so` → `2.3.2`，与驱动版本差距过大
3. `.rknn` 模型版本 = 6，compiler 2.3.2
4. 文件 MD5 一致，排除传输损坏
5. `librknnrt.so` 内含「driver version mismatch」校验逻辑
6. RK3576 + OBB 是官方支持组合，排除平台/任务不支持

---

## 5. 解决方向

| 方案 | 操作 | 难度 | 风险 |
|------|------|------|------|
| A. 升级板端内核/驱动 | 找 RK3576 新固件（含 rknpu driver 2.3.x）刷入 | 高 | 变砖风险 |
| B. 降级 toolkit 重转模型 | WSL 装 toolkit2 v2.0.0-beta0（最早支持 RK3576）重转 | 中 | 旧版 runtime 可能仍要求更高驱动 |
| C. 找厂商要配套 runtime | 向板子供应商要匹配 v0.9.8 驱动的旧 `librknnrt.so` | 低 | 取决于厂商 |

**前置确认（决定走 A 还是 B）**：driver v0.9.8 到底兼容哪个版本的 RKNN runtime。需查 Rockchip 官方文档或 GitHub issues 确认版本对应关系。

---

## 6. 关键命令速查

| 目的 | 命令 |
|------|------|
| NPU 驱动版本 | `cat /sys/kernel/debug/rknpu/version` |
| NPU 核数/负载 | `cat /sys/kernel/debug/rknpu/load` |
| librknnrt 版本 | `strings /usr/lib/librknnrt.so \| grep "librknnrt version"` |
| .rknn 模型版本 | `xxd -l 16 best_rk3576.rknn`（offset 8 的 uint32） |
| 复现 -7 | `python3 init_repro.py`（见 3.6） |
| 错误码定义 | `curl -sL <rknn_api.h URL> \| grep RKNN_ERR_CTX_INVALID` |
| 文件完整性 | `md5sum best_rk3576.rknn` |
