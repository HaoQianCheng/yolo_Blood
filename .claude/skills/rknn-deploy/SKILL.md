---
name: rknn-deploy
description: YOLO11-OBB 模型转换 + RK3576 板端部署完整流程（含已知坑清单）
metadata:
  author: yolo_Blood project
  version: "1.0.0"
---

# RKNN Deploy Skill

YOLO11-OBB 旋转框检测模型，从 PyTorch `.pt` 转换为 RKNN `.rknn`，部署到 RK3576 板端并启动 Web 实时检测服务。

## 前置检查

在开始任何操作前，先检查两个环境。

### 1. WSL2 转换环境（PC 端）

```bash
bash "${CLAUDE_SKILL_DIR}/scripts/check_wsl.sh"
```

检查项：WSL2 Ubuntu 22.04、Python 3.10、rknn-toolkit2 2.3.2、onnx 1.16.1、ultralytics、torch。

### 2. RK3576 板端

```bash
bash "${CLAUDE_SKILL_DIR}/scripts/check_board.sh"
```

检查项：SSH 连通、Python 3.10、pip、rknn-toolkit-lite2、摄像头、NPU 驱动、磁盘空间。

---

## 模型转换（WSL2 端）

### 路径

```
best.pt → ONNX → best_rk3576.rknn
```

在 WSL2 Ubuntu 中执行：

```bash
source /opt/rknn-env/bin/activate
python "${CLAUDE_SKILL_DIR}/scripts/convert.py" \
  --pt /path/to/best.pt \
  --out /path/to/output_dir \
  --target rk3576
```

### 转换关键点

1. **ONNX opset**: 用 opset 12（兼容性好），不要用过高版本
2. **onnx 版本**: 必须 `onnx==1.16.1`，不能更高（1.22 缺 `onnx.mapping`）
3. **simplify**: 可选，失败也能继续
4. **RKNN config**: 必须加 `disable_rules=['reduce_reshape_op_around_concat']`（OBB 专用，避 Issue #221）
5. **quantization**: FP16（`do_quantization=False`），无校准数据集时不要 INT8
6. **mean_values / std_values**: 设 `[[0,0,0]]` / `[[255,255,255]]`（模型内部做归一化）

### 转换产物

| 文件 | 说明 |
|------|------|
| `best.onnx` | 中间格式，可删 |
| `best_rk3576.rknn` | **最终产物**，传到板端 |

---

## 板端部署

### 一键部署

```bash
# 先把文件传到板端（用 scp 或 SSH MCP upload）
# 然后在板端执行：
bash deploy.sh
```

`deploy.sh` 做的事：
1. 换清华 apt 源（加速）
2. 装 pip、opencv、系统依赖
3. 装 rknn-toolkit-lite2（aarch64 wheel）
4. 装 Flask
5. 验证环境

### 启动 Web 服务

```bash
cd /root/blood_label
nohup python3 server.py > server.log 2>&1 &
```

访问 `http://<board_ip>:5000`。

---

## 已知坑清单（必读）

### 坑 1：`init_runtime(target='rk3576')` 导致 -7 错误

**现象**: `rknn_init` 返回 -7 (`RKNN_ERR_CTX_INVALID`)
**根因**: `target` 参数是 ADB 远程调试用的，板端本机推理不该传
**修复**: `rknn.init_runtime()` 不传任何参数
**来源**: ultralytics issue #420

```python
# ❌ 错误
rknn.init_runtime(target="rk3576")

# ✅ 正确
rknn.init_runtime()
```

### 坑 2：`rknn-toolkit-lite2` 平台检测 bug

**现象**: `Unsupported run platform: Linux aarch64`
**根因**: lite2 v2.3.2 的 `rknn_platform_utils` Cython 模块把 aarch64 误识别为 `Linux_x64`
**影响**: 只影响 `target` 参数非空时的初始化
**修复**: 不传 target（和坑 1 同一解法）

### 坑 3：双重归一化导致检测全部失败

**现象**: 检测结果 conf 极低（~0.001），或检测到大量低质量框
**根因**: 转换时配置了 `mean_values=[[0,0,0]], std_values=[[255,255,255]]`，模型内部会做 `/255`。如果推理代码又预处理了 `/255`，实际输入变成 `[0, 1/255]` ≈ 全黑
**修复**: 输入传 **UINT8 [0,255]**，不做 `/255` 归一化

```python
# ❌ 错误：双重归一化
inp = img.astype(np.float32) / 255.0

# ✅ 正确：直接传 UINT8
inp = img  # shape (1, H, W, 3), dtype uint8, range [0, 255]
```

### 坑 4：通道顺序搞反

**现象**: 检测到大量框，角度全错
**根因**: YOLO11-OBB ONNX 导出后通道顺序是 `[cx, cy, w, h, conf, angle]`，不是 `[cx, cy, w, h, angle, conf]`
**验证**: ultralytics NMS 对 ONNX 输出的结果和 `model.predict()` 完全一致

```python
# ✅ 正确通道顺序
cx, cy, bw, bh = o[:, 0], o[:, 1], o[:, 2], o[:, 3]
conf = o[:, 4]   # 直接用（已 sigmoid，0~1）
angle = o[:, 5]  # 弧度，直接用（已解码）
```

### 坑 5：ONNX 导出时角度假象

**现象**: 测试 ONNX 时 ch4（angle 通道）std≈0，看起来"坏了"
**根因**: 大部分 anchor 是背景，angle≈0。只有真实目标 anchor 有非零 angle
**验证**: 用 ultralytics `non_max_suppression(rotated=True)` 对 ONNX 输出跑 NMS，结果和 `model.predict()` 完全一致

### 坑 6：onnx 版本不能高于 1.16.1

**现象**: `AttributeError: module 'onnx' has no attribute 'mapping'`
**根因**: rknn-toolkit2 v2.3.2 用了 `onnx.mapping`，该属性在 onnx 1.16+ 被移除
**修复**: `pip install onnx==1.16.1`

### 坑 7：ultralytics 装不上（依赖解析地狱）

**现象**: `pip install ultralytics` 卡住，逐个下载 torch 2.5~2.12（每个 800MB+）
**根因**: ultralytics 要求 `torch>=1.8.0`（无上限），pip 做全量依赖解析
**修复**: `pip install "ultralytics" "torch<=2.4.0" "torchvision<=0.19.1"`

### 坑 8：rknn-toolkit-lite2 wheel 文件名必须保留平台标签

**现象**: `Invalid wheel filename (wrong number of parts)`
**根因**: 下载 wheel 时改了文件名，去掉了 `manylinux_2_17_aarch64.manylinux2014_aarch64` 平台标签
**修复**: 保留原始文件名

---

## 文件结构

```
/root/blood_label/
├── best_rk3576.rknn          # 模型文件
├── server.py                 # Web 服务
├── infer.py                  # 单图推理脚本
├── templates/
│   └── index.html            # Web 前端
└── server.log                # 运行日志
```

---

## 重新部署步骤

如果只换了模型（重新训练后），只需：

```bash
# 1. WSL 转换
source /opt/rknn-env/bin/activate
python convert.py --pt new_best.pt --out /mnt/g/WSL/model --target rk3576

# 2. 传到板端
scp /mnt/g/WSL/model/best_rk3576.rknn root@192.168.131.200:/root/blood_label/

# 3. 重启服务（先杀旧进程）
ssh root@192.168.131.200 "pkill -f 'python3 server.py'; cd /root/blood_label && nohup python3 server.py > server.log 2>&1 &"
```

---

## References

| 文件 | 何时加载 |
|------|---------|
| `${CLAUDE_SKILL_DIR}/scripts/convert.py` | 需要模型转换脚本时 |
| `${CLAUDE_SKILL_DIR}/scripts/check_wsl.sh` | 检查 WSL 环境时 |
| `${CLAUDE_SKILL_DIR}/scripts/check_board.sh` | 检查板端环境时 |
| `${CLAUDE_SKILL_DIR}/scripts/deploy.sh` | 板端一键部署时 |
