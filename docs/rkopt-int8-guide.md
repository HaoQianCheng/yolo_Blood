# YOLO11-OBB float16 -> int8 量化完整方案

**目标**: 板端推理从 float16 80ms/13fps 提速到 int8。最终 **RKOPT 方案成功，30-39fps（2.4x 提升）**。

## 环境

| 组件 | 版本 |
|------|------|
| rknn-toolkit2 | 2.3.2 |
| rknn-toolkit-lite2 | 2.3.2 |
| librknnrt | 2.3.2 |
| NPU 驱动 | v0.9.8 |
| 芯片 | RK3576 |
| ONNX opset | ≤19（2.3.2 限制） |

## 方案对比

| 方案 | 结果 | 原因 |
|------|------|------|
| 方案1: RKNN PTQ int8 | ❌ conf 全 0 | Sigmoid 输出 0~1 被 int8 q_zero 压死 |
| 方案2: 混合精度 | ❌ 死锁 | Sigmoid 是 element-wise，RKNN 不让标 float16 |
| 方案3: ultralytics quantize=8 | ❌ 不加速 | QDQ ONNX 被 RKNN 降级 float16 |
| 方案4: **RKOPT（官方）** | ✅ **成功** | 导出时移除 Sigmoid/DFL，CPU 后处理 |

## 完整步骤

### 步骤1: 克隆 airockchip/ultralytics_yolo11

```bash
git clone --depth 1 https://github.com/airockchip/ultralytics_yolo11.git
```

该仓库基于 ultralytics v8.3.9，导出时自动移除 Sigmoid/DFL/Concat 后处理。

### 步骤2: 修改 default.yaml，指定权重路径

编辑 `ultralytics/cfg/default.yaml`，将 `model:` 改为训练好的 best.pt 绝对路径：

```yaml
model: /mnt/c/Users/qianc/Projects/yolo_Blood/runs/obb/runs/obb/blood_label_n-2/weights/best.pt
```

### 步骤3: 导出 RKOPT ONNX

```bash
cd ultralytics_yolo11
export PYTHONPATH=./
python ./ultralytics/engine/exporter.py
```

产物在 best.pt 同目录，文件名 `best.onnx`（10.2MB，opset 12）。

**输出格式**（与标准导出不同）：

| 输出 | shape | 含义 |
|------|-------|------|
| output0 | [1, 65, 80, 80] | stride 8 检测头 (6400 anchors) |
| output1 | [1, 65, 40, 40] | stride 16 检测头 (1600 anchors) |
| output2 | [1, 65, 20, 20] | stride 32 检测头 (400 anchors) |
| output3 | [1, 1, 8400] | angle_feature (扁平) |

其中 65 = 64 DFL bins + 1 conf logit（未 sigmoid），单类别。

### 步骤4: 转 RKNN int8

用 `scripts/rknn/convert_rkopt.py`：

```bash
# WSL2 中运行
python scripts/rknn/convert_rkopt.py <best.onnx路径> [rk3576] [i8|fp]
```

关键配置：
- `mean_values=[[0,0,0]]`, `std_values=[[255,255,255]]`
- `disable_rules=['reduce_reshape_op_around_concat']`（Issue #221 OBB bug）
- `do_quantization=True`, `dataset=calib.txt`（至少 5 张校准图）

产物：`best_rkopt_rk3576.rknn`（7.5MB int8 vs 13MB float16）

### 步骤5: 板端部署

```python
# 加载
rknn = RKNNLite(verbose=False)
rknn.load_rknn("best_rkopt_rk3576.rknn")
rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1)

# 推理
outputs = rknn.inference(inputs=[img])  # 返回 4 个输出分支
```

### 步骤6: CPU 后处理（关键：先 sigmoid 过滤再 DFL）

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# 1. 拼接 3 检测头 -> [8400, 65]
o0 = outputs[0][0].transpose(1,2,0).reshape(6400, 65)
o1 = outputs[1][0].transpose(1,2,0).reshape(1600, 65)
o2 = outputs[2][0].transpose(1,2,0).reshape(400, 65)
box_dfl = np.concatenate([o0[:,:64], o1[:,:64], o2[:,:64]])  # [8400, 64]
conf_logit = np.concatenate([o0[:,64], o1[:,64], o2[:,64]])  # [8400]

# 2. sigmoid conf + 过滤 (8400 全做, ~0.24ms)
conf = sigmoid(conf_logit)
mask = conf > conf_thres  # 8400 -> 几十个

# 3. 只对幸存 anchor 做 DFL 解码 (<1ms)
b = box_dfl[mask].reshape(-1, 4, 16)
b -= b.max(axis=2, keepdims=True)
prob = np.exp(b) / np.exp(b).sum(axis=2, keepdims=True)
lt_rb = (prob * np.arange(16, dtype=np.float32)).sum(axis=2)  # [l,t,r,b]

# 4. 解码 box + angle
angle = (angle_feat[mask] - 0.25) * np.pi
# ... 旋转解码 + NMS (probiou 向量化)
```

⚠️ **必须**先 sigmoid 过滤再 DFL，不能全量 8400 DFL（27ms）。详见 `server_remote.py` 中的 `postprocess_rkopt` 函数。

## 踩坑记录

### 坑1: RKNN PTQ int8 conf 全 0
- **现象**: 标准 ONNX 转 RKNN int8，推理 conf 通道全 0，检测不到任何目标
- **排查**: PC 模拟器也全 0（排除板端）；accuracy_analysis 中间层 cos>0.98（量化精度 OK）；Sigmoid/Concat_7 层被 RKNN 移除（无 golden/simulator 文件）
- **根因**: YOLO 检测头 Sigmoid 输出 0~1，背景 anchor 占多数输出接近 0，int8 量化 q_zero 把小值全压成 0。GitHub airockchip/rknn_model_zoo #383 社区已报告未解决
- **结论**: RKNN PTQ 对 Sigmoid 是死穴，放弃此路

### 坑2: 混合精度死锁
- **思路**: 把 Sigmoid 标 float16 救 conf，其余 int8
- **现象**: `hybrid_quantization_step2` 报 `Invalid operands name '/model.23/Sub_2add_i1'`
- **根因**: Sigmoid 是 element-wise 算子，RKNN 不允许标记 float16
- **结论**: element-wise 算子不能标 float16，死锁

### 坑3: ultralytics export(quantize=8) 不是真 QAT
- **现象**: `train_qat.py` 用 `model.export(quantize=8)` 产出的 `best_int8.onnx`，转 RKNN 后算子全 float16（115 float16 vs 3 int8），不快
- **根因**: `quantize=8` 底层是 onnxruntime 的 QDQ 量化（PTQ）。RKNN 不融合 QDQ 节点成 int8，而是 DequantizeLinear 反量化成 float16
- **结论**: onnxruntime QDQ 对 RKNN 无效。真 QAT 要用 rknn-toolkit2 官方 QAT 工具链

### 坑4: ONNX opset 20 被 RKNN 拒绝
- **现象**: `load_onnx` 报 `Unsupport onnx opset 20, need <= 19!`
- **根因**: ultralytics 8.4+ 默认导出 opset 20，rknn-toolkit2 2.3.2 只支持 ≤19
- **解决**: 用 `onnx.version_converter` 降级；或用 airockchip/ultralytics_yolo11 导出（opset 12）

### 坑5: RKOPT 输出格式变了
- **现象**: 输出不再是 `[1,6,8400]`，是 4 分支
- **解决**: 后处理参考 rknn_model_zoo `yolov8_obb/python/yolov8_obb.py` 的 process 函数

### 坑6: 全量 DFL 解码 27ms
- **现象**: 8400 anchor 全做 DFL（softmax+加权）要 27ms，比 int8 inference 还大
- **根因**: 8400×4×16 softmax，ARM CPU 标量 exp 慢
- **解决**: 先 sigmoid conf 过滤（8400->几十），再对幸存 anchor 做 DFL（<1ms）。**后处理顺序必须: 先过滤再 DFL**

### 坑7: fast_nms 快速路径导致重复框
- **现象**: 10 个检测框里 7 个 conf 相同且重复
- **根因**: `fast_nms` 有快速路径 `if len(order) <= 20: return order[:max_det]`，候选≤20 直接返回跳过 NMS
- **解决**: 去掉快速路径，所有候选都走 NMS

### 坑8: postprocess 每帧生成 grid 耗时 2ms
- **现象**: postprocess 实测 4ms，分步骤计时总和只 1.8ms
- **根因**: 每帧重新 `np.meshgrid` + `np.concatenate` 生成网格坐标（固定值，不依赖输入）
- **解决**: 预计算 `_GRID, _STRIDES = _build_grid()` 全局变量，postprocess 直接引用

### 坑9: NMS 用 cv2 逐对计算慢
- **现象**: `compute_rotated_iou_matrix` 用 Python 双重循环 + cv2.rotatedRectangleIntersection，30×30=900 次
- **解决**: 改用 probiou（numpy 向量化旋转 IoU），基于 Bhattacharyya 距离矩阵一次算出

### 坑10: 文件传输到板端
- **scp**: 没密钥认证要密码，host key 变了要 `ssh-keygen -R`
- **HTTP server**: Windows 防火墙阻止入站，要 `New-NetFirewallRule` 放行
- **推荐**: MCP SSH `upload`/`download` 工具（路径须在工作目录内）

### 坑11: best.pt 位置
- 项目有两个 OBB 权重: `blood_label/weights/best.pt` 和 `blood_label_n-2/weights/best.pt`
- 当前板端部署的是 n-2 变体，RKOPT 导出也应用 n-2 保持一致

## 最终性能（RK3576 板端实测）

| 指标 | float16 | RKOPT int8 | 提升 |
|------|---------|------------|------|
| inference | 80ms | 19-22ms | 3.8x |
| postprocess | 3ms | 2-5ms | - |
| fps | 13 | 30-39 | 2.4x |
| 模型大小 | 13MB | 7.5MB | 42%↓ |
| conf 是否正常 | ✅ | ✅ | - |

## 文件索引

| 文件 | 用途 |
|------|------|
| `scripts/rknn/convert_rkopt.py` | RKOPT ONNX -> RKNN int8 转换脚本 |
| `scripts/rknn/check_rkopt_onnx.py` | 检查 ONNX 输入输出格式 |
| `scripts/rknn/fix_opset.py` | ONNX opset 降级工具 |
| `server_remote.py` | 板端服务（含 postprocess_rkopt 后处理） |
| `server_hd.py` | 板端服务（1280×720 + 拍照裁剪） |
| `convert_int8.py` | 旧方案：RKNN 自己 PTQ（conf 全 0，弃用） |
| `sim_test.py` | PC 模拟器测试脚本 |
| `calc_error.py` | accuracy_analysis 误差计算 |

## 关键约束

- **int8 速度必须 RKNN 自己量化**（do_quantization=True），外部 QDQ 会被降级 float16
- **Sigmoid/DFL 对量化不友好**，RKOPT 移到 CPU 是正解
- **后处理顺序**: 先 sigmoid 过滤再 DFL，不能全量
- **网格坐标预计算**: 固定值，别每帧重新生成