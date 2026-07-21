# yolo_Blood — 血袋标签旋转框检测 (YOLO11-OBB)

## 项目概述

用 YOLO11-OBB 检测血袋上的标签贴纸，输出**旋转框**（Oriented Bounding Box），单类别 `blood_label`。

数据来源：每张血袋照片有一张原图（`*.origin.jpg`）和一张人工标注图（`*.render.jpg`），标注图上画了绿色框线。脚本通过对比原图和标注图的差异（`cv2.absdiff`）自动提取标签框坐标。

实际场景中血袋总是倾斜摆放，因此必须用**旋转框**而非水平框——水平框会包裹额外背景，影响后续 OCR。

## 目录结构

```
yolo_Blood/
├── configs/
│   └── blood_label.yaml          # 数据集配置 (ultralytics 训练用)
├── scripts/
│   ├── extract_labels.py          # 核心: 从 render 图提取 OBB 标签
│   ├── split_dataset.py           # 划分数据集 + 归一化标签
│   ├── validate_data.py           # 校验图片对和 OBB 标签
│   ├── visualize_labels.py        # 把 OBB 标签画回原图
│   ├── compare_labels.py          # 三合一对比图 (提取vs人工vs原图)
│   ├── train.py                   # 训练脚本
│   ├── detect.py                  # 推理脚本 (PC ultralytics)
│   ├── demo.py                    # 摄像头实时检测演示 (PC)
│   └── rknn/                      # RK3576 板端部署
│       ├── infer.py               # 板端推理 (rknn-toolkit-lite2)
│       ├── infer_ctypes.py        # 板端推理 (ctypes + librknnrt.so C API)
│       ├── infer_min.py           # 最小验证脚本 (ctypes 加载 RKNN)
│       ├── server.py              # Web 检测服务 (Flask, :5000)
│       └── templates/
│           └── index.html         # Web 前端页面
├── data/
│   ├── raw/                       # 719 对原图+标注图 (原始数据, 不动)
│   ├── datasets/blood_label/     # 划分后的训练数据集
│   │   ├── train/  images/  labels/   (463 张)
│   │   ├── val/    images/  labels/   (132 张)
│   │   └── test/   images/  labels/   ( 67 张)
│   ├── manual_label/             # 57 张丢弃图 (标签超出大框) 供人工改
│   └── train_log.txt             # 最近一次训练日志
├── runtime/                        # RKNN 板端运行时
│   └── rknn/
│       ├── best_rk3576.rknn        # RKNN 模型 (13.3MB)
│       ├── librknnrt.so            # RKNN C 运行时 (7.4MB)
│       └── rknn_toolkit_lite2-*.whl
├── runs/obb/                      # 训练输出
├── .venv/                         # Python 虚拟环境
└── 训练素材/                       # 原始素材
```

## 技术栈

| 组件 | 版本/型号 |
|------|----------|
| Python | 3.12 |
| PyTorch | 2.13.0+cu126 (GPU) |
| CUDA | 12.6 |
| ultralytics | 8.4.102 |
| 显卡 | NVIDIA GeForce GTX 1660 SUPER 6GB |
| 模型 | yolo11n-obb (266 万参数, 6.7 GFLOPs) |

**注意**: venv 的 python 路径是 `.venv\Scripts\python.exe`，不要用系统 `python`（指向 Windows Store 占位符）。Windows 终端中如果 `python` 报 exit code 9009，需确认 venv 已激活或直接用 `.\.venv\Scripts\python.exe`。

## 数据流水线

### 1. 提取标签 — `scripts/extract_labels.py`

```bash
python scripts/extract_labels.py --raw data/raw [--output data/raw] [--workers N]
```

原理：
- `cv2.absdiff(origin, render)` → 差异 = 标注线
- 闭运算连接断线 → 找轮廓 → 识别"外层大框"和"内部标签框"
- 标签框 = 竖长方形 (h/w 1.2~2.5, 面积 ≥30000px), 完全在外框内
- `cv2.minAreaRect` 取旋转矩形 4 角点
- 角点任何超出大框 → **丢弃**（标签角度过大/标注异常）

输出：`data/raw/*.txt`，格式为 `class x1 y1 x2 y2 x3 y3 x4 y4`（**绝对像素坐标**）
空 txt = 丢弃的图

### 2. 划分数据集 — `scripts/split_dataset.py`

```bash
python scripts/split_dataset.py [--raw data/raw] [--output data/datasets/blood_label] [--manual data/manual_label]
```

- 只划 txt **非空**的图（662 张），7:2:1 → train/val/test
- **关键**：复制标签时自动将绝对像素坐标**归一化**为 0~1（除以图片宽/高），ultralytics OBB 要求归一化坐标，否则会被当 corrupt 丢弃
- 空 txt 的 57 张另存到 `data/manual_label/`（含 origin + render，供人工对照修改）

### 3. 校验 — `scripts/validate_data.py`

```bash
python scripts/validate_data.py [--raw data/raw]
```

检查图片对、标签格式、角点越界。

### 4. 可视化 — `scripts/visualize_labels.py` / `scripts/compare_labels.py`

```bash
python scripts/visualize_labels.py --raw data/raw --output data/annotated
python scripts/compare_labels.py --raw data/test_pair --output data/compare
```

visualize: 画绿旋转框回原图
compare: 三面板 (提取红框 | 人工 render | 原图)

## 训练

```bash
python scripts/train.py [--epochs 100] [--batch 16] [--imgsz 640] [--device 0]
```

- 权重: `yolo11n-obb.pt`（自动下载到项目根目录）
- 默认参数: 100 epochs, batch=16, imgsz=640, GPU
- 输出: `runs/obb/runs/obb/blood_label/`（路径嵌套是 ultralytics 行为，不管）
- GTX 1660 SUPER 不支持 AMP（混合精度），训练约 2.5-4 小时

**当前最佳结果** (epoch 98):

| 指标 | 值 |
|------|-----|
| mAP@50 | 0.9691 |
| mAP@50-95 | 0.2239 |
| Precision | 1.0 |
| Recall | 0.995 |
| best.pt | `runs/obb/runs/obb/blood_label/weights/best.pt` (5.47 MB) |

## 推理

```bash
# 推理单图
python scripts/detect.py --source data/test.jpg --weights runs/obb/.../best.pt

# 推理目录
python scripts/detect.py --source data/test_pair/ --conf 0.3

# 保存 OBB 坐标 txt
python scripts/detect.py --source data/test/ --save-txt
```

输出: `runs/detect/blood_label/det_*.jpg`（绿色旋转框 + 置信度文字）

### 摄像头实时演示 (PC)

```bash
python scripts/demo.py [--weights runs/obb/.../best.pt] [--cam 0]
```

按 Q 退出，右侧信息面板显示检测统计。

## 板端部署 (RK3576)

### 推理

```bash
# rknn-toolkit-lite2 方式
python scripts/rknn/infer.py --src test.jpg
python scripts/rknn/infer.py --cam               # 摄像头实时

# ctypes C API 方式 (绕过 toolkit 平台限制)
python scripts/rknn/infer_ctypes.py --src test.jpg

# 最小验证
python scripts/rknn/infer_min.py
```

### Web 检测服务

```bash
python scripts/rknn/server.py    # http://<board_ip>:5000
```

- 摄像头实时画面 + OBB 旋转框检测
- 支持在线调节置信度阈值
- MJPEG 流输出

## OBB 标签格式要点

- **ultralytics 训练要求归一化坐标**（0~1），非绝对像素
- `extract_labels.py` 输出绝对像素坐标（方便人工核对）
- `split_dataset.py` 复制到 datasets 时自动归一化
- 推理结果 `r.obb.xyxyxyxy` 是绝对像素坐标
- 格式: `class x1 y1 x2 y2 x3 y3 x4 y4`（4 角点，顺时针/逆时针均可）

## 已知问题/陷阱

1. **venv 的 python 路径**：必须用 `.\.venv\Scripts\python.exe`，系统 `python` 是 Windows Store 占位符，会报 exit code 9009
2. **render 文件名时间戳与 origin 不同**：毫秒位差异，不能简单 replace 匹配。`extract_labels.py` 用 sorted+zip 配对（已验证 719 对 0 错配）
3. **序号 `000004` 非唯一**：同序号多张图（时间戳不同），但 sorted 按完整文件名排序后 origin/render 对齐
4. **mAP@50-95 偏低**（~0.24）：正常，OBB 对角度精度敏感，463 张小数据环境下可接受。mAP@50=0.96 足以实用
5. **AMP 不支持**：GTX 1660 SUPER 无 Tensor Core，训练只能 float32，速度约为支持 AMP GPU 的一半
6. **57 张丢弃图在 `data/manual_label/`**：含 origin + render，改完后可重新提取并补入训练集
