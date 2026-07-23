# yolo_Blood — 血袋标签检测 + 血袋类型分类

## 项目概述

包含两个子项目：

| 子项目 | 目录 | 任务 | 模型 |
|--------|------|------|------|
| **血袋标签检测** | `src/blood_label/` | 检测血袋上的标签贴纸，输出旋转框 (OBB) | YOLO11n-OBB |
| **血袋类型分类** | `src/blood_classify/` | 识别血袋是红细胞还是血浆 | YOLO11n-cls |

### 血袋标签检测 (OBB)

用 YOLO11-OBB 检测血袋上的标签贴纸，输出**旋转框**（Oriented Bounding Box），单类别 `blood_label`。

数据来源：每张血袋照片有一张原图（`*.origin.jpg`）和一张人工标注图（`*.render.jpg`），标注图上画了绿色框线。脚本通过对比原图和标注图的差异（`cv2.absdiff`）自动提取标签框坐标。

实际场景中血袋总是倾斜摆放，因此必须用**旋转框**而非水平框——水平框会包裹额外背景，影响后续 OCR。

### 血袋类型分类

基于血袋的**颜色和形状**区分红细胞和血浆，不需要识别文字。
部署到 RK3576 板端实时推理。

## 目录结构

```
yolo_Blood/
├── src/
│   ├── blood_label/               ← OBB 标签检测 (所有脚本在此目录下运行)
│   │   ├── train.py               # 训练脚本
│   │   ├── detect.py              # 推理脚本 (PC ultralytics)
│   │   ├── demo.py                # 摄像头实时检测演示 (PC)
│   │   ├── server_pc.py           # Web 检测服务 (PC/CUDA, Flask :5000)
│   │   ├── extract_labels.py      # 核心: 从 render 图提取 OBB 标签
│   │   ├── split_dataset.py       # 划分数据集 + 归一化标签
│   │   ├── validate_data.py       # 校验图片对和 OBB 标签
│   │   ├── visualize_labels.py    # 把 OBB 标签画回原图
│   │   ├── compare_labels.py      # 三合一对比图 (提取vs人工vs原图)
│   │   ├── rknn/                  # RK3576 板端部署
│   │   │   ├── infer.py           # 板端推理 (rknn-toolkit-lite2)
│   │   │   ├── infer_ctypes.py    # 板端推理 (ctypes + librknnrt.so C API)
│   │   │   ├── infer_min.py       # 最小验证脚本 (ctypes 加载 RKNN)
│   │   │   ├── server.py          # Web 检测服务 (Flask, RK3576)
│   │   │   └── templates/
│   │   │       └── index.html     # Web 前端页面
│   │   ├── data/                  # OBB 数据
│   │   │   ├── raw/               # 719 对原图+标注图 (原始数据)
│   │   │   ├── datasets/
│   │   │   │   └── blood_label/   # OBB 训练数据集
│   │   │   │       ├── train/  images/  labels/   (463 张)
│   │   │   │       ├── val/    images/  labels/   (132 张)
│   │   │   │       └── test/   images/  labels/   ( 67 张)
│   │   │   ├── manual_label/      # 57 张丢弃图供人工改
│   │   │   ├── test_pair/         # 测试图片对
│   │   │   ├── sample30/          # 30 张样本
│   │   │   ├── sample30_viz/      # 30 张样本可视化
│   │   │   ├── sample_check/      # 样本检查
│   │   │   ├── sample_check_viz/  # 样本检查可视化
│   │   │   ├── compare/           # 三合一比较图输出
│   │   │   ├── verify_obb/        # OBB 验证
│   │   │   └── train_log.txt
│   │   ├── crops/                 # 标签裁剪
│   │   │   ├── crop_labels.py     # 裁剪脚本
│   │   │   ├── test_images/       # 测试输入图片
│   │   │   └── output/            # 裁剪输出
│   │
│   └── blood_classify/            ← 血袋类型分类
│       ├── train.py               # 分类模型训练
│       ├── detect.py              # PC 端分类推理
│       ├── rknn/
│       │   └── infer.py           # 板端分类推理 (rknn-toolkit-lite2)
│       └── data/                  # 分类数据集
│           └── datasets/
│               └── blood_type/
│                   ├── train/
│                   │   ├── red_blood_cell/    ← 红细胞图片
│                   │   └── plasma/            ← 血浆图片
│                   ├── val/
│                   └── test/
│
├── configs/                       ← 配置文件 (项目根目录)
│   ├── blood_label.yaml           # OBB 数据集配置
│   └── blood_type.yaml            # 分类数据集配置
│
├── runs/                          ← 训练/推理输出 (项目根目录)
│   ├── obb/                       # OBB 训练输出
│   └── detect/                    # 推理结果图
│
├── runtime/                       # RKNN 板端运行时
│   └── rknn/
│       ├── best_rk3576.rknn       # OBB RKNN 模型 (13.3MB)
│       ├── librknnrt.so           # RKNN C 运行时 (符号链接)
│       ├── librknnrt_v2.2.0.so    # RKNN C 运行时 v2.2.0
│       ├── rknn_toolkit_lite2-*.whl
│       └── README.md
│
├── docs/                          # 文档
│   ├── superpowers/plans/         # 计划文档
│   └── rk3576-rknn-init-7-排查.md # RKNN 排查记录
│
├── server_optimized.py            # 优化版 Web 服务脚本
├── upload.py                      # 上传脚本
├── best_n2_rk3576.rknn            # RKNN 模型 (n2 变体)
├── blood_label.conf               # 标签检测配置
├── check*.onnx                    # ONNX 调试模型
├── yolo11n-obb.pt                 # YOLO OBB 权重 (nano)
├── yolo11s-obb.pt                 # YOLO OBB 权重 (small)
│
├── templates/
│   └── index.html                 # Web 前端页面
│
├── .venv/                         # Python 虚拟环境
├── 训练素材/                       # 原始素材
└── 训练素材2/                      # 原始素材 (第二批)
```

## 技术栈

| 组件 | 版本/型号 |
|------|----------|
| Python | 3.12 |
| PyTorch | 2.13.0+cu126 (GPU) |
| CUDA | 12.6 |
| ultralytics | 8.4.102 |
| 显卡 | NVIDIA GeForce GTX 1660 SUPER 6GB |
| OBB 模型 | yolo11n-obb (266 万参数, 6.7 GFLOPs) |
| 分类模型 | yolo11n-cls |

**注意**: venv 的 python 路径是 `.venv\Scripts\python.exe`，不要用系统 `python`（指向 Windows Store 占位符）。

## 数据流水线 (OBB)

**所有 OBB 脚本在 `src/blood_label/` 目录下运行**，路径默认相对于该目录。

### 1. 提取标签 — `extract_labels.py`

```bash
cd src/blood_label
python extract_labels.py --raw data/raw [--output data/raw] [--workers N]
```

原理：
- `cv2.absdiff(origin, render)` → 差异 = 标注线
- 闭运算连接断线 → 找轮廓 → 识别"外层大框"和"内部标签框"
- 标签框 = 竖长方形 (h/w 1.2~2.5, 面积 ≥30000px), 完全在外框内
- `cv2.minAreaRect` 取旋转矩形 4 角点
- 角点任何超出大框 → **丢弃**（标签角度过大/标注异常）

输出：`data/raw/*.txt`，格式为 `class x1 y1 x2 y2 x3 y3 x4 y4`（**绝对像素坐标**）
空 txt = 丢弃的图

### 2. 划分数据集 — `split_dataset.py`

```bash
cd src/blood_label
python split_dataset.py --raw data/raw --output data/datasets/blood_label --manual data/manual_label
```

- 只划 txt **非空**的图（662 张），7:2:1 → train/val/test
- **关键**：复制标签时自动将绝对像素坐标**归一化**为 0~1（除以图片宽/高）
- 空 txt 的 57 张另存到 `data/manual_label/`

### 3. 校验 — `validate_data.py`

```bash
cd src/blood_label
python validate_data.py --raw data/raw
```

### 4. 可视化 — `visualize_labels.py` / `compare_labels.py`

```bash
cd src/blood_label
python visualize_labels.py --raw data/raw --output data/annotated
python compare_labels.py --raw data/test_pair --output data/compare
```

## 训练 (OBB)

```bash
cd src/blood_label
python train.py [--epochs 100] [--batch 16] [--imgsz 640] [--device 0]
```

- 权重: `yolo11n-obb.pt`（自动下载）
- 输出: `../../runs/obb/`（即项目根 `runs/obb/`）

**当前最佳结果** (epoch 98):

| 指标 | 值 |
|------|-----|
| mAP@50 | 0.9691 |
| mAP@50-95 | 0.2239 |
| Precision | 1.0 |
| Recall | 0.995 |

## 训练 (分类)

```bash
cd src/blood_classify
python train.py [--epochs 100] [--batch 32] [--imgsz 224] [--device 0]
```

- 权重: `yolo11n-cls.pt`（自动下载）
- 输出: `../../runs/classify/`（即项目根 `runs/classify/`）
- **前提**: `src/blood_classify/data/datasets/blood_type/` 下按文件夹放置图片

## 推理 (OBB)

```bash
cd src/blood_label
python detect.py --source data/test.jpg
python detect.py --source data/test_pair/ --conf 0.3
python detect.py --source data/test/ --save-txt
```

### 摄像头实时演示 (PC)

```bash
cd src/blood_label
python demo.py [--cam 0]
```

## 推理 (分类)

```bash
cd src/blood_classify
python detect.py --source path/to/image.jpg
python detect.py --source path/to/images/
```

## 板端部署 (RK3576)

### OBB 推理

```bash
python3 infer.py --src test.jpg       # 单图
python3 infer.py --cam                # 摄像头实时
python3 infer_ctypes.py --src test.jpg  # ctypes 方式
python3 infer_min.py                   # 最小验证
```

### 分类推理

```bash
python3 infer.py --src test.jpg       # 单图分类
python3 infer.py --cam                # 摄像头实时分类
```

### Web 检测服务 (OBB)

```bash
# RK3576 板端
python3 server.py    # http://<board_ip>:5000

# PC 端 (CUDA)
cd src/blood_label
python server_pc.py  # http://localhost:5000
```

## OBB 标签格式要点

- **ultralytics 训练要求归一化坐标**（0~1），非绝对像素
- `extract_labels.py` 输出绝对像素坐标
- `split_dataset.py` 复制到 datasets 时自动归一化
- 推理结果 `r.obb.xyxyxyxy` 是绝对像素坐标
- 格式: `class x1 y1 x2 y2 x3 y3 x4 y4`（4 角点）

## 已知问题/陷阱

1. **venv 的 python 路径**：必须用 `.\.venv\Scripts\python.exe`，系统 `python` 是 Windows Store 占位符
2. **render 文件名时间戳与 origin 不同**：毫秒位差异，`extract_labels.py` 用 sorted+zip 配对（已验证 0 错配）
3. **序号 `000004` 非唯一**：同序号多张图，sorted 按完整文件名排序后对齐
4. **mAP@50-95 偏低**（~0.24）：OBB 对角度精度敏感，小数据集下可接受
5. **AMP 不支持**：GTX 1660 SUPER 无 Tensor Core，训练只能 float32
6. **57 张丢弃图在 `data/manual_label/`**：改完后可重新提取并补入训练集
7. **脚本运行目录**：OBB 脚本在 `src/blood_label/` 下运行，分类脚本在 `src/blood_classify/` 下运行
