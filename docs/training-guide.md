# 训练指南

本文档说明血袋标签 OBB 检测模型与血袋类型分类模型的完整数据流水线与训练流程。

所有 OBB 脚本在 `src/blood_label/` 目录下运行，分类脚本在 `src/blood_classify/` 目录下运行（路径默认相对脚本目录）。

## 数据流水线（OBB）

### 原始数据格式

每张血袋照片一对图片：

- `*.origin.jpg` — 原图
- `*.render.jpg` — 人工标注渲染图（绿色框线）

> render 与 origin 文件名时间戳毫秒位不同，脚本按 sorted + zip 顺序配对（已验证 719 对 0 错配）。

### 步骤 1：自动提取标签 — `extract_labels.py`

```bash
cd src/blood_label
python extract_labels.py --raw data/raw [--output data/raw] [--workers N]
```

原理：

1. `cv2.absdiff(origin, render)` → 差异图 = 标注线
2. 闭运算连接断线 → 查找轮廓
3. 识别「外层大框」和「内部标签框」：标签框为竖长方形（h/w 1.2~2.5，面积 ≥ 30000px）且完全位于外框内
4. `cv2.minAreaRect` 取旋转矩形 4 角点
5. 角点任何超出大框 → 丢弃该图（输出空 txt）

输出：`data/raw/*.txt`，格式 `class x1 y1 x2 y2 x3 y3 x4 y4`（**绝对像素坐标**）。

经验数据：719 张中 662 张提取成功，57 张丢弃（见 `data/manual_label/`，人工修正后可重新提取补入训练集）。

### 步骤 2：划分数据集 — `split_dataset.py`

```bash
python split_dataset.py --raw data/raw --output data/datasets/blood_label --manual data/manual_label
```

- 仅保留 txt 非空的图片，按 7:2:1 划分 train(463) / val(132) / test(67)
- **关键**：复制标签时自动把绝对像素坐标归一化到 0~1（ultralytics 训练要求）

### 步骤 3：校验与可视化

```bash
python validate_data.py --raw data/raw
python visualize_labels.py --raw data/raw --output data/annotated
python compare_labels.py --raw data/test_pair --output data/compare   # 三合一对比：提取 vs 人工 vs 原图
```

## 训练（OBB）

```bash
cd src/blood_label
python train.py --epochs 100 --batch 16 --imgsz 640 --device 0
```

- 预训练权重 `yolo11n-obb.pt` 自动下载（约 266 万参数，6.7 GFLOPs）
- 输出到项目根 `runs/obb/`

实测结果（GTX 1660 SUPER 6GB，float32）：

| 指标 | 值 |
|------|-----|
| mAP@50 | 0.9691 |
| mAP@50-95 | 0.2239 |
| Precision | 1.000 |
| Recall | 0.995 |

### QAT 量化感知训练（可选）

```bash
python train_qat.py
```

> 注意：QAT 导出的 int8 ONNX 在 RKNN 2.3.2 上会被降级为 float16，**无法带来板端加速**，最终板端方案采用 RKOPT PTQ，见 [rkopt-int8-guide.md](rkopt-int8-guide.md)。

## 训练（分类）

```bash
cd src/blood_classify
python prepare_dataset.py      # 准备目录结构
python train.py --epochs 100 --batch 32 --imgsz 224 --device 0
```

数据集目录要求（按类别文件夹放置图片）：

```
data/datasets/blood_type/
├── train/
│   ├── red_blood_cell/    # 红细胞血袋图片
│   └── plasma/            # 血浆血袋图片
├── val/
└── test/
```

## 推理（PC）

```bash
# OBB
cd src/blood_label
python detect.py --source data/test_pair/ --conf 0.3 --save-txt
python demo.py --cam 0                # 摄像头实时
python server_pc.py                   # Web 服务 (CUDA, :5000)

# 分类
cd src/blood_classify
python detect.py --source path/to/images/
```

## OBB 标签格式要点

- 训练要求**归一化坐标**（0~1）；`extract_labels.py` 输出绝对像素坐标，`split_dataset.py` 归一化
- 推理结果 `r.obb.xyxyxyxy` 为绝对像素坐标
- 每行 9 列：`class x1 y1 x2 y2 x3 y3 x4 y4`

## 已知问题

1. **mAP@50-95 偏低（~0.22）**：OBB 对角度精度敏感，小数据集下属正常；mAP@50 0.97 已满足实际检测需求。
2. **AMP 不可用**：GTX 1660 SUPER 无 Tensor Core，只能 float32 训练。
3. **同序号多图**：`000004` 等序号非唯一，依赖完整文件名排序配对。
