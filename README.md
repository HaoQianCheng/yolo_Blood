# yolo_Blood — 血袋标签检测与类型分类

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![YOLO11](https://img.shields.io/badge/Model-YOLO11-00b3ff.svg)](https://docs.ultralytics.com/models/yolo11/)
[![RK3576](https://img.shields.io/badge/Target-RK3576_NPU-e92c2a.svg)](https://www.rock-chips.com/)
[![Board FPS](https://img.shields.io/badge/RK3576_INT8-30~39_FPS-brightgreen.svg)](docs/rkopt-int8-guide.md)

[English](README_EN.md) | 简体中文

基于 **YOLO11-OBB** 的血袋标签旋转框检测与血袋类型分类项目，完整覆盖 **数据自动标注 → 训练 → RKNN INT8 量化 → RK3576 板端部署** 全流程，板端实时推理达到 **30~39 FPS**。

![检测效果](docs/images/detection-demo.jpg)

> ⚠️ **免责声明**：本项目是计算机视觉实验项目。标签定位不等于 OCR，血袋类型分类也不是 ABO/Rh 血型检测；本项目未提供临床验证或医疗用途适用性证明，**不能据此进行诊断、配血或输血等医疗决策**。

## ✨ 特性

- 🔄 **旋转框 (OBB) 检测** — 血袋倾斜摆放时水平框会包裹大量背景、干扰后续 OCR，本项目使用 Oriented Bounding Box 精确贴合标签贴纸（单类别 `blood_label`）
- 🏷️ **自动标注流水线** — 原始数据自带人工标注渲染图，通过 `cv2.absdiff` 差分 + 轮廓分析自动提取 OBB 标签坐标，无需手工逐张标注
- 🩸 **血袋类型分类** — 基于颜色与形状区分 `plasma`（血浆）/ `red_blood_cell`（红细胞）（YOLO11n-cls），无需 OCR
- ⚡ **RKOPT INT8 量化** — 采用 airockchip 官方 RKOPT 导出方案（移除 Sigmoid/DFL 后处理），解决 RKNN int8 量化 conf 全 0 的问题，板端速度提升 **2.4×**（13 FPS → 30~39 FPS）
- 🖥️ **板端 Web 服务** — Flask 实时视频流 + 检测 API + 高清拍照，supervisor 常驻部署

## 📐 项目组成

| 子项目 | 目录 | 任务 | 模型 |
|--------|------|------|------|
| 血袋标签检测 | `src/blood_label/` | 检测血袋标签贴纸，输出四角点旋转框 | YOLO11n-OBB |
| 血袋类型分类 | `src/blood_classify/` | 区分血浆 / 红细胞 | YOLO11n-cls |

### 检测指标（yolo11n-obb，epoch 98）

| 指标 | 值 |
|------|-----|
| mAP@50 | **0.9691** |
| mAP@50-95 | 0.2239 |
| Precision | 1.000 |
| Recall | 0.995 |

> mAP@50-95 偏低是 OBB 任务对角度精度敏感所致，小数据集（662 张）下属正常水平；实际检测结果见上图。

## 🔧 环境要求

**PC 端（训练 / 转换）**

- Python ≥ 3.10（开发环境为 3.12）
- PyTorch ≥ 2.0（开发环境 2.13.0+cu126，GTX 1660 SUPER 6GB 即可训练；无 CUDA 时 `--device cpu`）
- [ultralytics](https://docs.ultralytics.com/) ≥ 8.3、OpenCV、NumPy；Web 演示另需 Flask
- RKNN 模型转换在 WSL2 / Linux 中进行，需 [rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2) 2.3.2

**板端（RK3576）**

- rknn-toolkit-lite2 2.3.2（`runtime/rknn/*.whl`）
- librknnrt 2.3.2、NPU 驱动 ≥ v0.9.8
- Python 3.10 (aarch64)、OpenCV、Flask

## 🚀 快速开始

```bash
git clone https://github.com/HaoQianCheng/yolo_Blood.git
cd yolo_Blood

# PC 端安装依赖
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

> 脚本工作目录很重要：OBB 命令在 `src/blood_label/` 下运行，分类命令在 `src/blood_classify/` 下运行（路径默认相对脚本目录）。

### 1. 数据准备（自动标注）

准备**有权使用且已去标识化**的成对图片——原图 `*.origin.jpg` + 人工标注渲染图 `*.render.jpg`，放入 `src/blood_label/data/raw/`：

```bash
cd src/blood_label

# 从 render 图差分提取 OBB 标签（绝对像素坐标 → *.txt）
python extract_labels.py --raw data/raw

# 划分数据集并归一化坐标（7:2:1 → train/val/test）
python split_dataset.py --raw data/raw --output data/datasets/blood_label

# 校验与可视化抽查
python validate_data.py --raw data/raw
python compare_labels.py --raw data/test_pair --output data/compare
```

提取原理：`cv2.absdiff(origin, render)` 差异即标注线 → 闭运算连接断线 → 找轮廓识别「外层大框」与「内部竖长标签框」（h/w 1.2~2.5，面积 ≥ 30000px）→ `cv2.minAreaRect` 取旋转矩形 4 角点；角点越界则丢弃（空 txt），丢弃图集中在 `data/manual_label/` 供人工复核。

> ⚠️ 注意事项：
> - 提取器依赖本项目的画框样式与几何规则，**不是通用标注格式转换器**
> - 图片按文件名排序后 `zip` 配对（非唯一标识匹配），新增数据须核对配对与数量；连续拍摄的近重复图应检查是否跨训练/验证集泄漏
> - 划分输出目录会被复用，不同数据版本请使用新的输出目录
> - 训练前编辑 [configs/blood_label.yaml](configs/blood_label.yaml)，将 `path` 指向数据集**绝对路径**

### 2. 训练

```bash
# OBB 标签检测
cd src/blood_label
python train.py --data ../../configs/blood_label.yaml --epochs 100 --batch 16 --imgsz 640 --device 0

# 血袋类型分类（需先在 data/datasets/blood_type/ 下按类别目录放置图片）
cd ../blood_classify
python train.py --epochs 100 --batch 32 --imgsz 224 --device 0
```

预训练权重（`yolo11n-obb.pt` 等）不存在时由 Ultralytics 联网自动下载；请只加载来源可信的权重。训练产物输出到项目根 `runs/` 目录。

### 3. 导出 RKOPT ONNX + INT8 量化

详见 [docs/rkopt-int8-guide.md](docs/rkopt-int8-guide.md)。核心流程：

```bash
# ① 用 airockchip/ultralytics_yolo11 导出 RKOPT ONNX（自动移除 Sigmoid/DFL）
git clone --depth 1 https://github.com/airockchip/ultralytics_yolo11.git
#    编辑 ultralytics/cfg/default.yaml 的 model: 指向 best.pt 后：
cd ultralytics_yolo11 && export PYTHONPATH=./
python ./ultralytics/engine/exporter.py

# ② ONNX → RKNN INT8（WSL / Linux，rknn-toolkit2）
cd ../../scripts/quant
python convert_int8.py <best.onnx> <calib_images_dir> <output_dir> --target rk3576

# ③ 量化精度对比（PC 模拟）
python calc_error.py
```

### 4. 板端部署（RK3576）

详见 [docs/deployment-rk3576.md](docs/deployment-rk3576.md)。速览：

```bash
# 上传运行时、模型与服务到板子
scp runtime/rknn/{best_rk3576.rknn,librknnrt.so} root@<board_ip>:/root/blood_label/
scp apps/server_hd.py root@<board_ip>:/root/blood_label/server.py
scp configs/blood_label.conf root@<board_ip>:/etc/supervisor/conf.d/

# 板端安装 lite2 运行库
pip3 install runtime/rknn/rknn_toolkit_lite2-2.3.2-*.whl

# 板端单图 / 摄像头推理验证
cd src/blood_label/rknn
python3 infer.py --src test.jpg
python3 infer.py --cam

# 启动 Web 检测服务（Flask，http://<board_ip>:5000）
supervisorctl update && supervisorctl start blood_label
```

PC 端 GPU 推理 Web 服务：`cd src/blood_label && python server_pc.py`。

## 📁 目录结构

```
yolo_Blood/
├── apps/                        # 板端应用（RK3576）
│   ├── server_hd.py             #   Web 检测服务（高清拍照版，主入口，:5000/:5001）
│   ├── server_remote.py         #   Web 检测服务（远程版）
│   ├── server_optimized.py      #   Web 检测服务（早期优化版）
│   ├── templates/index.html     #   Web 前端页面
│   └── index_remote.html        #   远程预览页
├── src/
│   ├── blood_label/             # OBB 标签检测子项目
│   │   ├── train.py             #   训练
│   │   ├── train_qat.py         #   QAT 量化感知训练
│   │   ├── detect.py / demo.py  #   PC 推理 / 摄像头演示
│   │   ├── server_pc.py         #   PC 端 Web 服务 (CUDA)
│   │   ├── extract_labels.py    #   自动标注（diff 提取 OBB）
│   │   ├── split_dataset.py     #   数据集划分 + 坐标归一化
│   │   ├── validate_data.py     #   数据校验
│   │   ├── visualize_labels.py / compare_labels.py   # 可视化
│   │   ├── crops/crop_labels.py #   标签区域裁剪（供 OCR）
│   │   └── rknn/                #   板端推理（lite2 / ctypes）+ Web 服务
│   └── blood_classify/          # 血袋类型分类子项目
│       ├── train.py / detect.py #   训练 / PC 推理
│       ├── server_pc.py         #   PC 端 Web 服务
│       ├── prepare_dataset.py   #   数据集准备
│       └── rknn/infer.py        #   板端推理
├── scripts/
│   ├── rknn/                    # RKNN 转换工具
│   │   ├── convert_rkopt.py     #   RKOPT ONNX → RKNN int8 PTQ
│   │   ├── check_rkopt_onnx.py  #   ONNX 结构检查
│   │   └── fix_opset.py         #   opset 降级 (20 → 19)
│   ├── quant/                   # INT8 量化工具
│   │   ├── convert_int8.py      #   ONNX → RKNN INT8 转换
│   │   ├── calc_error.py        #   量化误差评估
│   │   ├── sim_test.py          #   PC 端模拟测试
│   │   └── prep_calib.sh        #   校准图片集准备
│   └── upload_model.py          #   模型上传到板子（SFTP）
├── configs/                     # 配置
│   ├── blood_label.yaml         #   OBB 数据集配置
│   ├── blood_type.yaml          #   分类数据集配置
│   └── blood_label.conf         #   supervisor 部署配置
├── runtime/rknn/                # 板端运行时（librknnrt.so、lite2 whl、rknn 模型）
├── docs/                        # 文档（见下）
├── models/                      # 本地模型产物（不入库）
└── assets/                      # 示例图片
```

## 📚 文档

| 文档 | 说明 |
|------|------|
| [RKOPT INT8 量化完整方案](docs/rkopt-int8-guide.md) | 4 种量化方案对比、RKOPT 导出细节、板端 30+ FPS 实测 |
| [RK3576 部署指南](docs/deployment-rk3576.md) | 板端环境、模型上传、supervisor 常驻、常见部署问题 |
| [训练指南](docs/training-guide.md) | 数据流水线细节、训练参数、指标解读 |
| [RKNN 初始化排查记录](docs/troubleshooting-rknn-init.md) | RKNN init 失败 -7 等问题排查过程 |

## ❓ FAQ

- **为什么用 OBB 而不是普通水平框？** 实际场景血袋总是倾斜摆放，水平框会带入大量背景，影响后续 OCR；旋转框能精确贴合标签贴纸。
- **RKNN INT8 量化后 conf 全是 0 怎么办？** 这是 Sigmoid 输出被 int8 量化压死的已知问题，请使用 RKOPT 导出方案（移除 Sigmoid/DFL，CPU 后处理），见量化文档。
- **训练时 AMP 报错？** GTX 1660 SUPER 无 Tensor Core，不支持 AMP，使用 float32 训练即可。
- **OBB 标签格式？** `class x1 y1 x2 y2 x3 y3 x4 y4`（4 角点）；ultralytics 训练要求归一化坐标，`split_dataset.py` 会自动完成归一化；推理结果 `r.obb.xyxyxyxy` 为绝对像素坐标。

## 🤝 贡献

欢迎提交 Issue 与 Pull Request，参见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 📄 许可证

[MIT License](LICENSE) © HaoQianCheng
