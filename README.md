# yolo_Blood

中文 | [English](README_EN.md)

基于 Ultralytics YOLO11 的血袋图像实验项目，包含两个独立任务：

- **标签检测**：YOLO11-OBB 检测血袋上的标签贴纸，输出四角点旋转框；只有 `blood_label` 一个检测类别。
- **类型分类**：YOLO11-cls 区分 `plasma`（血浆）和 `red_blood_cell`（红细胞），输出整张图的类别与置信度。

另有 PC 摄像头/Web 演示及 RK3576 RKNN 推理、量化转换和排查脚本。标签定位不等于 OCR，血袋类型分类也不是 ABO/Rh 血型检测。本项目未在本 README 中提供临床验证或医疗用途适用性证明，不能据此进行诊断、配血或输血决策。

## 目录导航

```text
yolo_Blood/
├── src/blood_label/         # OBB 数据处理、训练、PC 推理与演示
│   ├── crops/              # 标签裁剪工具
│   ├── raw/                # 仓库中的部分原始样本/标签，不等于完整训练集
│   └── rknn/               # 标准 OBB 板端推理、Web 服务及模板
├── src/blood_classify/      # 分类数据准备、训练、PC 与 RKNN 推理
├── configs/                # 数据集配置
├── scripts/rknn/           # RKOPT 转换、ONNX 检查及 opset 工具
├── runtime/rknn/           # 已提交的 OBB RKNN 模型、运行库、Lite2 wheel
├── runs/                   # 已提交的部分训练输出、权重和 ONNX 文件
├── templates/              # 根目录 Web 服务模板
├── server_remote.py        # RKOPT Web 服务，端口 5000
├── server_hd.py            # RKOPT 高清摄像头/拍照裁剪服务，端口 5001
└── docs/                   # 方案与排查记录
```

默认工作目录很重要：OBB 命令在 `src/blood_label/` 下运行，分类命令在 `src/blood_classify/` 下运行。下面每组示例会标明目录；`/absolute/path/...` 是需要替换的本机路径。

## 环境准备（PC）

核心依赖由代码导入确定：Python、PyTorch、Ultralytics、OpenCV、NumPy；Web 演示另需 Flask。仓库没有统一的 `requirements.txt`、`pyproject.toml` 或依赖锁文件，以下是起步安装示例，不是经过验证的锁定环境。

```bash
# 仓库根目录
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell 则使用：.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install ultralytics opencv-python numpy flask
```

先根据本机 CPU/CUDA 环境选择匹配的 PyTorch 安装方式；安装完成后检查实际版本和 GPU 是否可用。训练默认 `--device 0`，没有可用 CUDA GPU 时传 `--device cpu`。摄像头窗口需要可用摄像头和图形桌面，不能以无 GUI 的 OpenCV 包代替窗口演示依赖。

[CLAUDE.md](CLAUDE.md) 记录了原开发环境，但不是可复现依赖清单。预训练权重不存在时，Ultralytics 可能需要联网下载。只加载来源可信的模型权重。

## 1. 血袋标签检测（OBB）

### 准备数据

准备有权使用且已去标识化的 `*.origin.jpg` 原图与 `*.render.jpg` 人工画框图，放到 `src/blood_label/data/raw/`。提取器依赖该项目的画框样式和几何规则，并非通用标注格式转换器。

```bash
# 从仓库根目录进入
cd src/blood_label
python extract_labels.py --raw data/raw --output data/raw
python validate_data.py --raw data/raw
python split_dataset.py --raw data/raw --output data/datasets/blood_label --manual data/manual_label --seed 42
```

- 提取标签为 `class x1 y1 x2 y2 x3 y3 x4 y4`，坐标是**绝对像素**。
- 划分脚本按约 7:2:1 分为 train/val/test，并将坐标归一化到 0–1，生成各子集的 `images/` 与 `labels/`。
- 空标签样本被另存到人工检查输出目录，不作为训练子集中的负样本使用。
- 图片配对按文件名排序后 `zip`，不是按唯一标识匹配；新增数据必须核对配对和数量。连续拍摄的近重复图也应检查是否跨训练/验证集泄漏。
- 输出目录会被复用，重新划分不会清理旧文件；不同数据版本请使用新的输出目录。

编辑 [configs/blood_label.yaml](configs/blood_label.yaml)，将 `path` 指向生成数据集的**绝对路径**，避免 Ultralytics 数据目录设置与相对路径解析差异。保留 `train: train/images`、`val: val/images`、`test: test/images` 和 `0: blood_label`。

仓库中没有完整的默认 `data/datasets/` 训练目录；历史文档中的样本数量不是本次运行的保证。

### 训练

```bash
# 工作目录：src/blood_label
python train.py --data ../../configs/blood_label.yaml --weights yolo11n-obb.pt --epochs 100 --imgsz 640 --batch 16 --device 0
```

脚本默认 `--project ../../runs/obb --name blood_label`，并启用旋转/翻转增强。以训练完成时打印的 `best.pt` 路径为准；重复运行可能产生递增目录。CPU 运行可将 `--device 0` 换成 `--device cpu`。

### 图片、目录、摄像头与 Web 推理

以下使用仓库中已提交的 n-2 OBB 权重路径；用自己的训练结果时替换 `--weights`：

```bash
# 工作目录：src/blood_label
python detect.py --weights ../../runs/obb/runs/obb/blood_label_n-2/weights/best.pt --source /absolute/path/image.jpg --conf 0.25
python detect.py --weights ../../runs/obb/runs/obb/blood_label_n-2/weights/best.pt --source /absolute/path/images --save-txt
python demo.py --weights ../../runs/obb/runs/obb/blood_label_n-2/weights/best.pt --camera 0
python server_pc.py --weights ../../runs/obb/runs/obb/blood_label_n-2/weights/best.pt --cam 0 --port 5000
```

`detect.py` 默认输出到 `../../runs/detect/blood_label`，生成 `det_*` 标注图；`--save-txt` 还会请求 Ultralytics 保存坐标。不同输出写入方式可能使用不同的递增目录，请检查日志。摄像头窗口按 Q 退出；PC Web 页面为 `http://localhost:5000`。

注意：`demo.py` 的摄像头参数是 `--camera`，`server_pc.py` 则是 `--cam`。训练、图片推理和演示脚本的默认权重路径并不完全一致，因此建议始终显式传入权重。

## 2. 血袋类型分类

准备如下目录；类别文件夹名决定类别，当前推理脚本固定映射 `0 = plasma`、`1 = red_blood_cell`：

```text
src/blood_classify/data/datasets/blood_type/
├── train/{plasma,red_blood_cell}/
├── val/{plasma,red_blood_cell}/
└── test/{plasma,red_blood_cell}/
```

`prepare_dataset.py` 依赖源码中写死的原始素材目录，使用前需检查这些目录；它不是任意输入路径的 CLI。也可以自行按上述结构整理数据。分类训练的 `--data` 接收**目录，不是 YAML 文件**。

```bash
# 从仓库根目录进入
cd src/blood_classify
python train.py --data data/datasets/blood_type --weights yolo11n-cls.pt --epochs 100 --imgsz 224 --batch 32 --device 0
python detect.py --weights /absolute/path/classification/best.pt --source /absolute/path/image.jpg
```

训练默认输出到 `../../runs/classify/blood_type`；Windows 多进程数据加载有问题时可尝试 `--workers 0`。图片推理打印类别和置信度。仓库没有提交分类 `best.pt` 或分类 RKNN 权重，须先训练/准备对应模型。改变类别时还需核对固定类别映射和导出模型的输出顺序。

## 3. RK3576 部署

### 运行环境与模型约定

- 转换端使用 RKNN-Toolkit2；板端使用 RKNN-Toolkit-Lite2、OpenCV、NumPy，Web 服务另需 Flask。
- [runtime/rknn](runtime/rknn/) 提供 OBB 模型与运行时文件。已提交 Lite2 wheel 标记为 `cp310`、`aarch64`，只能在相应 Python ABI/架构环境使用；不要安装到 PC 的 Python 3.12 环境。
- 确认模型目标芯片、Toolkit、Lite2、`librknnrt.so` 和板端驱动的兼容性。不要仅比较不同组件的版本数字大小或直接覆盖系统运行库。
- 下列脚本没有 `--model` 参数，模型路径写死在源码中。先将**匹配任务和输出格式**的模型放到指定路径，或在自己的部署副本中调整常量。

| 入口 | 期望的模型路径 | 任务/格式 |
| --- | --- | --- |
| `src/blood_label/rknn/infer.py`、`server.py` | `/root/blood_label/best_rk3576.rknn` | 标准 OBB |
| `src/blood_classify/rknn/infer.py` | `/root/blood_classify/best_rk3576.rknn` | 两类分类 |
| `server_remote.py`、`server_hd.py` | `/root/blood_label/best_rkopt_rk3576.rknn` | RKOPT 四分支 OBB |

### 标准模型推理

在已配置好运行环境、模型和摄像头的 RK3576 上，以下从仓库根目录运行：

```bash
python3 src/blood_label/rknn/infer.py --src /absolute/path/image.jpg --conf 0.25
python3 src/blood_label/rknn/infer.py --cam 0
python3 src/blood_label/rknn/server.py

# 需要另行准备分类 RKNN 模型
python3 src/blood_classify/rknn/infer.py --src /absolute/path/image.jpg
python3 src/blood_classify/rknn/infer.py --cam 0
```

OBB Web 服务使用端口 5000。板端单图脚本会保存标注结果；摄像头窗口同样需要图形环境。分类预处理采用 224×224 RGB 输入，部署时必须与模型转换的归一化和输出约定一致。`infer_ctypes.py`、`infer_min.py` 是较底层的 C API/最小排查入口，不应视为所有板卡的通用替代方案。

### RKOPT INT8 转换与 Web 服务

先阅读 [RKOPT 量化方案](docs/rkopt-int8-guide.md)。该路线使用 Rockchip 适配导出器，将部分检测头运算移至 CPU 后处理；不能把任意标准 ONNX 或标准 OBB RKNN 模型当作 RKOPT 模型。

当前 [convert_rkopt.py](scripts/rknn/convert_rkopt.py) 的主程序写死了本机 WSL 路径，**没有解析文档示例中的位置参数**。可在已装 Toolkit2 的转换环境中，从仓库根目录直接调用函数，并替换以下路径：

```bash
python - <<'PY'
from pathlib import Path
from scripts.rknn.convert_rkopt import convert
out = Path('/absolute/path/rknn_output')
out.mkdir(parents=True, exist_ok=True)
convert('/absolute/path/rkopt.onnx', '/absolute/path/calibration', str(out), target='rk3576')
PY
```

校准目录需要至少 5 张 `.jpg`/`.png` 图片，以及列出有效图片路径的 `calib.txt`。数量要求来自脚本检查，不代表量化质量保证。函数固定进行 INT8 量化，输出 `best_rkopt_rk3576.rknn`；当前仓库未提交该产物。

板端放好 RKOPT 模型后，可从仓库根目录运行其中一个服务，保留同目录的 `templates/`：

```bash
python3 server_remote.py  # 端口 5000
# 或
python3 server_hd.py      # 端口 5001，高清采集与拍照裁剪
```

它们的后处理固定按 640 输入、单类别、三个检测头加一个角度分支解析（`[1,65,80,80]`、`[1,65,40,40]`、`[1,65,20,20]`、`[1,1,8400]`）。更换导出方式、输入尺寸或类别数时必须重新核对，不能直接复用。

## 限制与排查

- **Web 安全**：这些 Flask 演示默认绑定 `0.0.0.0`，代码没有提供完整认证/TLS 部署方案。只在可信隔离网络使用，不要直接暴露到公网。摄像头、截图和标签裁剪可能包含个人信息，应先去标识化并控制保存、访问与分享范围。
- **版本与历史记录**：[初始化 -7 排查记录](docs/rk3576-rknn-init-7-排查.md) 与较新的 RKOPT 记录对同一驱动版本有不同结论。它们是排查历史，不是兼容性矩阵；以实际板卡日志和厂商配套版本为准。
- **实验脚本**：`convert_int8.py`、`train_qat.py` 等保留实验路线，名称不构成有效 QAT、量化正确性或加速效果的保证。
- **结果解释**：本 README 依据代码和配置整理，未重新训练、执行推理或在 RK3576 上验证。不将历史文档中的精度/FPS作为复现保证；部署前需自行验证准确性、鲁棒性、速度和数据划分。

更多入口：[项目开发说明](CLAUDE.md) · [RKNN 文件说明](runtime/rknn/README.md) · [RKOPT 指南](docs/rkopt-int8-guide.md)
