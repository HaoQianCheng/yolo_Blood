# 血袋标签检测项目实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从绿色框标注图自动提取 YOLO 格式标注，训练血袋标签检测模型

**Architecture:** 先通过 OpenCV 颜色提取从 `.render.jpg` 中解析绿色矩形框坐标，转为 YOLO 归一化格式；校验后划分数据集，用 yolo11n 预训练权重微调

**Tech Stack:** Python 3.12, OpenCV, Ultralytics YOLO11, PyTorch CUDA

## Global Constraints

- 项目路径：`C:\Users\qianc\Desktop\yolo_Blood`
- 与 `C:\Users\qianc\Desktop\yolo` 项目完全独立
- 图片对匹配规则：`*.origin.jpg`（原图）↔ `*.render.jpg`（标注图），中间时间戳部分不同
- 类别：1 类 `blood_label`
- GPU：GTX 1660 SUPER 6GB
- 输入尺寸：640

---

### Task 1: 项目脚手架

**Files:**
- Create: `C:\Users\qianc\Desktop\yolo_Blood\scripts\__init__.py`
- Create: `C:\Users\qianc\Desktop\yolo_Blood\scripts\extract_labels.py`
- Create: `C:\Users\qianc\Desktop\yolo_Blood\scripts\validate_data.py`
- Create: `C:\Users\qianc\Desktop\yolo_Blood\scripts\split_dataset.py`
- Create: `C:\Users\qianc\Desktop\yolo_Blood\scripts\train.py`
- Create: `C:\Users\qianc\Desktop\yolo_Blood\scripts\detect.py`
- Create: `C:\Users\qianc\Desktop\yolo_Blood\configs\blood_label.yaml`
- Create: `C:\Users\qianc\Desktop\yolo_Blood\models\base\.gitkeep`
- Create: `C:\Users\qianc\Desktop\yolo_Blood\models\trained\.gitkeep`
- Create: `C:\Users\qianc\Desktop\yolo_Blood\data\raw\.gitkeep`
- Create: `C:\Users\qianc\Desktop\yolo_Blood\data\datasets\.gitkeep`

**Interfaces:**
- Produces: `scripts/extract_labels.py` 空框架，Task 2 填充

- [ ] **Step 1: 创建目录结构**

```powershell
cd C:\Users\qianc\Desktop\yolo_Blood
New-Item -ItemType Directory -Force -Path "scripts","configs","models\base","models\trained","data\raw","data\datasets"
```

- [ ] **Step 2: 创建空占位文件**

```powershell
cd C:\Users\qianc\Desktop\yolo_Blood
New-Item -ItemType File -Force -Path "scripts\__init__.py","models\base\.gitkeep","models\trained\.gitkeep","data\raw\.gitkeep","data\datasets\.gitkeep"
$files = @("scripts\extract_labels.py","scripts\validate_data.py","scripts\split_dataset.py","scripts\train.py","scripts\detect.py")
foreach ($f in $files) { New-Item -ItemType File -Force -Path $f | Out-Null }
Write-Output "done"
```

- [ ] **Step 3: 创建数据集配置文件**

`configs/blood_label.yaml`:
```yaml
train: data/datasets/blood_label/train/images
val: data/datasets/blood_label/val/images
test: data/datasets/blood_label/test/images

nc: 1
names: ['blood_label']
```

- [ ] **Step 4: 移动已有图片到 data/raw**

```powershell
cd C:\Users\qianc\Desktop\yolo_Blood
Move-Item *.origin.jpg data\raw\ -Force
Move-Item *.render.jpg data\raw\ -Force
```

- [ ] **Step 5: 初始化 git 并提交**

```powershell
cd C:\Users\qianc\Desktop\yolo_Blood
git init
git add -A
git commit -m "feat: 项目脚手架 - 血袋标签检测"
```

---

### Task 2: 绿色框提取 + 实际图片测试

**Files:**
- Write: `C:\Users\qianc\Desktop\yolo_Blood\scripts\extract_labels.py`

**Interfaces:**
- Consumes: `data/raw/` 下的 `*.origin.jpg` 和 `*.render.jpg`
- Produces:
  - `extract_green_boxes(render_path: str) -> list[tuple[float,float,float,float]]` — 归一化 `(cx, cy, w, h)` 列表
  - `match_image_pairs(raw_dir: str) -> list[tuple[Path, Path]]`
  - `process_all(raw_dir: str, output_dir: str) -> None`
- CLI: `uv run python scripts/extract_labels.py --raw data/raw`

- [ ] **Step 1: 用实际图片测试绿色提取，确认 HSV 阈值**

```bash
cd C:\Users\qianc\Desktop\yolo_Blood
uv run python -c "
import cv2
import numpy as np

img = cv2.imread('data/raw/20260720.160406.991.000009.render.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_green = (35, 43, 46)
upper_green = (85, 255, 255)
mask = cv2.inRange(hsv, lower_green, upper_green)

kernel = np.ones((3, 3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

h, w = img.shape[:2]
print(f'图片尺寸: {w}x{h}')
print(f'检测到 {len(contours)} 个绿色区域')
for i, c in enumerate(contours):
    area = cv2.contourArea(c)
    if area < 50:
        continue
    x, y, bw, bh = cv2.boundingRect(c)
    cx = (x + bw/2) / w
    cy = (y + bh/2) / h
    nw = bw / w
    nh = bh / h
    print(f'框{i+1}: cx={cx:.4f} cy={cy:.4f} w={nw:.4f} h={nh:.4f}')
    print(f'  像素: x={x} y={y} w={bw} h={bh} area={area:.0f}')
"
```

观察输出，确认框数量和位置合理。如检测不到或数量不对，调整 `lower_green` / `upper_green` 阈值。

- [ ] **Step 2: 阈值确认后，写入 extract_labels.py**

```python
"""从 render 标注图提取绿色框，生成 YOLO 格式标注文件"""
import argparse
from pathlib import Path
import cv2
import numpy as np

# 绿色 HSV 阈值（根据实际测试调整）
LOWER_GREEN = (35, 43, 46)
UPPER_GREEN = (85, 255, 255)


def extract_green_boxes(render_path: str) -> list[tuple[float, float, float, float]]:
    """从绿色框标注图中提取所有矩形框。

    Returns:
        [(cx, cy, w, h), ...] 归一化坐标，值域 0-1
    """
    img = cv2.imread(str(render_path))
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {render_path}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = img.shape[:2]
    boxes = []
    for contour in contours:
        if cv2.contourArea(contour) < 50:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        cx = (x + bw / 2) / w
        cy = (y + bh / 2) / h
        nw = bw / w
        nh = bh / h
        boxes.append((cx, cy, nw, nh))

    return boxes


def match_image_pairs(raw_dir: str) -> list[tuple[Path, Path]]:
    """匹配原图和标注图对。按文件名排序后配对。"""
    raw = Path(raw_dir)
    origins = sorted(raw.glob("*.origin.jpg"))
    renders = sorted(raw.glob("*.render.jpg"))

    if len(origins) != len(renders):
        print(f"警告: 原图 {len(origins)} 张 ≠ 标注图 {len(renders)} 张")

    pairs = list(zip(origins, renders))
    print(f"匹配到 {len(pairs)} 对图片")
    return pairs


def process_all(raw_dir: str, output_dir: str = None) -> None:
    """批量处理：为每张原图生成 YOLO 标注 txt。"""
    if output_dir is None:
        output_dir = raw_dir
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pairs = match_image_pairs(raw_dir)
    success = 0
    empty = 0

    for origin_path, render_path in pairs:
        boxes = extract_green_boxes(str(render_path))
        txt_name = origin_path.name.replace(".origin.jpg", ".txt")
        txt_path = out / txt_name

        if not boxes:
            empty += 1
            txt_path.write_text("")
            continue

        lines = [f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cx, cy, w, h in boxes]
        txt_path.write_text("\n".join(lines))
        success += 1

    print(f"处理完成: 有框 {success} 张, 无框 {empty} 张")


def main():
    parser = argparse.ArgumentParser(description="从绿色框标注图提取 YOLO 格式标注")
    parser.add_argument("--raw", default="data/raw", help="原始图片目录")
    parser.add_argument("--output", default=None, help="标注输出目录（默认同 --raw）")
    args = parser.parse_args()
    process_all(args.raw, args.output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 运行提取脚本**

```bash
cd C:\Users\qianc\Desktop\yolo_Blood
uv run python scripts/extract_labels.py --raw data/raw
```

- [ ] **Step 4: 验证提取结果——在原图上画出框，肉眼确认**

```bash
cd C:\Users\qianc\Desktop\yolo_Blood
uv run python -c "
import cv2
from pathlib import Path
from scripts.extract_labels import extract_green_boxes, match_image_pairs

pairs = match_image_pairs('data/raw')
if not pairs:
    print('没有匹配的图片对！')
    exit(1)

origin_path, render_path = pairs[0]
img = cv2.imread(str(origin_path))
h, w = img.shape[:2]

boxes = extract_green_boxes(str(render_path))
for cx, cy, bw, bh in boxes:
    x1 = int((cx - bw/2) * w)
    y1 = int((cy - bh/2) * h)
    x2 = int((cx + bw/2) * w)
    y2 = int((cy + bh/2) * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

out_path = 'data/raw/_verify_result.jpg'
cv2.imwrite(out_path, img)
print(f'验证图已保存: {out_path}, 框数: {len(boxes)}')
"
```

打开 `data/raw/_verify_result.jpg` 检查框位置是否与原始标注一致。

- [ ] **Step 5: 提交**

```bash
cd C:\Users\qianc\Desktop\yolo_Blood
git add scripts/extract_labels.py
git commit -m "feat: 绿色框提取脚本 - 自动生成 YOLO 标注"
```

---

### Task 3: 数据校验脚本

**Files:**
- Write: `C:\Users\qianc\Desktop\yolo_Blood\scripts\validate_data.py`

**Interfaces:**
- Consumes: `extract_green_boxes()` from Task 2
- Produces: 校验报告（控制台输出）

- [ ] **Step 1: 写入 validate_data.py**

```python
"""数据校验脚本：检查图片对和标注有效性"""
import argparse
from pathlib import Path
import cv2
from extract_labels import extract_green_boxes, match_image_pairs


def validate(raw_dir: str) -> dict:
    raw = Path(raw_dir)
    pairs = match_image_pairs(raw_dir)

    stats = {
        "total": len(pairs), "ok": 0,
        "no_box": [], "size_mismatch": [], "box_out_of_range": [],
        "box_too_small": [], "box_too_large": [], "read_error": [],
    }

    for origin_path, render_path in pairs:
        origin_img = cv2.imread(str(origin_path))
        render_img = cv2.imread(str(render_path))
        if origin_img is None or render_img is None:
            stats["read_error"].append(str(origin_path))
            continue

        oh, ow = origin_img.shape[:2]
        rh, rw = render_img.shape[:2]
        if oh != rh or ow != rw:
            stats["size_mismatch"].append(f"{origin_path.name}: 原图{ow}x{oh} vs 标注图{rw}x{rh}")
            continue

        try:
            boxes = extract_green_boxes(str(render_path))
        except Exception as e:
            stats["read_error"].append(f"{origin_path.name}: {e}")
            continue

        if not boxes:
            stats["no_box"].append(origin_path.name)
            continue

        all_ok = True
        for cx, cy, nw, nh in boxes:
            area = nw * nh
            if area < 0.01:
                stats["box_too_small"].append(f"{origin_path.name}: {nw:.3f}x{nh:.3f}")
                all_ok = False
            elif area > 0.8:
                stats["box_too_large"].append(f"{origin_path.name}: {nw:.3f}x{nh:.3f}")
                all_ok = False
            if not (0 <= cx <= 1 and 0 <= cy <= 1):
                stats["box_out_of_range"].append(f"{origin_path.name}: cx={cx:.3f} cy={cy:.3f}")
                all_ok = False

        if all_ok:
            stats["ok"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="校验 YOLO 标注数据")
    parser.add_argument("--raw", default="data/raw", help="原始图片目录")
    args = parser.parse_args()

    stats = validate(args.raw)

    print("=" * 60)
    print("数据校验报告")
    print("=" * 60)
    print(f"总数: {stats['total']}, 通过: {stats['ok']}")
    print(f"读取错误: {len(stats['read_error'])}")
    print(f"尺寸不匹配: {len(stats['size_mismatch'])}")
    print(f"未检测到框: {len(stats['no_box'])}")
    print(f"框过小(<1%): {len(stats['box_too_small'])}")
    print(f"框过大(>80%): {len(stats['box_too_large'])}")
    print(f"框越界: {len(stats['box_out_of_range'])}")

    if stats["no_box"]:
        print(f"\n无框图片 (前10):")
        for n in stats["no_box"][:10]:
            print(f"  {n}")

    if stats["ok"] == stats["total"]:
        print("\n全部通过!")
    elif stats["ok"] == 0:
        print("\n没有图片通过校验，请检查 HSV 阈值或数据")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行校验**

```bash
cd C:\Users\qianc\Desktop\yolo_Blood
uv run python scripts/validate_data.py --raw data/raw
```

- [ ] **Step 3: 提交**

```bash
git add scripts/validate_data.py
git commit -m "feat: 数据校验脚本"
```

---

### Task 4: 数据集划分脚本

**Files:**
- Write: `C:\Users\qianc\Desktop\yolo_Blood\scripts\split_dataset.py`

**Interfaces:**
- Consumes: `data/raw/` 下通过校验的 `.origin.jpg` + `.txt` 对
- Produces: `data/datasets/blood_label/{train,val,test}/{images,labels}/`

- [ ] **Step 1: 写入 split_dataset.py**

```python
"""数据集划分：7:2:1 拆分为 train/val/test"""
import argparse
import random
import shutil
from pathlib import Path


def split_dataset(raw_dir: str, output_dir: str,
                  train_r: float = 0.7, val_r: float = 0.2, seed: int = 42):
    raw = Path(raw_dir)
    out = Path(output_dir)

    origins = sorted(raw.glob("*.origin.jpg"))
    valid = []
    for origin in origins:
        txt_name = origin.name.replace(".origin.jpg", ".txt")
        if (raw / txt_name).exists():
            valid.append(origin)

    random.seed(seed)
    random.shuffle(valid)

    n = len(valid)
    n_train = int(n * train_r)
    n_val = int(n * val_r)

    splits = {
        "train": valid[:n_train],
        "val": valid[n_train:n_train + n_val],
        "test": valid[n_train + n_val:],
    }

    for split_name, files in splits.items():
        img_dir = out / split_name / "images"
        lbl_dir = out / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for origin in files:
            txt_name = origin.name.replace(".origin.jpg", ".txt")
            dest_img_name = origin.name.replace(".origin", "")
            shutil.copy2(origin, img_dir / dest_img_name)
            shutil.copy2(raw / txt_name, lbl_dir / dest_img_name.replace(".jpg", ".txt"))

        print(f"{split_name}: {len(files)} 张")

    print(f"总计: {n} 张")


def main():
    parser = argparse.ArgumentParser(description="划分数据集")
    parser.add_argument("--raw", default="data/raw", help="原始数据目录")
    parser.add_argument("--output", default="data/datasets/blood_label", help="输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()
    split_dataset(args.raw, args.output, seed=args.seed)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行划分**

```bash
cd C:\Users\qianc\Desktop\yolo_Blood
uv run python scripts/split_dataset.py --raw data/raw --output data/datasets/blood_label
```

- [ ] **Step 3: 验证目录结构**

```bash
Get-ChildItem data\datasets\blood_label -Recurse -Depth 3
```

- [ ] **Step 4: 提交**

```bash
git add scripts/split_dataset.py
git commit -m "feat: 数据集划分脚本 - 7:2:1 拆分"
```

---

### Task 5: 训练脚本

**Files:**
- Write: `C:\Users\qianc\Desktop\yolo_Blood\scripts\train.py`

**Interfaces:**
- Consumes: `configs/blood_label.yaml`, 划分好的数据集, `models/base/yolo11n.pt`
- Produces: `runs/detect/blood_label_*/weights/best.pt`

- [ ] **Step 1: 下载基础模型**

```bash
cd C:\Users\qianc\Desktop\yolo_Blood
uv run python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
Move-Item -Force yolo11n.pt models/base/
```

- [ ] **Step 2: 写入 train.py**

```python
"""YOLO 训练脚本"""
import argparse
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser(description="血袋标签检测训练")
    parser.add_argument("--config", default="blood_label", help="数据集配置名")
    parser.add_argument("--model", default="yolo11n", help="基础模型名")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch", type=int, default=8, help="批大小")
    parser.add_argument("--device", default="cuda", help="训练设备")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图片大小")
    parser.add_argument("--patience", type=int, default=15, help="早停轮数")
    args = parser.parse_args()

    config_path = ROOT / "configs" / f"{args.config}.yaml"
    base_model = ROOT / "models" / "base" / f"{args.model}.pt"

    if not base_model.exists():
        print(f"下载 {args.model}...")
        YOLO(f"{args.model}.pt")

    print(f"配置: {config_path}")
    print(f"模型: {base_model}")
    print(f"{args.epochs} epochs, batch={args.batch}, device={args.device}")

    model = YOLO(str(base_model))
    model.train(
        data=str(config_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        name=f"blood_label_{args.epochs}ep",
        verbose=True,
        plots=True,
    )

    print("训练完成!")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 更新 config 使用绝对路径**

`configs/blood_label.yaml`:
```yaml
train: C:\Users\qianc\Desktop\yolo_Blood\data\datasets\blood_label\train\images
val: C:\Users\qianc\Desktop\yolo_Blood\data\datasets\blood_label\val\images
test: C:\Users\qianc\Desktop\yolo_Blood\data\datasets\blood_label\test\images

nc: 1
names: ['blood_label']
```

- [ ] **Step 4: 提交**

```bash
git add scripts/train.py configs/blood_label.yaml
git commit -m "feat: 训练脚本"
```

---

### Task 6: 检测脚本

**Files:**
- Write: `C:\Users\qianc\Desktop\yolo_Blood\scripts\detect.py`

- [ ] **Step 1: 写入 detect.py**

```python
"""血袋标签检测脚本"""
import argparse
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser(description="血袋标签检测")
    parser.add_argument("--model", default="blood_label_best", help="模型名或路径")
    parser.add_argument("--source", required=True, help="输入图片/视频/摄像头(0)")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--save", action="store_true", help="保存结果图片")
    parser.add_argument("--device", default="cuda", help="设备")
    args = parser.parse_args()

    model_path = args.model
    if "/" not in model_path and "\\" not in model_path:
        for c in [ROOT / "models" / "trained" / f"{model_path}.pt",
                  ROOT / "models" / "base" / f"{model_path}.pt"]:
            if c.exists():
                model_path = str(c)
                break

    print(f"模型: {model_path}")
    model = YOLO(model_path)
    results = model.predict(args.source, conf=args.conf, device=args.device, save=args.save)

    if not args.save:
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                print(f"检测到 {len(r.boxes)} 个标签:")
                for i, box in enumerate(r.boxes):
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    print(f"  [{i+1}] ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f}) conf={float(box.conf[0]):.3f}")
            else:
                print("未检测到标签")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 提交**

```bash
git add scripts/detect.py
git commit -m "feat: 检测脚本"
```

---

### Task 7: README 文档

**Files:**
- Create: `C:\Users\qianc\Desktop\yolo_Blood\README.md`

- [ ] **Step 1: 写入 README.md**

```markdown
# 血袋标签检测

基于 YOLO11 的血袋标签检测项目。

## 流程

```bash
# 1. 提取标注
uv run python scripts/extract_labels.py --raw data/raw

# 2. 校验
uv run python scripts/validate_data.py --raw data/raw

# 3. 划分数据集
uv run python scripts/split_dataset.py --raw data/raw

# 4. 训练
uv run python scripts/train.py --epochs 100

# 5. 检测
uv run python scripts/detect.py --source 图片.jpg --save
```

## 目录

```
yolo_Blood/
├── scripts/
│   ├── extract_labels.py    # 从 render 图提取 YOLO 标注
│   ├── validate_data.py     # 校验
│   ├── split_dataset.py     # 划分
│   ├── train.py             # 训练
│   └── detect.py            # 检测
├── configs/                 # 配置文件
├── data/
│   ├── raw/                 # 原始图片
│   └── datasets/            # 划分后数据集
├── models/
│   ├── base/                # 预训练权重
│   └── trained/             # 训练后权重
└── runs/                    # 训练输出
```
```

- [ ] **Step 2: 最终提交**

```bash
git add README.md
git commit -m "docs: 项目 README"
```

---