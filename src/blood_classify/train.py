"""YOLO11 分类模型训练: 血袋类型识别 (红细胞 vs 血浆)

ultralytics classify 模式要求数据集目录结构:
    data/datasets/blood_type/
    ├── train/
    │   ├── red_blood_cell/   ← 红细胞图片
    │   └── plasma/           ← 血浆图片
    ├── val/
    │   ├── red_blood_cell/
    │   └── plasma/
    └── test/
        ├── red_blood_cell/
        └── plasma/

文件夹名即类别名，ultralytics 按字母序分配 class id。
"""
import os
import argparse
from pathlib import Path

# 禁用 ultralytics 的 git 检查，避免 git 状态异常导致的错误
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
os.environ["GIT_DISCOVERY_ACROSS_FILESYSTEM"] = "0"
os.environ["ULTRALYTICS_DISABLE_GIT"] = "1"

from ultralytics import YOLO


def train(weights: str, data: str, epochs: int, imgsz: int,
          batch: int, device: str, project: str, name: str, workers: int = 8) -> None:
    model = YOLO(weights)
    results = model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        workers=workers,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\n训练完成, 最佳权重: {best}")
    print(f"  验证集结果目录: {results.save_dir}")


def main():
    parser = argparse.ArgumentParser(description="YOLO11 血袋类型分类训练")
    parser.add_argument("--weights", default="yolo11n-cls.pt",
                        help="预训练权重(默认 yolo11n-cls.pt, 自动下载)")
    parser.add_argument("--data", default="../../src/blood_classify/data/datasets/blood_type",
                        help="数据集目录路径(不是 yaml 文件)")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=224, help="训练图片尺寸(分类默认224)")
    parser.add_argument("--batch", type=int, default=32, help="batch size")
    parser.add_argument("--device", default="0", help="设备(0=GPU, cpu=CPU)")
    parser.add_argument("--project", default="../../runs/classify", help="输出项目目录")
    parser.add_argument("--name", default="blood_type", help="本次运行名")
    parser.add_argument("--workers", type=int, default=8, help="数据加载 workers 数量")
    args = parser.parse_args()

    # 验证数据集目录存在
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"错误: 数据集目录不存在: {data_path}")
        return

    train(args.weights, args.data, args.epochs, args.imgsz,
          args.batch, args.device, args.project, args.name, args.workers)


if __name__ == "__main__":
    main()
