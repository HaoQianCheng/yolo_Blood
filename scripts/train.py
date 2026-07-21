"""YOLO11-OBB 训练脚本: 血袋标签旋转框检测

用 yolo11n-obb.pt 预训练权重在 blood_label 数据集上微调。
默认 GPU 训练 (device=0), 可用 --device cpu 回退。
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def train(weights: str, data: str, epochs: int, imgsz: int,
          batch: int, device: str, project: str, name: str) -> None:
    model = YOLO(weights)
    results = model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
    )
    # best.pt 路径
    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\n训练完成, 最佳权重: {best}")
    print(f"  验证集结果目录: {results.save_dir}")


def main():
    parser = argparse.ArgumentParser(description="YOLO11-OBB 血袋标签训练")
    parser.add_argument("--weights", default="yolo11n-obb.pt",
                        help="预训练权重(默认 yolo11n-obb.pt, 自动下载)")
    parser.add_argument("--data", default="configs/blood_label.yaml",
                        help="数据集配置 yaml")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=640, help="训练图片尺寸")
    parser.add_argument("--batch", type=int, default=16, help="batch size")
    parser.add_argument("--device", default="0", help="设备(0=GPU, cpu=CPU)")
    parser.add_argument("--project", default="runs/obb", help="输出项目目录")
    parser.add_argument("--name", default="blood_label", help="本次运行名")
    args = parser.parse_args()
    train(args.weights, args.data, args.epochs, args.imgsz,
          args.batch, args.device, args.project, args.name)


if __name__ == "__main__":
    main()
