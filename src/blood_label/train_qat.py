"""YOLO11-OBB QAT/INT8 训练+导出血袋标签检测模型

流程: 加载预训练权重 → 微调 → 验证 → 导出 ONNX INT8 模型
导出的 ONNX 可直接用 RKNN-Toolkit2 转为 .rknn 部署到 RK3576。

用法:
    cd src/blood_label
    python train_qat.py                          # 默认参数
    python train_qat.py --epochs 30 --batch 8    # 自定义
    python train_qat.py --no-export              # 只训练不导出
"""
import argparse
import tempfile
from pathlib import Path

import yaml
from ultralytics import YOLO

# 脚本所在目录, 用于将相对路径转为绝对路径
_SCRIPT_DIR = Path(__file__).resolve().parent


def _abs(path: str) -> str:
    """将相对于脚本目录的路径转为绝对路径, 避免 ultralytics datasets_dir 拼接问题"""
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str((_SCRIPT_DIR / p).resolve())


def _fix_data_yaml(yaml_path: str) -> str:
    """读取数据集 YAML, 将 path 字段转为绝对路径后写入临时文件,
    避免 ultralytics 将相对 path 与 datasets_dir 错误拼接。"""
    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 将 path 字段从 "相对于项目根" 转为绝对路径
    # YAML 的 path 是相对于项目根 (yolo_Blood/) 的, 不是相对于 YAML 文件
    project_root = _SCRIPT_DIR.parent.parent          # src/blood_label → yolo_Blood
    cfg["path"] = str((project_root / cfg["path"]).resolve())

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8",
    )
    yaml.dump(cfg, tmp, default_flow_style=False, allow_unicode=True)
    tmp.close()
    return tmp.name


def train_and_export(weights: str, data: str, epochs: int, imgsz: int,
                     batch: int, device: str, project: str, name: str,
                     export: bool = True) -> None:
    model = YOLO(_abs(weights))
    abs_data = _abs(data)
    project = _abs(project)

    # 修复 YAML 中的 path 字段 (绝对路径)
    fixed_yaml = _fix_data_yaml(abs_data)

    try:
        # ---- 1. 微调训练 ----
        results = model.train(
            data=fixed_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=project,
            name=name,
            degrees=180,       # 全角度旋转增强
            flipud=0.5,        # 上下翻转 50%
            fliplr=0.5,        # 左右翻转 50%
        )
        best = Path(results.save_dir) / "weights" / "best.pt"
        print(f"\n训练完成, 最佳权重: {best}")

        # ---- 2. 验证 ----
        val_model = YOLO(str(best))
        metrics = val_model.val(data=fixed_yaml, imgsz=imgsz, device=device)
        print(f"验证结果: mAP@50={metrics.box.map50:.4f}  "
              f"mAP@50-95={metrics.box.map:.4f}")

        # ---- 3. 导出 ONNX INT8 ----
        if export:
            print("\n正在导出 ONNX INT8 模型 (含校准量化)...")
            onnx_path = val_model.export(
                format="onnx",
                quantize=8,            # INT8 量化
                data=fixed_yaml,       # 校准数据 (自动取 val 集)
                imgsz=imgsz,
                simplify=True,
            )
            print(f"ONNX INT8 模型已导出: {onnx_path}")
            print(f"\n下一步: 用 RKNN-Toolkit2 将 ONNX 转为 .rknn:")
            print(f"  python convert_to_rknn.py --onnx {onnx_path}")
    finally:
        Path(fixed_yaml).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="YOLO11-OBB QAT/INT8 训练+导出")
    parser.add_argument("--weights", default="../../runs/obb/runs/obb/blood_label/weights/best.pt",
                        help="预训练权重 (默认用之前训练好的 best.pt)")
    parser.add_argument("--data", default="../../configs/blood_label.yaml",
                        help="数据集配置 yaml")
    parser.add_argument("--epochs", type=int, default=50, help="微调轮数")
    parser.add_argument("--imgsz", type=int, default=640, help="训练图片尺寸")
    parser.add_argument("--batch", type=int, default=16, help="batch size")
    parser.add_argument("--device", default="0", help="设备 (0=GPU, cpu=CPU)")
    parser.add_argument("--project", default="../../runs/obb", help="输出项目目录")
    parser.add_argument("--name", default="blood_label_qat", help="本次运行名")
    parser.add_argument("--no-export", action="store_true",
                        help="只训练, 不导出 ONNX")
    args = parser.parse_args()

    train_and_export(
        weights=args.weights,
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        export=not args.no_export,
    )


if __name__ == "__main__":
    main()
