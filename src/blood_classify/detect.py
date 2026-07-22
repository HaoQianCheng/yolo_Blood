"""YOLO11 分类推理脚本: 血袋类型识别 (红细胞 vs 血浆)

对图片/目录进行分类推理，输出类别和置信度。
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


CLASS_NAMES = {0: "plasma", 1: "red_blood_cell"}
CLASS_NAMES_CN = {"plasma": "血浆", "red_blood_cell": "红细胞"}


def classify(weights: str, source: str, conf: float,
             project: str, name: str) -> None:
    model = YOLO(weights)
    results = model.predict(
        source=source,
        conf=conf,
        project=project,
        name=name,
    )

    for r in results:
        img_name = Path(r.path).name
        if r.probs is not None:
            top1_id = int(r.probs.top1)
            top1_conf = float(r.probs.top1conf)
            cls_name = CLASS_NAMES.get(top1_id, f"cls_{top1_id}")
            cls_cn = CLASS_NAMES_CN.get(cls_name, cls_name)
            print(f"{img_name}: {cls_cn} ({cls_name}) 置信度={top1_conf:.3f}")
        else:
            print(f"{img_name}: 无法分类")

    print(f"\n完成: {len(results)} 张图")


def main():
    parser = argparse.ArgumentParser(description="YOLO11 血袋类型分类推理")
    parser.add_argument("--weights", default="../../runs/classify/blood_type/weights/best.pt",
                        help="训练好的权重(默认 best.pt)")
    parser.add_argument("--source", required=True,
                        help="输入图片或目录")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--project", default="../../runs/classify", help="输出项目目录")
    parser.add_argument("--name", default="predict", help="本次运行名")
    args = parser.parse_args()
    classify(args.weights, args.source, args.conf,
             args.project, args.name)


if __name__ == "__main__":
    main()
