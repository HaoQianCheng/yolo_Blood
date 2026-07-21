"""YOLO11-OBB 推理脚本: 血袋标签旋转框检测

用训练好的 best.pt 对图片/目录推理, 输出画旋转框的图, 可选保存 OBB 坐标 txt。
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def _draw_obb(img, obb):
    """在图上画一个旋转框(绿色多边形 + 类别+置信度文字)。"""
    # obb.cxywhr / obb.xyxyxyxy: ultralytics OBB 结果对象
    pts = obb.xyxyxyxy[0].cpu().numpy().astype(np.int32)  # 4 角点 (4,2)
    cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
    cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
    conf = float(obb.conf[0])
    cls = int(obb.cls[0])
    cv2.putText(img, f"blood_label {conf:.2f}", (cx, max(cy - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


def detect(weights: str, source: str, conf: float,
           save_txt: bool, project: str, name: str) -> None:
    model = YOLO(weights)
    results = model.predict(
        source=source,
        conf=conf,
        save=not save_txt,          # 默认只保存画框图(save_txt 时改存 txt)
        save_txt=save_txt,
        project=project,
        name=name,
    )

    # 自己再画一遍带中文友好标注的图(ultralytics 默认图也在 save_dir)
    src = Path(source)
    out_dir = Path(project) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        img = r.orig_img.copy()
        for obb in r.obb:
            _draw_obb(img, obb)
        out_path = out_dir / f"det_{Path(r.path).name}"
        cv2.imwrite(str(out_path), img)
        n = len(r.obb) if r.obb is not None else 0
        print(f"{Path(r.path).name}: 检测到 {n} 个标签 -> {out_path}")

    print(f"\n完成: {len(results)} 张图, 输出目录 {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="YOLO11-OBB 血袋标签推理")
    parser.add_argument("--weights", default="runs/obb/blood_label/weights/best.pt",
                        help="训练好的权重(默认 best.pt)")
    parser.add_argument("--source", required=True,
                        help="输入图片或目录")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--save-txt", action="store_true", help="保存 OBB 坐标 txt")
    parser.add_argument("--project", default="runs/detect", help="输出项目目录")
    parser.add_argument("--name", default="blood_label", help="本次运行名")
    args = parser.parse_args()
    detect(args.weights, args.source, args.conf,
           args.save_txt, args.project, args.name)


if __name__ == "__main__":
    main()
