"""YOLO11-OBB 裁剪脚本: 从原图中截取检测到的标签区域

用训练好的 best.pt 推理，将 OBB 旋转框内的标签区域旋转校正后裁剪出来，保存为 JPG。
用于后续 OCR 识别。
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def crop_obb_region(img, pts):
    """用 OBB 旋转角度校正，裁剪出水平的标签区域。

    Args:
        img: 原图 (BGR)
        pts: OBB 四个角点 (4, 2) int32

    Returns:
        裁剪后且旋转摆正的图像 (BGR)
    """
    rect = cv2.minAreaRect(pts.astype(np.float32))
    center, (w, h), angle = rect

    # minAreaRect 角度约定: angle∈(-90,0] 时 width 对应水平边
    # 当 h > w 时说明矩形是"竖着"的，需要把角度旋转 90°
    if w < h:
        w, h = h, w
        angle += 90

    # 旋转整张图使标签水平
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 计算旋转后画布大小，避免裁切
    img_h, img_w = img.shape[:2]
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(img_h * sin_a + img_w * cos_a)
    new_h = int(img_h * cos_a + img_w * sin_a)
    M[0, 2] += (new_w - img_w) / 2
    M[1, 2] += (new_h - img_h) / 2

    rotated = cv2.warpAffine(img, M, (new_w, new_h))

    # 将中心点变换到旋转后的坐标系
    new_cx = M[0, 0] * center[0] + M[0, 1] * center[1] + M[0, 2]
    new_cy = M[1, 0] * center[0] + M[1, 1] * center[1] + M[1, 2]

    # 裁剪
    x1 = max(0, int(new_cx - w / 2))
    y1 = max(0, int(new_cy - h / 2))
    x2 = min(new_w, int(new_cx + w / 2))
    y2 = min(new_h, int(new_cy + h / 2))

    return rotated[y1:y2, x1:x2]


def crop_labels(weights, source, conf, output):
    model = YOLO(weights)
    src = Path(source)
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有图片
    if src.is_file():
        images = [src]
    elif src.is_dir():
        images = sorted(src.glob("*.jpg")) + sorted(src.glob("*.png"))
    else:
        print(f"错误: {source} 不存在")
        return

    total_crops = 0
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"跳过: {img_path} (无法读取)")
            continue

        results = model.predict(source=str(img_path), conf=conf, verbose=False)
        r = results[0]

        if r.obb is None or len(r.obb) == 0:
            print(f"{img_path.name}: 未检测到标签")
            continue

        stem = img_path.stem
        for i, obb in enumerate(r.obb):
            pts = obb.xyxyxyxy[0].cpu().numpy().astype(np.int32)  # (4, 2)
            cropped = crop_obb_region(img, pts)
            out_path = out_dir / f"{stem}_crop_{i}.jpg"
            cv2.imwrite(str(out_path), cropped)
            conf_val = float(obb.conf[0])
            print(f"{img_path.name} -> {out_path.name} (置信度: {conf_val:.2f})")
            total_crops += 1

    print(f"\n完成: {len(images)} 张图, 裁剪 {total_crops} 个标签, 输出目录 {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="YOLO11-OBB 标签裁剪")
    parser.add_argument("--weights", default="../../runs/obb/runs/obb/blood_label/weights/best.pt",
                        help="训练好的权重(默认 best.pt)")
    parser.add_argument("--source", required=True,
                        help="输入图片或目录")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--output", default="crops", help="输出目录(默认 crops)")
    args = parser.parse_args()
    crop_labels(args.weights, args.source, args.conf, args.output)


if __name__ == "__main__":
    main()
