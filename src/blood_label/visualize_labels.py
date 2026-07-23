"""将 YOLO-OBB 旋转框标注画回原图，生成可视化标注图，用于校验标注质量"""
import argparse
import shutil
from pathlib import Path
import cv2
import numpy as np


def _parse_obb_line(line: str):
    parts = line.split()
    if len(parts) != 9:
        return None
    cls = int(parts[0])
    coords = [(float(parts[i]), float(parts[i + 1])) for i in range(1, 9, 2)]
    return cls, coords


def draw_boxes(raw_dir: str, output_dir: str) -> None:
    raw = Path(raw_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    images = sorted(raw.glob("*.origin.jpg"))
    done = empty = 0

    for img_path in images:
        txt_path = raw / img_path.name.replace(".origin.jpg", ".txt")
        if not txt_path.exists():
            shutil.copy2(img_path, out / img_path.name.replace(".origin.jpg", ".annotated.jpg"))
            empty += 1
            continue
        content = txt_path.read_text().strip()
        if not content:
            shutil.copy2(img_path, out / img_path.name.replace(".origin.jpg", ".annotated.jpg"))
            empty += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"无法读取: {img_path.name}")
            continue

        h, w = img.shape[:2]
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            parsed = _parse_obb_line(line)
            if parsed is None:
                continue
            cls, coords = parsed
            pts = np.array(coords, dtype=np.int32)
            # 画旋转多边形(绿色，3px)
            cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
            # 标签文字
            cx = int(pts[:, 0].mean())
            cy = int(pts[:, 1].mean())
            label = f"cls={cls}"
            font_scale = max(0.5, min(w, h) / 1200)
            cv2.putText(img, label, (cx, max(cy - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)

        out_name = img_path.name.replace(".origin.jpg", ".annotated.jpg")
        cv2.imwrite(str(out / out_name), img)
        done += 1

    print(f"完成: {done} 张有标注图, {empty} 张无标注")


def main():
    parser = argparse.ArgumentParser(description="将 YOLO-OBB 标注画回原图")
    parser.add_argument("--raw", default="data/raw", help="原始数据目录")
    parser.add_argument("--output", default="data/annotated", help="标注图输出目录")
    args = parser.parse_args()
    draw_boxes(args.raw, args.output)


if __name__ == "__main__":
    main()
