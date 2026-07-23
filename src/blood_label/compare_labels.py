"""三合一对比图：原图+提取框 | 人工标注图(render) | 纯原图
用于校验 extract_labels.py 提取效果（OBB 旋转框版）
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
from extract_labels import extract_label_obb


def make_comparison(raw_dir: str, output_dir: str) -> None:
    raw = Path(raw_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    origins = sorted(raw.glob("*.origin.jpg"))
    renders_by_num = {}
    for r in sorted(raw.glob("*.render.jpg")):
        m = re.search(r'\.(\d{6})\.render', r.name)
        if m:
            renders_by_num[m.group(1)] = r

    done = 0
    for origin_path in origins:
        m = re.search(r'\.(\d{6})\.origin', origin_path.name)
        if not m:
            continue
        num = m.group(1)
        render_path = renders_by_num.get(num)
        if render_path is None:
            print(f"缺少标注图: {origin_path.name}")
            continue

        origin = cv2.imread(str(origin_path))
        render = cv2.imread(str(render_path))
        if origin is None or render is None:
            print(f"无法读取: {origin_path.name}")
            continue

        h, w = origin.shape[:2]

        # 面板1：原图 + 脚本提取的红色旋转框
        panel1 = origin.copy()
        result = extract_label_obb(str(render_path), str(origin_path))
        if result is not None:
            corners, _ = result
            pts = np.array(corners, dtype=np.int32)
            cv2.polylines(panel1, [pts], isClosed=True, color=(0, 0, 255), thickness=5)
            cv2.putText(panel1, "extracted", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # 面板2：人工标注图 render（含绿色框）
        panel2 = render.copy()

        # 面板3：纯原图
        panel3 = origin.copy()

        for panel, title, color in [
            (panel1, "extracted (red)", (0, 0, 255)),
            (panel2, "human label (render)", (0, 255, 0)),
            (panel3, "origin (clean)", (255, 255, 255)),
        ]:
            cv2.putText(panel, title, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        scale = 600 / w
        new_w = 600
        new_h = int(h * scale)
        panels = [
            cv2.resize(panel1, (new_w, new_h)),
            cv2.resize(panel2, (new_w, new_h)),
            cv2.resize(panel3, (new_w, new_h)),
        ]
        combined = np.hstack(panels)

        out_name = origin_path.name.replace(".origin.jpg", ".compare.jpg")
        cv2.imwrite(str(out / out_name), combined)
        done += 1

    print(f"生成 {done} 张对比图 -> {out}")


def main():
    parser = argparse.ArgumentParser(description="生成三合一对比图校验 OBB 标注提取")
    parser.add_argument("--raw", default="data/test_pair", help="含 origin+render 的目录")
    parser.add_argument("--output", default="data/compare", help="对比图输出目录")
    args = parser.parse_args()
    make_comparison(args.raw, args.output)


if __name__ == "__main__":
    main()
