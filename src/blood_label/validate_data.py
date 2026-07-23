"""数据校验脚本：检查图片对和 OBB 旋转框标注有效性"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cv2
from extract_labels import extract_label_obb, match_image_pairs


def _parse_obb_line(line: str):
    """解析 OBB 行: class x1 y1 x2 y2 x3 y3 x4 y4 -> (cls, [(x,y)*4])"""
    parts = line.split()
    if len(parts) != 9:
        return None
    cls = int(parts[0])
    coords = [(float(parts[i]), float(parts[i + 1])) for i in range(1, 9, 2)]
    return cls, coords


def validate(raw_dir: str) -> dict:
    pairs = match_image_pairs(raw_dir)
    stats = {
        "total": len(pairs), "ok": 0,
        "no_box": [], "size_mismatch": [], "bad_format": [],
        "corner_out_of_range": [], "read_error": [],
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

        txt_path = Path(str(origin_path).replace(".origin.jpg", ".txt"))
        if not txt_path.exists():
            stats["no_box"].append(origin_path.name)
            continue
        content = txt_path.read_text().strip()
        if not content:
            stats["no_box"].append(origin_path.name)
            continue

        parsed = _parse_obb_line(content)
        if parsed is None:
            stats["bad_format"].append(origin_path.name)
            continue
        _, coords = parsed

        all_ok = True
        for x, y in coords:
            if not (0 <= x <= ow and 0 <= y <= oh):
                stats["corner_out_of_range"].append(f"{origin_path.name}: ({x:.0f},{y:.0f})")
                all_ok = False
                break
        if all_ok:
            stats["ok"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="校验 YOLO-OBB 标注数据")
    parser.add_argument("--raw", default="data/raw", help="原始图片目录")
    args = parser.parse_args()

    stats = validate(args.raw)

    print("=" * 60)
    print("数据校验报告 (OBB)")
    print("=" * 60)
    print(f"总数: {stats['total']}, 通过: {stats['ok']}")
    print(f"读取错误: {len(stats['read_error'])}")
    print(f"尺寸不匹配: {len(stats['size_mismatch'])}")
    print(f"格式错误(非9字段): {len(stats['bad_format'])}")
    print(f"角点越界: {len(stats['corner_out_of_range'])}")
    print(f"无框/丢弃: {len(stats['no_box'])}")

    if stats["bad_format"]:
        print(f"\n格式错误 (前10):")
        for n in stats["bad_format"][:10]:
            print(f"  {n}")
    if stats["corner_out_of_range"]:
        print(f"\n角点越界 (前10):")
        for n in stats["corner_out_of_range"][:10]:
            print(f"  {n}")

    if stats["ok"] == stats["total"]:
        print("\n全部通过!")
    elif stats["ok"] == 0:
        print("\n没有图片通过校验，请检查参数")


if __name__ == "__main__":
    main()
