"""数据集划分：7:2:1 拆分为 train/val/test
空 txt(被丢弃的图)不进任何子集，另存到手动标记文件夹供人工处理。
复制到 datasets 的 OBB 标签归一化为 ultralytics 要求的格式(坐标除以图片宽/高)。
"""
import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cv2


def _normalize_obb_label(txt_path: Path, img_path: Path) -> str:
    """读 OBB 标签(绝对像素 class x1 y1..x4 y4), 转归一化(class nx1 ny1..nx4 ny4)。
    ultralytics OBB 要求坐标 ∈ [0,1], 否则标签被当 corrupt 丢弃。"""
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {img_path}")
    h, w = img.shape[:2]
    lines = []
    for line in txt_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) != 9:
            continue
        cls = parts[0]
        coords = [(float(parts[i]) / w, float(parts[i + 1]) / h) for i in range(1, 9, 2)]
        flat = " ".join(f"{x:.6f} {y:.6f}" for x, y in coords)
        lines.append(f"{cls} {flat}")
    return "\n".join(lines) + "\n" if lines else ""

from extract_labels import match_image_pairs


def split_dataset(raw_dir: str, output_dir: str, manual_dir: str,
                  train_r: float = 0.7, val_r: float = 0.2, seed: int = 42):
    raw = Path(raw_dir)
    out = Path(output_dir)
    manual = Path(manual_dir)

    # 复用 extract_labels 的配对逻辑(sorted+zip, 已验证 0 错配)
    # 建 origin 名 -> render 路径 映射, 供丢弃图找对应 render
    pairs = match_image_pairs(raw_dir)
    render_map = {o.name: r for o, r in pairs}

    valid = []
    discarded = []
    for origin in sorted(raw.glob("*.origin.jpg")):
        txt_name = origin.name.replace(".origin.jpg", ".txt")
        txt_path = raw / txt_name
        if not txt_path.exists():
            continue
        # 空 txt = 被丢弃(标签超出大框)→ 不进子集, 另存手动标记
        if txt_path.read_text().strip():
            valid.append(origin)
        else:
            discarded.append(origin)

    print(f"有效标注: {len(valid)} 张, 丢弃(另存手动标记): {len(discarded)} 张")

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
            # 标签归一化后写入(不直接复制绝对坐标原版)
            dest_txt = lbl_dir / dest_img_name.replace(".jpg", ".txt")
            dest_txt.write_text(_normalize_obb_label(raw / txt_name, origin))

        print(f"{split_name}: {len(files)} 张")

    # 丢弃图(空 txt)另存到手动标记文件夹: 原图 + 标注图, 供人工对照修改
    # render 时间戳与 origin 不同(毫秒位差异), 按 match_image_pairs 的配对找对应 render
    if discarded:
        manual.mkdir(parents=True, exist_ok=True)
        for origin in discarded:
            shutil.copy2(origin, manual / origin.name)
            render = render_map.get(origin.name)
            if render is not None:
                shutil.copy2(render, manual / render.name)
        print(f"手动标记(丢弃图): {len(discarded)} 张 -> {manual}")

    print(f"总计: {n} 张")


def main():
    parser = argparse.ArgumentParser(description="划分数据集")
    parser.add_argument("--raw", default="data/raw", help="原始数据目录")
    parser.add_argument("--output", default="data/datasets/blood_label", help="输出目录")
    parser.add_argument("--manual", default="data/manual_label", help="丢弃图(空txt)手动标记目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()
    split_dataset(args.raw, args.output, args.manual, seed=args.seed)


if __name__ == "__main__":
    main()
