"""
准备血袋分类数据集

从训练素材中收集红细胞和血浆图片，组织成 YOLO 分类格式：
datasets/blood_type/
├── train/
│   ├── red_blood_cell/
│   └── plasma/
├── val/
│   ├── red_blood_cell/
│   └── plasma/
└── test/
    ├── red_blood_cell/
    └── plasma/
"""
import os
import shutil
import random
from pathlib import Path

def collect_images(src_dirs, pattern="*.origin.jpg"):
    """从多个目录收集原图（只收集 origin 结尾的图片）"""
    images = []
    for src_dir in src_dirs:
        src_dir = Path(src_dir)
        if src_dir.exists():
            images.extend(list(src_dir.glob(pattern)))
            # 也支持其他格式的 origin 图片
            images.extend(list(src_dir.glob("*.origin.jpeg")))
            images.extend(list(src_dir.glob("*.origin.png")))
    return images

def split_dataset(images, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    """划分数据集"""
    random.seed(seed)
    random.shuffle(images)

    n = len(images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = images[:n_train]
    val = images[n_train:n_train + n_val]
    test = images[n_train + n_val:]

    return train, val, test

def copy_images(images, dst_dir, desc="复制图片"):
    """复制图片到目标目录"""
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    total = len(images)
    for i, img_path in enumerate(images, 1):
        if i % 100 == 0 or i == total:
            print(f"\r{desc}: {i}/{total}", end="", flush=True)
        dst_path = dst_dir / img_path.name
        # 如果文件名冲突，添加前缀
        if dst_path.exists():
            dst_path = dst_dir / f"{img_path.parent.name}_{img_path.name}"
        shutil.copy2(img_path, dst_path)
    print()  # 换行

def main():
    # 项目根目录
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "src" / "blood_classify" / "data" / "datasets" / "blood_type"

    # 红细胞素材来源
    red_cell_sources = [
        project_root / "训练素材" / "训练素材" / "红细胞B 形状素材（斜）",
        project_root / "训练素材" / "训练素材" / "红细胞B 形状素材（斜2）",
        project_root / "训练素材" / "训练素材" / "红细胞B 形状素材（正）",
        project_root / "训练素材2" / "训练素材2" / "红细胞训练素材",
        project_root / "训练素材2" / "训练素材2" / "红细胞训练集",
    ]

    # 血浆素材来源
    plasma_sources = [
        project_root / "训练素材" / "训练素材" / "血浆A 形状素材",
        project_root / "训练素材2" / "训练素材2" / "血浆训练集",
        project_root / "训练素材2" / "训练素材2" / "血浆小袋O形状素材",
    ]

    print("收集红细胞图片...")
    red_cell_images = collect_images(red_cell_sources)
    print(f"找到 {len(red_cell_images)} 张红细胞图片")

    print("\n收集血浆图片...")
    plasma_images = collect_images(plasma_sources)
    print(f"找到 {len(plasma_images)} 张血浆图片")

    print("\n划分数据集...")
    red_train, red_val, red_test = split_dataset(red_cell_images)
    plasma_train, plasma_val, plasma_test = split_dataset(plasma_images)

    print(f"\n红细胞: train={len(red_train)}, val={len(red_val)}, test={len(red_test)}")
    print(f"血浆:   train={len(plasma_train)}, val={len(plasma_val)}, test={len(plasma_test)}")

    print("\n复制图片到数据集目录...")

    # 复制红细胞
    copy_images(red_train, output_dir / "train" / "red_blood_cell", "红细胞 train")
    copy_images(red_val, output_dir / "val" / "red_blood_cell", "红细胞 val")
    copy_images(red_test, output_dir / "test" / "red_blood_cell", "红细胞 test")

    # 复制血浆
    copy_images(plasma_train, output_dir / "train" / "plasma", "血浆 train")
    copy_images(plasma_val, output_dir / "val" / "plasma", "血浆 val")
    copy_images(plasma_test, output_dir / "test" / "plasma", "血浆 test")

    print(f"\n数据集准备完成！")
    print(f"输出目录: {output_dir}")

    # 统计最终结果
    for split in ["train", "val", "test"]:
        print(f"\n{split}:")
        for cls in ["red_blood_cell", "plasma"]:
            count = len(list((output_dir / split / cls).glob("*.*")))
            print(f"  {cls}: {count} 张")

if __name__ == "__main__":
    main()
