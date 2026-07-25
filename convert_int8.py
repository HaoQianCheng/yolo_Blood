#!/usr/bin/env python3
"""
YOLO11-OBB INT8 量化转换脚本
"""

import os
import sys

def convert_int8(onnx_path, calib_dir, out_dir, target="rk3576"):
    """ONNX → RKNN (INT8 量化)"""
    from rknn.api import RKNN

    print(f"===== INT8 量化转换 (target={target}) =====")

    # 收集校准图片
    calib_images = []
    for f in os.listdir(calib_dir):
        if f.endswith('.jpg') or f.endswith('.png'):
            calib_images.append(os.path.join(calib_dir, f))

    if len(calib_images) < 5:
        print(f"错误: 校准图片太少 ({len(calib_images)}张), 至少需要5张")
        sys.exit(1)

    print(f"校准图片: {len(calib_images)}张")

    rknn = RKNN(verbose=True)

    # mean/std: 模型内部做归一化 (输入 UINT8 [0,255])
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=target,
        disable_rules=["reduce_reshape_op_around_concat"],
    )

    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print(f"load_onnx 失败: {ret}")
        sys.exit(1)

    # INT8 量化
    dataset_txt = os.path.join(calib_dir, "calib.txt")
    ret = rknn.build(
        do_quantization=True,
        dataset=dataset_txt  # 校准数据集txt文件
    )
    if ret != 0:
        print(f"build 失败: {ret}")
        sys.exit(1)

    rknn_path = os.path.join(out_dir, f"best_{target}_int8.rknn")
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print(f"export_rknn 失败: {ret}")
        sys.exit(1)

    rknn.release()
    print(f"INT8 RKNN 导出完成: {rknn_path}")
    return rknn_path


if __name__ == "__main__":
    onnx_path = "/mnt/g/WSL/model/best.onnx"
    calib_dir = "/mnt/g/WSL/calib_images"
    out_dir = "/mnt/g/WSL/model"

    # 创建校准目录
    os.makedirs(calib_dir, exist_ok=True)

    # 检查文件
    if not os.path.exists(onnx_path):
        print(f"错误: 找不到 {onnx_path}")
        sys.exit(1)

    convert_int8(onnx_path, calib_dir, out_dir)
