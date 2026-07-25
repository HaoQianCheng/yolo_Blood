#!/usr/bin/env python3
"""
RKOPT ONNX -> RKNN (int8 PTQ)

Sigmoid/DFL 已移出模型, RKNN 自己做 int8 量化不会 conf 全 0
输出 4 分支: 3 检测头 [1,65,H,W] + angle_feature [1,1,8400]
"""

import os
import sys


def convert(onnx_path, calib_dir, out_dir, target="rk3576"):
    from rknn.api import RKNN

    print(f"===== RKOPT ONNX -> RKNN int8 (target={target}) =====")
    print(f"ONNX: {onnx_path}")

    calib_images = [os.path.join(calib_dir, f) for f in os.listdir(calib_dir)
                    if f.endswith(('.jpg', '.png'))]
    if len(calib_images) < 5:
        print(f"错误: 校准图片太少 ({len(calib_images)}张), 至少需要5张")
        sys.exit(1)
    print(f"校准图片: {len(calib_images)}张")

    rknn = RKNN(verbose=True)

    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=target,
        disable_rules=["reduce_reshape_op_around_concat"],  # Issue #221 OBB bug
    )

    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print(f"load_onnx 失败: {ret}")
        sys.exit(1)

    dataset_txt = os.path.join(calib_dir, "calib.txt")
    if not os.path.exists(dataset_txt):
        print(f"错误: 找不到校准列表 {dataset_txt}")
        sys.exit(1)

    ret = rknn.build(do_quantization=True, dataset=dataset_txt)
    if ret != 0:
        print(f"build 失败: {ret}")
        sys.exit(1)

    rknn_path = os.path.join(out_dir, f"best_rkopt_{target}.rknn")
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print(f"export_rknn 失败: {ret}")
        sys.exit(1)

    rknn.release()
    print(f"\nRKNN 导出完成: {rknn_path}")
    print(f"文件大小: {os.path.getsize(rknn_path)/1024/1024:.2f} MB")
    return rknn_path


if __name__ == "__main__":
    onnx_path = "/mnt/c/Users/qianc/Projects/yolo_Blood/runs/obb/runs/obb/blood_label_n-2/weights/best.onnx"
    calib_dir = "/mnt/g/WSL/calib_images"
    out_dir = "/mnt/g/WSL/model"

    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(onnx_path):
        print(f"错误: 找不到 {onnx_path}")
        sys.exit(1)

    convert(onnx_path, calib_dir, out_dir)
