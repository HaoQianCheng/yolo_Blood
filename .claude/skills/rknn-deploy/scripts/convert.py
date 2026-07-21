#!/usr/bin/env python3
"""
YOLO11-OBB 模型转换脚本: .pt → ONNX → .rknn
用法: python convert.py --pt best.pt --out ./output --target rk3576

前置条件 (WSL2 Ubuntu):
  source /opt/rknn-env/bin/activate
  pip install ultralytics "torch<=2.4.0" onnx==1.16.1
"""

import os
import sys
import argparse

def convert_pt_to_onnx(pt_path, out_dir, opset=12):
    """Step 1: PyTorch → ONNX"""
    from ultralytics import YOLO

    print("===== Step 1: .pt → .onnx =====")
    model = YOLO(pt_path)

    try:
        onnx_path = model.export(format="onnx", opset=opset, simplify=True)
    except Exception as e:
        print(f"simplify 失败 ({e}), 改用无 simplify 导出")
        onnx_path = model.export(format="onnx", opset=opset, simplify=False)

    # 确保 onnx 在 out_dir
    if not os.path.exists(onnx_path):
        onnx_path = str(Path(pt_path).with_suffix(".onnx"))

    print(f"ONNX 导出完成: {onnx_path}")
    return onnx_path


def convert_onnx_to_rknn(onnx_path, out_dir, target="rk3576"):
    """Step 2: ONNX → RKNN (OBB 专用)"""
    from rknn.api import RKNN

    print(f"===== Step 2: .onnx → .rknn (target={target}) =====")
    rknn = RKNN(verbose=True)

    # mean/std: 模型内部做归一化 (输入 UINT8 [0,255])
    # disable_rules: OBB 专用，避 Issue #221
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

    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print(f"build 失败: {ret}")
        sys.exit(1)

    rknn_path = os.path.join(out_dir, f"best_{target}.rknn")
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print(f"export_rknn 失败: {ret}")
        sys.exit(1)

    rknn.release()
    print(f"RKNN 导出完成: {rknn_path}")
    return rknn_path


def main():
    parser = argparse.ArgumentParser(description="YOLO11-OBB 模型转换: pt → ONNX → RKNN")
    parser.add_argument("--pt", required=True, help="输入 .pt 模型路径")
    parser.add_argument("--out", default=".", help="输出目录 (默认当前目录)")
    parser.add_argument("--target", default="rk3576", help="目标平台 (默认 rk3576)")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset 版本 (默认 12)")
    parser.add_argument("--skip-onnx", action="store_true", help="跳过 ONNX 导出 (直接用已有 .onnx)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.skip_onnx:
        onnx_path = os.path.join(args.out, "best.onnx")
        if not os.path.exists(onnx_path):
            print(f"错误: --skip-onnx 但找不到 {onnx_path}")
            sys.exit(1)
        print(f"跳过 ONNX 导出，使用: {onnx_path}")
    else:
        onnx_path = convert_pt_to_onnx(args.pt, args.out, args.opset)

    rknn_path = convert_onnx_to_rknn(onnx_path, args.out, args.target)

    print(f"\n===== 转换完成 =====")
    for f in [onnx_path, rknn_path]:
        if os.path.exists(f):
            size_mb = os.path.getsize(f) / 1024 / 1024
            print(f"  {os.path.basename(f)}: {size_mb:.2f} MB")


if __name__ == "__main__":
    from pathlib import Path
    main()
