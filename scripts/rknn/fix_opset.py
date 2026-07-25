#!/usr/bin/env python3
"""把 best_int8.onnx 的 opset 从 20 降到 19 (rknn-toolkit2 2.3.2 要求 <=19)"""
import onnx
from onnx import version_converter

src = "/mnt/c/Users/qianc/Projects/yolo_Blood/runs/obb/blood_label_qat/weights/best_int8.onnx"
dst = "/mnt/g/WSL/model/best_int8_op19.onnx"

m = onnx.load(src)
print(f"原始 opset: {[o.version for o in m.opset_import]}")
print(f"输入: {[i.name + str([d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input]}")
print(f"输出: {[o.name + str([d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output]}")

try:
    m19 = version_converter.convert_version(m, 19)
    onnx.save(m19, dst)
    print(f"已保存 opset 19: {dst}")
    print(f"文件大小: {__import__('os').path.getsize(dst)/1024/1024:.2f} MB")
except Exception as e:
    print(f"转换失败: {e}")
