#!/usr/bin/env python3
"""
RK3576 板端 OBB 推理 — 基于 C API (ctypes + librknnrt.so)
绕过 rknn-toolkit-lite2 的平台限制
"""
import os
import sys
import time
import math
import ctypes
import argparse
from pathlib import Path

import cv2
import numpy as np

# ==================== C API 定义 ====================
LIB_PATH = "/usr/lib/librknnrt.so"
lib = ctypes.CDLL(LIB_PATH)

# 常量
RKNN_SUCC = 0
RKNN_TENSOR_FORMAT_NHWC = 1
RKNN_TENSOR_FORMAT_NCHW = 0
RKNN_TENSOR_TYPE_FLOAT32 = 0
RKNN_TENSOR_TYPE_UINT8 = 2
RKNN_TENSOR_TYPE_INT8 = 3

# RKNNContext 指针
class _rknn_context(ctypes.Structure):
    pass
RKNNContext = ctypes.POINTER(_rknn_context)

# rknn_input
class rknn_input(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("type", ctypes.c_uint32),
        ("fmt", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 4),
    ]

# rknn_output
class rknn_output(ctypes.Structure):
    _fields_ = [
        ("want_float", ctypes.c_uint8),
        ("is_prealloc", ctypes.c_uint8),
        ("index", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 8),
    ]

# rknn_tensor_attr
class rknn_tensor_attr(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("n_dims", ctypes.c_uint32),
        ("dims", ctypes.c_uint32 * 8),
        ("name", ctypes.c_char * 256),
        ("n_elems", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("fmt", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("qnt_type", ctypes.c_uint32),
        ("zp", ctypes.c_int32),
        ("scale", ctypes.c_float),
        ("w_stride", ctypes.c_uint32),
        ("size_with_stride", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("h_stride", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 8),
    ]

# 函数签名
rknn_init = lib.rknn_init
rknn_init.argtypes = [RKNNContext, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
rknn_init.restype = ctypes.c_int

rknn_destroy = lib.rknn_destroy
rknn_destroy.argtypes = [RKNNContext]
rknn_destroy.restype = ctypes.c_int

rknn_export_model = lib.rknn_export_model  # actually query io num
# Let's redefine proper functions:
lib.rknn_query.argtypes = [RKNNContext, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
lib.rknn_query.restype = ctypes.c_int

lib.rknn_inputs_set.argtypes = [RKNNContext, ctypes.c_uint32, ctypes.POINTER(rknn_input)]
lib.rknn_inputs_set.restype = ctypes.c_int

lib.rknn_run.argtypes = [RKNNContext, ctypes.c_void_p]
lib.rknn_run.restype = ctypes.c_int

lib.rknn_outputs_get.argtypes = [RKNNContext, ctypes.c_uint32, ctypes.POINTER(rknn_output), ctypes.c_void_p]
lib.rknn_outputs_get.restype = ctypes.c_int

lib.rknn_outputs_release.argtypes = [RKNNContext, ctypes.c_uint32, ctypes.POINTER(rknn_output)]
lib.rknn_outputs_release.restype = ctypes.c_int

RKNN_QUERY_INPUT_NUM = 0
RKNN_QUERY_OUTPUT_NUM = 1
RKNN_QUERY_INPUT_ATTR = 2
RKNN_QUERY_OUTPUT_ATTR = 3


class RKNNBackend:
    def __init__(self, model_path, target="rk3576"):
        with open(model_path, "rb") as f:
            self.model_data = f.read()
        self.ctx = RKNNContext()
        ret = rknn_init(self.ctx, self.model_data, len(self.model_data), 0, None)
        if ret != 0:
            raise RuntimeError(f"rknn_init 失败: {ret}")

        # 查询输入输出
        io_num = ctypes.c_uint32()
        lib.rknn_query(self.ctx, RKNN_QUERY_INPUT_NUM, ctypes.byref(io_num), ctypes.sizeof(io_num))
        self.input_num = io_num.value
        lib.rknn_query(self.ctx, RKNN_QUERY_OUTPUT_NUM, ctypes.byref(io_num), ctypes.sizeof(io_num))
        self.output_num = io_num.value

        # 输入属性
        self.input_attrs = []
        for i in range(self.input_num):
            attr = rknn_tensor_attr()
            attr.index = i
            lib.rknn_query(self.ctx, RKNN_QUERY_INPUT_ATTR, ctypes.byref(attr), ctypes.sizeof(attr))
            self.input_attrs.append(attr)
            print(f"输入[{i}]: {attr.name.decode() if attr.name else 'N/A'} "
                  f"dims={list(attr.dims[:attr.n_dims])} size={attr.size} fmt={attr.fmt} type={attr.type}")

        # 输出属性
        self.output_attrs = []
        for i in range(self.output_num):
            attr = rknn_tensor_attr()
            attr.index = i
            lib.rknn_query(self.ctx, RKNN_QUERY_OUTPUT_ATTR, ctypes.byref(attr), ctypes.sizeof(attr))
            self.output_attrs.append(attr)
            print(f"输出[{i}]: {attr.name.decode() if attr.name else 'N/A'} "
                  f"dims={list(attr.dims[:attr.n_dims])} size={attr.size} fmt={attr.fmt} type={attr.type}")

    def run(self, input_array):
        """
        input_array: numpy array (HWC, uint8 or float32)
        返回: list of numpy arrays (outputs)
        """
        # 设置输入
        p_input = rknn_input()
        p_input.index = 0
        p_input.type = RKNN_TENSOR_TYPE_FLOAT32 if input_array.dtype == np.float32 else RKNN_TENSOR_TYPE_UINT8
        p_input.fmt = RKNN_TENSOR_FORMAT_NHWC
        p_input.buf = input_array.ctypes.data_as(ctypes.c_void_p)
        p_input.size = input_array.nbytes
        p_input.pass_through = 0

        ret = lib.rknn_inputs_set(self.ctx, 1, ctypes.byref(p_input))
        if ret != 0:
            raise RuntimeError(f"rknn_inputs_set 失败: {ret}")

        # 推理
        ret = lib.rknn_run(self.ctx, None)
        if ret != 0:
            raise RuntimeError(f"rknn_run 失败: {ret}")

        # 获取输出
        outputs = []
        output_ptrs = (rknn_output * self.output_num)()
        for i in range(self.output_num):
            output_ptrs[i].want_float = 1  # FLOAT32
            output_ptrs[i].index = i
            output_ptrs[i].is_prealloc = 0

        ret = lib.rknn_outputs_get(self.ctx, self.output_num, output_ptrs, None)
        if ret != 0:
            raise RuntimeError(f"rknn_outputs_get 失败: {ret}")

        for i in range(self.output_num):
            size = output_ptrs[i].size
            buf = output_ptrs[i].buf
            data = ctypes.cast(buf, ctypes.POINTER(ctypes.c_float * (size // 4))).contents
            arr = np.ctypeslib.as_array(data).copy()
            arr = arr.reshape(list(self.output_attrs[i].dims[:self.output_attrs[i].n_dims]))
            outputs.append(arr)

        lib.rknn_outputs_release(self.ctx, self.output_num, output_ptrs)
        return outputs

    def close(self):
        rknn_destroy(self.ctx)


# ==================== OBB 后处理 ====================
IMGSZ = 640
STRIDES = [8, 16, 32]
OBJ_CLASS = "blood_label"

def make_anchors(feats, strides, grid_cell_offset=0.5):
    anchor_points, stride_tensor = [], []
    for i, stride in enumerate(strides):
        h, w = feats[i]
        sx = np.arange(w) + grid_cell_offset
        sy = np.arange(h) + grid_cell_offset
        sy, sx = np.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(np.stack([sx.ravel(), sy.ravel()], axis=-1))
        stride_tensor.append(np.full((h * w, 1), stride))
    return np.concatenate(anchor_points), np.concatenate(stride_tensor)


def postprocess_obb(output, img_shape, conf_thres=0.25, max_det=100):
    """YOLO11-OBB 后处理: output=(1,6,8400)"""
    h, w = img_shape[:2]
    output = output[0].transpose(1, 0)  # (8400, 6)

    feats = [(IMGSZ // s, IMGSZ // s) for s in STRIDES]
    anchor_points, stride_tensor = make_anchors(feats, STRIDES)

    cx = output[:, 0]
    cy = output[:, 1]
    bw = output[:, 2]
    bh = output[:, 3]
    angle = output[:, 4]
    conf = output[:, 5]

    # sigmoid
    cx = 1 / (1 + np.exp(-np.clip(cx, -20, 20)))
    cy = 1 / (1 + np.exp(-np.clip(cy, -20, 20)))
    bw = 1 / (1 + np.exp(-np.clip(bw, -20, 20)))
    bh = 1 / (1 + np.exp(-np.clip(bh, -20, 20)))
    conf = 1 / (1 + np.exp(-np.clip(conf, -20, 20)))
    angle = 1 / (1 + np.exp(-np.clip(angle, -20, 20)))

    stride_flat = stride_tensor[:, 0]
    anchor_x = anchor_points[:, 0]
    anchor_y = anchor_points[:, 1]

    cx = (cx * 2 - 0.5 + anchor_x) * stride_flat
    cy = (cy * 2 - 0.5 + anchor_y) * stride_flat
    bw = (bw * 2) ** 2 * stride_flat
    bh = (bh * 2) ** 2 * stride_flat
    angle = (angle * 2 - 1) * math.pi / 4

    mask = conf > conf_thres
    if not np.any(mask):
        return []

    cx_filtered = cx[mask]
    cy_filtered = cy[mask]
    bw_filtered = bw[mask]
    bh_filtered = bh[mask]
    angle_filtered = angle[mask]
    conf_filtered = conf[mask]

    detections = []
    for i in range(len(cx_filtered)):
        c = math.cos(angle_filtered[i])
        s = math.sin(angle_filtered[i])
        w2, h2 = bw_filtered[i] / 2, bh_filtered[i] / 2

        corners = np.array([[-w2, -h2], [w2, -h2], [w2, h2], [-w2, h2]])
        rot = np.array([[c, -s], [s, c]])
        corners = corners @ rot.T + np.array([cx_filtered[i], cy_filtered[i]])
        corners[:, 0] = np.clip(corners[:, 0], 0, w)
        corners[:, 1] = np.clip(corners[:, 1], 0, h)

        detections.append({
            'xyxyxyxy': corners.reshape(-1).tolist(),
            'conf': float(conf_filtered[i]),
            'cls': OBJ_CLASS,
            'angle_rad': float(angle_filtered[i]),
        })

    detections.sort(key=lambda x: x['conf'], reverse=True)
    return detections[:max_det]


def draw_obb(img, detections):
    canvas = img.copy()
    colors = [(0, 255, 0), (0, 255, 255), (0, 0, 255)]
    for i, det in enumerate(detections):
        pts = np.array(det['xyxyxyxy']).reshape(4, 2).astype(np.int32)
        color = colors[i % len(colors)]
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)
        center = np.mean(pts, axis=0).astype(np.int32)
        cv2.circle(canvas, tuple(center), 3, color, -1)
        label = f"{det['cls']} {det['conf']:.2f}"
        cv2.putText(canvas, label, (int(pts[0][0]), int(pts[0][1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return canvas


def run_inference(src, conf_thres=0.25, save_txt=False):
    print(f"加载模型: {LIB_PATH}")
    backend = RKNNBackend("/root/blood_label/best_rk3576.rknn")

    if isinstance(src, str):
        img = cv2.imread(src)
        src_name = Path(src).stem
    else:
        img = src
        src_name = "capture"

    orig_h, orig_w = img.shape[:2]
    print(f"图像: {orig_w}x{orig_h}")

    scale = IMGSZ / max(orig_h, orig_w)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((IMGSZ, IMGSZ, 3), 114, dtype=np.uint8)
    pad_x, pad_y = (IMGSZ - new_w) // 2, (IMGSZ - new_h) // 2
    canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized

    input_data = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    input_data = input_data.astype(np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)

    t0 = time.time()
    outputs = backend.run(input_data)
    elapsed = time.time() - t0
    print(f"推理耗时: {elapsed*1000:.1f}ms")
    print(f"输出形状: {outputs[0].shape}")

    detections = postprocess_obb(outputs[0], (IMGSZ, IMGSZ), conf_thres)
    print(f"检测到: {len(detections)} 个目标")

    for det in detections:
        pts = np.array(det['xyxyxyxy']).reshape(4, 2)
        pts[:, 0] = (pts[:, 0] - pad_x) / scale
        pts[:, 1] = (pts[:, 1] - pad_y) / scale
        pts[:, 0] = np.clip(pts[:, 0], 0, orig_w)
        pts[:, 1] = np.clip(pts[:, 1], 0, orig_h)
        det['xyxyxyxy'] = pts.reshape(-1).tolist()

    result_img = draw_obb(img, detections)
    out_img = f"{src_name}_det.jpg"
    cv2.imwrite(out_img, result_img)
    print(f"结果已保存: {out_img}")

    if save_txt:
        out_txt = f"{src_name}_det.txt"
        with open(out_txt, 'w') as f:
            for det in detections:
                pts = det['xyxyxyxy']
                f.write(f"0 {' '.join(f'{p:.2f}' for p in pts)} {det['conf']:.4f}\n")
        print(f"标签已保存: {out_txt}")

    for det in detections:
        print(f"  [{det['cls']}] conf={det['conf']:.3f} angle={det['angle_rad']:.3f}rad")

    backend.close()
    return result_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--save-txt", action="store_true")
    args = parser.parse_args()

    if args.src:
        run_inference(args.src, args.conf, args.save_txt)
    else:
        print("用法: python3 infer.py --src <image.jpg>")