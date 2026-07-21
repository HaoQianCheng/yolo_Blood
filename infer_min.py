#!/usr/bin/env python3
"""最小验证: ctypes + librknnrt.so 加载 best_rk3576.rknn 跑一帧推理"""
import ctypes
import numpy as np
import cv2

LIB = "/usr/lib/librknnrt.so"
MODEL = "/root/blood_label/best_rk3576.rknn"
IMG = "/root/blood_label/test.jpg"

lib = ctypes.CDLL(LIB)

# --- 结构体 ---
class _rknn_context(ctypes.Structure):
    pass
RKNNContext = ctypes.POINTER(_rknn_context)

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

class rknn_output(ctypes.Structure):
    _fields_ = [
        ("want_float", ctypes.c_uint8),
        ("is_prealloc", ctypes.c_uint8),
        ("index", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 8),
    ]

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

# --- 函数签名 ---
lib.rknn_init.argtypes = [RKNNContext, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
lib.rknn_init.restype = ctypes.c_int
lib.rknn_destroy.argtypes = [RKNNContext]
lib.rknn_destroy.restype = ctypes.c_int
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

RKNN_QUERY_IN_NUM = 0
RKNN_QUERY_OUT_NUM = 1
RKNN_QUERY_IN_ATTR = 2
RKNN_QUERY_OUT_ATTR = 3

# NHWC=1, NCHW=0; FLOAT32=0, UINT8=2
FMT_NHWC = 1
TYPE_FLOAT32 = 0
TYPE_UINT8 = 2

# --- 1. 加载模型 ---
with open(MODEL, "rb") as f:
    data = f.read()
print(f"模型大小: {len(data)} bytes")

# 用显式 buffer 持有模型数据, 避免被 GC
model_buf = ctypes.create_string_buffer(data, len(data))
ctx = RKNNContext()
# 先用 COLLECT_MODEL_INFO_ONLY flag 尝试, 只收集信息不完整初始化
RKNN_FLAG_COLLECT_MODEL_INFO_ONLY = 0x00000100
ret = lib.rknn_init(ctx, model_buf, len(data), RKNN_FLAG_COLLECT_MODEL_INFO_ONLY, None)
print(f"rknn_init (MODEL_INFO_ONLY): {ret}")
if ret != 0:
    # 再试普通 flag=0
    ctx2 = RKNNContext()
    ret = lib.rknn_init(ctx2, model_buf, len(data), 0, None)
    print(f"rknn_init (flag=0): {ret}")
    if ret == 0:
        ctx = ctx2
    else:
        # 查 SDK 版本 (用一个 dummy init? 不行, init 失败无法 query)
        # 直接看 .rknn 文件头里的版本信息
        print(f"\n=== .rknn 文件头 (前 64 字节) ===")
        print(' '.join(f'{b:02x}' for b in data[:64]))
        print(f"=== 模型 magic/version 字符串 ===")
        import re
        # 找文件里的版本字符串
        for m in re.finditer(rb'RKNN[\x00-\xff]{0,20}', data[:1024]):
            print(f"  offset {m.start()}: {m.group()[:32]}")
        # 找 toolkit version
        for s in [b'2.3.2', b'2.3.', b'rknn', b'RKNN', b'toolkit']:
            idx = data.find(s)
            if idx >= 0 and idx < 4096:
                print(f"  找到 {s} @ offset {idx}: {data[idx:idx+40]}")
        raise SystemExit(f"rknn_init 失败: {ret}")
print(f"rknn_init 成功: {ret}")

# --- 2. 查询 IO ---
io_num = ctypes.c_uint32()
lib.rknn_query(ctx, RKNN_QUERY_IN_NUM, ctypes.byref(io_num), ctypes.sizeof(io_num))
in_num = io_num.value
lib.rknn_query(ctx, RKNN_QUERY_OUT_NUM, ctypes.byref(io_num), ctypes.sizeof(io_num))
out_num = io_num.value
print(f"输入数: {in_num}, 输出数: {out_num}")

for i in range(in_num):
    a = rknn_tensor_attr()
    a.index = i
    lib.rknn_query(ctx, RKNN_QUERY_IN_ATTR, ctypes.byref(a), ctypes.sizeof(a))
    print(f"  输入[{i}] name={a.name.decode(errors='ignore')} dims={list(a.dims[:a.n_dims])} "
          f"fmt={a.fmt} type={a.type} size={a.size}")

out_attrs = []
for i in range(out_num):
    a = rknn_tensor_attr()
    a.index = i
    lib.rknn_query(ctx, RKNN_QUERY_OUT_ATTR, ctypes.byref(a), ctypes.sizeof(a))
    out_attrs.append(a)
    print(f"  输出[{i}] name={a.name.decode(errors='ignore')} dims={list(a.dims[:a.n_dims])} "
          f"fmt={a.fmt} type={a.type} size={a.size} n_elems={a.n_elems}")

# --- 3. 准备输入 (640x640 NHWC uint8) ---
img = cv2.imread(IMG)
h, w = img.shape[:2]
print(f"原图: {w}x{h}")
scale = 640 / max(h, w)
nw, nh = int(w * scale), int(h * scale)
resized = cv2.resize(img, (nw, nh))
canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
px, py = (640 - nw) // 2, (640 - nh) // 2
canvas[py:py+nh, px:px+nw] = resized
inp = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)  # (640,640,3) uint8

# --- 4. 推理 ---
p_in = rknn_input()
p_in.index = 0
p_in.buf = inp.ctypes.data_as(ctypes.c_void_p)
p_in.size = inp.nbytes
p_in.pass_through = 0
p_in.type = TYPE_UINT8
p_in.fmt = FMT_NHWC

ret = lib.rknn_inputs_set(ctx, 1, ctypes.byref(p_in))
print(f"rknn_inputs_set: {ret}")

ret = lib.rknn_run(ctx, None)
print(f"rknn_run: {ret}")

outs = (rknn_output * out_num)()
for i in range(out_num):
    outs[i].want_float = 1
    outs[i].index = i
    outs[i].is_prealloc = 0

ret = lib.rknn_outputs_get(ctx, out_num, outs, None)
print(f"rknn_outputs_get: {ret}")

for i in range(out_num):
    n = out_attrs[i].n_dims
    dims = list(out_attrs[i].dims[:n])
    n_elems = out_attrs[i].n_elems
    size = outs[i].size
    nfloat = size // 4
    buf = ctypes.cast(outs[i].buf, ctypes.POINTER(ctypes.c_float * nfloat)).contents
    arr = np.ctypeslib.as_array(buf).copy()
    # want_float=1 时输出已是 float32, 按 n_elems reshape
    try:
        arr = arr.reshape(dims)
    except Exception:
        arr = arr.reshape(-1)
    print(f"  输出[{i}] dims={dims} 实际size={size} reshape后shape={arr.shape}")
    print(f"    前 10 值: {arr.flatten()[:10]}")
    print(f"    min={arr.min():.4f} max={arr.max():.4f} mean={arr.mean():.4f}")

lib.rknn_outputs_release(ctx, out_num, outs)
lib.rknn_destroy(ctx)
print("✅ ctypes 推理流程跑通")
