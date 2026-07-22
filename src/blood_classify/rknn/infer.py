#!/usr/bin/env python3
"""
YOLO11-cls 血袋类型分类 RKNN 板端推理
RK3576 / rknn-toolkit-lite2 v2.3.2

用法:
  python3 infer.py --src test.jpg          # 单图推理
  python3 infer.py --cam                   # 摄像头0实时分类
  python3 infer.py --cam 1 --conf 0.6      # 摄像头1
"""

import sys
import time
import argparse
from pathlib import Path

import cv2
import numpy as np
from rknnlite.api import RKNNLite

# ==================== 配置 ====================
MODEL_PATH = "/root/blood_classify/best_rk3576.rknn"
IMGSZ = 224
CLASS_NAMES = ["plasma", "red_blood_cell"]
CLASS_NAMES_CN = ["血浆", "红细胞"]


# ==================== 后处理 ====================

def softmax(logits):
    """数值稳定的 softmax。"""
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


def postprocess_cls(output, conf_thres=0.25):
    """分类后处理: output shape (1, num_classes) -> (class_id, class_name, conf)"""
    logits = output[0].flatten()
    probs = softmax(logits)
    cls_id = int(np.argmax(probs))
    conf = float(probs[cls_id])
    if conf < conf_thres:
        return None, conf, probs
    return cls_id, conf, probs


# ==================== 预处理 ====================

def preprocess(img):
    """预处理: resize -> 224x224, RGB, uint8"""
    resized = cv2.resize(img, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)
    inp = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return np.expand_dims(inp, axis=0)


# ==================== 绘制 ====================

def draw_result(img, cls_id, conf, probs):
    """在图上绘制分类结果。"""
    canvas = img.copy()
    h, w = canvas.shape[:2]

    # 半透明背景条
    bar_h = 60
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)

    if cls_id is not None:
        name_cn = CLASS_NAMES_CN[cls_id]
        name_en = CLASS_NAMES[cls_id]
        color = (0, 255, 0) if cls_id == 1 else (0, 200, 255)
        text = f"{name_cn} ({name_en}) {conf:.1%}"
        cv2.putText(canvas, text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    else:
        cv2.putText(canvas, f"Uncertain (max={max(probs):.1%})", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return canvas


# ==================== 推理 ====================

def run_inference(src, conf_thres=0.25):
    rknn = RKNNLite(verbose=False)
    ret = rknn.load_rknn(MODEL_PATH)
    if ret != 0:
        print(f"加载 RKNN 失败: {ret}")
        return
    ret = rknn.init_runtime()
    if ret != 0:
        print(f"初始化 runtime 失败: {ret}")
        return
    print(f"RKNN 模型加载成功: {MODEL_PATH}")

    img = cv2.imread(src)
    if img is None:
        print(f"无法读取图像: {src}")
        return
    src_name = Path(src).stem
    print(f"图像: {img.shape[1]}x{img.shape[0]}")

    inp = preprocess(img)
    print(f"输入: shape={inp.shape} dtype={inp.dtype}")

    t0 = time.time()
    outputs = rknn.inference(inputs=[inp])
    infer_ms = (time.time() - t0) * 1000
    print(f"推理耗时: {infer_ms:.1f}ms")

    cls_id, conf, probs = postprocess_cls(outputs[0], conf_thres)
    if cls_id is not None:
        print(f"结果: {CLASS_NAMES_CN[cls_id]} ({CLASS_NAMES[cls_id]}) 置信度={conf:.3f}")
    else:
        print(f"结果: 不确定 (最大置信度={max(probs):.3f})")

    # 打印所有类别概率
    for i, (name, cn, p) in enumerate(zip(CLASS_NAMES, CLASS_NAMES_CN, probs)):
        print(f"  {cn}({name}): {p:.4f}")

    result = draw_result(img, cls_id, conf, probs)
    out_img = f"{src_name}_cls.jpg"
    cv2.imwrite(out_img, result)
    print(f"保存: {out_img}")

    rknn.release()
    return cls_id, conf


def run_camera(cam_id=0, conf_thres=0.25):
    rknn = RKNNLite(verbose=False)
    ret = rknn.load_rknn(MODEL_PATH)
    if ret != 0:
        print(f"加载 RKNN 失败: {ret}")
        return
    ret = rknn.init_runtime()
    if ret != 0:
        print(f"初始化 runtime 失败: {ret}")
        return

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"无法打开摄像头: {cam_id}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("摄像头已启动，按 Q 退出")
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        inp = preprocess(frame)
        outputs = rknn.inference(inputs=[inp])
        cls_id, conf, probs = postprocess_cls(outputs[0], conf_thres)
        dt = time.time() - t0

        fps = 0.9 * fps + 0.1 / dt if fps > 0 else 1.0 / dt

        result = draw_result(frame, cls_id, conf, probs)

        # 底部信息
        info = f"FPS:{fps:.1f} Infer:{dt*1000:.0f}ms"
        cv2.putText(result, info, (10, result.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Blood Type Classifier - RK3576", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    rknn.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO11-cls 血袋类型分类 RK3576 推理")
    parser.add_argument("--src", type=str, default=None)
    parser.add_argument("--cam", type=int, default=None, nargs='?', const=0)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    if args.cam is not None:
        run_camera(args.cam, args.conf)
    elif args.src:
        run_inference(args.src, args.conf)
    else:
        print("用法: python3 infer.py --src <image.jpg> 或 --cam [id]")
