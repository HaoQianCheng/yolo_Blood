#!/usr/bin/env python3
"""
YOLO11-OBB RKNN 板端推理脚本 (v2 - 正确后处理)
RK3576 / rknn-toolkit-lite2 v2.3.2

输出格式 (nms=False 导出):
  shape (1, 6, 8400) → 转置 (8400, 6)
  通道: [cx, cy, w, h, angle, conf]
  - cx/cy/w/h: 已是 640 像素坐标
  - angle: 已是弧度 [-pi/4, 3pi/4)
  - conf: 已 sigmoid 过 (0~1)

用法:
  python3 infer.py --src test.jpg          # 单图推理
  python3 infer.py --cam                   # 摄像头0实时检测
  python3 infer.py --cam 1 --conf 0.3      # 摄像头1
"""

import os
import sys
import time
import math
import argparse
from pathlib import Path

import cv2
import numpy as np
from rknnlite.api import RKNNLite

# ==================== 配置 ====================
MODEL_PATH = "/root/blood_label/best_rk3576.rknn"
IMGSZ = 640
CONF_THRES = 0.25
IOU_THRES = 0.45
MAX_DET = 100
OBJ_CLASS = "blood_label"

# ==================== 旋转 NMS ====================

def batch_probiou(box1, box2, eps=1e-7):
    """
    计算旋转框 IoU (cv2.rotatedRectangleIntersection)
    box1: (N, 5) [cx, cy, w, h, angle_rad]
    box2: (M, 5) [cx, cy, w, h, angle_rad]
    返回: (N, M) IoU 矩阵
    """
    N, M = box1.shape[0], box2.shape[0]
    iou_matrix = np.zeros((N, M), dtype=np.float32)

    for i in range(N):
        c1 = box1[i]
        r1 = ((c1[0], c1[1]), (c1[2], c1[3]), math.degrees(c1[4]))
        a1 = c1[2] * c1[3]
        for j in range(M):
            c2 = box2[j]
            r2 = ((c2[0], c2[1]), (c2[2], c2[3]), math.degrees(c2[4]))
            a2 = c2[2] * c2[3]

            if a1 < eps or a2 < eps:
                continue

            inter_val, inter_pts = cv2.rotatedRectangleIntersection(r1, r2)
            if inter_val == cv2.INTERSECT_NONE:
                continue

            inter_area = cv2.contourArea(inter_pts)
            union = a1 + a2 - inter_area
            if union > eps:
                iou_matrix[i, j] = inter_area / union

    return iou_matrix


def rotated_nms(boxes, scores, iou_thres=0.45, max_det=100):
    """
    旋转 NMS
    boxes: (N, 5) [cx, cy, w, h, angle_rad]
    scores: (N,)
    返回: 保留的 index
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    order = np.argsort(-scores)
    if len(order) > 500:
        order = order[:500]

    keep = []
    while len(order) > 0 and len(keep) < max_det:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break

        remaining = order[1:]
        ious = batch_probiou(boxes[i:i+1], boxes[remaining], eps=1e-7)[0]
        order = remaining[ious <= iou_thres]

    return np.array(keep, dtype=np.int64)


# ==================== 后处理 ====================

def postprocess_obb(output, img_shape, conf_thres=0.25, iou_thres=0.45, max_det=100):
    """
    YOLO11-OBB 后处理 (导出已解码，直接用值)
    output: (1, 6, 8400) numpy array
    """
    h, w = img_shape[:2]
    o = output[0].transpose(1, 0)  # (8400, 6)

    cx, cy, bw, bh = o[:, 0], o[:, 1], o[:, 2], o[:, 3]
    conf = o[:, 4]   # 直接用 (已 sigmoid 过，0~1)
    angle = o[:, 5]  # 弧度，直接用

    mask = conf > conf_thres
    if not np.any(mask):
        return []

    cx, cy, bw, bh = cx[mask], cy[mask], bw[mask], bh[mask]
    angle, conf = angle[mask], conf[mask]

    boxes = np.stack([cx, cy, bw, bh, angle], axis=1)
    keep = rotated_nms(boxes, conf, iou_thres, max_det)

    detections = []
    for idx in keep:
        c = math.cos(angle[idx])
        s = math.sin(angle[idx])
        w2, h2 = bw[idx] / 2, bh[idx] / 2

        corners = np.array([[-w2, -h2], [w2, -h2], [w2, h2], [-w2, h2]])
        rot = np.array([[c, -s], [s, c]])
        corners = corners @ rot.T + np.array([cx[idx], cy[idx]])
        corners[:, 0] = np.clip(corners[:, 0], 0, w)
        corners[:, 1] = np.clip(corners[:, 1], 0, h)

        detections.append({
            'xyxyxyxy': corners.reshape(-1).tolist(),
            'conf': float(conf[idx]),
            'cls': OBJ_CLASS,
            'angle_rad': float(angle[idx]),
        })

    return detections


def draw_obb(img, detections):
    canvas = img.copy()
    colors = [(0, 255, 0), (0, 255, 255), (0, 0, 255), (255, 0, 255)]

    for i, det in enumerate(detections):
        pts = np.array(det['xyxyxyxy']).reshape(4, 2).astype(np.int32)
        color = colors[i % len(colors)]
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)
        center = np.mean(pts, axis=0).astype(np.int32)
        cv2.circle(canvas, tuple(center), 3, color, -1)
        label = f"{det['conf']:.2f} {math.degrees(det['angle_rad']):.0f}deg"
        cv2.putText(canvas, label, (int(pts[0][0]), int(pts[0][1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return canvas


# ==================== 推理 ====================

def preprocess(img):
    """预处理: 填充+缩放 -> 640x640, 返回 input_data, scale, pad_x, pad_y"""
    h, w = img.shape[:2]
    scale = IMGSZ / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((IMGSZ, IMGSZ, 3), 114, dtype=np.uint8)
    pad_x = (IMGSZ - new_w) // 2
    pad_y = (IMGSZ - new_h) // 2
    canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
    inp = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)  # UINT8 [0,255], 模型内部有 mean/std 归一化
    return np.expand_dims(inp, axis=0), scale, pad_x, pad_y


def postprocess_detections(detections, scale, pad_x, pad_y, orig_w, orig_h):
    """坐标还原到原图"""
    for det in detections:
        pts = np.array(det['xyxyxyxy']).reshape(4, 2)
        pts[:, 0] = (pts[:, 0] - pad_x) / scale
        pts[:, 1] = (pts[:, 1] - pad_y) / scale
        pts[:, 0] = np.clip(pts[:, 0], 0, orig_w)
        pts[:, 1] = np.clip(pts[:, 1], 0, orig_h)
        det['xyxyxyxy'] = pts.reshape(-1).tolist()
    return detections


def run_inference(src, conf_thres=0.25, save_txt=False):
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

    inp, scale, pad_x, pad_y = preprocess(img)
    orig_h, orig_w = img.shape[:2]
    print(f"图像: {orig_w}x{orig_h}")

    t0 = time.time()
    outputs = rknn.inference(inputs=[inp])
    print(f"推理耗时: {(time.time()-t0)*1000:.1f}ms")

    detections = postprocess_obb(outputs[0], (IMGSZ, IMGSZ), conf_thres)
    detections = postprocess_detections(detections, scale, pad_x, pad_y, orig_w, orig_h)
    print(f"检测到: {len(detections)} 个")

    for i, det in enumerate(detections):
        print(f"  [{i}] conf={det['conf']:.3f} angle={math.degrees(det['angle_rad']):.1f}deg")

    result = draw_obb(img, detections)
    out_img = f"{src_name}_det.jpg"
    cv2.imwrite(out_img, result)
    print(f"保存: {out_img}")

    if save_txt:
        out_txt = f"{src_name}_det.txt"
        with open(out_txt, 'w') as f:
            for det in detections:
                pts = det['xyxyxyxy']
                f.write(f"0 {' '.join(f'{p:.2f}' for p in pts)} {det['conf']:.4f}\n")
        print(f"标签: {out_txt}")

    rknn.release()
    return result, detections


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
        inp, scale, px, py = preprocess(frame)
        outputs = rknn.inference(inputs=[inp])
        dets = postprocess_obb(outputs[0], (IMGSZ, IMGSZ), conf_thres)
        h, w = frame.shape[:2]
        dets = postprocess_detections(dets, scale, px, py, w, h)
        result = draw_obb(frame, dets)

        dt = time.time() - t0
        fps = 0.9 * fps + 0.1 / dt if fps > 0 else 1.0 / dt

        cv2.putText(result, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(result, f"Det: {len(dets)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(result, f"Infer: {dt*1000:.0f}ms", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Blood Label OBB - RK3576", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    rknn.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO11-OBB RK3576 推理")
    parser.add_argument("--src", type=str, default=None)
    parser.add_argument("--cam", type=int, default=None, nargs='?', const=0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--save-txt", action="store_true")
    args = parser.parse_args()

    if args.cam is not None:
        run_camera(args.cam, args.conf)
    elif args.src:
        run_inference(args.src, args.conf, args.save_txt)
    else:
        print("用法: python3 infer.py --src <image.jpg> 或 --cam [id]")
