#!/usr/bin/env python3
"""
YOLO11-OBB 实时检测 Web 服务 (优化版)
RK3576 / rknn-toolkit-lite2 v2.3.2
"""

import os
import time
import math
import threading

import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
from rknnlite.api import RKNNLite

# ==================== 配置 ====================
MODEL_PATH = "/root/blood_label/best_n2_rk3576.rknn"
CAM_ID = 0
IMGSZ = 640
DEFAULT_CONF = 0.25
IOU_THRES = 0.45

# ==================== 全局状态 ====================
app = Flask(__name__)

class DetectionState:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.annotated = None
        self.detections = []
        self.fps = 0.0
        self.infer_ms = 0.0
        self.conf_thres = DEFAULT_CONF
        self.running = True

state = DetectionState()

# ==================== 向量化旋转 IoU ====================

def compute_rotated_iou_matrix(boxes):
    """向量化计算旋转框 IoU 矩阵，比纯 Python 循环快 10x+"""
    N = len(boxes)
    if N <= 1:
        return np.zeros((N, N), dtype=np.float32)

    # 提取参数: cx, cy, w, h, angle
    cx, cy, w, h, angle = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
    areas = w * h

    # 预计算旋转框的4个角点 (向量化)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    w2, h2 = w / 2, h / 2

    # 4个角点相对于中心的偏移
    corners_x = np.stack([-w2, w2, w2, -w2], axis=1)  # (N, 4)
    corners_y = np.stack([-h2, -h2, h2, h2], axis=1)  # (N, 4)

    # 旋转
    rot_x = corners_x * cos_a[:, None] - corners_y * sin_a[:, None] + cx[:, None]
    rot_y = corners_x * sin_a[:, None] + corners_y * cos_a[:, None] + cy[:, None]

    # 计算每对框的 IoU
    iou_matrix = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i+1, N):
            r1 = ((cx[i], cy[i]), (w[i], h[i]), math.degrees(angle[i]))
            r2 = ((cx[j], cy[j]), (w[j], h[j]), math.degrees(angle[j]))
            inter_val, inter_pts = cv2.rotatedRectangleIntersection(r1, r2)
            if inter_val == cv2.INTERSECT_NONE:
                continue
            inter_area = cv2.contourArea(inter_pts)
            union = areas[i] + areas[j] - inter_area
            if union > 1e-7:
                iou = inter_area / union
                iou_matrix[i, j] = iou
                iou_matrix[j, i] = iou
    return iou_matrix


def fast_nms(boxes, scores, iou_thres=0.45, max_det=100):
    """快速 NMS：单类别直接用置信度贪心"""
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    # 按置信度降序排序
    order = np.argsort(-scores)

    # 快速路径：候选少时直接取 top-K
    if len(order) <= 20:
        return order[:max_det]

    # 限制 NMS 输入数量
    if len(order) > 30:
        order = order[:30]

    selected_boxes = boxes[order]
    iou_matrix = compute_rotated_iou_matrix(selected_boxes)

    keep = []
    suppressed = np.zeros(len(order), dtype=bool)
    for i in range(len(order)):
        if suppressed[i]:
            continue
        keep.append(order[i])
        if len(keep) >= max_det:
            break
        # 抑制与当前框 IoU 过高的框
        suppressed[i+1:] |= (iou_matrix[i, i+1:] > iou_thres)

    return np.array(keep, dtype=np.int64)


# ==================== 后处理 ====================

def postprocess_obb(output, img_shape, conf_thres=0.25):
    h, w = img_shape[:2]
    o = output[0].transpose(1, 0)  # (8400, 6)

    cx, cy, bw, bh = o[:, 0], o[:, 1], o[:, 2], o[:, 3]
    conf = o[:, 4]   # 直接用 (已 sigmoid, 0~1)
    angle = o[:, 5]  # 弧度，直接用

    # 置信度过滤
    mask = conf > conf_thres
    if not np.any(mask):
        return []

    cx, cy, bw, bh = cx[mask], cy[mask], bw[mask], bh[mask]
    angle, conf = angle[mask], conf[mask]

    boxes = np.stack([cx, cy, bw, bh, angle], axis=1)
    keep = fast_nms(boxes, conf, IOU_THRES)

    # 向量化计算角点
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
        label = f"blood_label {det['conf']:.2f}"
        cv2.putText(canvas, label, (int(pts[0][0]), int(pts[0][1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return canvas


# ==================== 预处理 ====================

def preprocess(img):
    h, w = img.shape[:2]
    scale = IMGSZ / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((IMGSZ, IMGSZ, 3), 114, dtype=np.uint8)
    pad_x = (IMGSZ - new_w) // 2
    pad_y = (IMGSZ - new_h) // 2
    canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
    inp = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return np.expand_dims(inp, axis=0), scale, pad_x, pad_y


# ==================== 推理线程 ====================

def inference_loop(rknn):
    fps_timer = time.time()
    frame_count = 0

    while state.running:
        with state.lock:
            frame = state.frame
        if frame is None:
            time.sleep(0.002)
            continue

        inp, scale, px, py = preprocess(frame)

        t0 = time.time()
        outputs = rknn.inference(inputs=[inp])
        infer_ms = (time.time() - t0) * 1000

        conf_thres = state.conf_thres
        detections = postprocess_obb(outputs[0], (IMGSZ, IMGSZ), conf_thres)

        # 坐标还原到原图
        h, w = frame.shape[:2]
        for det in detections:
            pts = np.array(det['xyxyxyxy']).reshape(4, 2)
            pts[:, 0] = (pts[:, 0] - px) / scale
            pts[:, 1] = (pts[:, 1] - py) / scale
            pts[:, 0] = np.clip(pts[:, 0], 0, w)
            pts[:, 1] = np.clip(pts[:, 1], 0, h)
            det['xyxyxyxy'] = pts.reshape(-1).tolist()

        annotated = draw_obb(frame, detections)

        # 信息面板
        frame_count += 1
        now = time.time()
        if now - fps_timer >= 1.0:
            state.fps = frame_count / (now - fps_timer)
            frame_count = 0
            fps_timer = now
        state.infer_ms = infer_ms

        # 画面板
        panel_h = 90
        panel = np.zeros((panel_h, annotated.shape[1], 3), dtype=np.uint8)
        cv2.putText(panel, f"FPS: {state.fps:.1f}", (15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(panel, f"Detections: {len(detections)}", (15, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(panel, f"Infer: {infer_ms:.0f}ms", (250, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(panel, f"Conf: {conf_thres:.2f}", (250, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(panel, "Blood Label OBB - RK3576 NPU", (450, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)
        annotated = np.vstack([annotated, panel])

        with state.lock:
            state.annotated = annotated
            state.detections = detections

        time.sleep(0.002)


def camera_loop():
    cap = cv2.VideoCapture(CAM_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print(f"无法打开摄像头 {CAM_ID}")
        return

    print(f"摄像头 {CAM_ID} 已打开")

    while state.running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        with state.lock:
            state.frame = frame

    cap.release()


# ==================== MJPEG 流 ====================

def generate_mjpeg():
    while state.running:
        with state.lock:
            annotated = state.annotated
        if annotated is None:
            time.sleep(0.05)
            continue

        _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
        jpg_bytes = buf.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpg_bytes + b'\r\n')

        time.sleep(0.005)


# ==================== 路由 ====================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/status')
def api_status():
    with state.lock:
        return jsonify({
            'fps': round(state.fps, 1),
            'infer_ms': round(state.infer_ms, 1),
            'detections': len(state.detections),
            'conf_thres': state.conf_thres,
        })


@app.route('/api/conf', methods=['POST'])
def api_set_conf():
    data = request.get_json()
    if data and 'conf' in data:
        state.conf_thres = max(0.05, min(0.99, float(data['conf'])))
        return jsonify({'ok': True, 'conf': state.conf_thres})
    return jsonify({'ok': False}), 400


# ==================== 启动 ====================

if __name__ == '__main__':
    print("加载 RKNN 模型...")
    rknn = RKNNLite(verbose=False)
    ret = rknn.load_rknn(MODEL_PATH)
    if ret != 0:
        print(f"加载失败: {ret}")
        exit(1)
    ret = rknn.init_runtime()
    if ret != 0:
        print(f"初始化失败: {ret}")
        exit(1)
    print("RKNN 模型就绪")

    threading.Thread(target=inference_loop, args=(rknn,), daemon=True).start()
    threading.Thread(target=camera_loop, daemon=True).start()

    print("Web 服务启动: http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)
