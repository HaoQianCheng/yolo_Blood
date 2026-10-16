#!/usr/bin/env python3
"""
YOLO11-OBB 实时检测 Web 服务 (优化版)
RK3576 / rknn-toolkit-lite2 v2.3.2
"""

import os
import time
import math
import threading
import base64

import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
from rknnlite.api import RKNNLite

# ==================== 配置 ====================
MODEL_PATH = "/root/blood_label/best_rkopt_rk3576.rknn"
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
        self.pre_ms = 0.0
        self.post_ms = 0.0
        self.draw_ms = 0.0
        self.conf_thres = DEFAULT_CONF
        self.running = True
        self.is_cropping = False
        self.crop_buffer = []

state = DetectionState()
crop_lock = threading.Lock()

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

    # probiou 向量化旋转 IoU (替代 cv2 逐对循环, 快 10x+)
    a = (w**2) / 4 * cos_a**2 + (h**2) / 4 * sin_a**2
    b = (w**2) / 4 * sin_a**2 + (h**2) / 4 * cos_a**2
    c = (w**2 - h**2) / 4 * sin_a * cos_a

    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    a_sum = a[:, None] + a[None, :]
    b_sum = b[:, None] + b[None, :]
    c_sum = c[:, None] + c[None, :]
    ab1 = (a + b)[:, None]
    ab2 = (a + b)[None, :]
    denom = ab1 * ab2 - c_sum**2 + 1e-7
    t1 = (a_sum * dy**2 + b_sum * dx**2 - 2 * c_sum * dx * dy) / denom
    t2 = (a_sum * dx**2 + b_sum * dy**2 + 2 * c_sum * dx * dy) / denom
    bdd = (np.sqrt(np.clip(t1, 0, None)) + np.sqrt(np.clip(t2, 0, None))) / 2
    return (1 - bdd / 2).astype(np.float32)


def fast_nms(boxes, scores, iou_thres=0.45, max_det=100):
    """快速 NMS：单类别直接用置信度贪心"""
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    # 按置信度降序排序
    order = np.argsort(-scores)

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

# 预计算网格坐标 (固定, 不依赖输入, 省 ~2ms/帧)
def _build_grid():
    grids, strides = [], []
    for n, s in [(80, 8), (40, 16), (20, 32)]:
        gy, gx = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        grids.append(np.stack([gx.flatten(), gy.flatten()], axis=1).astype(np.float32))
        strides.append(np.full(n * n, s, dtype=np.float32))
    return np.concatenate(grids), np.concatenate(strides)
_GRID, _STRIDES = _build_grid()


def postprocess_rkopt(outputs, img_shape, conf_thres=0.25):
    """RKOPT 后处理: 4 分支输出, CPU sigmoid + DFL 解码 (先过滤再 DFL)
    outputs: [out0(1,65,80,80), out1(1,65,40,40), out2(1,65,20,20), angle(1,1,8400)]
    65 = 64 DFL + 1 conf; 先 sigmoid 过滤再 DFL, 后处理 <2ms
    """
    w_img, h_img = img_shape[1], img_shape[0]

    # 1. 拼接 3 个检测头 -> [8400, 65] (grid/strides 预计算)
    o0 = outputs[0][0].transpose(1, 2, 0).reshape(6400, 65)
    o1 = outputs[1][0].transpose(1, 2, 0).reshape(1600, 65)
    o2 = outputs[2][0].transpose(1, 2, 0).reshape(400, 65)
    box_dfl = np.concatenate([o0[:, :64], o1[:, :64], o2[:, :64]])  # [8400, 64]
    conf_logit = np.concatenate([o0[:, 64], o1[:, 64], o2[:, 64]])  # [8400]
    grid = _GRID
    strides = _STRIDES
    angle_feat = outputs[3].flatten()           # [8400]

    # 2. sigmoid conf + 过滤 (8400 全做, ~0.24ms)
    conf = 1.0 / (1.0 + np.exp(-conf_logit))
    mask = conf > conf_thres
    if not np.any(mask):
        return []

    # 3. 只对幸存 anchor 做 DFL 解码 (M 个, <1ms)
    b = box_dfl[mask].reshape(-1, 4, 16)  # [M, 4, 16]
    b = b - b.max(axis=2, keepdims=True)
    e = np.exp(b)
    prob = e / e.sum(axis=2, keepdims=True)
    proj = np.arange(16, dtype=np.float32)
    lt_rb = (prob * proj).sum(axis=2)  # [M, 4] = [l, t, r, b]

    # 4. 解码 (向量化)
    l, t, r, b = lt_rb[:, 0], lt_rb[:, 1], lt_rb[:, 2], lt_rb[:, 3]
    cx_off = (r - l) / 2
    cy_off = (b - t) / 2
    angle = (angle_feat[mask] - 0.25) * np.pi
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    grid_m = grid[mask]
    strides_m = strides[mask]
    conf_m = conf[mask]

    cx = (cx_off * cos_a - cy_off * sin_a + grid_m[:, 0] + 0.5) * strides_m
    cy = (cx_off * sin_a + cy_off * cos_a + grid_m[:, 1] + 0.5) * strides_m
    bw = (l + r) * strides_m
    bh = (t + b) * strides_m

    # 5. NMS (旋转框)
    boxes = np.stack([cx, cy, bw, bh, angle], axis=1)
    keep = fast_nms(boxes, conf_m, IOU_THRES)

    # 6. 计算角点
    detections = []
    for idx in keep:
        c, s = cos_a[idx], sin_a[idx]
        w2, h2 = bw[idx] / 2, bh[idx] / 2
        corners = np.array([[-w2, -h2], [w2, -h2], [w2, h2], [-w2, h2]])
        rot = np.array([[c, -s], [s, c]])
        corners = corners @ rot.T + np.array([cx[idx], cy[idx]])
        corners[:, 0] = np.clip(corners[:, 0], 0, w_img)
        corners[:, 1] = np.clip(corners[:, 1], 0, h_img)
        detections.append({
            'xyxyxyxy': corners.reshape(-1).tolist(),
            'conf': float(conf_m[idx]),
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


def crop_obb_region(img, pts):
    """用 OBB 旋转角度校正，裁剪出水平的标签区域 (供 OCR)"""
    rect = cv2.minAreaRect(pts.astype(np.float32))
    center, (w, h), angle = rect
    if w < h:
        w, h = h, w
        angle += 90
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img_h, img_w = img.shape[:2]
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(img_h * sin_a + img_w * cos_a)
    new_h = int(img_h * cos_a + img_w * sin_a)
    M[0, 2] += (new_w - img_w) / 2
    M[1, 2] += (new_h - img_h) / 2
    rotated = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_CUBIC)
    new_cx = M[0, 0] * center[0] + M[0, 1] * center[1] + M[0, 2]
    new_cy = M[1, 0] * center[0] + M[1, 1] * center[1] + M[1, 2]
    x1 = max(0, int(new_cx - w / 2))
    y1 = max(0, int(new_cy - h / 2))
    x2 = min(new_w, int(new_cx + w / 2))
    y2 = min(new_h, int(new_cy + h / 2))
    return rotated[y1:y2, x1:x2]


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

        t0 = time.time()
        inp, scale, px, py = preprocess(frame)
        pre_ms = (time.time() - t0) * 1000

        t0 = time.time()
        outputs = rknn.inference(inputs=[inp])
        infer_ms = (time.time() - t0) * 1000

        conf_thres = state.conf_thres
        t0 = time.time()
        detections = postprocess_rkopt(outputs, (IMGSZ, IMGSZ), conf_thres)

        # 坐标还原到原图
        h, w = frame.shape[:2]
        for det in detections:
            pts = np.array(det['xyxyxyxy']).reshape(4, 2)
            pts[:, 0] = (pts[:, 0] - px) / scale
            pts[:, 1] = (pts[:, 1] - py) / scale
            pts[:, 0] = np.clip(pts[:, 0], 0, w)
            pts[:, 1] = np.clip(pts[:, 1], 0, h)
            det['xyxyxyxy'] = pts.reshape(-1).tolist()
        post_ms = (time.time() - t0) * 1000

        # 标签裁剪采集 (去重: 同位置 <50px 只存一张)
        if state.is_cropping:
            with crop_lock:
                buf_snapshot = list(state.crop_buffer)
            new_crops = []
            for det in detections:
                pts = np.array(det['xyxyxyxy']).reshape(4, 2)
                cx_det = float(pts[:, 0].mean())
                cy_det = float(pts[:, 1].mean())
                if any(abs(c['cx'] - cx_det) < 50 and abs(c['cy'] - cy_det) < 50
                       for c in buf_snapshot):
                    continue
                cropped = crop_obb_region(frame, pts)
                if cropped.size == 0:
                    continue
                _, buf = cv2.imencode('.jpg', cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
                new_crops.append({
                    'jpg_b64': base64.b64encode(buf.tobytes()).decode(),
                    'conf': det['conf'],
                    'cx': cx_det,
                    'cy': cy_det,
                })
            with crop_lock:
                state.crop_buffer.extend(new_crops)

        t0 = time.time()
        annotated = draw_obb(frame, detections)
        draw_ms = (time.time() - t0) * 1000

        # 信息面板
        frame_count += 1
        now = time.time()
        if now - fps_timer >= 1.0:
            state.fps = frame_count / (now - fps_timer)
            frame_count = 0
            fps_timer = now
        state.infer_ms = infer_ms
        state.pre_ms = pre_ms
        state.post_ms = post_ms
        state.draw_ms = draw_ms

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


def open_camera():
    """打开摄像头：优先 CAM_ID，失败则探测 0/1/2"""
    for idx in [CAM_ID] + [i for i in range(3) if i != CAM_ID]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # 自动曝光 (15fps 亮度够)
            ret, _ = cap.read()
            if ret:
                print(f"摄像头 {idx} 已打开: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
                return cap
        cap.release()
    return None


def camera_loop():
    while state.running:
        cap = open_camera()
        if cap is None:
            print("无法打开摄像头，5秒后重试...")
            time.sleep(5)
            continue

        while state.running:
            ret, frame = cap.read()
            if not ret:
                break  # 摄像头掉线，外层循环重新打开
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
            'pre_ms': round(state.pre_ms, 1),
            'infer_ms': round(state.infer_ms, 1),
            'post_ms': round(state.post_ms, 1),
            'draw_ms': round(state.draw_ms, 1),
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


@app.route('/api/crop/start', methods=['POST'])
def api_crop_start():
    with crop_lock:
        state.is_cropping = True
        state.crop_buffer = []
    return jsonify({'ok': True, 'msg': '裁剪采集已开始'})


@app.route('/api/crop/stop', methods=['POST'])
def api_crop_stop():
    with crop_lock:
        state.is_cropping = False
        count = len(state.crop_buffer)
    return jsonify({'ok': True, 'count': count})


@app.route('/api/crop/images')
def api_crop_images():
    with crop_lock:
        return jsonify({'images': list(state.crop_buffer), 'count': len(state.crop_buffer)})


@app.route('/api/capture', methods=['POST'])
def api_capture():
    """拍照: 用当前帧 + 最近检测结果裁剪高清标签图 (不重新推理, 避免RKNN并发)"""
    with state.lock:
        frame = state.frame.copy() if state.frame is not None else None
        dets = [dict(d) for d in state.detections]
    if frame is None:
        return jsonify({'ok': False, 'msg': '无画面'})
    crops = []
    for det in dets:
        pts = np.array(det['xyxyxyxy']).reshape(4, 2).astype(np.float32)
        cropped = crop_obb_region(frame, pts)
        if cropped.size == 0:
            continue
        _, buf = cv2.imencode('.jpg', cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
        crops.append({
            'jpg_b64': base64.b64encode(buf.tobytes()).decode(),
            'conf': det['conf'],
        })
    return jsonify({'ok': True, 'count': len(crops), 'crops': crops, 'frame_shape': list(frame.shape)})


# ==================== 启动 ====================

if __name__ == '__main__':
    print("加载 RKNN 模型...")
    rknn = RKNNLite(verbose=False)
    ret = rknn.load_rknn(MODEL_PATH)
    if ret != 0:
        print(f"加载失败: {ret}")
        exit(1)
    ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1)
    if ret != 0:
        print(f"初始化失败: {ret}")
        exit(1)
    print("RKNN 模型就绪")

    threading.Thread(target=inference_loop, args=(rknn,), daemon=True).start()
    threading.Thread(target=camera_loop, daemon=True).start()

    print("Web 服务启动: http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5001, threaded=True)
