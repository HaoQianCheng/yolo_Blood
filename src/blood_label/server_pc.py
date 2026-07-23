#!/usr/bin/env python3
"""
YOLO11-OBB 实时检测 Web 服务 (PC / CUDA)
基于 ultralytics，支持摄像头实时检测

访问: http://localhost:5000
"""

import os
import time
import argparse
import threading
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO

# ==================== 配置 ====================
DEFAULT_WEIGHTS = "../../runs/obb/runs/obb/blood_label_n-2/weights/best.pt"
CAM_ID = 0
IMGSZ = 640
DEFAULT_CONF = 0.5
IOU_THRES = 0.45

# ==================== 全局状态 ====================
app = Flask(__name__, template_folder=str(Path(__file__).parent / "rknn" / "templates"))


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


# ==================== 绘制 OBB ====================

def draw_obb(img, detections):
    canvas = img.copy()
    colors = [(0, 255, 0), (0, 255, 255), (0, 0, 255), (255, 0, 255)]
    for i, det in enumerate(detections):
        pts = np.array(det["xyxyxyxy"]).reshape(4, 2).astype(np.int32)
        color = colors[i % len(colors)]
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)
        center = np.mean(pts, axis=0).astype(np.int32)
        cv2.circle(canvas, tuple(center), 4, color, -1)
        label = f"blood_label {det['conf']:.2f}"
        cv2.putText(canvas, label, (int(pts[0][0]), int(pts[0][1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return canvas


# ==================== 摄像头 + 推理线程 ====================

def inference_loop(model):
    fps_timer = time.time()
    frame_count = 0

    while state.running:
        with state.lock:
            frame = state.frame
        if frame is None:
            time.sleep(0.002)
            continue

        conf_thres = state.conf_thres

        t0 = time.time()
        results = model.predict(
            source=frame,
            imgsz=IMGSZ,
            conf=conf_thres,
            iou=IOU_THRES,
            verbose=False,
        )
        infer_ms = (time.time() - t0) * 1000

        detections = []
        if results and results[0].obb is not None:
            obb = results[0].obb
            for j in range(len(obb)):
                pts = obb.xyxyxyxy[j].cpu().numpy().reshape(-1).tolist()
                conf = float(obb.conf[j])
                detections.append({"xyxyxyxy": pts, "conf": conf})

        annotated = draw_obb(frame, detections)

        # 信息面板
        frame_count += 1
        now = time.time()
        if now - fps_timer >= 1.0:
            state.fps = frame_count / (now - fps_timer)
            frame_count = 0
            fps_timer = now
        state.infer_ms = infer_ms

        panel_h = 90
        panel = np.zeros((panel_h, annotated.shape[1], 3), dtype=np.uint8)
        cv2.putText(panel, f"FPS: {state.fps:.1f}", (15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(panel, f"Detection: {len(detections)}", (15, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(panel, f"Infer: {infer_ms:.0f}ms", (250, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(panel, f"Conf: {conf_thres:.2f}", (250, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(panel, "Blood Label OBB - CUDA", (450, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)
        annotated = np.vstack([annotated, panel])

        with state.lock:
            state.annotated = annotated
            state.detections = detections

        time.sleep(0.005)


def camera_loop(cam_id):
    cap = cv2.VideoCapture(cam_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"无法打开摄像头 {cam_id}")
        return

    print(f"摄像头 {cam_id} 已打开")

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

        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
        jpg_bytes = buf.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpg_bytes + b"\r\n")

        time.sleep(0.008)


# ==================== 路由 ====================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video():
    return Response(generate_mjpeg(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    with state.lock:
        return jsonify({
            "fps": round(state.fps, 1),
            "infer_ms": round(state.infer_ms, 1),
            "detections": len(state.detections),
            "conf_thres": state.conf_thres,
        })


@app.route("/api/conf", methods=["POST"])
def api_set_conf():
    data = request.get_json()
    if data and "conf" in data:
        state.conf_thres = max(0.05, min(0.99, float(data["conf"])))
        return jsonify({"ok": True, "conf": state.conf_thres})
    return jsonify({"ok": False}), 400


# ==================== 启动 ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="模型权重路径")
    parser.add_argument("--cam", type=int, default=CAM_ID, help="摄像头 ID")
    parser.add_argument("--port", type=int, default=5000, help="服务端口")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="默认置信度")
    args = parser.parse_args()

    state.conf_thres = args.conf

    # 加载模型
    print(f"加载模型: {args.weights}")
    model = YOLO(args.weights)
    # 预热
    dummy = np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8)
    model.predict(source=dummy, imgsz=IMGSZ, verbose=False)
    print("模型就绪 (CUDA)")

    # 启动线程
    threading.Thread(target=inference_loop, args=(model,), daemon=True).start()
    threading.Thread(target=camera_loop, args=(args.cam,), daemon=True).start()

    print(f"Web 服务启动: http://0.0.0.0:{args.port}")
    app.run(host="0.0.0.0", port=args.port, threaded=True)
