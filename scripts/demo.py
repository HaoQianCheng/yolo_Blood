"""实时摄像头 YOLO11-OBB 血袋标签检测演示
按 Q 退出，右侧信息面板显示检测统计。
"""
import argparse
import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO


# ---- 视觉常量 ----
PANEL_WIDTH = 260
PANEL_ALPHA = 0.85
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
FONT_SCALE_LARGE = 0.9
FONT_SCALE_TITLE = 0.7
FONT_SCALE_SMALL = 0.45
COLOR_GREEN = (0, 230, 60)
COLOR_WHITE = (255, 255, 255)
COLOR_PANEL_BG = (20, 25, 35)
COLOR_DIVIDER = (60, 70, 90)
COLOR_DIM = (160, 170, 190)
THICKNESS = 2
THICKNESS_BOLD = 3
LINE_HEIGHT = 22


def _draw_obb(img: np.ndarray, pts: np.ndarray, conf: float, color: tuple) -> None:
    """在图上画旋转框 + 置信度标签。"""
    pts_i = pts.astype(np.int32)
    cv2.polylines(img, [pts_i], isClosed=True, color=color, thickness=THICKNESS_BOLD)
    cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
    label = f"{conf:.2f}"
    cv2.putText(img, label, (cx - 18, cy - 10),
                FONT, FONT_SCALE, color, THICKNESS)


def _draw_panel(canvas: np.ndarray, det_count: int, conf: float,
                fps: float, infer_ms: float) -> np.ndarray:
    """在画布右侧绘制半透明信息面板。"""
    h, w = canvas.shape[:2]
    panel = np.zeros((h, PANEL_WIDTH, 3), dtype=np.uint8)
    panel[:] = COLOR_PANEL_BG

    y = 30

    # 标题
    cv2.putText(panel, "YOLO11-OBB", (20, y), FONT, FONT_SCALE_TITLE, COLOR_GREEN, THICKNESS)
    y += 28
    cv2.putText(panel, "血袋标签检测", (20, y), FONT, FONT_SCALE_SMALL, COLOR_DIM, 1)
    y += 35

    # 分隔线
    cv2.line(panel, (16, y), (PANEL_WIDTH - 16, y), COLOR_DIVIDER, 1)
    y += 20

    # 检测状态
    if det_count > 0:
        status_text = "DETECTED"
        status_color = COLOR_GREEN
    else:
        status_text = "NO DETECTION"
        status_color = (80, 120, 220)
    cv2.putText(panel, status_text, (20, y), FONT, FONT_SCALE_LARGE, status_color, THICKNESS_BOLD)
    y += 35

    # 统计卡片
    stats = [
        ("检测数", f"{det_count}"),
        ("平均置信度", f"{conf:.3f}" if det_count > 0 else "--"),
        ("FPS", f"{fps:.1f}"),
        ("推理耗时", f"{infer_ms:.0f} ms"),
    ]

    for label, value in stats:
        cv2.putText(panel, label, (20, y), FONT, FONT_SCALE_SMALL, COLOR_DIM, 1)
        y += 20
        cv2.putText(panel, value, (20, y), FONT, FONT_SCALE_LARGE, COLOR_WHITE, THICKNESS)
        y += 28

    y += 5
    cv2.line(panel, (16, y), (PANEL_WIDTH - 16, y), COLOR_DIVIDER, 1)
    y += 18

    cv2.putText(panel, "按 Q 退出", (20, y), FONT, FONT_SCALE_SMALL, COLOR_DIM, 1)

    return np.hstack([canvas, panel])


def _create_window() -> None:
    """创建可调整大小的窗口。"""
    cv2.namedWindow("血袋标签检测 - YOLO11-OBB", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("血袋标签检测 - YOLO11-OBB", 1100, 600)


def run_demo(weights: str, camera_id: int, conf_threshold: float) -> None:
    """主循环: 打开摄像头, 逐帧推理, 叠加显示。"""
    model = YOLO(weights)
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"错误: 无法打开摄像头 (ID={camera_id})")
        return

    _create_window()

    fps_queue = deque(maxlen=30)
    infer_queue = deque(maxlen=30)

    print(f"摄像头已打开 (ID={camera_id}), 按 Q 退出...")

    while True:
        t0 = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            print("警告: 读取帧失败")
            continue

        # 推理
        t_infer_start = time.perf_counter()
        results = model.predict(frame, conf=conf_threshold, verbose=False)
        infer_ms = (time.perf_counter() - t_infer_start) * 1000
        infer_queue.append(infer_ms)

        result = results[0]

        # 画旋转框
        det_count = 0
        conf_sum = 0.0
        if result.obb is not None:
            for obb in result.obb:
                pts = obb.xyxyxyxy[0].cpu().numpy()
                conf_val = float(obb.conf[0])
                _draw_obb(frame, pts, conf_val, COLOR_GREEN)
                det_count += 1
                conf_sum += conf_val

        avg_conf = conf_sum / det_count if det_count > 0 else 0.0
        avg_infer = float(np.mean(infer_queue)) if infer_queue else 0.0

        # FPS
        fps_queue.append(time.perf_counter() - t0)
        fps = 1.0 / (sum(fps_queue) / len(fps_queue)) if fps_queue else 0.0

        # 拼接面板
        display = _draw_panel(frame, det_count, avg_conf, fps, avg_infer)
        cv2.imshow("血袋标签检测 - YOLO11-OBB", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("演示结束")


def main():
    parser = argparse.ArgumentParser(description="YOLO11-OBB 血袋标签实时检测演示")
    parser.add_argument("--weights", default="runs/obb/runs/obb/blood_label/weights/best.pt",
                        help="模型权重路径")
    parser.add_argument("--camera", type=int, default=0, help="摄像头 ID (默认 0)")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    args = parser.parse_args()
    run_demo(args.weights, args.camera, args.conf)


if __name__ == "__main__":
    main()
