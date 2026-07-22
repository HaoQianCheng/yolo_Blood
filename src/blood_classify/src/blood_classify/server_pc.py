"""
血袋类型分类 Web 服务 (PC端)
接入摄像头实时分类，显示结果
"""
import os
import time
import threading

import cv2
import numpy as np
from flask import Flask, render_template_string, Response, request, jsonify
from ultralytics import YOLO
from pathlib import Path

# ==================== 配置 ====================
# 使用项目根目录的相对路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = str(PROJECT_ROOT / "runs" / "classify" / "blood_type-7" / "weights" / "best.pt")
CAM_ID = 0
IMGSZ = 224
DEFAULT_CONF = 0.25

# 类别名称
CLASS_NAMES = {0: "plasma", 1: "red_blood_cell"}
CLASS_NAMES_CN = {0: "血浆", 1: "红细胞"}
CLASS_COLORS = {0: (255, 165, 0), 1: (0, 0, 255)}  # 橙色，蓝色

# ==================== 全局状态 ====================
app = Flask(__name__)

class ClassifyState:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.annotated = None
        self.result = None
        self.fps = 0.0
        self.infer_ms = 0.0
        self.conf_thres = DEFAULT_CONF
        self.running = True

state = ClassifyState()

# ==================== 模型加载 ====================
print(f"加载模型: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print("模型加载完成")

# ==================== 摄像头线程 ====================
def camera_thread():
    """摄像头捕获线程"""
    cap = cv2.VideoCapture(CAM_ID)
    if not cap.isOpened():
        print(f"错误: 无法打开摄像头 {CAM_ID}")
        return

    print(f"摄像头 {CAM_ID} 已打开")

    while state.running:
        ret, frame = cap.read()
        if not ret:
            print("警告: 无法读取摄像头帧")
            time.sleep(0.1)
            continue

        with state.lock:
            state.frame = frame.copy()

        time.sleep(0.01)  # ~100 FPS 采集

    cap.release()
    print("摄像头已释放")

def inference_thread():
    """推理线程"""
    frame_count = 0
    start_time = time.time()

    while state.running:
        # 获取当前帧
        with state.lock:
            if state.frame is None:
                time.sleep(0.01)
                continue
            frame = state.frame.copy()

        # 推理
        t0 = time.time()
        results = model.predict(
            source=frame,
            imgsz=IMGSZ,
            conf=state.conf_thres,
            verbose=False,
        )
        infer_ms = (time.time() - t0) * 1000

        # 处理结果
        if results and len(results) > 0:
            r = results[0]
            if r.probs is not None:
                top1_id = int(r.probs.top1)
                top1_conf = float(r.probs.top1conf)

                # 绘制结果
                annotated = frame.copy()
                h, w = annotated.shape[:2]

                # 绘制类别和置信度
                cls_cn = CLASS_NAMES_CN.get(top1_id, f"类别{top1_id}")
                cls_en = CLASS_NAMES.get(top1_id, f"class_{top1_id}")
                color = CLASS_COLORS.get(top1_id, (255, 255, 255))

                # 背景框
                text = f"{cls_cn} ({cls_en}): {top1_conf:.2%}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                thickness = 3
                (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

                # 绘制背景矩形
                cv2.rectangle(annotated, (10, 10), (10 + text_w + 20, 10 + text_h + 20), (0, 0, 0), -1)
                cv2.rectangle(annotated, (10, 10), (10 + text_w + 20, 10 + text_h + 20), color, 3)

                # 绘制文字
                cv2.putText(annotated, text, (20, 10 + text_h + 5), font, font_scale, color, thickness)

                # 绘制置信度条
                bar_y = h - 50
                bar_w = int(w * 0.6)
                bar_h = 30
                bar_x = (w - bar_w) // 2

                # 背景
                cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
                # 进度条
                fill_w = int(bar_w * top1_conf)
                cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
                # 边框
                cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)
                # 置信度文字
                conf_text = f"{top1_conf:.2%}"
                cv2.putText(annotated, conf_text, (bar_x + bar_w // 2 - 30, bar_y + bar_h - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # FPS 和推理时间
                fps_text = f"FPS: {state.fps:.1f} | Infer: {infer_ms:.1f}ms"
                cv2.putText(annotated, fps_text, (w - 350, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                with state.lock:
                    state.annotated = annotated
                    state.result = {
                        'class_id': top1_id,
                        'class_name': cls_en,
                        'class_cn': cls_cn,
                        'confidence': top1_conf,
                        'infer_ms': infer_ms
                    }
            else:
                with state.lock:
                    state.annotated = frame.copy()
                    state.result = None

        # 计算 FPS
        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            state.fps = frame_count / elapsed

        state.infer_ms = infer_ms
        time.sleep(0.01)  # 控制推理频率

# ==================== Web 路由 ====================
@app.route('/')
def index():
    """主页"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    """视频流"""
    def generate():
        while state.running:
            with state.lock:
                if state.annotated is None:
                    time.sleep(0.01)
                    continue
                frame = state.annotated.copy()

            # 编码为 JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            time.sleep(0.03)  # ~30 FPS 视频流

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    """获取状态"""
    with state.lock:
        result = state.result
        fps = state.fps
        infer_ms = state.infer_ms

    return jsonify({
        'result': result,
        'fps': fps,
        'infer_ms': infer_ms,
        'conf_thres': state.conf_thres
    })

@app.route('/api/conf', methods=['POST'])
def api_set_conf():
    """设置置信度阈值"""
    data = request.json
    if 'conf' in data:
        state.conf_thres = float(data['conf'])
        return jsonify({'status': 'ok', 'conf': state.conf_thres})
    return jsonify({'status': 'error', 'message': 'missing conf parameter'}), 400

# ==================== HTML 模板 ====================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>血袋类型分类 - 实时检测</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #fff;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
        }
        h1 {
            font-size: 2.5em;
            margin-bottom: 20px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        .container {
            display: flex;
            gap: 30px;
            max-width: 1400px;
            width: 100%;
        }
        .video-container {
            flex: 1;
            background: #0f0f23;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        }
        .video-container img {
            width: 100%;
            height: auto;
            display: block;
        }
        .info-panel {
            width: 350px;
            background: #0f0f23;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        }
        .info-title {
            font-size: 1.5em;
            margin-bottom: 20px;
            color: #00d4ff;
            border-bottom: 2px solid #00d4ff;
            padding-bottom: 10px;
        }
        .info-item {
            margin-bottom: 20px;
            padding: 15px;
            background: #1a1a3e;
            border-radius: 10px;
        }
        .info-label {
            font-size: 0.9em;
            color: #aaa;
            margin-bottom: 5px;
        }
        .info-value {
            font-size: 1.8em;
            font-weight: bold;
        }
        .class-plasma {
            color: #ffa500;
        }
        .class-red_blood_cell {
            color: #00bfff;
        }
        .confidence-bar {
            width: 100%;
            height: 30px;
            background: #333;
            border-radius: 15px;
            overflow: hidden;
            margin-top: 10px;
        }
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #00ff00, #00cc00);
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 0.9em;
        }
        .controls {
            margin-top: 30px;
        }
        .control-item {
            margin-bottom: 15px;
        }
        .control-item label {
            display: block;
            margin-bottom: 5px;
            color: #aaa;
        }
        .control-item input[type="range"] {
            width: 100%;
            height: 10px;
            -webkit-appearance: none;
            background: #333;
            border-radius: 5px;
            outline: none;
        }
        .control-item input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 20px;
            height: 20px;
            background: #00d4ff;
            border-radius: 50%;
            cursor: pointer;
        }
        .conf-value {
            text-align: center;
            font-size: 1.2em;
            color: #00d4ff;
            margin-top: 5px;
        }
        .stats {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 20px;
        }
        .stat-item {
            background: #1a1a3e;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        .stat-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #00ff00;
        }
        .stat-label {
            font-size: 0.8em;
            color: #aaa;
            margin-top: 5px;
        }
        .footer {
            margin-top: 30px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <h1>🔬 血袋类型分类系统</h1>

    <div class="container">
        <div class="video-container">
            <img src="/video_feed" alt="Video Feed">
        </div>

        <div class="info-panel">
            <div class="info-title">📊 分类结果</div>

            <div class="info-item">
                <div class="info-label">预测类别</div>
                <div class="info-value" id="class-name">等待中...</div>
            </div>

            <div class="info-item">
                <div class="info-label">置信度</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" id="conf-bar" style="width: 0%">0%</div>
                </div>
            </div>

            <div class="stats">
                <div class="stat-item">
                    <div class="stat-value" id="fps">0</div>
                    <div class="stat-label">FPS</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="infer-time">0</div>
                    <div class="stat-label">推理时间 (ms)</div>
                </div>
            </div>

            <div class="controls">
                <div class="info-title">⚙️ 设置</div>
                <div class="control-item">
                    <label>置信度阈值</label>
                    <input type="range" id="conf-slider" min="0" max="100" value="25">
                    <div class="conf-value" id="conf-value">25%</div>
                </div>
            </div>
        </div>
    </div>

    <div class="footer">
        血袋类型分类系统 - YOLO11n-cls | 实时摄像头检测
    </div>

    <script>
        // 更新状态
        function updateStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    if (data.result) {
                        const classEl = document.getElementById('class-name');
                        classEl.textContent = `${data.result.class_cn} (${data.result.class_name})`;
                        classEl.className = `info-value class-${data.result.class_name}`;

                        const confBar = document.getElementById('conf-bar');
                        const confPercent = (data.result.confidence * 100).toFixed(1);
                        confBar.style.width = `${confPercent}%`;
                        confBar.textContent = `${confPercent}%`;
                    }

                    document.getElementById('fps').textContent = data.fps.toFixed(1);
                    document.getElementById('infer-time').textContent = data.infer_ms.toFixed(1);
                })
                .catch(error => console.error('Error:', error));
        }

        // 置信度滑块
        const confSlider = document.getElementById('conf-slider');
        const confValue = document.getElementById('conf-value');

        confSlider.addEventListener('input', function() {
            const value = this.value;
            confValue.textContent = `${value}%`;

            fetch('/api/conf', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({conf: value / 100})
            });
        });

        // 定时更新
        setInterval(updateStatus, 100);
        updateStatus();
    </script>
</body>
</html>
"""

# ==================== 主函数 ====================
if __name__ == '__main__':
    # 启动摄像头线程
    cam_thread = threading.Thread(target=camera_thread, daemon=True)
    cam_thread.start()

    # 启动推理线程
    infer_thread = threading.Thread(target=inference_thread, daemon=True)
    infer_thread.start()

    # 等待摄像头初始化
    time.sleep(1)

    print("\n" + "="*50)
    print("血袋类型分类 Web 服务已启动")
    print("="*50)
    print(f"访问地址: http://localhost:5000")
    print(f"模型: {MODEL_PATH}")
    print(f"摄像头: {CAM_ID}")
    print("="*50 + "\n")

    # 启动 Flask
    app.run(host='0.0.0.0', port=5000, threaded=True)
