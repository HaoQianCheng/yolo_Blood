# RK3576 板端部署指南

本文档介绍如何将训练并量化好的 RKNN 模型部署到 RK3576 开发板，运行实时推理与 Web 检测服务。

## 环境要求

| 组件 | 版本 |
|------|------|
| 芯片 | RK3576（NPU 6 TOPS） |
| 系统 | Debian / Ubuntu (aarch64) |
| Python | 3.10 |
| rknn-toolkit-lite2 | 2.3.2（`runtime/rknn/*.whl`） |
| librknnrt | 2.3.2（`runtime/rknn/librknnrt.so`） |
| NPU 驱动 | ≥ v0.9.8 |

## 1. 安装板端运行时

```bash
# 安装 Python 推理库（whl 随仓库提供）
pip3 install runtime/rknn/rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.whl

# 安装系统依赖
pip3 install opencv-python flask numpy

# 更新 NPU 运行时库（替换系统自带版本）
cp runtime/rknn/librknnrt.so /usr/lib/
```

> librknnrt.so 版本必须与 rknn-toolkit2 转换版本一致（本项目为 2.3.2），否则会初始化失败。

## 2. 上传模型与服务

```bash
# 方式一：scp
scp runtime/rknn/best_rk3576.rknn root@<board_ip>:/root/blood_label/
scp apps/server_hd.py root@<board_ip>:/root/blood_label/server.py

# 方式二：使用仓库脚本（按需修改 IP 与路径）
python scripts/upload_model.py
```

板端目录约定：

```
/root/blood_label/
├── best_rkopt_rk3576.rknn    # RKOPT INT8 模型
├── server.py                 # Web 检测服务
└── librknnrt.so              # 运行时库（可选，已装入 /usr/lib 则无需）
```

## 3. 推理验证

```bash
cd src/blood_label/rknn
python3 infer.py --src test.jpg        # 单图推理 (rknn-toolkit-lite2)
python3 infer.py --cam                 # 摄像头实时检测
python3 infer_ctypes.py --src test.jpg # ctypes + librknnrt C API 方式
python3 infer_min.py                   # 最小加载验证
```

若 `infer_min.py` 输出正常张量形状，说明 NPU 驱动、运行时库、模型三者匹配。

## 4. Web 检测服务

`apps/server_hd.py`（主入口）提供：

- 实时视频流页面（`http://<board_ip>:5000`）
- 检测 API（JSON 返回 OBB 角点、置信度）
- 高清拍照接口（1920×1080）

手动运行：

```bash
cd /root/blood_label
python3 server.py
```

## 5. supervisor 常驻部署

使用 `configs/blood_label.conf`：

```bash
scp configs/blood_label.conf root@<board_ip>:/etc/supervisor/conf.d/
supervisorctl update
supervisorctl status blood_label    # 确认 RUNNING
```

日志：`/root/blood_label/server.log`。

## 性能参考

| 方案 | 精度 | 板端速度 |
|------|------|----------|
| float16 RKNN | FP16 | ~80ms / 13 FPS |
| QAT INT8（ultralytics 导出） | INT8 | 不加速（被降级 FP16） |
| **RKOPT INT8（本项目采用）** | INT8 | **~26-33ms / 30-39 FPS** |

量化方案选型与完整转换步骤见 [rkopt-int8-guide.md](rkopt-int8-guide.md)。

## 常见问题

1. **RKNN init 返回 -7** — 检查 librknnrt.so 版本与 NPU 驱动版本匹配，详见 [troubleshooting-rknn-init.md](troubleshooting-rknn-init.md)。
2. **检测 conf 全 0** — 使用了标准 ultralytics 导出的 ONNX 直接 int8 量化所致，须使用 RKOPT 导出方案。
3. **摄像头打不开** — 确认 `/dev/video*` 权限，MJPEG 格式相机需在脚本中保持 FOURCC 设置。
4. **服务启动报 ImportError** — 板端 Python 版本需为 3.10（whl 为 cp310），或从 [rknn-toolkit2 releases](https://github.com/airockchip/rknn-toolkit2/releases) 下载对应版本 whl。
