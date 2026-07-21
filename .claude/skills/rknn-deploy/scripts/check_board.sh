#!/bin/bash
# RK3576 板端环境检查脚本（通过 SSH MCP 执行）
# 用法: bash check_board.sh

echo "=============================="
echo "  RK3576 板端环境检查"
echo "=============================="

# 1. 系统
echo "--- 系统信息 ---"
uname -a
cat /etc/os-release | head -2

# 2. Python
echo -n "Python: "
python3 --version 2>&1 || echo "❌"

# 3. pip
echo -n "pip: "
python3 -m pip --version 2>&1 || echo "❌ pip 未安装"

# 4. NPU 驱动
echo -n "NPU 驱动: "
cat /sys/kernel/debug/rknpu/version 2>/dev/null || echo "❌ 无 rknpu"

# 5. rknn-toolkit-lite2
echo -n "rknn-toolkit-lite2: "
python3 -c "from rknnlite.api import RKNNLite; print('✅ OK')" 2>/dev/null || echo "❌ 未安装"

# 6. Flask
echo -n "Flask: "
python3 -c "import flask; print('✅', flask.__version__)" 2>/dev/null || echo "❌ 未安装 (pip install flask)"

# 7. OpenCV
echo -n "OpenCV: "
python3 -c "import cv2; print('✅', cv2.__version__)" 2>/dev/null || echo "❌ 未安装"

# 8. 摄像头
echo -n "摄像头: "
if [ -e /dev/video0 ]; then
    echo "✅ /dev/video0 存在"
else
    echo "⚠️ 无 /dev/video0"
fi

# 9. 磁盘
echo "--- 磁盘 ---"
df -h / | tail -1

# 10. 内存
echo "--- 内存 ---"
free -h | head -2

# 11. 网络
echo -n "网络 (pypi): "
python3 -c "import urllib.request; urllib.request.urlopen('https://pypi.org', timeout=5); print('✅ 通')" 2>/dev/null || echo "❌ 不通"

# 12. 模型文件
echo -n "模型文件: "
if [ -f /root/blood_label/best_rk3576.rknn ]; then
    SIZE=$(du -h /root/blood_label/best_rk3576.rknn | cut -f1)
    echo "✅ $SIZE"
else
    echo "❌ /root/blood_label/best_rk3576.rknn 不存在"
fi

echo ""
echo "=============================="
echo "检查完成"
