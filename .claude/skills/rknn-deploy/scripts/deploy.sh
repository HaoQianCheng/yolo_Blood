#!/bin/bash
# RK3576 板端一键部署脚本
# 在板端执行: bash deploy.sh
#
# 前提: best_rk3576.rknn, server.py, infer.py, templates/ 已传到 /root/blood_label/

set -e
cd /root/blood_label

echo "===== 1. 换清华 apt 源 ====="
if ! grep -q "tuna.tsinghua" /etc/apt/sources.list 2>/dev/null; then
    cp /etc/apt/sources.list /etc/apt/sources.list.bak
    sed -i 's|http://ports.ubuntu.com/ubuntu-ports|https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports|g' /etc/apt/sources.list
    echo "源已换清华"
else
    echo "已是清华源"
fi
apt-get update -qq

echo "===== 2. 装系统依赖 ====="
DEBIAN_FRONTEND=noninteractive apt-get install -y python3-pip python3-venv libgl1-mesa-glx libglib2.0-0 2>&1 | tail -3

echo "===== 3. 配置 pip 清华源 ====="
mkdir -p ~/.config/pip
cat > ~/.config/pip/pip.conf << 'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple/
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF
python3 -m pip install --upgrade pip 2>&1 | tail -1

echo "===== 4. 装 rknn-toolkit-lite2 ====="
if python3 -c "from rknnlite.api import RKNNLite" 2>/dev/null; then
    echo "已安装，跳过"
else
    # 下载 wheel (如果本地没有)
    WHEEL="rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"
    if [ ! -f "$WHEEL" ]; then
        echo "下载 lite2 wheel..."
        curl -sL -o "$WHEEL" "https://ghproxy.net/https://raw.githubusercontent.com/airockchip/rknn-toolkit2/master/rknn-toolkit-lite2/packages/$WHEEL"
    fi
    # 下载 librknnrt.so (runtime)
    if [ ! -f /usr/lib/librknnrt.so ]; then
        echo "下载 librknnrt.so..."
        curl -sL -o /usr/lib/librknnrt.so "https://ghproxy.net/https://raw.githubusercontent.com/airockchip/rknn-toolkit2/master/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so"
    fi
    python3 -m pip install "$WHEEL" 2>&1 | tail -3
fi

echo "===== 5. 装 numpy + opencv + flask ====="
python3 -m pip install numpy opencv-python-headless flask 2>&1 | tail -3

echo "===== 6. 验证环境 ====="
python3 << 'PYEOF'
import sys
ok = True
try:
    from rknnlite.api import RKNNLite
    print("  rknn-toolkit-lite2: ✅")
except: print("  rknn-toolkit-lite2: ❌"); ok = False
try:
    import cv2; print(f"  opencv: ✅ {cv2.__version__}")
except: print("  opencv: ❌"); ok = False
try:
    import flask; print(f"  flask: ✅ {flask.__version__}")
except: print("  flask: ❌"); ok = False
try:
    import numpy; print(f"  numpy: ✅ {numpy.__version__}")
except: print("  numpy: ❌"); ok = False
if ok:
    print("\n✅ 环境就绪")
else:
    print("\n❌ 有依赖缺失")
    sys.exit(1)
PYEOF

echo "===== 7. 检查文件 ====="
for f in best_rk3576.rknn server.py infer.py templates/index.html; do
    if [ -f "$f" ]; then
        echo "  ✅ $f"
    else
        echo "  ❌ $f 缺失"
    fi
done

echo ""
echo "===== 部署完成 ====="
echo "启动 Web 服务: nohup python3 server.py > server.log 2>&1 &"
echo "访问: http://$(hostname -I | awk '{print $1}'):5000"
