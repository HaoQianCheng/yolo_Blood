#!/bin/bash
# WSL2 转换环境检查脚本
# 用法: bash check_wsl.sh

echo "=============================="
echo "  WSL2 转换环境检查"
echo "=============================="

ERRORS=0

# 1. WSL2
echo -n "WSL2: "
if command -v wsl &>/dev/null; then
    echo "✅ 已安装"
else
    echo "❌ 未安装"
    ERRORS=$((ERRORS+1))
fi

# 2. WSL Ubuntu 发行版
echo -n "Ubuntu 发行版: "
if wsl -l -v 2>/dev/null | grep -q "Ubuntu"; then
    wsl -l -v 2>/dev/null | grep "Ubuntu" | head -1
else
    echo "❌ 未找到 Ubuntu 发行版"
    ERRORS=$((ERRORS+1))
fi

# 3. Python + rknn-toolkit2（在 WSL 内检查）
echo ""
echo "--- WSL 内部检查 ---"

CHECK_SH='
source /opt/rknn-env/bin/activate 2>/dev/null || true

echo -n "Python: "
python3 --version 2>&1 || echo "❌ Python 未找到"

echo -n "rknn-toolkit2: "
python3 -c "from rknn.api import RKNN; print(\"✅\", RKNN.__module__)" 2>/dev/null || echo "❌ 未安装"

echo -n "onnx 版本: "
python3 -c "import onnx; v=onnx.__version__; ok=v.startswith(\"1.16\"); print(f\"{v} {'✅' if ok else '❌ 需要 1.16.x, 当前 ' + v}\")" 2>/dev/null || echo "❌ onnx 未安装"

echo -n "onnx.mapping: "
python3 -c "import onnx; print('✅' if hasattr(onnx, 'mapping') else '❌ 缺失')"

echo -n "ultralytics: "
python3 -c "import ultralytics; print('✅', ultralytics.__version__)" 2>/dev/null || echo "❌ 未安装"

echo -n "torch: "
python3 -c "import torch; print('✅', torch.__version__)" 2>/dev/null || echo "❌ 未安装"

echo -n "torch 版本兼容: "
python3 -c "import torch; v=torch.__version__; ok=v.startswith(\"2.4\") or v.startswith(\"2.3\"); print(f\"{v} {'✅' if ok else '⚠️ 建议 2.4.x'}\")" 2>/dev/null || true
'

wsl -d Ubuntu-22.04 -- bash -c "$CHECK_SH" 2>/dev/null

echo ""
echo "=============================="
if [ $ERRORS -gt 0 ]; then
    echo "⚠️ 有 $ERRORS 项未通过，请先修复"
else
    echo "✅ 环境检查通过，可以开始转换"
fi
