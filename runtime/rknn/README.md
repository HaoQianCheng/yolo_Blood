# runtime/rknn

RKNN 板端部署所需的二进制文件。

| 文件 | 说明 |
|------|------|
| `best_rk3576.rknn` | YOLO11-OBB 模型 (RK3576, FP16, 13.3MB) |
| `librknnrt.so` | RKNN Runtime v2.3.2 (aarch64, 板端推理用) |
| `librknnrt_v2.2.0.so` | RKNN Runtime v2.2.0 (备用，与 driver 0.9.8 兼容性待验证) |
| `rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl` | rknn-toolkit-lite2 Python 包 (板端安装用) |

## 部署到板端

```bash
# 传模型
scp runtime/rknn/best_rk3576.rknn root@192.168.131.200:/root/blood_label/

# 传 lite2 wheel + librknnrt.so (首次部署)
scp runtime/rknn/rknn_toolkit_lite2-*.whl root@192.168.131.200:/root/blood_label/
scp runtime/rknn/librknnrt.so root@192.168.131.200:/usr/lib/
```
