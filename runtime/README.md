# runtime

板端部署所需的运行时文件和二进制包。

```
runtime/
└── rknn/               # RKNN 相关
    ├── best_rk3576.rknn    # 转换好的模型
    ├── librknnrt.so        # Runtime 库 (aarch64)
    └── *.whl               # lite2 Python 包
```
