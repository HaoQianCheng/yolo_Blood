# 贡献指南

感谢关注 yolo_Blood！欢迎通过以下方式参与贡献。

## 🐛 提交 Issue

提交 Issue 前请先搜索已有 Issue，避免重复。请注意附带：

- **Bug 报告**：Python / ultralytics / rknn-toolkit2 版本、运行环境（PC 或板端）、完整报错堆栈、复现步骤
- **功能建议**：使用场景、期望行为

## 🔀 提交 Pull Request

1. Fork 本仓库并创建特性分支：`git checkout -b feature/your-feature`
2. 保持改动聚焦，一个 PR 只做一件事
3. 代码风格与现有代码保持一致（脚本头部包含 docstring 说明用途）
4. 如涉及转换 / 部署流程改动，请同步更新 `docs/` 下对应文档
5. 提交 PR 时描述清楚改动内容与测试情况

## 📋 约定

- 提交信息使用中文或英文均可，格式建议：`类型: 简述`（如 `docs: 补充部署文档`、`fix: 修复量化校准路径`）
- 请勿提交数据集、模型权重等大文件（`.pt` / `.onnx` / `.rknn` 已在 `.gitignore` 中忽略）
- 板端相关脚本请注明目标环境（RK3576 / rknn-toolkit-lite2 2.3.2）

## 📄 许可证

提交 PR 即表示你同意将贡献内容以 [MIT License](LICENSE) 授权。
