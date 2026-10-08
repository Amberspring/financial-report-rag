# 上传 GitHub

仓库名建议 `financial-research-copilot`。目录已经是独立 Git 仓库；ZIP 不包含 Git 历史或模型权重。

新建空的 GitHub 仓库后，在本目录运行：

```bash
git remote add origin https://github.com/YOUR_ACCOUNT/financial-research-copilot.git
git push -u origin main
```

不提交 .env、API key、真实客户资料、artifacts、模型、原 PDF。实验结果与不含隐私的案例已纳入版本控制。上传前可运行 `python -m pytest -q`。原始结果、状态和限制应与 README 一起保留。
