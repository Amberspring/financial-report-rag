# 独立报告级留出：FinQA 100 题（2026-10-12）

历史 100 题以及 2026-10-09 冻结的 200 题都已被本项目审查过，不能再作为修复后的全新留出。新集合在开发数值推理路线前冻结，只读取官方 test 的 ID、问题和报告文件名；排除历史 dev/test ID 与公司年份报告，再排除上一批 200 题的全部 ID 和公司年份报告。剩余 168 题，无法在报告级隔离下再取 200 题，因此按 `SHA256(example_id)` 固定顺序取 100 题，不按答案、程序或模型成绩挑题。

结果是 100 题、55 个 FinQA 页面转录、51 个公司年份报告、45 家公司、20 个年份；与上一批留出的报告、ID 和标准化问题各有 0 个重叠。无文本的冻结 manifest 在 `results/protocol-20261012-independent-100/holdout-manifest.json`，SHA-256 为 `c13f3497ad5905092fe7fee5891d2c1045195fa0c7d47f2debe555453853df16`；物化文件的各自 SHA-256 在同目录 `materialized-manifest.json`。原始 test、题目、页面转录和评分标签留在 gitignore 的 `artifacts/`，预测器只读取文档和 A/B 查询，评分时才读取标签。

本集合适合验证**给定报告页面的数值推理**。其中 B 代理查询虽然加了公司和年份，语料仍只有选中的页面转录，不能声称测过完整年报库检索或原 PDF 表格解析。公开 test 可能出现在模型预训练中，这里的“独立”只表示本项目的修复与旧实验在 ID、公司年份报告及标准化问题上隔离。

```sh
python scripts/freeze_holdout.py --source artifacts/finqa-source/test.json --exclude-manifest results/protocol-20261009/holdout-manifest.json --count 100 --output-dir artifacts/holdout-20261012-independent-100/frozen
python scripts/materialize_holdout.py --source artifacts/finqa-source/test.json --manifest artifacts/holdout-20261012-independent-100/frozen/holdout-manifest.json --output-dir artifacts/holdout-20261012-independent-100/materialized
python -m pytest -q tests/test_freeze_holdout.py
```

上述脚本拒绝覆盖已有目录。复现时将两个 `--output-dir` 换成新的 run ID，并校对生成的哈希；历史目录和本轮冻结目录不可复写。当前尚无模型在这 100 题上的预测或准确率，后续必须先保存完整逐题输出，再解封评分标签。
