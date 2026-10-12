# 冻结 200 题 CPU 规则基线的分阶段归因

输入是已经落盘的 `results/protocol-20261009/cpu-rule-proxy.predictions.jsonl`，不是重新预测。诊断脚本在读取全部原始预测后才读标签，输出 `results/protocol-20261009/cpu-rule-proxy.pipeline-20261012.json`，其中固定记录三个输入 SHA-256；历史 100 题的 2 对、27 错、71 拒和本次 200 题的 0 对均未改写。

| 模式 | 目标页 Top1 | 金标证据全部在 Top5 | 已回答 | 答对 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 200/200 | 154/200 | 17/200 | 0/200 |
| Dense | 196/200 | 51/200 | 1/200 | 0/200 |
| Hybrid | 197/200 | 63/200 | 8/200 | 0/200 |

这组 B 代理问题已写明公司与报告年份，而且检索库只含选中 FinQA 页面转录；目标页 Top1 很高是这种范围约束下的结果，不能写成完整年报库的检索能力。BM25 虽有 154 题把全部金标证据放进 Top5，但答案仍 0/200，说明瓶颈主要在从证据选择正确操作数、期间、单位与运算；已回答的 17 题全部错，增加拒答或调高 Recall 均不足以达成业务目标。`gold_evidence_selected_proxy` 仅比对表格行或文本段落的金标键，不核对单元格、数字和单位；即使 BM25 有 3/200，也不能称为证据正确率或数值计算准确率。

下一步必须先在**给定报告**协议 A 中用真实模型和冻结 200 题验证表格与文本操作数、单步/多步计划、单位及端到端准确率，再建立完整 PDF 年报库和独立标注做协议 B。A 的任务输入允许提供报告页 ID，B 的预测则必须自行检索正确公司、年份和完整报告，两者不能混报。模型运行尚未完成，因此本诊断仅用于定位修复顺序。

```sh
python -m pip install -e .
python scripts/diagnose_pipeline.py --predictions results/protocol-20261009/cpu-rule-proxy.predictions.jsonl --queries artifacts/holdout-20261009/queries-B-proxy.json --labels artifacts/holdout-20261009/labels-scoring-only.json --output results/protocol-20261009/cpu-rule-proxy.pipeline-<new-run-id>.json
python -m pytest -q tests/test_pipeline_diagnosis.py
```
