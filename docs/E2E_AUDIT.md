# 从检索指标到端到端答案评测

FinQA 官方原始来源：[czyssrs/FinQA](https://github.com/czyssrs/FinQA)。`scripts/audit_sources.py` 下载官方 dev/test 文件，验证已有 manifest SHA-256，并逐条比对问题、答案、程序、证据标签与正文表格。2026-10-08 本机审计确认 131 页转录、dev40/test100 与源文件一致；这不能代替原 PDF 页序与标签正确性的人工核验。

```sh
python scripts/audit_sources.py
python scripts/evaluate_e2e.py --output results/e2e-<unique-run-id>.json
python scripts/evaluate_e2e.py --embedding sbert:BAAI/bge-small-en-v1.5 --reranker ce:cross-encoder/ms-marco-MiniLM-L-6-v2 --output results/e2e-neural-<unique-run-id>.json
```

新评测不向预测函数传递正确 doc_id，也不读取 gold program 来构造计划。预测先在全库检索，再用明确公司符号与年份定位候选表格；计算操作数必须在已召回表格中，多个可计算文档则拒答。标注答案、证据与 doc_id 仅在预测后用于打分；查询缺少公司上下文时不会自动补入正确公司。

本次 CPU TF-IDF 全链路结果如实保留：100 问中数值回答覆盖率 0%，raw execution accuracy 0%，Dense/BM25/Hybrid 的 evidence Recall@5 分别为 21.67%/63.83%/38.17%。问题通常依赖给定财报上下文，且窄规则规划器不覆盖多数 FinQA 运算，所以这个保守版本不能作为通用金融数值问答模型。旧的“给定正确文档”评测与全库端到端评测必须分开报告，不能利用标签回填范围来提高成绩。

API 的 `/query` 支持 `calculate=true` 且不提供 doc_id 走新路径；显式 doc_id 仍是用户提供的范围，属于另一种协议。证据摘录不是生成式答案准确率。GraphRAG 只有在真实多跳标签与消融结果充分时才作为有效增强，不预填收益。

可选模型规划器通过 `--url http://127.0.0.1:8000/v1 --model Qwen3-8B` 启用，API 对应环境变量 `RAG_MODEL_URL` 与 `RAG_MODEL`。模型仅看检索所得表格，并输出单步运算及单元格引用；程序在执行前检查每个引用属于召回证据，运算采用 Decimal，不执行模型代码，不允许常量或跨文档运算。每条预测和原始模型响应即时写入 `.predictions.jsonl`，评分仍在预测后进行；覆盖范围有限，配置好接口不代表已完成模型实测。
