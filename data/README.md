# 数据、版权与推理边界

FinQA 官方来源 https://github.com/czyssrs/FinQA ，官方仓库 MIT 许可见 FinQA-LICENSE.txt。注明原报告公司、年份、页码；保留原作者和数据集归属，不声明原报告版权归本项目所有。固定子集和源文件指纹见 finqa-manifest.json。131 个唯一文档页面；40 dev / 100 test 问题；重建见 scripts/rebuild_finqa.py。

finqa-documents.json 是正文与表格，queries 是仅问题与文档范围，labels 是评测专用答案/程序/证据标签。推理模块不读取 labels。全库检索评测不把 query.doc_id 作为检索过滤；数值规则解析单独披露给定文档范围。

blocks.json、queries.json、synthetic-report.pdf 为旧的小型合成测试 fixture，只用于单元测试/解析演示，不参与 FinQA 主实验，也不冒充真实公司财报。原 Microsoft 10-K PDF 下载到 artifacts，不再分发；解析审计与 FinQA 评测分开。

FinQA 引用中的 page 来自原数据 filename（page_N.pdf），没有在本次逐条核对其与下载原 PDF 的物理页序是否一致；不能宣称原 PDF 引用已人工验真。
