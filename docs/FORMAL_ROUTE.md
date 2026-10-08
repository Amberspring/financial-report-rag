# 正式同技术路线｜执行与验收

本文件定义正式版；现有 CPU/MiniLM/mock 结果只作为辅助验证。执行命令存在不代表结果已产生。当前机器没有 NVIDIA GPU、Docker 或已安装 Linux 子系统，不将 dry-run 写成实测。

## 1. Qwen3-8B 后训练

Linux NVIDIA 训练环境，进入 financial-service-llm：

```bash
python -m pip install -e '.[gpu,eval]'
python -m ecom.data
python scripts/ablation.py --stage sft --execute
python -m ecom.train --config configs/lora-r16-qv.json --stage dpo
python scripts/merge_adapter.py --adapter artifacts/lora-r16-qv/dpo/policy --output artifacts/qwen3-financial-merged
```

四组 SFT：LoRA r8/qv、r16/qv、r16/all 与 QLoRA r16/qv。只有训练产品的偏好对进入 DPO；检查实际训练参数更新、参考适配器未变，保存模型及日志。DPO 的 policy 子目录是合并入口，不能合并冻结的 reference。

使用第一项目 scripts/predict.py 采集实际服务回复，指定 --model financial-model 和独立 --output；再用 python -m ecom.evaluate 在同一 data/eval.jsonl 上计算规则指标。规则指标不能代替人工事实与安全评审。量化后重复同一评测，保留不同模型的结果文件。

验收：基座/SFT/DPO 在相同测试集生成，固定生成参数，检查事实、数字、拒答和行为边界；报告模型版本、数据指纹、每组显存和耗时。当前 192 条合成 FAQ 只是最小实验集，不能包装成真实银行大规模训练数据。扩大数据规模与中文任务时需另外记录来源和许可。

## 2. 正式 BGE reranker

进入 financial-research-copilot：

```bash
python -m pip install -e '.[neural,bge]'
python scripts/benchmark_finqa.py --neural --reranker BAAI/bge-reranker-v2-m3 --flag-reranker
RAG_CONFIG=configs/formal-bge.json python -m uvicorn finrag.api:app --host 127.0.0.1 --port 8001
```

embedding 使用 BGE，重排使用 BAAI/bge-reranker-v2-m3。记录真实型号、Recall@k 与延迟；与 MiniLM 结果分开，避免混用模型名称。保留表格保护、引用、拒答、受控计算与标签隔离。旧结果请先备份，benchmark 会重写结果文件。

## 3. vLLM、量化与 Redis

独立 vLLM 环境安装 requirements-gpu.txt；独立量化环境安装 requirements-quantization.txt，不覆盖训练环境依赖。

```bash
# financial-ai-serving；FP16 模型由项目一合并得到
python scripts/serve_vllm.py --model ../financial-service-llm/artifacts/qwen3-financial-merged --prefix-cache off --max-num-batched-tokens 4096 --execute
# 另一个终端、相同 vLLM 环境
python scripts/benchmark_vllm.py --model ../financial-service-llm/artifacts/qwen3-financial-merged --label fp16 --execute
```

量化环境分别执行 quantize.py 的 awq、gptq、int8。校准文件为第一项目 artifacts/data/train.jsonl，禁止用测试集校准；压缩产物使用 vLLM compressed-tensors 加载。每次只启动一个服务，重复同一批容量工作负载，保存模型质量、P95、TTFT/TPOT、输出 tokens/s、显存、错误数。TTFT 包含排队与 prefill，不能冒充纯 prefill 的 GPU 时间。

再比较 prefix cache 开/关及 max_num_batched_tokens 参数；每次重启服务、保留启动命令，固定输入输出长度及并发。随机 token 压测只测容量，金融问答质量另测。若需精确 prefill/decode 阶段开销，使用 vLLM/Torch profiler。

Redis 使用 compose.yaml 的真实服务；单独启动 `docker compose up -d redis`，再运行 `python scripts/benchmark_redis.py`。基准仅写 UUID 前缀的临时键，清理自己的键，不清空数据库。配置网关 REDIS_URL 后另做端到端热点压测；Redis 单次 SET/GET 不能代替完整模型性能。

正式网关 MODEL=financial-model，与 vLLM 服务名一致。完整服务还需测试过期规则隔离、限流、OOM/超时/断连降级、恢复和监控。

## 完成定义

- 正式模型/算法/服务实际运行成功，保存真实结果和失败案例；不能用低配替代结果充当正式结果。
- 质量和性能使用同一数据、同一生成设置和明确缓存状态，报告基线、样本数、硬件及版本。
- 不能事先承诺比截图的百分比大；不同数据与指标之间无法直接比较。
- 只有达到这些条件，简历才可以写完成 Qwen3-8B、QLoRA、vLLM 与量化实验。

依据：
- BGE reranker 官方：https://huggingface.co/BAAI/bge-reranker-v2-m3
- vLLM 0.11 原生压测：https://docs.vllm.ai/en/v0.11.0/cli/bench/serve.html
