# 结果摘要：树方法与原始 baseline 对照

本文档比较当前方法和所需的以下 baseline：

```text
CoT / Self-Consistency / Consensus / MAD / dMAD / CDSD
```


## 1. Qwen2.5-14B full500 主表

这是当前最严格、最适合作论文主表的一组：同一模型、同一批四个多跳数据集、每个数据集 500 条、同一 Qwen2.5-32B LLM-judge。

来源：`tables/qwen25_14b_full500_against_baselines.csv`

| 方法 | HotpotQA | 2Wiki | MuSiQue | StrategyQA | Macro |
| --- | ---: | ---: | ---: | ---: | ---: |
| CoT | 76.4 | 78.2 | 54.0 | 90.4 | 74.75 |
| Self-Consistency | 80.2 | 79.2 | 56.4 | 91.8 | 76.90 |
| Consensus | 79.6 | 78.6 | 57.4 | 91.0 | 76.65 |
| MAD | 79.4 | 77.0 | 55.0 | 89.6 | 75.25 |
| dMAD | 72.6 | 70.6 | 47.0 | 62.6 | 63.20 |
| CDSD-Soft | 81.2 | 81.4 | 57.8 | 93.6 | 78.50 |
| CDSD-JudgeFix | 81.0 | 81.6 | 57.2 | 93.6 | 78.35 |
| CDSD-Granularity+Type | 80.6 | 80.4 | 59.6 | 93.4 | 78.50 |
| v6.2 Tri-Projection Evidence Debate Tree | 82.2 | 81.8 | 61.2 | 94.0 | 79.80 |

结论：在 Qwen2.5-14B full500 上，v6.2 是目前唯一严格超过全部原始 baseline 的树版本。相对最强单个 CDSD 变体，macro 提升 +1.30；相对每个数据集 oracle 取最强 CDSD，macro 仍提升 +0.80。


## 2. Qwen3 full500 baseline

2026-07-01 已补齐 Qwen3-4B、Qwen3-8B、Qwen3-14B 的 full500 原始 baseline，均使用同一个 Qwen2.5-32B LLM-judge 复评。

来源：

- `raw_metrics/qwen3_full500_baselines_q32judge_summary.csv`
- `raw_metrics/qwen3_full500_baselines_q32judge.jsonl`

Macro judge accuracy:

| 模型 | CoT | Self-Consistency | Consensus | MAD | dMAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-4B | 73.45 | 73.55 | 75.05 | 72.80 | 69.35 |
| Qwen3-8B | 76.25 | 76.10 | 77.40 | 75.80 | 65.45 |
| Qwen3-14B | 78.20 | 77.85 | 78.05 | 77.10 | 66.90 |

Qwen3-32B baseline 正在主项目中补跑，尚未写入本包。

## 3. Qwen3-8B：v8 / v6.2 full500 与原始 baseline

现在 Qwen3-8B 的原始 baseline、v8、v6.2 都是 full500，并用同一个 Qwen2.5-32B LLM-judge 进行评估，因此这组可以作为同模型、同规模的直接对照。

| 方法 | n | HotpotQA | 2Wiki | MuSiQue | StrategyQA | Macro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CoT | 500 | 79.00 | 79.60 | 54.60 | 91.80 | 76.25 |
| Self-Consistency | 500 | 80.80 | 76.40 | 55.80 | 91.40 | 76.10 |
| Consensus | 500 | 79.60 | 80.20 | 59.00 | 90.80 | 77.40 |
| MAD | 500 | 79.00 | 78.60 | 57.60 | 88.00 | 75.80 |
| dMAD | 500 | 72.40 | 73.20 | 47.80 | 68.40 | 65.45 |
| v6.2  | 500 | 82.00 | 83.40 | 59.60 | 93.00 | 79.50 |
| v8  | 500 | 77.20 | 73.40 | 50.40 | 91.60 | 73.15 |

结论：在 Qwen3-8B full500 上，v6.2 的 Macro 为 79.50，高于最强原始 baseline Consensus 的 77.40；v8 Macro 为 73.15，低于原始 baseline。v8 更适合作为“自由抽点、合并、冲突判断路线”的失败/诊断版本，而不是当前性能主结果。

## 4. Qwen3.5-9B：已有 baseline 参考

当前 Qwen3.5-9B 的原始 baseline 是 n=150，v6.2 是 full500，同样只能看趋势。

| 方法 | n | HotpotQA | 2Wiki | MuSiQue | StrategyQA | Macro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CoT | 150 | 81.33 | 82.67 | 58.00 | 92.00 | 78.50 |
| Self-Consistency | 150 | 78.00 | 80.67 | 57.33 | 92.67 | 77.17 |
| Consensus | 150 | 80.67 | 80.67 | 62.67 | 91.33 | 78.84 |
| MAD | 150 | 80.67 | 82.67 | 61.33 | 89.33 | 78.50 |
| dMAD | 150 | 70.00 | 74.00 | 40.67 | 64.00 | 62.17 |
| CDSD best | 150 | 84.00 | 82.67 | 64.67 | 94.00 | 81.34 |
| v6.2 Tri-Projection | 500 | 84.60 | 82.40 | 71.40 | 95.80 | 83.55 |

保守结论：Qwen3.5-9B 上 v6.2 的主要优势来自 MuSiQue；如果要把 Qwen3.5 写进正式主表，需要补跑 full500 的 CoT、SC、Consensus、MAD、dMAD、CDSD。
