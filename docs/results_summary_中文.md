# 结果摘要

## 1. 刚跑完的 v23 full500

模型：Qwen2.5-14B  
judge：Qwen2.5-32B LLM-judge  
文件：`tables/v23_qwen25_14b_full500_latest.csv`

| 数据集 | v23 local EM | v23 judge | 最强 CDSD judge | 差值 |
| --- | ---: | ---: | ---: | ---: |
| HotpotQA | 69.0 | 77.8 | 81.2 | -3.4 |
| 2WikiMultiHopQA | 73.6 | 73.2 | 81.6 | -8.4 |

结论：v23 full50 的乐观结果没有稳定迁移到 full500。尤其 2Wiki full500 下降很明显，说明 role-edge/open-frontier 等 guard 并没有形成通用收益。

## 2. v6.2 full500 仍是当前最强树版本

文件：`tables/v62_multimodel_full500_against_tree_cdsd.csv`

| 模型 | Macro v6.2 | Macro CDSD | 差值 |
| --- | ---: | ---: | ---: |
| Qwen3.5-9B | 83.55 | 81.60 | +1.95 |
| Qwen3-8B | 79.50 | 78.95 | +0.55 |
| Qwen2.5-14B | 79.80 | 78.75 | +1.05 |

结论：如果目标是当前论文主表，v6.2 比 v23 更值得保留。建议把它重构叙述为“围绕 evidence projections 的结构化多 agent debate tree”。

## 3. v8 是最干净的 debate 版本

文件：`tables/selected_tree_versions_full50_judge.csv`

| 模型 | v7 judge macro | v8 judge macro | v23 full50 judge macro |
| --- | ---: | ---: | ---: |
| Qwen3-8B | 76.0 | 78.5 | 79.0 |
| Qwen2.5-14B | 73.0 | 78.0 | 80.5 |

这个表只能作为小规模算法证据，因为 v7/v8/v23 多数是 full50。v8 的价值主要在于设计干净、组件可消融、case analysis 清楚。

## 4. Qwen2.5-14B full500 对照

文件：`tables/qwen25_14b_full500_against_baselines.csv`

关键点：

- v6.2 在 HotpotQA、2Wiki、MuSiQue、StrategyQA 上都不低于或高于 CDSD 主版本。
- v23 只完成 HotpotQA 和 2Wiki full500，且两者都明显输给 CDSD。
- CDSD 在 Qwen2.5-14B 上仍然是非常强的 chain-style debate baseline。

## 5. 现在应如何使用这些结果

建议论文叙事暂时分成两条：

1. **干净算法线**：以 v8 为核心，强调 terminality debate、contested terminal branch、树上局部冲突裁决。
2. **强结果线**：以 v6.2 为核心，把三投影树改写成 evidence-projection structured debate。

v23 则保留为失败分析：它证明了单纯把失败 case 写成 guard 容易损害通用性，也提示下一版应该引入更统一的 Tree Moderator，而不是继续堆分散规则。

