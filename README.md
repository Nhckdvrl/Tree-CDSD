# Tree-CDSD 

当前的一些和树相关的 多agent辩论 算法迭代


## 版本

| 版本 | 定位 | 代码位置 | 主要结论 |
| --- | --- | --- | --- |
| v0 initial LDT | 最初的 Layerwise Debate Tree 忠实实现 | `code/selected_versions/v0_initial/src/ldt` | 用少量 gold-context 样本验证了树节点、合并、关系判断和局部 debate 的基本可行性 |
| v6.2 Tri-Projection Evidence Debate Tree | 当前 full500 最强版本 | `code/selected_versions/v62_triproj/src/ldt_v5` | 多模型 full500 上整体强于 CDSD，可重述为 structured debate over evidence projections |
| v7 Debate Tree | 回到显式多 agent 树上辩论 | `code/selected_versions/v7_debate_tree/src/ldt_debate_tree` | 组件清楚，但性能仍弱 |
| v8 Contested Terminal Debate Tree | 最干净、最符合辩论主线的版本 | `code/selected_versions/v8_contested_terminal/src/ldt_debate_tree` | terminality debate 和 contested terminal branch 带来可解释收益 |
| v23 Adaptive Role-Edge Debate Tree | 最新 guarded 版本 | `code/selected_versions/v23_adaptive_role_edge/src/ldt_debate_tree` | full50 看起来强，full500 最新结果明显输给 CDSD，适合作为失败分析 |

`code/runnable_snapshot/src` 保留了原始 import 结构所需的最小运行快照，如果要跑，可以基于这个目录整理成正式包。

v7、v8、v23 现在已经拆成三个独立代码快照。每个目录里的 `README.md` 说明了该版本默认锁定的组件开关，trace 里也会写入对应的 `algorithm_version`。

## 结果表

- `tables/qwen25_14b_full500_against_baselines.csv`：Qwen2.5-14B 上 CoT、SC、Consensus、MAD、dMAD、CDSD 和树方法 full500 对照。
- `tables/v62_multimodel_full500_against_tree_cdsd.csv`：v6.2 在 Qwen3.5-9B、Qwen3-8B、Qwen2.5-14B 上的 full500 主表。
- `tables/qwen3_8b_v8_v62_full500_against_baselines.csv`：Qwen3-8B 上 CoT、SC、Consensus、MAD、dMAD、v6.2、v8 的 full500 同模型同 judge 对照。
- `raw_metrics/qwen3_full500_baselines_q32judge_summary.csv`：Qwen3-4B、Qwen3-8B、Qwen3-14B 的 CoT、SC、Consensus、MAD、dMAD full500 baseline，Qwen2.5-32B judge 复评。
- `raw_metrics/qwen3_full500_baselines_q32judge.jsonl`：上述 Qwen3 full500 baseline 的逐条 judge 结果。
- `tables/selected_tree_versions_full50_judge.csv`：v7、v8、v23 的 full50 小规模算法对照。
- `tables/v23_qwen25_14b_full500_latest.csv`：刚跑完的 v23 Qwen2.5-14B full500 最新结果。
- `tables/v0_initial_pilot_summary.csv`：v0 初版 gold-context 小样本摘要。

主要结果解读见 `docs/results_summary.md`，该文档只把树方法和原始 baseline 比较，不再把 v3/v4 等内部迭代版放进主表。

## 轨迹

- `traces/raw_examples`：完整原始 JSON 轨迹，适合细看 prompt、节点、合并、debate 和最终裁决。
- `traces/compact`：压缩后的轨迹摘要，适合快速看树节点、终点辩论、三投影选择和失败模式。
- `traces/README.md`：每个轨迹文件的用途说明。

代表样例包括：

- v8: Fletcher Webster、Knowsley、Adelphia successor 失败。
- v23: Rupert father-in-law 成功、Hotpot 成功样例、Whiston/Knowsley 回归失败、full500 首批失败摘要。
- v6.2: Fletcher Webster、Knowsley 三投影成功样例。

## 当前判断

最适合论文的两个候选是：

1. **v8 Contested Terminal Debate Tree**：最干净，最能讲清楚“树上多 agent debate”，适合做主算法叙事和消融设计。
2. **v6.2 Tri-Projection Evidence Debate Tree**：结果最强，应重新叙述成“围绕 evidence projection 的结构化多 agent debate”，不要轻易丢掉。
