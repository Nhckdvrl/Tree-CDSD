# Tree-CDSD 预备上传包

这个目录是给 `Nhckdvrl/Tree-CDSD` 准备的本地预览版，当前还没有上传远端。它只保留几条值得论文叙事和复查的主线版本，不再展示 60 多轮细碎迭代。

## 版本范围

| 版本 | 定位 | 代码位置 | 主要结论 |
| --- | --- | --- | --- |
| v0 initial LDT | 最初的 Layerwise Debate Tree 忠实实现 | `code/selected_versions/v0_initial/src/ldt` | 用少量 gold-context 样本验证了树节点、合并、关系判断和局部 debate 的基本可行性 |
| v6.2 Tri-Projection Evidence Debate Tree | 当前 full500 最强版本 | `code/selected_versions/v62_triproj/src/ldt_v5` | 多模型 full500 上整体强于 CDSD，可重述为 structured debate over evidence projections |
| v7 Debate Tree | 回到显式多 agent 树上辩论 | `code/selected_versions/v7_v8_v23_debate_tree/src/ldt_debate_tree` | 组件清楚，但性能仍弱 |
| v8 Contested Terminal Debate Tree | 最干净、最符合辩论主线的版本 | `code/selected_versions/v7_v8_v23_debate_tree/src/ldt_debate_tree` | terminality debate 和 contested terminal branch 带来可解释收益 |
| v23 Adaptive Role-Edge Debate Tree | 最新 guarded 版本 | `code/selected_versions/v7_v8_v23_debate_tree/src/ldt_debate_tree` | full50 看起来强，full500 最新结果明显输给 CDSD，适合作失败分析和 guard 边界讨论 |

`code/runnable_snapshot/src` 保留了原始 import 结构所需的最小运行快照，后续如果要复跑，可以基于这个目录整理成正式包。

注意：v7、v8、v23 属于同一条连续修改的代码线，当前工作区只保留了最新代码状态。具体说明见 `docs/version_code_note_中文.md`。

## 结果表

- `tables/qwen25_14b_full500_against_baselines.csv`：Qwen2.5-14B 上 CoT、SC、Consensus、MAD、dMAD、CDSD 和树方法 full500 对照。
- `tables/v62_multimodel_full500_against_tree_cdsd.csv`：v6.2 在 Qwen3.5-9B、Qwen3-8B、Qwen2.5-14B 上的 full500 主表。
- `tables/selected_tree_versions_full50_judge.csv`：v7、v8、v23 的 full50 小规模算法对照。
- `tables/v23_qwen25_14b_full500_latest.csv`：刚跑完的 v23 Qwen2.5-14B full500 最新结果。
- `tables/v0_initial_pilot_summary.csv`：v0 初版 gold-context 小样本摘要。

## 轨迹

- `traces/raw_examples`：完整原始 JSON 轨迹，适合细看 prompt、节点、合并、debate 和最终裁决。
- `traces/compact`：压缩后的轨迹摘要，适合快速看树节点、终点辩论、三投影选择和失败模式。
- `traces/README.md`：每个轨迹文件的用途说明。

代表样例包括：

- v8: Fletcher Webster、Knowsley、Adelphia successor 失败。
- v23: Rupert father-in-law 成功、Hotpot 成功样例、Whiston/Knowsley 回归失败、full500 首批失败摘要。
- v6.2: Fletcher Webster、Knowsley 三投影成功样例。

## 当前判断

最适合论文主线的两个候选是：

1. **v8 Contested Terminal Debate Tree**：最干净，最能讲清楚“树上多 agent debate”，适合做主算法叙事和消融设计。
2. **v6.2 Tri-Projection Evidence Debate Tree**：结果最强，应重新叙述成“围绕 evidence projection 的结构化多 agent debate”，不要轻易丢掉。

v23 不建议作为主算法直接宣传。它暴露了一个重要边界：在 full50 上有效的 role-edge、open-frontier、granularity、binary vote 等 guard，full500 上并没有稳定超过 CDSD，且有脱离干净辩论叙事的风险。
