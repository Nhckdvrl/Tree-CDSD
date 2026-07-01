# v7 Debate Tree

v7 是回到显式多 agent debate 主线后的第一版树算法。

## 固定算法行为

- 多个 agents 在每个 frontier node 下提出 atomic next-hop。
- 合并等价候选，并用 relation judge 判断 compatible / uncertain / conflict。
- 对 conflict / uncertain group 做局部 debate。
- adversarial challenger 检查多数 next-hop 是否漏掉竞争分支。
- compatible synthesis 可作为消融开关，但默认关闭。
- 最终在 terminal paths 之间做 path-level debate。

## 和后续版本的区别

v7 不启用 terminality debate，也不保留 contested terminal branch；它更像“局部冲突树 + 最终路径辩论”的基线版本。

## 默认锁定

`src/ldt_debate_tree/algorithm.py` 中默认：

- `ldt_use_terminality_debate=False`
- `ldt_keep_contested_terminals=False`
- `ldt_open_frontier_patience=False`
- `ldt_strict_role_edges=False`
- `ldt_use_granularity_guard=False`
- `ldt_use_binary_stance_vote=False`

trace 中会写入 `algorithm_version="v7_debate_tree"`。
