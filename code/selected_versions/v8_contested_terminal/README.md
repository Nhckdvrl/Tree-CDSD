# v8 Contested Terminal Debate Tree

v8 是最干净、最适合论文叙事的树上多 agent debate 版本。

## 固定算法行为

- 继承 v7 的 frontier next-hop proposal、候选合并、关系判断、局部 debate、path-level debate。
- 新增 terminality debate：当某个候选声称已经是 final answer 时，让 agents 专门辩论它是否真的满足问题槽位。
- 新增 contested terminal branch：被 terminality referee 质疑的终点不会被硬删除，而是保留为低分争议终点，同时原节点重新打开继续扩展。

## 和 v23 的区别

v8 不启用 v23 的 guarded 组件：

- 不启用 strict role-edge audit；
- 不启用 open-frontier patience；
- 不启用 location granularity guard；
- 不启用 yes/no terminal stance vote。

因此 v8 是更干净的算法主线，适合做机制解释和消融。

## 默认锁定

`src/ldt_debate_tree/algorithm.py` 中默认：

- `ldt_use_terminality_debate=True`
- `ldt_keep_contested_terminals=True`
- `ldt_open_frontier_patience=False`
- `ldt_strict_role_edges=False`
- `ldt_use_granularity_guard=False`
- `ldt_use_binary_stance_vote=False`

trace 中会写入 `algorithm_version="v8_contested_terminal"`。
