# v23 Adaptive Role-Edge Debate Tree

v23 是在 v8 上继续加入多个 guarded audit 后的最新版本。

## 固定算法行为

- 保留 v8 的 terminality debate 和 contested terminal branch。
- 对角色关系题自动启用 strict role-edge terminality / final debate。
- 当强 open frontier 与当前 final 分数接近时启用 open-frontier patience，继续扩展竞争分支。
- 对地点粒度题启用 granularity guard。
- 对 yes/no 普通冲突启用 terminal stance vote。

## 论文定位

v23 不是最干净的主算法，更适合作为失败分析和 guard 边界讨论。它在 full50 上看起来更强，但 Qwen2.5-14B full500 已完成的 HotpotQA 和 2Wiki 都明显弱于 CDSD。

## 默认锁定

`src/ldt_debate_tree/algorithm.py` 中默认：

- `ldt_use_terminality_debate=True`
- `ldt_keep_contested_terminals=True`
- `ldt_open_frontier_patience=True`
- `ldt_strict_role_edges=None`
- `ldt_use_granularity_guard=True`
- `ldt_use_binary_stance_vote=True`

trace 中会写入 `algorithm_version="v23_adaptive_role_edge"`。
