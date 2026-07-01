# 代码索引

## 阅读用版本代码

The same list is also summarized in `selected_versions/README.md`.

- `selected_versions/v0_initial/src/ldt`
  - 最初的 Layerwise Debate Tree。
  - 包含 proposal、merge、relation judge、local debate、beam 选择、final path selection。

- `selected_versions/v62_triproj/src/ldt_v5`
  - v6.2 三投影证据树。
  - 核心入口是 `algorithm.py` 中的 evidence plan、resolved plan、三投影 reasoning tree、terminal leaf selection。
  - 详见 `selected_versions/v62_triproj/README.md`。

- `selected_versions/v7_debate_tree/src/ldt_debate_tree`
  - v7 显式树上多 agent debate。
  - 默认关闭 terminality debate、contested terminal、role-edge、open-frontier、granularity、binary stance vote。
  - trace 写入 `algorithm_version="v7_debate_tree"`。

- `selected_versions/v8_contested_terminal/src/ldt_debate_tree`
  - v8 争议终点辩论树。
  - 默认开启 terminality debate 和 contested terminal branch。
  - 默认关闭 v23 的 role-edge、open-frontier、granularity、binary stance vote。
  - trace 写入 `algorithm_version="v8_contested_terminal"`。

- `selected_versions/v23_adaptive_role_edge/src/ldt_debate_tree`
  - v23 adaptive role-edge guarded 版本。
  - 默认开启 terminality debate、contested terminal、open-frontier patience、auto strict role-edge、granularity guard、binary stance vote。
  - trace 写入 `algorithm_version="v23_adaptive_role_edge"`。

- `selected_versions/chain_structured_nodes/src/cdsd`
  - chain-CDSD 的结构化中间节点实现。
  - 包含 `cdsdj`、v12 indexed segment debate、`cdsdnga` 的 node extractor / conflict localization / local segment debate / commit gate。
  - 后续 Tree/Graph-CDSD 重新设计时，应优先复用这里的 typed node chain，而不是继续 v8 的自由 next-hop 抽点。

## 可运行快照

`runnable_snapshot/src` 保留了原项目的 import 结构：

- `ldt`
- `ldt_v5`
- `ldt_debate_tree`
- `cdsd`
- `agents`
- `llm`
- `eval`
- `data`
- `methods`
- `run.py`

后续正式上传前，建议把 `runnable_snapshot/src` 提升为仓库主 `src`，再把 `selected_versions` 作为版本对照或 paper artifact。

## 配置

`configs` 中保留了代表性模型配置：

- `qwen3_8b.yaml`
- `qwen3_4b.yaml`
- `qwen25_14b_gpu3.yaml`
- `qwen35_9b.yaml`
- `base.yaml`

当前多 agent 设定是同一基础模型的 3 个 persona agent，温度通常为 `[0.6, 0.6, 0.8]` 或 `[0.7, 0.7, 0.9]`。referee、merger、relation judge 通常使用同一 client 的低温调用，不是单独的强模型。
