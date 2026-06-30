# 代码索引

## 阅读用版本代码

- `selected_versions/v0_initial/src/ldt`
  - 最初的 Layerwise Debate Tree。
  - 包含 proposal、merge、relation judge、local debate、beam 选择、final path selection。

- `selected_versions/v62_triproj/src/ldt_v5`
  - v6.2 三投影证据树。
  - 核心入口是 `algorithm.py` 中的 evidence plan、resolved plan、三投影 reasoning tree、terminal leaf selection。

- `selected_versions/v7_v8_v23_debate_tree/src/ldt_debate_tree`
  - v7/v8/v23 共用的树上多 agent debate 代码线。
  - 当前文件状态对应最新 v23，v7/v8 的差异在文档和运行配置中说明。
  - v8 主体对应 terminality debate 和 contested terminal branch。
  - v23 在此基础上加入 strict role-edge、open-frontier patience、granularity guard、binary stance vote 等 guarded 组件。

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

