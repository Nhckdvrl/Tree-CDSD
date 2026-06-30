# 上传前检查清单

当前目录只是预备包，尚未 push。

建议正式上传前检查：

1. 是否要把 `code/runnable_snapshot/src` 提升为仓库主 `src`。
2. 是否要保留 `code/selected_versions`，还是只保留 paper artifact。
3. 是否要补一个 `requirements.txt` 或 `environment.yml`。
4. 是否要把大体积 raw trace 降采样，避免仓库太重。
5. 是否要把 v7/v8/v23 当前共用代码拆成真正的版本快照。
6. 是否要把 v62 改名为 `TriProjectionEvidenceDebateTree`，让代码命名更贴近论文叙事。
7. 是否要在 README 中明确：v23 latest full500 不好，当前推荐主线是 v6.2 强结果 + v8 干净机制。

