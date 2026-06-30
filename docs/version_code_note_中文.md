# 版本代码说明

`src/ldt` 和 `src/ldt_v5` 是相对独立的代码目录，分别对应 v0 和 v6.2。

`src/ldt_debate_tree` 这条线从 v7 开始连续演化到 v23，当前工作区中没有独立 git commit 保存 v7、v8、v23 三个冻结快照。因此本预备包采用如下方式：

1. `code/selected_versions/v7_v8_v23_debate_tree/src/ldt_debate_tree` 保留当前最新代码，对应 v23。
2. v7、v8 的算法差异主要保存在中文文档和结果文件中。
3. 正式上传前如果需要严格复现 v7/v8，应从当前代码中拆出两个冻结分支：
   - `v8_clean`: 保留 terminality debate、contested terminal branch、atomic quantity closure。
   - `v23_guarded`: 在 v8_clean 上加入 role-edge、open-frontier、bare-entity、granularity、binary vote。
4. 论文叙事建议不要把 v23 guard 混入 v8 主流程，否则消融和 case analysis 会变得不干净。

这个说明故意放在文档里，避免上传后让读者误以为 v7/v8/v23 都有完全独立源码快照。

