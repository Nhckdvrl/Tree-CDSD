# LDT v0 算法实现与真实轨迹迭代记录

日期：2026-06-29

## 目标

实现一个不覆盖原 CDSD/MAD/SC 等 baseline 的新算法：Layerwise Debate Tree, LDT。v0 的目标不是追求最终 SOTA，而是把参考文档中的推荐版本忠实落到代码里，并用少量真实多跳样本检查树、节点、边、冲突组和 debate 是否真的符合算法思想。

## 新增文件

- `src/ldt/`：独立 LDT 算法包。
- `src/run.py`：只新增 `ldt` 方法注册和 LDT 参数透传。
- `results/ldt/v0/`：v0 真实轨迹输出目录。
- `scripts/run_ldt_v0_smoke.sh`：150 条 v0 轨迹验证入口。
- `scripts/ldt_trace_audit.py`：结构化 trace 审计工具。
- `docs/ldt/`：LDT 独立文档目录。

## v0 算法

当前版本采用 online step sampling：

1. 每个 frontier 父节点让 3 个 agent 各生成一个 next-hop。
2. 规则合并 + LLM merge refine；LLM 拆散的完全同义候选会被 deterministic coalesce 再合并。
3. LLM 判断同父候选间的 compatible/conflict/uncertain 关系。
4. conflict group 触发局部 debate，再由 referee 输出 accepted/uncertain/rejected。
5. compatible/uncertain group 轻量打分。
6. 每父节点保留 `per_parent_beam=2`，每层全局保留 `global_beam=6`。
7. 多跳 QA 默认 `min_depth_before_final=2`，避免把完整 CoT 压成单节点。
8. 最终答案由 top terminal paths 选择，路径分数采用 weakest-node score。

## 真实轨迹迭代

已跑真实小样本：

- `hotpotqa`, gold context, n=5：`results/ldt/v0/hotpotqa_ldt_v0_dev5_gold.jsonl`
- `2wiki`, gold context, n=3：`results/ldt/v0/2wiki_ldt_v0_dev3_gold.jsonl`
- 综合 audit：`docs/ldt/v0_trace_audit_dev.md`

当前综合 pilot：

- 8 条真实样本，本地 EM/acc = 0.875，F1 = 0.875。
- 平均非 root 节点数 = 2.50。
- 平均最大深度 = 2.25。
- 平均 debate groups/question = 0.50。
- trace 结构错误 = 0/8。

## 修复记录

1. Qwen 输出 Python 风格 `False/True` 导致 JSON 解析失败。新增 loose JSON parser。
2. 模型输出 YAML-ish `node: ...` 时，原 fallback 会把 `confidence: 1.0` 误读成 answer。修复 line parser 禁止跨行吞字段。
3. yes/no 问题中，模型会把 `American` 这类中间属性当 final answer。新增 answer contract 和 final format repair。
4. LLM merge 会把 `2. Ed Wood...` 和 `1. Ed Wood...` 拆开。新增编号归一化和 post-LLM exact coalesce。
5. final selector 过度相信节点 answer 字段。改为 `answer_hint`，并在 prompt 中声明必须重新验证 question relation。
6. 早期 final 让树退化成单节点。新增 `min_depth_before_final=2` guard。
7. QA 答案编号残留如 `3. Animorphs`、纯编号 `3.`。新增统一 answer cleanup。

## 观察

LDT 的树结构已经能真实表达逐层推理。例如 Hotpot 的 nationality 样本形成：

- depth 1：`Scott Derrickson is American.`
- depth 2：`Ed Wood is American.`
- final：`yes`

2Wiki 样本也能保留并比较多条同层候选，例如 release-year 比较题中同时保留 `Blind Shaft` 和 `The Mask of Fu Manchu` 分支，再选正确路径。

## 已知问题

- 个别 candidate 的 `answer_hint` 仍可能来自 agent 的不可靠短答案；final selector 通常能纠正，但后续应加入 candidate-answer grounding verifier。
- Hotpot `Shirley Temple` 样本存在 benchmark 标注歧义：`United States ambassador` 和 `Chief of Protocol` 都是 passage 中的 government position，本地 EM 只接受后者。
- 当前 v0 仍偏依赖最终 selector 的关系理解，下一步应把 relation preservation 前移到 candidate scoring/debate resolution。
