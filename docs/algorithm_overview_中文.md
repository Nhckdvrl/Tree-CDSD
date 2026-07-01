# Tree-CDSD 算法迭代主线
记录了当前的几个算法版本

## 1. 主线

本项目的研究主线是多 agent debate，并且是基于树的。这个树的作用不是枚举多个 thought 后打分，而是记录 agent 对局部推理边的立场、冲突、裁决和复审。

一个大致的算法思路是：

```text
多 agent 提出局部证据边或证据路线
→ moderator 合并等价立场并识别冲突
→ agents 围绕局部冲突辩论
→ 裁决写回树结构
→ terminal path 进入最终裁决
```

## 2. v0 Initial LDT

v0 是最初忠实实现的版本：

1. 每个 frontier 父节点让 3 个 agent 各生成一个 atomic next-hop。
2. 对候选做 deterministic merge 和 LLM merge。
3. 对同父候选判断 equivalent、compatible、uncertain、conflict。
4. conflict 或 uncertain group 触发局部 debate。
5. 每个父节点保留少量 child，整层保留全局 beam。
6. 最终从 terminal paths 中选答案。

v0 的价值是验证树结构和局部 debate 是否能真实落地。但不是性能主版本。

## 3. v6.2 Tri-Projection Evidence Debate Tree

v6.2 是我放着没管让ai自动迭代出来的产物，没想到和最初的主线思路截然不同了，但是这确实是当前刷榜最强的树方法了。早期文档把它写成“三投影树”，好像和辩论关系不大，但后来想了想也可以和树辩论扯上关系，它可以重新叙述成一种 **structured debate over evidence projections**。

具体参考：/docs/v6.2_三投影证据辩论树.md

目前觉得如果其他实在刷不出来 v6.2 不必被放弃。

## 4. v7 Debate Tree

我发现 v6.2 有点奇葩后重新做的 v7 ，把主线从三投影重新拉回显式的多 agent debate：

1. 可选 root contract debate，定义答案槽位和桥接关系。
2. 每层 agents 提出 atomic next-hop。
3. 合并候选并判定关系。
4. 引入 adversarial challenger 处理假共识。
5. conflict / uncertain 进入 local debate。
6. compatible synthesis 作为可消融模块。
7. terminal paths 进入最终 path-level debate。

v7 的结构很适合论文叙事，但苦于性能不足，尤其 MuSiQue 和 Qwen2.5-14B 泛化弱。

## 5. v8 Contested Terminal Debate Tree

v8 是目前最干净的多 agent debate 主线版本。它的核心贡献是把“是否已经到终点”也变成树上的辩论对象。

流程：

```text
agents 提出 next-hop
→ merge / relation judge
→ local debate
→ terminality debate
→ contested terminal branch 保留败方立场
→ 原节点重开继续扩展
→ final path-level debate
```

关键设计：

- **terminality debate**：判断 candidate final 是否真的满足问题槽位，还是只到了中间实体。
- **contested terminal branch**：被质疑的 final 不被硬删除，而是作为低分争议终点保留，最终辩论仍可复审。
- **atomic quantity closure**：数字、日期、年份等显式原子答案不被过度重开。

v8 的优点是算法边界清楚、方便消融、case analysis 好讲。缺点是 full50 有收益，但尚未证明 full500 稳定强于 CDSD。

## 6. v23 Adaptive Role-Edge Debate Tree

让ai迭代出来的屎，我去睡觉了没看，结果这个 v23 是在 v8 上加入多个 guarded 组件后搞出来的，明明已经叫不要搞很多工程兜底了：

- strict role-edge terminality
- open-frontier patience
- bare-entity guard
- granularity guard
- binary stance vote

这些组件都来自真实失败轨迹，所以它们有明显的 benchmark-facing 味道。并且在 Qwen2.5-14B full 500 结果在 HotpotQA 和 2Wiki 上明显输给 CDSD，所以真的是屎。



