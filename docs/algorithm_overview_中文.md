# Tree-CDSD 算法版本脉络

## 1. 主线

本项目的研究主线是多 agent debate，不是普通 Tree-of-Thought 搜索。树的作用不是枚举多个 thought 后打分，而是记录 agent 对局部推理边的立场、冲突、裁决和复审。

一个更准确的总描述是：

```text
多 agent 提出局部证据边或证据路线
→ moderator 合并等价立场并识别冲突
→ agents 围绕局部冲突辩论
→ 裁决写回树结构
→ terminal path 进入最终裁决
```

## 2. v0 Initial LDT

v0 是最初忠实实现参考文档的版本：

1. 每个 frontier 父节点让 3 个 agent 各生成一个 atomic next-hop。
2. 对候选做 deterministic merge 和 LLM merge。
3. 对同父候选判断 equivalent、compatible、uncertain、conflict。
4. conflict 或 uncertain group 触发局部 debate。
5. 每个父节点保留少量 child，整层保留全局 beam。
6. 最终从 terminal paths 中选答案。

v0 的价值是验证树结构和局部 debate 是否能真实落地。它不是性能主版本。

## 3. v6.2 Tri-Projection Evidence Debate Tree

v6.2 是当前 full500 表现最强的树方法。虽然早期文档把它写成“三投影树”，但它可以重新叙述成一种 **structured debate over evidence projections**。

对应关系如下：

| v6.2 组件 | 多 agent debate 解释 |
| --- | --- |
| 多个 evidence plans | 多个 debater 提出不同证据路线或假设 |
| resolved evidence plan | debate 后形成 shared agenda / provisional motion |
| plan validation | 对主张做 evidence cross-examination |
| full / evidence / contract 三投影树 | 三种论证视角：开放上下文、证据聚焦、关系约束 |
| terminal leaf selection | adjudicator 在终端论证间裁决 |
| projection-agreement audit | 带 evidence guard 的 majority / consistency check |
| terminal span cleaning | verdict normalization / answer span adjudication |

因此 v6.2 不必被放弃。更好的论文叙事是：v8 是最干净的树上局部辩论，v6.2 是结果最强的投影式结构化辩论树。

## 4. v7 Debate Tree

v7 把主线从三投影重新拉回显式多 agent debate：

1. 可选 root contract debate，定义答案槽位和桥接关系。
2. 每层 agents 提出 atomic next-hop。
3. 合并候选并判定关系。
4. 引入 adversarial challenger 处理假共识。
5. conflict / uncertain 进入 local debate。
6. compatible synthesis 作为可消融模块。
7. terminal paths 进入最终 path-level debate。

v7 的结构很适合论文叙事，但当时性能不足，尤其 MuSiQue 和 Qwen2.5-14B 泛化弱。

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

v23 是在 v8 上加入多个 guarded 组件后的最新版本：

- strict role-edge terminality
- open-frontier patience
- bare-entity guard
- granularity guard
- binary stance vote

这些组件都来自真实失败轨迹，但它们有明显的 benchmark-facing 味道。最新 Qwen2.5-14B full500 结果显示，v23 在 HotpotQA 和 2Wiki 上明显输给 CDSD，所以不建议把 v23 作为主算法直接宣传。

更合适的位置是：

1. 作为 v8 后续 guard 研究的失败/边界版本；
2. 用于分析为什么“堆规则式 guard”会破坏通用性；
3. 为下一版更干净的 controller/moderator agent 设计提供证据。

## 7. 和 Tree-of-Thoughts 的区别

Tree-of-Thoughts 主要是生成多个 thought、评估、搜索。Tree-CDSD 的核心是 debate：

- 候选节点是 agent 的局部立场；
- 边是可被质询的推理关系；
- conflict / uncertain 是局部辩论触发器；
- terminality 是对“是否已经回答问题”的辩论；
- final path selection 是对完整论证路径的裁决。

