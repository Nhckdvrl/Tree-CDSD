# Selected Algorithm Versions

This directory keeps the core code for the tree / graph CDSD line as paper
artifacts. Each subdirectory is intentionally small: it contains the algorithm
module needed to inspect the mechanism, plus a README when the version needs
extra context.

## Versions

| Version | Core module | Role |
| --- | --- | --- |
| `v0_initial` | `src/ldt` | Original layerwise debate tree: next-hop proposals, merge, relation judging, local debate, beam search, final path selection. |
| `v62_triproj` | `src/ldt_v5` | Tri-projection evidence debate tree: full tree, evidence-focused tree, contract-focused tree, terminal leaf arbitration. |
| `v7_debate_tree` | `src/ldt_debate_tree` | Explicit debate-tree baseline with local conflict debate and final path debate. |
| `v8_contested_terminal` | `src/ldt_debate_tree` | Adds terminality debate and contested terminal branches; this is the cleanest version of the original open-node debate idea. |
| `v23_adaptive_role_edge` | `src/ldt_debate_tree` | Adds role-edge, open-frontier, granularity, and binary stance guards; kept mainly for failure/boundary analysis. |

## Shared Runtime Snapshot

`../runnable_snapshot/src` keeps the surrounding runtime used by these modules:
data loaders, agents, LLM client, evaluators, baseline methods, CDSD methods,
and `run.py`. The selected versions are easier to read; the runtime snapshot is
closer to the original executable layout.
