# Chain Structured Nodes

This directory preserves the clean chain-CDSD implementation that makes
intermediate reasoning debate operational through structured node extraction.
It is kept here as the intended base layer for future Tree/Graph-CDSD work.

## Core Idea

The chain method does not ask weak solver agents to directly maintain a tree or
graph. It uses a more conservative two-stage protocol:

1. Each solver writes an ordinary step-by-step rationale.
2. A low-temperature controller using the same backbone extracts an ordered
   chain of small typed nodes.
3. A conflict judge compares the node chains and locates the earliest material
   disagreement.
4. Debate is restricted to the local segment ending at that indexed conflict.
5. A resolver commits only evidence-supported, relation-preserving resolved
   nodes.

The resulting object is:

```text
ordered typed node chain
-> indexed conflict localization
-> local segment debate
-> conservative commit gate
```

This is the part of CDSD that is most relevant for future graph redesign: a
graph version should align and connect these structured chains instead of
returning to free-form next-hop node generation.

## Files

| File | Purpose |
| --- | --- |
| `src/cdsd/components.py` | Shared node extraction, fallback parsing, conflict detection, debate, and resolver helpers. |
| `src/cdsd/prompts.py` | Prompts for reasoning, typed node extraction, indexed conflict judging, local segment debate, and resolution. |
| `src/cdsd/v9_judgefix.py` | `cdsdj`: structured nodes plus judge-consensus fix; robust select-style chain baseline. |
| `src/cdsd/v12_indexed.py` | Indexed faithful segment debate: conflict judge returns per-agent node indices and debate is scoped to each prefix segment. |
| `src/cdsd/v15_structured_nodes.py` | `cdsdnga`: v12-style indexed debate with repaired structured-node decomposition and answer-type granularity guard. |
| `src/cdsd/__init__.py` | Clean registry names for the paper-facing chain variants. |

## Method Roles

`cdsdj` and `cdsdnga` are the main chain methods to compare against Qwen3
baselines.

- `cdsdj` is the simpler robust controller. It uses structured node extraction,
  detects conflict, performs single-claim soft debate, and avoids trusting the
  judge's synthesized final answer on consensus by selecting from actual agent
  answers.
- `cdsdnga` is the mechanism-complete chain method. It keeps indexed conflict
  localization and local segment debate, then adds an answer-type-aware guard to
  avoid over-specific or wrong-granularity answers.

`v15_structured_nodes.py` is an implementation filename, not a separate paper
method name. In paper prose, describe it as the repaired implementation of
indexed structured-node CDSD.

## Reuse for Tree/Graph-CDSD

The next graph method should reuse this layer as follows:

```text
agent CoT samples
-> chain structured nodes
-> cross-agent node alignment
-> typed evidence/conflict graph
-> local segment debate on conflicting aligned nodes
-> gated commit and path selection
```

This avoids the main failure mode of the v7/v8/v23 free-node route: unstable
debate objects that are hard to merge, verify, and compare.
