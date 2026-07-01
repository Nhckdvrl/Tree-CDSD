# v6.2 Tri-Projection Evidence Debate Tree

v6.2 is the strongest full500 tree result currently kept in this repository.
It should be read as a structured evidence-projection variant of intermediate
process debate, rather than as a free-form open-node tree.

## Core Code

- Entry module: `src/ldt_v5/algorithm.py`
- Public entry point: `run(...)`
- Runtime import path in the original project: `src.ldt_v5`

## Mechanism

The algorithm builds and compares three constrained projections:

1. `full`: the ordinary reasoning tree.
2. `focused_evidence`: a tree guided by a resolved evidence plan.
3. `focused_contract`: a tree guided by the answer-role contract.

The final answer is selected by terminal leaf arbitration and projection
agreement audit. This makes the intermediate reasoning process easier to judge:
the debate is no longer over arbitrary natural-language nodes, but over
structured evidence projections and terminal leaves.

## Paper Role

Use v6.2 as the current performance-oriented tree method:

- It wins the Qwen2.5-14B full500 main table against the selected baselines.
- It also beats the Qwen3-8B full500 baseline macro in the current Qwen3 table.
- Its narrative should emphasize typed/evidence-constrained intermediate
  process management, not unconstrained node discovery.
