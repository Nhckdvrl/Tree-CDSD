# CDSD Algorithm Family

CDSD means **Conflict-Driven Stepwise Debate**: multiple agents first reason independently, a judge finds the first claim-level conflict, agents debate only the local segment that led to that conflict, and the resolved segment becomes the agreed prefix for the next reasoning step.

This directory now separates variants by research role. The code files stay flat for import compatibility while long experiments are running; use this README and `registry.py` as the source of truth for what is mainline versus archived.

## Paper-Facing Methods

| paper name | code | version | role | status |
| --- | --- | --- | --- | --- |
| CDSD-Hard | `cdsd` | v1 | naive hard-commit original | historical baseline |
| CDSD-Soft | `cdsds` | v2 | soft-commit process debate | mainline |
| CDSD-Select | `cdsdj` | v9 | v2 plus judge-free final selection | mainline robustness |
| CDSD-Faithful | `cdsdfi` | v12 | indexed segment debate with evidence and relation gates | faithful ablation |
| CDSD-Faithful, repaired claim decomposition | `cdsdn`/`cdsdnga` | v12 repair | same v12 controller, but replaces the weak claim-decomposition stage | current main v12 implementation |

Do not introduce a separate paper method name for the repaired claim-decomposition line. In notes and paper tables it is still **v12 / CDSD-Faithful**; `cdsdn` and `cdsdnga` are implementation tags left only for reproducibility.

## The Original Algorithm Story

1. Agents independently answer and expose chain-of-thought as numbered claim nodes.
2. A judge compares the claim chains and finds the first material conflict.
3. The agents debate the segment from the last agreed/debated node to that first conflict.
4. A resolver produces one or more agreed nodes for that segment.
5. Safe nodes are committed into the agreed prefix.
6. Agents continue reasoning from the agreed prefix.
7. The loop repeats until consensus or a safe fallback answer.

The important claim is **process-level, conflict-localized debate**, not answer-level arguing or final-answer voting.

## Mainline

### `v2_soft.py` / `cdsds` / CDSD-Soft

This is the first stable implementation. It keeps the conflict-localized loop but fixes v1's main failure mode:

- no hard-freezing of unsafe claims;
- commit only when the resolver is confident;
- short-circuit when all agents already agree on a concrete answer;
- fallback to agent answer aggregation when debate cannot safely resolve the segment.

Paper role: strong process-debate baseline.

### `v9_judgefix.py` / `cdsdj` / CDSD-Select

This is a robustness fix on top of v2. The trace diagnosis showed that judge-consensus exits were often lost because the judge synthesized a short, evasive, or minority final answer. v9 keeps the v2 loop but ignores judge-generated final answers on consensus exits and selects from agent answers instead.

Paper role: protocol robustness. Do not present it as a new intermediate-debate mechanism.

### `v12_indexed.py:run` / `cdsdfi` / CDSD-Faithful

This repairs old v3 so it finally matches the original segment-debate idea:

- judge returns explicit claim indices;
- each agent debates only the segment up to its first conflicting claim;
- resolver returns resolved nodes plus exact evidence quotes;
- programmatic gates require quote support and relation preservation;
- unsafe commits fallback instead of freezing an error.

Paper role: faithful ablation and next-iteration base. Current q32/q72 full evidence does not justify promoting it above CDSD-Select as the main method yet.

### `v15_structured_nodes.py` implementation tags / v12 repaired claim decomposition

This targets the weakest part of v12: turning each solver's reasoning into comparable claim nodes. Instead of asking small models to directly output `Claim 1`, `Claim 2`, it uses a two-stage protocol:

- solver writes a normal step-by-step solution;
- a deterministic extractor rewrites that solution into typed nodes with fields for QA, math, and multiple-choice tasks;
- the v12 indexed conflict finder and segment resolver operate on the extracted node text;
- traces store node payloads and node-quality metrics for audit.

Paper role: current v12 repair. Do not call it a new algorithm. When a table needs one method column, report it as **v12** and note in a footnote that the implementation tag is `cdsdnga`.

### Other structured-node implementation tags

Trace analysis produced several implementation tags such as `cdsdnp`, `cdsdng`, `cdsdngs`, and `cdsdnga`. These are not separate methods. Keep them in manifests and appendix-level ablations only.

## Historical Baselines

| code | version | status | why kept |
| --- | --- | --- | --- |
| `v1_hardcommit.py` / `cdsd` | v1 | historical | shows hard commit amplifies errors |
| `v3_faithful.py` / `cdsdf` | v3-old | historical | closest early attempt, but lacks reliable indices/evidence gates |

## Archived / Diagnostic Variants

These are kept for auditability and appendix-level ablations. They should not be renamed into paper methods unless new evidence promotes them.

| code | version | status | reason |
| --- | --- | --- | --- |
| `v4_anchored.py` / `cdsda` | v4 | rejected | CoT anchor can override correct loop answers |
| `v5_grounded.py` / `cdsde` | v5 | rejected/diagnostic | grounding and extra rounds did not beat v2 enough |
| `v6_canonical.py` / `cdsdc` | v6 | rejected | answer shortening hurts containment-based EM |
| `v7_aggregate.py` / `cdsdg` | v7 | diagnostic | EM-oriented containment aggregation with F1 cost |
| `v8_rerank.py` / `cdsdr` | v8 | rejected/diagnostic | rerank has targeted gains but unstable regressions |
| `v10_judgegate.py` / `cdsdk` | v10 | rejected | tied v9; v9 is simpler |
| `v11_judgeverbatim.py` / `cdsdv` | v11 | rejected | too conservative |
| `v12_indexed.py:run_boost` / `cdsdx` | v13 | rejected | q72 gold/targeted gains, but q7/q32/q72 full regressions |
| `v12_indexed.py:run_bridge` / `cdsdfb` | v14 | rejected smoke | safe but no gain over `cdsdfi`, higher calls |
| `v15_structured_nodes.py:run_select` / `cdsdns` | v15-select | diagnostic | same structured nodes plus final rerank, useful only if `cdsdn` improves node quality |

## File Map

| file | purpose |
| --- | --- |
| `components.py` | shared claim parsing, calls, judge conflict, debate, resolver, aggregation |
| `prompts.py` | all prompt templates |
| `registry.py` | human-readable method taxonomy |
| `__init__.py` | runtime import registry for method names |
| `eval.py` | legacy CDSD-only runner |

## Naming Rule Going Forward

Use paper-facing names in notes and tables:

- `CDSD-Soft`, not `v2` when writing the paper.
- `CDSD-Select`, not `v9`.
- `CDSD-Faithful`, not `cdsdfi`.
- `CDSD-Node`, not `cdsdn`, only after validation promotes it.

Keep raw code names only in reproducibility commands and result file tags.
