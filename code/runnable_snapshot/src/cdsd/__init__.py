"""CDSD algorithm family (Conflict-Driven Stepwise Debate).

Paper-facing methods:
  CDSD-Soft     -> cdsds   (v2)
  CDSD-Select   -> cdsdj   (v9)
  CDSD-Faithful -> cdsdfi  (v12 faithful ablation)
  CDSD-Faithful -> cdsdnga (v12 with repaired claim decomposition; implementation tag only)

Historical/diagnostic variants remain importable for reproducibility, but should not be
treated as equal mainline methods. See `src/cdsd/README.md` and `src/cdsd/registry.py`.
"""
from src.cdsd.v1_hardcommit import run as run_cdsd
from src.cdsd.v2_soft import run as run_cdsd_soft
from src.cdsd.v3_faithful import run as run_cdsd_faithful
from src.cdsd.v4_anchored import run as run_cdsd_anchored
from src.cdsd.v5_grounded import run as run_cdsd_grounded
from src.cdsd.v6_canonical import run as run_cdsd_canonical
from src.cdsd.v7_aggregate import run as run_cdsd_aggregate
from src.cdsd.v8_rerank import run as run_cdsd_rerank
from src.cdsd.v9_judgefix import run as run_cdsd_judgefix
from src.cdsd.v10_judgegate import run as run_cdsd_judgegate
from src.cdsd.v11_judgeverbatim import run as run_cdsd_judgeverbatim
from src.cdsd.v12_indexed import run as run_cdsd_faithful_indexed
from src.cdsd.v12_indexed import run_boost as run_cdsd_x
from src.cdsd.v12_indexed import run_bridge as run_cdsd_bridge
from src.cdsd.v15_structured_nodes import run as run_cdsd_structured_nodes
from src.cdsd.v15_structured_nodes import run_select as run_cdsd_structured_select
from src.cdsd.v15_structured_nodes import run_preserve_answer as run_cdsd_structured_preserve
from src.cdsd.v15_structured_nodes import run_bridge as run_cdsd_structured_bridge
from src.cdsd.v15_structured_nodes import run_claim_clean as run_cdsd_structured_claim_clean
from src.cdsd.v15_structured_nodes import run_nonanswer_recover as run_cdsd_structured_nonanswer_recover
from src.cdsd.v15_structured_nodes import run_granularity_guard as run_cdsd_structured_granularity_guard
from src.cdsd.v15_structured_nodes import run_granularity_guard_strict as run_cdsd_structured_granularity_guard_strict
from src.cdsd.v15_structured_nodes import run_granularity_guard_answer_type as run_cdsd_structured_granularity_guard_answer_type
from src.cdsd.v15_structured_nodes import run_granularity_guard_answer_type_soft_commit as run_cdsd_structured_granularity_guard_answer_type_soft_commit
from src.cdsd.v15_structured_nodes import run_granularity_guard_answer_type_concrete_commit as run_cdsd_structured_granularity_guard_answer_type_concrete_commit
from src.cdsd.v15_structured_nodes import run_granularity_guard_answer_type_always_commit as run_cdsd_structured_granularity_guard_answer_type_always_commit
from src.cdsd.v15_structured_nodes import run_granularity_guard_answer_type_debate_select as run_cdsd_structured_granularity_guard_answer_type_debate_select
from src.cdsd.v15_structured_nodes import run_audited_consensus as run_cdsd_structured_audited_consensus
from src.cdsd.v15_structured_nodes import run_verified_select as run_cdsd_structured_verified_select

VARIANTS = {
    "v1": run_cdsd, "cdsd": run_cdsd,
    "v2": run_cdsd_soft, "cdsds": run_cdsd_soft,
    "v3": run_cdsd_faithful, "cdsdf": run_cdsd_faithful,
    "v4": run_cdsd_anchored, "cdsda": run_cdsd_anchored,
    "v5": run_cdsd_grounded, "cdsde": run_cdsd_grounded,
    "v6": run_cdsd_canonical, "cdsdc": run_cdsd_canonical,
    "v7": run_cdsd_aggregate, "cdsdg": run_cdsd_aggregate,
    "v8": run_cdsd_rerank, "cdsdr": run_cdsd_rerank,
    "v9": run_cdsd_judgefix, "cdsdj": run_cdsd_judgefix,
    "v10": run_cdsd_judgegate, "cdsdk": run_cdsd_judgegate,
    "v11": run_cdsd_judgeverbatim, "cdsdv": run_cdsd_judgeverbatim,
    "v12": run_cdsd_faithful_indexed, "cdsdfi": run_cdsd_faithful_indexed,
    "cdsdi": run_cdsd_faithful_indexed,
    "v13": run_cdsd_x, "cdsdx": run_cdsd_x,
    "v14": run_cdsd_bridge, "cdsdfb": run_cdsd_bridge,
    "v15": run_cdsd_structured_nodes, "cdsdn": run_cdsd_structured_nodes,
    "cdsdns": run_cdsd_structured_select,
    "v16": run_cdsd_structured_preserve, "cdsdnp": run_cdsd_structured_preserve,
    "v17": run_cdsd_structured_bridge, "cdsdnb": run_cdsd_structured_bridge,
    "v18": run_cdsd_structured_claim_clean, "cdsdnc": run_cdsd_structured_claim_clean,
    "v19": run_cdsd_structured_nonanswer_recover, "cdsdnr": run_cdsd_structured_nonanswer_recover,
    "v20": run_cdsd_structured_granularity_guard, "cdsdng": run_cdsd_structured_granularity_guard,
    "v21": run_cdsd_structured_granularity_guard_strict, "cdsdngs": run_cdsd_structured_granularity_guard_strict,
    "v22": run_cdsd_structured_granularity_guard_answer_type, "cdsdnga": run_cdsd_structured_granularity_guard_answer_type,
    "cdsdngac": run_cdsd_structured_granularity_guard_answer_type_soft_commit,
    "cdsdngax": run_cdsd_structured_granularity_guard_answer_type_concrete_commit,
    "cdsdngaa": run_cdsd_structured_granularity_guard_answer_type_always_commit,
    "cdsdngad": run_cdsd_structured_granularity_guard_answer_type_debate_select,
    "v23": run_cdsd_structured_audited_consensus, "cdsdna": run_cdsd_structured_audited_consensus,
    "v24": run_cdsd_structured_verified_select, "cdsdnv": run_cdsd_structured_verified_select,
}

__all__ = ["run_cdsd", "run_cdsd_soft", "run_cdsd_faithful", "run_cdsd_anchored", "run_cdsd_grounded",
           "run_cdsd_canonical", "run_cdsd_aggregate", "run_cdsd_rerank",
           "run_cdsd_faithful_indexed", "run_cdsd_x", "run_cdsd_bridge",
           "run_cdsd_structured_nodes", "run_cdsd_structured_select",
           "run_cdsd_structured_preserve", "run_cdsd_structured_bridge",
           "run_cdsd_structured_claim_clean", "run_cdsd_structured_nonanswer_recover",
           "run_cdsd_structured_granularity_guard", "run_cdsd_structured_granularity_guard_strict",
           "run_cdsd_structured_granularity_guard_answer_type",
           "run_cdsd_structured_granularity_guard_answer_type_soft_commit",
           "run_cdsd_structured_granularity_guard_answer_type_concrete_commit",
           "run_cdsd_structured_granularity_guard_answer_type_always_commit",
           "run_cdsd_structured_granularity_guard_answer_type_debate_select",
           "run_cdsd_structured_audited_consensus",
           "run_cdsd_structured_verified_select",
           "VARIANTS"]
