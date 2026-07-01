"""Clean CDSD runtime registry for the extracted research project.

Only paper-facing or actively iterated variants are exposed here:
- cdsds: CDSD-Soft / v2
- cdsdj: CDSD-Select / v9
- cdsdfi: faithful indexed v12
- cdsdnga: current v12 repaired claim/node decomposition
"""

from src.cdsd.v2_soft import run as run_cdsd_soft
from src.cdsd.v9_judgefix import run as run_cdsd_judgefix
from src.cdsd.v12_indexed import run as run_cdsd_faithful_indexed
from src.cdsd.v15_structured_nodes import (
    run_granularity_guard_answer_type as run_cdsd_structured_granularity_guard_answer_type,
    run_granularity_guard_answer_type_soft_commit as run_cdsd_structured_granularity_guard_answer_type_soft_commit,
)
from src.cdsd.general_verify import run_general_verify_select as run_cdsd_general_verify
from src.cdsd.conflict_iter import run_conflict_iter as run_cdsd_conflict_iter
from src.cdsd.router import run_router as run_cdsd_router, run_conflict_iter_anchor as run_cdsd_conflict_iter_anchor

VARIANTS = {
    "v2": run_cdsd_soft,
    "cdsds": run_cdsd_soft,
    "v9": run_cdsd_judgefix,
    "cdsdj": run_cdsd_judgefix,
    "v12": run_cdsd_faithful_indexed,
    "cdsdfi": run_cdsd_faithful_indexed,
    "cdsdnga": run_cdsd_structured_granularity_guard_answer_type,
    "cdsdngac": run_cdsd_structured_granularity_guard_answer_type_soft_commit,
    "cdsdg": run_cdsd_general_verify,
    "cdsdi": run_cdsd_conflict_iter,
    "cdsdx": run_cdsd_router,
    "cdsdia": run_cdsd_conflict_iter_anchor,
}

__all__ = [
    "run_cdsd_soft",
    "run_cdsd_judgefix",
    "run_cdsd_faithful_indexed",
    "run_cdsd_structured_granularity_guard_answer_type",
    "run_cdsd_structured_granularity_guard_answer_type_soft_commit",
    "run_cdsd_general_verify",
    "run_cdsd_conflict_iter",
    "run_cdsd_router",
    "VARIANTS",
]
