"""Human-readable registry for CDSD variants.

This file is intentionally metadata-only: it does not drive imports or experiments.
Its job is to keep the research story legible as the codebase accumulates variants.
"""

MAINLINE = {
    "cdsds": {
        "paper_name": "CDSD-Soft",
        "version": "v2",
        "status": "mainline",
        "role": "strong soft-commit baseline",
        "story": "Naive conflict-localized debate, repaired so unresolved or unsafe claims are not frozen.",
    },
    "cdsdj": {
        "paper_name": "CDSD-Select",
        "version": "v9",
        "status": "mainline",
        "role": "protocol-robust selection baseline",
        "story": "CDSD-Soft plus judge-free final answer selection on judge-consensus exits.",
    },
    "cdsdfi": {
        "paper_name": "CDSD-Faithful",
        "version": "v12",
        "status": "faithful_ablation",
        "role": "closest implementation of the original segment-debate idea",
        "story": "Indexed conflict claims, segment-level debate, exact evidence quotes, and relation-preserving commit.",
    },
    "cdsdnga": {
        "paper_name": "CDSD-Faithful",
        "version": "v12",
        "status": "main_current",
        "role": "v12 with repaired claim decomposition",
        "story": "Same indexed v12 controller; the weak initial claim-decomposition stage is replaced by task-aware CoT-to-node extraction. Do not present as a separate algorithm.",
    },
}

HISTORICAL_BASE = {
    "cdsd": {
        "paper_name": "CDSD-Hard",
        "version": "v1",
        "status": "historical",
        "role": "naive original implementation",
        "story": "Hard-commits resolved claims and demonstrates error amplification.",
    },
    "cdsdf": {
        "paper_name": "CDSD-Faithful-Old",
        "version": "v3",
        "status": "historical",
        "role": "old faithful attempt",
        "story": "Segment debate without reliable claim indices or exact evidence gating.",
    },
}

REJECTED_OR_DIAGNOSTIC = {
    "cdsda": "v4 CoT anchor; rejected because the anchor can override correct loop answers.",
    "cdsde": "v5 grounded 2-round resolver; roughly ties v2 but does not justify extra complexity.",
    "cdsdc": "v6 canonicalization; rejected because shorter spans hurt containment-based EM.",
    "cdsdg": "v7 containment-aware aggregation; EM-oriented diagnostic with F1 trade-off.",
    "cdsdr": "v8 evidence rerank; useful diagnostic, not stable enough for mainline.",
    "cdsdk": "v10 judge gate; tied v9 in A/B, so v9 is preferred for simplicity.",
    "cdsdv": "v11 judge verbatim; too conservative.",
    "cdsdx": "v13 cdsdfi plus rerank; gold/targeted gains but full-setting regressions.",
    "cdsdfb": "v14 bridge recovery; safe smoke test but no gain over cdsdfi and higher calls.",
}

PAPER_METHODS = [
    "CoT",
    "SC",
    "MAD",
    "dMAD",
    "Consensus",
    "CDSD-Soft",
    "CDSD-Select",
    "CDSD-Faithful",
]

IMPLEMENTATION_ONLY = {
    "cdsdn": "v12 claim-decomposition repair without the final answer-type guard.",
    "cdsdnga": "current v12 claim-decomposition repair used in main small-model tables.",
    "cdsdngad": "cdsdnga plus debate-aware final answer selection when a localized debate fails the commit gate.",
}
