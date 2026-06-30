"""CDSD v6 — v2 + answer canonicalization. Targets the EM-granularity artifacts found in the
corruption analysis (~half of "corruptions" are paraphrase/over- or under-specified spans, e.g.
'1978' vs '7 October 1978', 'Portugal' vs 'Lisbon District', long descriptive vs short span).

Mechanism: run the v2 soft loop unchanged, then one cheap deterministic call normalizes the
settled answer to the shortest exact phrase the question asks for. QA only (math/mc unchanged).
"""
from src.cdsd import components as C
from src.cdsd.v2_soft import run as run_v2
from src.llm.client import Tracker


def run(agents, ex, dtype, max_iters=6, debate_rounds=1, final_agents=False, **kw):
    res = run_v2(agents, ex, dtype, max_iters=max_iters, debate_rounds=debate_rounds,
                 final_agents=final_agents, **kw)
    if dtype != "qa":
        return res
    client = agents[0].client
    tr = Tracker()
    canon = C.canonicalize(client, tr, ex.question, res["pred"], dtype)
    res["trace"]["canonicalized"] = {"from": res["pred"], "to": canon}
    res["pred"] = canon
    # fold the extra call into the cost accounting
    extra = tr.as_dict()
    res["calls"] = res.get("calls", 0) + extra.get("calls", 0)
    res["prompt_tokens"] = res.get("prompt_tokens", 0) + extra.get("prompt_tokens", 0)
    res["completion_tokens"] = res.get("completion_tokens", 0) + extra.get("completion_tokens", 0)
    return res
