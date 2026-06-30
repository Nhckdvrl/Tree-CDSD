"""CDSD v2 — soft: single-claim debate but CONFIDENCE-GATED commit, non-answer guard +
concrete fallback, consensus short-circuit, stop-on-unresolved. Fixes v1's error amplification."""
from src.llm.client import Tracker
from src.util import is_nonanswer
from src.cdsd import components as C


def run(agents, ex, dtype, max_iters=6, debate_rounds=1, final_agents=False, **kw):
    client = agents[0].client
    tr = Tracker()
    premises, debated, iters_log = [], [], []
    final, agents_answers = None, []

    for it in range(max_iters):
        generated = [
            C.gen_structured_nodes(
                a, ex.question, premises, dtype, tr,
                prefer_extractor_answer=True)
            for a in agents
        ]
        agents_node_payloads = [g[0] for g in generated]
        agents_claims = [C.node_claims(nodes) for nodes in agents_node_payloads]
        agents_answers = [g[1] for g in generated]
        node_meta = [g[2] for g in generated]

        concrete = [a for a in agents_answers if a and not is_nonanswer(a)]
        if concrete and len(concrete) == len(agents) and len({C._norm(a, dtype) for a in concrete}) == 1:
            final = concrete[0]
            iters_log.append({"iter": it, "answers": agents_answers, "shortcircuit": True})
            break

        verdict = C.find_conflict(client, tr, ex.question, premises, agents_claims, agents_answers,
                                  dtype, constructive=True)
        log = {
            "iter": it,
            "answers": agents_answers,
            "node_payloads": agents_node_payloads,
            "node_quality": [m.get("quality", {}) for m in node_meta],
            "node_schema_ok": [bool(m.get("schema_ok")) for m in node_meta],
            "node_generation": node_meta,
            "verdict": verdict,
        }
        if verdict.get("status") == "consensus":
            if final_agents:
                final = C.majority_concrete(agents_answers, dtype)
            else:
                fa = verdict.get("final_answer")
                final = fa if (fa and not is_nonanswer(fa)) else C.majority_concrete(agents_answers, dtype)
            iters_log.append(log)
            break

        resolved, commit, transcript = C.debate_resolve(
            client, tr, agents, ex.question, premises, verdict, dtype, rounds=debate_rounds, soft=True)
        log.update({"resolved": resolved, "commit": commit, "debate": transcript})
        iters_log.append(log)
        if commit:
            premises.append(resolved)
            debated.append(resolved)
        else:
            final = C.majority_concrete(agents_answers, dtype)
            break

    if final is None:
        final = C.majority_concrete(agents_answers, dtype)
    if is_nonanswer(final):
        final = C.majority_concrete(agents_answers, dtype) or final
    return {"pred": final, "trace": {"premises": premises, "iters": iters_log},
            "n_debates": len(debated), "debated_claims": debated, **tr.as_dict()}
