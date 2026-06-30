"""CDSD v3 — faithful to the original 8-step spec, with v2's soft fixes:
  1-2 generate + decompose into claim NODES
  3   judge finds the FIRST conflicting node (or consensus)
  4   debate the SEGMENT from the last agreed node to the conflict -> 1-3 agreed nodes
  5/6 mark debated, extend the agreed prefix, CONTINUE reasoning from it
  7   find the next conflict (loop)
  8   emit the answer once no conflict remains
"""
from src.llm.client import Tracker
from src.eval.graders import extract_answer
from src.util import is_nonanswer
from src.cdsd import components as C


def run(agents, ex, dtype, max_iters=6, debate_rounds=1, final_agents=False, **kw):
    client = agents[0].client
    tr = Tracker()
    agreed_prefix, debated, iters_log = [], [], []
    final, agents_answers = None, []

    for it in range(max_iters):
        # step 1-2 (+6: continue from the agreed prefix)
        raws = [C.gen_claims(a, ex.question, agreed_prefix, dtype, tr) for a in agents]
        parsed = [C.parse_claims(r) for r in raws]
        agents_nodes = [p[0] for p in parsed]
        agents_answers = [p[1] or extract_answer(raws[i], dtype) for i, p in enumerate(parsed)]

        concrete = [a for a in agents_answers if a and not is_nonanswer(a)]
        if concrete and len(concrete) == len(agents) and len({C._norm(a, dtype) for a in concrete}) == 1:
            final = concrete[0]
            iters_log.append({"iter": it, "answers": agents_answers, "shortcircuit": True})
            break

        # step 3
        verdict = C.find_conflict(client, tr, ex.question, agreed_prefix, agents_nodes, agents_answers,
                                  dtype, constructive=True)
        log = {"iter": it, "answers": agents_answers, "verdict": verdict}
        if verdict.get("status") == "consensus":  # step 8
            if final_agents:
                final = C.majority_concrete(agents_answers, dtype)
            else:
                fa = verdict.get("final_answer")
                final = fa if (fa and not is_nonanswer(fa)) else C.majority_concrete(agents_answers, dtype)
            iters_log.append(log)
            break

        # step 4: debate the segment -> agreed nodes
        nodes, commit, transcript = C.debate_resolve_segment(
            client, tr, agents, ex.question, agreed_prefix, verdict, agents_nodes, dtype, rounds=debate_rounds)
        log.update({"resolved_nodes": nodes, "commit": commit, "debate": transcript})
        iters_log.append(log)
        if commit:  # step 5 + 6
            agreed_prefix.extend(nodes)
            debated.extend(nodes)
        else:
            final = C.majority_concrete(agents_answers, dtype)
            break

    if final is None:
        final = C.majority_concrete(agents_answers, dtype)
    if is_nonanswer(final):
        final = C.majority_concrete(agents_answers, dtype) or final
    return {"pred": final, "trace": {"agreed_prefix": agreed_prefix, "iters": iters_log},
            "n_debates": len(debated), "debated_claims": debated, **tr.as_dict()}
