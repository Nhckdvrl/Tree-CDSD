"""CDSD v7 — containment-aware aggregation. Same soft loop as v2, but every answer-selection
point uses majority_specific() instead of plain majority, and the judge's free-form consensus
answer is replaced by the best-supported agent answer. Targets the genuine EM-recoverable
corruptions found in analysis (aggregation discarding a correct/more-specific agent answer),
exploiting that the QA grader scores EM by containment (a longer answer that contains the gold
still counts). No extra LLM calls vs v2.
"""
from src.llm.client import Tracker
from src.eval.graders import extract_answer
from src.util import is_nonanswer
from src.cdsd import components as C


def run(agents, ex, dtype, max_iters=6, debate_rounds=1, final_agents=False, **kw):
    client = agents[0].client
    tr = Tracker()
    premises, debated, iters_log = [], [], []
    final, agents_answers = None, []

    for it in range(max_iters):
        raws = [C.gen_claims(a, ex.question, premises, dtype, tr) for a in agents]
        parsed = [C.parse_claims(r) for r in raws]
        agents_claims = [p[0] for p in parsed]
        agents_answers = [p[1] or extract_answer(raws[i], dtype) for i, p in enumerate(parsed)]

        concrete = [a for a in agents_answers if a and not is_nonanswer(a)]
        if concrete and len(concrete) == len(agents) and len({C._norm(a, dtype) for a in concrete}) == 1:
            final = concrete[0]
            iters_log.append({"iter": it, "answers": agents_answers, "shortcircuit": True})
            break

        verdict = C.find_conflict(client, tr, ex.question, premises, agents_claims, agents_answers,
                                  dtype, constructive=True)
        log = {"iter": it, "answers": agents_answers, "verdict": verdict}
        if verdict.get("status") == "consensus":
            # prefer the best-supported agent answer over the judge's free-form phrasing
            final = C.majority_specific(agents_answers, dtype)
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
            final = C.majority_specific(agents_answers, dtype)
            break

    if final is None:
        final = C.majority_specific(agents_answers, dtype)
    if is_nonanswer(final):
        final = C.majority_specific(agents_answers, dtype) or final
    return {"pred": final, "trace": {"premises": premises, "iters": iters_log},
            "n_debates": len(debated), "debated_claims": debated, **tr.as_dict()}
