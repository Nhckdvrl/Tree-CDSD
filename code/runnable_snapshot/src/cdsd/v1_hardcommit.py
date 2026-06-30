"""CDSD v1 — original: single-claim debate, HARD commit (resolved claim is frozen as a
premise unconditionally). Baseline for the family; tends to amplify a wrong commit."""
from src.llm.client import Tracker
from src.eval.graders import extract_answer
from src.cdsd import components as C


def run(agents, ex, dtype, max_iters=6, debate_rounds=1, **kw):
    client = agents[0].client
    tr = Tracker()
    premises, debated, iters_log = [], [], []
    final, agents_answers = None, []

    for it in range(max_iters):
        raws = [C.gen_claims(a, ex.question, premises, dtype, tr) for a in agents]
        parsed = [C.parse_claims(r) for r in raws]
        agents_claims = [p[0] for p in parsed]
        agents_answers = [p[1] or extract_answer(raws[i], dtype) for i, p in enumerate(parsed)]

        verdict = C.find_conflict(client, tr, ex.question, premises, agents_claims, agents_answers, dtype)
        log = {"iter": it, "answers": agents_answers, "verdict": verdict, "raws": raws}
        if verdict.get("status") == "consensus":
            final = verdict.get("final_answer") or C.majority(agents_answers, dtype)
            iters_log.append(log)
            break

        resolved, _commit, transcript = C.debate_resolve(
            client, tr, agents, ex.question, premises, verdict, dtype, rounds=debate_rounds, soft=False)
        premises.append(resolved)
        debated.append(resolved)
        log["resolved"] = resolved
        log["debate"] = transcript
        iters_log.append(log)

    if final is None:
        final = C.majority(agents_answers, dtype)
    return {"pred": final, "trace": {"premises": premises, "iters": iters_log},
            "n_debates": len(debated), "debated_claims": debated, **tr.as_dict()}
