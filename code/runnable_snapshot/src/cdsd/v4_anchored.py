"""CDSD v4 — CoT-anchored. Targets the #1 failure mode found in analysis: 55-76% of
corruptions happen with ZERO debates, i.e. the claim-decomposition majority is worse than a
clean free-form CoT on questions CoT already gets right.

Fix: anchor to free-form CoT.
  Phase A: each agent emits a plain CoT answer (like SC). If unanimous & concrete -> emit (cheap).
  Phase B: otherwise run the v2 soft conflict loop on decomposed claims.
  Decision: if the loop actually COMMITTED a confident premise (debate did real work),
            trust the loop's answer; otherwise fall back to the CoT-anchored majority
            (cot answers + loop answers) so claim-format noise can't corrupt easy cases.
"""
from src.llm.client import Tracker
from src.eval.graders import extract_answer
from src.util import is_nonanswer
from src.cdsd import components as C


def run(agents, ex, dtype, max_iters=6, debate_rounds=1, final_agents=False, **kw):
    client = agents[0].client
    tr = Tracker()

    # --- Phase A: free-form CoT anchors (SC-style) ---
    cot_raws = [a.solve_cot(ex.question, dtype, tr) for a in agents]
    cot_answers = [extract_answer(r, dtype) for r in cot_raws]
    concrete = [a for a in cot_answers if a and not is_nonanswer(a)]
    if concrete and len(concrete) == len(agents) and len({C._norm(a, dtype) for a in concrete}) == 1:
        return {"pred": concrete[0],
                "trace": {"anchor": "cot-unanimous", "cot_answers": cot_answers, "iters": []},
                "n_debates": 0, "debated_claims": [], **tr.as_dict()}

    # --- Phase B: v2 soft conflict loop ---
    premises, debated, iters_log = [], [], []
    final, agents_answers = None, []
    committed_confident = False

    for it in range(max_iters):
        raws = [C.gen_claims(a, ex.question, premises, dtype, tr) for a in agents]
        parsed = [C.parse_claims(r) for r in raws]
        agents_claims = [p[0] for p in parsed]
        agents_answers = [p[1] or extract_answer(raws[i], dtype) for i, p in enumerate(parsed)]

        conc = [a for a in agents_answers if a and not is_nonanswer(a)]
        if conc and len(conc) == len(agents) and len({C._norm(a, dtype) for a in conc}) == 1:
            final = conc[0]
            iters_log.append({"iter": it, "answers": agents_answers, "shortcircuit": True})
            break

        verdict = C.find_conflict(client, tr, ex.question, premises, agents_claims, agents_answers,
                                  dtype, constructive=True)
        log = {"iter": it, "answers": agents_answers, "verdict": verdict}
        if verdict.get("status") == "consensus":
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
            committed_confident = True
        else:
            final = C.majority_concrete(agents_answers, dtype)
            break

    loop_answer = final if final is not None else C.majority_concrete(agents_answers, dtype)

    # --- Decision: trust debate only if it did real work; else anchor to CoT ---
    if committed_confident and loop_answer and not is_nonanswer(loop_answer):
        chosen, anchor = loop_answer, "debate-committed"
    else:
        pool = cot_answers + ([loop_answer] if loop_answer else [])
        chosen, anchor = C.majority_concrete(pool, dtype), "cot-anchored"
    if is_nonanswer(chosen):
        chosen = C.majority_concrete(cot_answers, dtype) or chosen

    return {"pred": chosen,
            "trace": {"anchor": anchor, "cot_answers": cot_answers,
                      "loop_answer": loop_answer, "premises": premises, "iters": iters_log},
            "n_debates": len(debated), "debated_claims": debated, **tr.as_dict()}
