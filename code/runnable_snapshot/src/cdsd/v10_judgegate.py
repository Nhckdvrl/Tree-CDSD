"""CDSD v10 cdsdk — judge.final_answer as auxiliary signal, gated by agreement with majority.

More conservative than v9: use judge.final_answer ONLY when it agrees (via containment or
exact match) with the majority_specific agent pick. Otherwise treat as suspect and use the
agent pick directly. Trades a slightly higher acceptance of judge synthesis for robustness
against judge picking minority/evasive/paraphrased outputs.
"""
from src.llm.client import Tracker
from src.eval.graders import extract_answer, normalize_qa
from src.util import is_nonanswer
from src.cdsd import components as C


def _judge_agrees(fa, agent_pick, dtype):
    if not fa or not agent_pick or dtype != "qa":
        return False
    a, b = normalize_qa(fa), normalize_qa(agent_pick)
    if not a or not b:
        return False
    return a == b or a in b or b in a


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
            agent_pick = C.majority_specific(agents_answers, dtype)
            fa = verdict.get("final_answer")
            if fa and not is_nonanswer(fa) and _judge_agrees(fa, agent_pick, dtype):
                final = agent_pick if dtype != "qa" else (fa if len(fa) >= len(agent_pick) else agent_pick)
            else:
                final = agent_pick
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
        final = C.majority_concrete(agents_answers, dtype) or final
    return {"pred": final, "trace": {"premises": premises, "iters": iters_log},
            "n_debates": len(debated), "debated_claims": debated, **tr.as_dict()}
