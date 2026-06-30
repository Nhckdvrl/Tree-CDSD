"""CDSD v11 cdsdv — judge.final_answer must be VERBATIM one of the agent answers, else
fall back to majority_specific. Strictest gate: judge can only "select", never "synthesize".
"""
from src.llm.client import Tracker
from src.eval.graders import extract_answer, normalize_qa
from src.util import is_nonanswer
from src.cdsd import components as C


def _verbatim_match(fa, agent_answers, dtype):
    """Return the agent answer that judge selected (verbatim, by normalized equality)."""
    if not fa or dtype != "qa":
        return None
    fn = normalize_qa(fa)
    if not fn:
        return None
    for a in agent_answers:
        if a and normalize_qa(a) == fn:
            return a
    return None


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
            fa = verdict.get("final_answer")
            verbatim = _verbatim_match(fa, agents_answers, dtype) if fa and not is_nonanswer(fa) else None
            final = verbatim if verbatim else C.majority_specific(agents_answers, dtype)
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
