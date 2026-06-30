"""CDSD v5 — evidence-grounded resolution. Targets the #2 failure mode: when debate fires,
EM is only 37-60% (the resolution step is weak). Same soft structure as v2 but:
  - debate runs 2 rounds (more refinement) instead of 1
  - resolution must CITE the supporting passage/fact (grounded prompt); if nothing supports
    a position, confident=false -> don't commit, fall back to concrete majority.
  - commit cap + oscillation guard: stop committing once we hit `commit_cap` nodes or a new
    node duplicates a prior one (the sanity check showed grounded+2rounds can run away,
    oscillating Tanzania/Niassa/Both... for 6 commits / 48 calls).
This makes commits evidence-backed rather than rhetorical, and keeps v2's guards.
"""
from src.llm.client import Tracker
from src.eval.graders import extract_answer, normalize_qa
from src.util import is_nonanswer
from src.cdsd import components as C


def run(agents, ex, dtype, max_iters=6, debate_rounds=2, final_agents=False, commit_cap=3, **kw):
    client = agents[0].client
    tr = Tracker()
    premises, debated, iters_log = [], [], []
    final, agents_answers = None, []
    committed_norm = set()

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
            final = fa if (fa and not is_nonanswer(fa)) else C.majority_concrete(agents_answers, dtype)
            iters_log.append(log)
            break

        resolved, commit, transcript = C.debate_resolve(
            client, tr, agents, ex.question, premises, verdict, dtype,
            rounds=debate_rounds, soft=True, grounded=True)
        rn = normalize_qa(resolved)
        dup = rn in committed_norm
        log.update({"resolved": resolved, "commit": commit, "dup": dup, "debate": transcript})
        iters_log.append(log)
        if commit and not dup and len(debated) < commit_cap:
            premises.append(resolved)
            debated.append(resolved)
            committed_norm.add(rn)
        else:
            # not confident, oscillating, or cap reached -> settle on concrete majority
            final = C.majority_concrete(agents_answers, dtype)
            break

    if final is None:
        final = C.majority_concrete(agents_answers, dtype)
    if is_nonanswer(final):
        final = C.majority_concrete(agents_answers, dtype) or final
    return {"pred": final, "trace": {"premises": premises, "iters": iters_log},
            "n_debates": len(debated), "debated_claims": debated, **tr.as_dict()}
