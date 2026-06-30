from collections import Counter

from src.llm.client import Tracker
from src.eval.graders import extract_answer, normalize_qa


def _norm(ans, dtype):
    return (ans or "") if dtype == "math" else normalize_qa(ans)


SC_BUDGET_K = 10  # budget-matched SC sample count, ~ CDSD's average calls/question


def run_self_consistency_budget(agents, ex, dtype, sc_temperature=0.8, **kw):
    """Budget-matched SC: same vote rule but with SC_BUDGET_K samples (~CDSD's calls/question), so
    an accuracy gain over this isn't just 'CDSD spent more compute'. Forces k=SC_BUDGET_K."""
    kw.pop("sc_samples", None)
    return run_self_consistency(agents, ex, dtype, sc_samples=SC_BUDGET_K,
                                sc_temperature=sc_temperature, **kw)


def run_self_consistency(agents, ex, dtype, sc_samples=3, sc_temperature=0.8, **kw):
    tr = Tracker()
    sols = [agents[k % len(agents)].solve_cot(ex.question, dtype, tr, temperature=sc_temperature)
            for k in range(sc_samples)]
    answers = [extract_answer(s, dtype) for s in sols]
    norm = [_norm(a, dtype) for a in answers]
    cnt = Counter([n for n in norm if n])
    if cnt:
        winner = cnt.most_common(1)[0][0]
        pred = answers[norm.index(winner)]
    else:
        pred = answers[0] if answers else ""
    return {"pred": pred, "trace": {"solutions": sols, "answers": answers}, **tr.as_dict()}
