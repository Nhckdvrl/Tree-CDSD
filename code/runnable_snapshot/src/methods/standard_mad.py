from collections import Counter

from src.llm.client import Tracker
from src.eval.graders import extract_answer, normalize_qa


def _majority(answers, dtype):
    norm = [(a or "") if dtype == "math" else normalize_qa(a) for a in answers]
    cnt = Counter([n for n in norm if n])
    if not cnt:
        return answers[0] if answers else ""
    return answers[norm.index(cnt.most_common(1)[0][0])]


def run_standard_mad(agents, ex, dtype, mad_rounds=2, **kw):
    """Du et al. 2024: independent round 0, then `mad_rounds` of answer-level debate."""
    tr = Tracker()
    sols = {a.idx: a.solve_cot(ex.question, dtype, tr) for a in agents}
    history = [dict(sols)]
    for _ in range(mad_rounds):
        new = {}
        for a in agents:
            others = [(i, sols[i]) for i in sols if i != a.idx]
            new[a.idx] = a.mad_update(ex.question, dtype, others, tr)
        sols = new
        history.append(dict(sols))
    answers = [extract_answer(sols[a.idx], dtype) for a in agents]
    return {"pred": _majority(answers, dtype),
            "trace": {"rounds": history, "final_answers": answers}, **tr.as_dict()}


def run_standard_mad_self(agents, ex, dtype, mad_rounds=2, **kw):
    """MAD variant where each solver also sees its own previous solution when revising."""
    tr = Tracker()
    sols = {a.idx: a.solve_cot(ex.question, dtype, tr) for a in agents}
    history = [dict(sols)]
    for _ in range(mad_rounds):
        new = {}
        for a in agents:
            others = [(i, sols[i]) for i in sols if i != a.idx]
            new[a.idx] = a.mad_update(ex.question, dtype, others, tr, own=sols[a.idx])
        sols = new
        history.append(dict(sols))
    answers = [extract_answer(sols[a.idx], dtype) for a in agents]
    return {"pred": _majority(answers, dtype),
            "trace": {"rounds": history, "final_answers": answers, "includes_self": True},
            **tr.as_dict()}
