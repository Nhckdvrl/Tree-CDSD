"""Consensus-protocol baseline, after "Voting or Consensus? Decision-Making in Multi-Agent Debate"
(Findings ACL 2025). Contrasts with one-shot VOTING (= self-consistency / MAD-majority): N agents
solve, then iteratively revise while seeing each other's solutions, and the debate STOPS as soon as
they reach consensus (all answers agree). If no consensus by max_rounds, fall back to majority.
This isolates the "keep debating until agreement" decision rule as its own baseline.
"""
from collections import Counter

from src.llm.client import Tracker
from src.eval.graders import extract_answer, normalize_qa


def _norm(a, dtype):
    return (a or "") if dtype == "math" else normalize_qa(a)


def _majority(answers, dtype):
    norm = [_norm(a, dtype) for a in answers]
    cnt = Counter([n for n in norm if n])
    if not cnt:
        return answers[0] if answers else ""
    return answers[norm.index(cnt.most_common(1)[0][0])]


def run_consensus(agents, ex, dtype, max_rounds=3, mad_rounds=None, **kw):
    if mad_rounds is not None:
        max_rounds = mad_rounds
    tr = Tracker()
    sols = {a.idx: a.solve_cot(ex.question, dtype, tr) for a in agents}
    history = [dict(sols)]
    rounds_used = 0
    for r in range(max_rounds):
        answers = [extract_answer(sols[a.idx], dtype) for a in agents]
        if len({_norm(a, dtype) for a in answers if a}) <= 1:
            break  # consensus reached
        rounds_used += 1
        new = {}
        for a in agents:
            others = [(i, sols[i]) for i in sols if i != a.idx]
            new[a.idx] = a.mad_update(ex.question, dtype, others, tr)
        sols = new
        history.append(dict(sols))
    answers = [extract_answer(sols[a.idx], dtype) for a in agents]
    norm = {_norm(a, dtype) for a in answers if a}
    return {"pred": _majority(answers, dtype),
            "trace": {"rounds": history, "final_answers": answers,
                      "rounds_used": rounds_used, "consensus": len(norm) <= 1},
            **tr.as_dict()}


def run_consensus_self(agents, ex, dtype, max_rounds=3, mad_rounds=None, **kw):
    """Consensus baseline where revisions include each agent's own prior solution."""
    if mad_rounds is not None:
        max_rounds = mad_rounds
    tr = Tracker()
    sols = {a.idx: a.solve_cot(ex.question, dtype, tr) for a in agents}
    history = [dict(sols)]
    rounds_used = 0
    for _ in range(max_rounds):
        answers = [extract_answer(sols[a.idx], dtype) for a in agents]
        if len({_norm(a, dtype) for a in answers if a}) <= 1:
            break
        rounds_used += 1
        new = {}
        for a in agents:
            others = [(i, sols[i]) for i in sols if i != a.idx]
            new[a.idx] = a.mad_update(ex.question, dtype, others, tr, own=sols[a.idx])
        sols = new
        history.append(dict(sols))
    answers = [extract_answer(sols[a.idx], dtype) for a in agents]
    norm = {_norm(a, dtype) for a in answers if a}
    return {"pred": _majority(answers, dtype),
            "trace": {"rounds": history, "final_answers": answers,
                      "rounds_used": rounds_used, "consensus": len(norm) <= 1,
                      "includes_self": True},
            **tr.as_dict()}
