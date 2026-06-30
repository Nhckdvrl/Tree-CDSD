"""Self-Refine (Madaan et al. 2023): a SINGLE agent iteratively critiques and revises its own
solution — no other agents. This is the control that separates "multiple agents debating" from
"one model just iterating more": same model, same extra calls, but feedback comes from itself.
"""
from src.llm.client import Tracker, ask
from src.eval.graders import extract_answer
from src.util import is_nonanswer
from src.agents.agent import instr

FEEDBACK_SYS = ("You are a careful self-reviewer. You critique a candidate solution and point out "
                "concrete errors (arithmetic slips, misread facts, wrong logic) or confirm it is correct.")
REFINE_SYS = "You revise a solution using the feedback, fixing any real errors."


def run_self_refine(agents, ex, dtype, refine_rounds=2, mad_rounds=None, **kw):
    """One agent: initial CoT, then up to `refine_rounds` of (self-feedback -> revise).
    Budget ~ 1 + 2*rounds calls, comparable to MAD/Consensus at the same round count."""
    if mad_rounds is not None:          # share the round budget knob with the debate baselines
        refine_rounds = mad_rounds
    a = agents[0]
    tr = Tracker()
    ins = instr(dtype)
    q = ex.question
    sol = a.solve_cot(q, dtype, tr)
    history = [sol]
    for r in range(refine_rounds):
        fb = ask(a.client, tr, [{"role": "system", "content": FEEDBACK_SYS},
                                {"role": "user", "content":
                                 f"Problem: {q}\n\nCandidate solution:\n{sol}\n\n"
                                 f"List any concrete errors. If the solution is fully correct, reply "
                                 f"exactly 'NO ERRORS'."}],
                 temperature=0.0, max_tokens=a.max_tokens)
        if "no error" in (fb or "").lower():
            break
        sol = ask(a.client, tr, [{"role": "system", "content": REFINE_SYS},
                                 {"role": "user", "content":
                                  f"Problem: {q}\n\nPrevious solution:\n{sol}\n\nReviewer feedback:\n{fb}\n\n"
                                  f"Produce an improved, corrected solution. {ins}"}],
                  temperature=a.temperature, max_tokens=a.max_tokens)
        history.append(sol)
    pred = extract_answer(sol, dtype)
    return {"pred": pred, "trace": {"history": history}, **tr.as_dict()}
