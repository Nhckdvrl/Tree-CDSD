"""Liang et al. 2024 (EMNLP), "Encouraging Divergent Thinking in LLMs through Multi-Agent Debate".

Two debaters with OPPOSING stances (affirmative + negative/devil's-advocate) argue across rounds
in a shared transcript; a judge/moderator reads the debate and emits the final answer. The negative
debater is explicitly pushed to disagree to counter "Degeneration-of-Thought" (premature consensus).
Distinct from Du et al. MAD (which is N symmetric agents revising toward agreement).
"""
from src.llm.client import Tracker, ask
from src.eval.graders import extract_answer
from src.util import extract_json
from src.agents.agent import instr

AFF_SYS = ("You are the affirmative debater. Propose and defend a correct, complete solution with "
           "clear step-by-step reasoning.")
NEG_SYS = ("You are the negative debater — a rigorous devil's advocate. Challenge the affirmative "
           "solution: expose arithmetic slips, misread details, wrong facts, or overlooked cases, "
           "and propose your own corrected solution. Do not concede just to agree.")
MOD_SYS = ("You are the moderator. Read the full debate and decide the single correct final answer "
           "on the merits of the arguments.")
# Liang et al.'s judge runs every round in an "extract-or-continue" mode: it ends the debate as
# soon as the arguments support a confident answer, otherwise it lets the debate continue.
JUDGE_SYS = ("You are the debate judge. After each round you decide whether the debate has produced "
             "a clear, well-supported final answer yet. You output only valid JSON. /no_think")


def _judge_round(client, tr, q, debate, ins, dtype, max_tokens):
    out = ask(client, tr, [{"role": "system", "content": JUDGE_SYS},
                           {"role": "user", "content":
                            f"Problem: {q}\n\nDebate so far:\n{debate}\n\n"
                            f"Has the debate reached a clear, well-supported answer? If yes, give it; "
                            f"if the debaters still genuinely disagree or evidence is missing, continue.\n"
                            f"Respond ONLY JSON: {{\"decided\":true|false,\"answer\":\"<final answer or empty>\"}}"}],
              temperature=0.0, max_tokens=max_tokens)
    d = extract_json(out) or {}
    decided = d.get("decided") in (True, "true", "True", "yes", 1)
    return decided, (d.get("answer") or ""), out


def run_divergent_mad(agents, ex, dtype, mad_rounds=2, **kw):
    client = agents[0].client
    tr = Tracker()
    ins = instr(dtype)
    q = ex.question
    mt = agents[0].max_tokens

    aff = ask(client, tr, [{"role": "system", "content": AFF_SYS},
                           {"role": "user", "content": f"{ins}\n\nProblem: {q}"}],
              temperature=0.7, max_tokens=mt)
    transcript = [("affirmative", aff)]
    debate = f"Affirmative:\n{aff}"

    decided_answer, judge_log = None, []
    for r in range(mad_rounds):
        neg = ask(client, tr, [{"role": "system", "content": NEG_SYS},
                               {"role": "user", "content":
                                f"Problem: {q}\n\nDebate so far:\n{debate}\n\n"
                                f"Challenge the latest affirmative solution and give your own. {ins}"}],
                  temperature=0.7, max_tokens=mt)
        transcript.append(("negative", neg))
        debate += f"\n\nNegative:\n{neg}"

        # Judge after each round: stop early if a confident answer has emerged (Liang et al.).
        decided, ans, jraw = _judge_round(client, tr, q, debate, ins, dtype, mt)
        judge_log.append({"round": r, "decided": decided, "answer": ans})
        if decided and ans.strip():
            decided_answer = ans
            break

        if r < mad_rounds - 1:
            aff = ask(client, tr, [{"role": "system", "content": AFF_SYS},
                                   {"role": "user", "content":
                                    f"Problem: {q}\n\nDebate so far:\n{debate}\n\n"
                                    f"Respond to the negative debater; defend or revise your solution. {ins}"}],
                      temperature=0.7, max_tokens=mt)
            transcript.append(("affirmative", aff))
            debate += f"\n\nAffirmative:\n{aff}"

    if decided_answer is not None:
        pred = extract_answer(decided_answer, dtype) or decided_answer
        verdict = decided_answer
    else:
        verdict = ask(client, tr, [{"role": "system", "content": MOD_SYS},
                                   {"role": "user", "content":
                                    f"Problem: {q}\n\nFull debate:\n{debate}\n\n"
                                    f"Decide the correct final answer. {ins}"}],
                      temperature=0.0, max_tokens=mt)
        pred = extract_answer(verdict, dtype)
    return {"pred": pred,
            "trace": {"transcript": transcript, "verdict": verdict, "judge": judge_log,
                      "early_stop": decided_answer is not None},
            **tr.as_dict()}
