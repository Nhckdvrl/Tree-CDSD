from src.llm.client import ask

MATH_INSTR = "Solve the problem step by step. End with a line exactly: 'Answer: <number>'."
QA_INSTR = ("Answer the question. Reason step by step using intermediate facts, "
            "then end with a line exactly: 'Answer: <short answer>'.")
MC_INSTR = ("Reason step by step, then choose the correct option. "
            "End with a line exactly: 'Answer: <letter A/B/C/D>'.")


MATHX_INSTR = ("Solve the competition math problem step by step. "
               "End with your final answer in \\boxed{...}.")


def instr(dtype):
    if dtype == "mathx":
        return MATHX_INSTR
    if dtype == "math":
        return MATH_INSTR
    if dtype == "mc":
        return MC_INSTR
    return QA_INSTR


class Agent:
    """A homogeneous LLM agent differentiated only by persona (system prompt) + temperature.

    Baseline-facing methods live here; CDSD-specific generation/argue helpers live in
    src/cdsd/components.py (they read persona/temperature/client off this object).
    """

    def __init__(self, idx, client, persona, temperature=0.7, max_tokens=1024):
        self.idx = idx
        self.client = client
        self.persona = persona
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _sys(self):
        return {"role": "system", "content": self.persona}

    def solve_cot(self, question, dtype, tr, temperature=None):
        user = f"{instr(dtype)}\n\nProblem: {question}"
        t = self.temperature if temperature is None else temperature
        return ask(self.client, tr, [self._sys(), {"role": "user", "content": user}],
                   temperature=t, max_tokens=self.max_tokens)

    def mad_update(self, question, dtype, others, tr, own=None):
        joined = "\n\n".join(f"Solver {i} solution:\n{s}" for i, s in others)
        if own is None:
            user = (
                f"Problem: {question}\n\n"
                f"These are solutions from other solvers:\n{joined}\n\n"
                f"Using their solutions as additional information, critically re-examine the problem and "
                f"give your own updated step-by-step solution. {instr(dtype)}"
            )
        else:
            user = (
                f"Problem: {question}\n\n"
                f"Your previous solution:\n{own}\n\n"
                f"Solutions from other solvers:\n{joined}\n\n"
                f"Using both your previous reasoning and the other solvers' solutions, critically "
                f"re-examine the problem and give your updated step-by-step solution. {instr(dtype)}"
            )
        return ask(self.client, tr, [self._sys(), {"role": "user", "content": user}],
                   temperature=self.temperature, max_tokens=self.max_tokens)
