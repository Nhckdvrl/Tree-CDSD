from src.llm.client import Tracker


def run_cot(agents, ex, dtype, cot_temperature=0.0, **kw):
    tr = Tracker()
    text = agents[0].solve_cot(ex.question, dtype, tr, temperature=cot_temperature)
    return {"pred": text, "trace": {"solution": text}, **tr.as_dict()}
