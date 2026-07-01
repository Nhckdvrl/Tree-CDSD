import re
import string
from collections import Counter


def _last_number(text):
    if not text:
        return None
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace("$", ""))
    return nums[-1].replace(",", "") if nums else None


def _extract_boxed(text):
    """Return the content of the LAST \\boxed{...} with balanced braces, or None."""
    key = "\\boxed"
    start = text.rfind(key)
    if start < 0:
        return None
    i = text.find("{", start)
    if i < 0:
        return None
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
        j += 1
    return None


def _normalize_math(s):
    """Light LaTeX normalization for MATH-style answers (not full symbolic equivalence)."""
    s = str(s or "").strip()
    b = _extract_boxed(s)
    if b is not None:
        s = b
    s = s.replace("π", "\\pi")
    s = re.sub(r"(?<![A-Za-z\\])pi(?![A-Za-z])", r"\\pi", s)
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\(", "").replace("\\)", "").replace("\\[", "").replace("\\]", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\ ", " ").replace("\\;", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = s.replace("\\%", "").replace("%", "").replace("$", "").replace("^\\circ", "")
    s = s.replace(" ", "").replace("\\\\", "").rstrip(".")
    # \frac{a}{b} -> a/b ; remove remaining simple braces
    s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"\1/\2", s)
    s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s)
    s = re.sub(r"√\s*\{?([A-Za-z0-9+\-*/^]+)\}?", r"sqrt(\1)", s)
    s = re.sub(r"(\d)\*sqrt", r"\1sqrt", s)
    s = s.replace("{", "").replace("}", "")
    return s.strip().lower()


def _as_float(s):
    """Parse a plain number or simple a/b fraction string to float, else None."""
    s = s.strip().strip("()")
    try:
        return float(s)
    except Exception:
        pass
    m = re.fullmatch(r"\(?(-?\d+(?:\.\d+)?)\)?/\(?(-?\d+(?:\.\d+)?)\)?", s)
    if m and float(m.group(2)) != 0:
        return float(m.group(1)) / float(m.group(2))
    return None


def _math_equiv(pred, gold):
    p, g = _normalize_math(pred), _normalize_math(gold)
    if p == g and p != "":
        return True
    fp, fg = _as_float(p), _as_float(g)
    if fp is not None and fg is not None:
        return abs(fp - fg) < 1e-4
    return False


def extract_answer(text, dtype):
    """Pull the final answer string out of a model/method output."""
    if not text:
        return ""
    if dtype == "mathx":
        b = _extract_boxed(text)
        if b is not None:
            return b.strip()
        m = re.search(r"answer\s*(?:is|:|=)\s*(.+)", text, re.IGNORECASE)
        if m:
            cand = m.group(1).strip().splitlines()[0].strip().rstrip(".")
            if cand and cand not in ("$", "$$", r"\[", r"\]"):
                return cand.strip("$").strip()
        displays = re.findall(r"\$\$(.*?)\$\$", text, flags=re.DOTALL)
        if displays:
            cand = displays[-1].strip()
            b = _extract_boxed(cand)
            return (b if b is not None else cand).strip()
        # Already-short answer (e.g. a prior extraction like "(3, \\frac{\\pi}{2})" or "p - q"):
        # return it as-is. Only fall back to a bare number for long free-form text.
        t = text.strip()
        if "\n" not in t and len(t) <= 60:
            return t.rstrip(".")
        n = _last_number(text)
        return n if n is not None else ""
    if dtype == "mc":
        m = re.search(r"answer\s*(?:is|:|=)?\s*\(?\*?\s*([A-H])\b", text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        ms = re.findall(r"(?<![A-Za-z])([A-H])(?![A-Za-z])", text)
        return ms[-1].upper() if ms else ""
    m = re.search(r"answer\s*(?:is|:|=)\s*(.+)", text, re.IGNORECASE)
    cand = m.group(1).strip().splitlines()[0].strip() if m else None
    if dtype == "math":
        if cand:
            n = _last_number(cand)
            if n is not None:
                return n
        n = _last_number(text)
        return n if n is not None else ""
    if cand:
        return cand.strip().strip(string.punctuation + " ")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1].strip(string.punctuation + " ") if lines else ""


def _num_eq(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-4
    except Exception:
        return str(a).strip() == str(b).strip()


def normalize_qa(s):
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def em_score(pred, gold):
    return float(normalize_qa(pred) == normalize_qa(gold))


def f1_score(pred, gold):
    p, g = normalize_qa(pred).split(), normalize_qa(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    prec, rec = ns / len(p), ns / len(g)
    return 2 * prec * rec / (prec + rec)


def grade(pred_answer, gold, dtype):
    """pred_answer is already the extracted answer string. Returns correctness metrics."""
    if dtype == "mathx":
        ok = bool(pred_answer) and _math_equiv(pred_answer, gold)
        return {"correct": float(ok), "em": float(ok), "f1": float(ok)}
    if dtype == "math":
        pn = _last_number(pred_answer) if pred_answer else None
        ok = pn is not None and _num_eq(pn, gold)
        return {"correct": float(ok), "em": float(ok), "f1": float(ok)}
    if dtype == "mc":
        ok = bool(pred_answer) and pred_answer.strip().upper()[:1] == str(gold).strip().upper()[:1]
        return {"correct": float(ok), "em": float(ok), "f1": float(ok)}
    em = em_score(pred_answer, gold)
    f1 = f1_score(pred_answer, gold)
    if not em and normalize_qa(gold) and normalize_qa(gold) in normalize_qa(pred_answer):
        em = 1.0  # accept entity containment
    return {"correct": em, "em": em, "f1": f1}
