import json
import re
import time
from concurrent.futures import ThreadPoolExecutor


def now():
    return time.time()


def extract_json(text):
    """Extract the first balanced JSON object from text.

    Tolerates ```json fences, Qwen3 empty thinking tags, and common LaTeX commands
    emitted with single backslashes inside JSON strings.
    """
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.IGNORECASE | re.DOTALL).strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    start = t.find("{")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(t)):
        c = t[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    chunk = t[start:i + 1]
                    latex_safe = re.sub(
                        r"\\+(?!u[0-9a-fA-F]{4})(?=[A-Za-z])",
                        r"\\\\",
                        chunk,
                    )
                    latex_safe = re.sub(
                        r"\\+(?![\"\\/bfnrtu])",
                        r"\\\\",
                        latex_safe,
                    )
                    for cand in (chunk, chunk.replace("\n", " "),
                                 latex_safe, latex_safe.replace("\n", " ")):
                        try:
                            return json.loads(cand)
                        except Exception:
                            continue
                    return None
    return None


def extract_after(text, label):
    """Return the text after `label:` (single line); fall back to the last non-empty line."""
    if not text:
        return ""
    m = re.search(label + r"\s*[:=]\s*(.+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().splitlines()[0].strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else ""


_NONANSWER_RE = re.compile(
    r"cannot (be )?determin|insufficient|no (information|info|data|relevant)|"
    r"not (provided|enough|specified|available|stated|mentioned)|unknown|"
    r"cannot (identify|establish|find|conclude|be found)|unable to|"
    r"does not (provide|establish|specify|mention|indicate|contain)|"
    r"there is no (information|mention|indication|evidence)|"
    r"can'?t be (determined|established|identified)",
    re.IGNORECASE)


def is_nonanswer(s):
    """True if the string is empty or an evasive 'cannot determine / insufficient info' answer."""
    s = (s or "").strip()
    return (not s) or bool(_NONANSWER_RE.search(s))


def parallel_map(fn, items, workers=8):
    items = list(items)
    if workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))
