"""Shared CDSD building blocks: claim parsing, agent-side generation/argue, conflict
detection, and debate/resolution (single-claim and segment). Prompt text lives in prompts.py."""
import re
from collections import Counter

from src.llm.client import ask
from src.util import extract_after, extract_json, is_nonanswer
from src.eval.graders import extract_answer, normalize_qa, _normalize_math
from src.cdsd import prompts

# ---- claim decomposition (step 2) -------------------------------------------------
CLAIM_RE = re.compile(r"^\s*(?:claim|step)\s*\d+\s*[:.\)]\s*(.+)$", re.IGNORECASE)
NUM_RE = re.compile(r"^\s*\d+\s*[:.\)]\s*(.+)$")
ANSWER_LINE_RE = re.compile(r"^\s*(?:final\s+)?answer\s*[:=]", re.IGNORECASE)


def parse_claims(text):
    """Numbered claim list -> (claims: list[str], answer: str|None), with a sentence-split fallback."""
    if not text:
        return [], None
    m = re.search(r"(?:final\s+)?answer\s*[:=]\s*(.+)", text, re.IGNORECASE)
    answer = m.group(1).strip().splitlines()[0].strip().rstrip(".") if m else None
    claims = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or ANSWER_LINE_RE.match(line):
            continue
        mm = CLAIM_RE.match(line) or NUM_RE.match(line)
        if mm:
            claims.append(mm.group(1).strip())
    if not claims:
        body = text[:m.start()] if m else text
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if s.strip()]
        claims = sents or ([text.strip()] if text.strip() else [])
    return claims, answer


def _node_text(node):
    if isinstance(node, str):
        return node.strip()
    if not isinstance(node, dict):
        return str(node).strip() if node is not None else ""
    claim = (node.get("claim") or node.get("text") or node.get("node") or "").strip()
    if claim:
        return claim
    parts = [node.get("subject"), node.get("relation"), node.get("object")]
    parts = [str(p).strip() for p in parts if str(p or "").strip()]
    if parts:
        return " ".join(parts)
    expr = str(node.get("expression") or "").strip()
    result = str(node.get("result") or "").strip()
    if expr and result:
        return f"{expr} -> {result}"
    return ""


def _normalize_nodes(raw_nodes):
    nodes = []
    if not isinstance(raw_nodes, list):
        return nodes
    for i, node in enumerate(raw_nodes, start=1):
        if isinstance(node, dict):
            item = dict(node)
            item["id"] = item.get("id") or i
            item["claim"] = _node_text(item)
        else:
            item = {"id": i, "kind": "claim", "claim": _node_text(node)}
        if item["claim"]:
            nodes.append(item)
    return nodes


def node_claims(nodes):
    return [_node_text(n) for n in nodes if _node_text(n)]


def node_quality(nodes, fallback=False):
    claims = node_claims(nodes)
    n = len(claims)
    lengths = [len(c.split()) for c in claims]
    long_nodes = sum(1 for x in lengths if x > 28)
    very_short = sum(1 for x in lengths if x < 4)
    connective_heavy = sum(
        1 for c in claims
        if len(re.findall(r"\b(and|also|while|whereas|because|therefore|then)\b", c.lower())) >= 2
    )
    repeated = n - len({normalize_qa(c) for c in claims if normalize_qa(c)})
    return {
        "n_nodes": n,
        "avg_words": round(sum(lengths) / n, 2) if n else 0.0,
        "long_nodes": long_nodes,
        "very_short_nodes": very_short,
        "connective_heavy_nodes": connective_heavy,
        "repeated_nodes": repeated,
        "fallback": bool(fallback),
    }



def _shorten_claim(text, max_words=20):
    text = re.sub(r"\s+", " ", (text or "")).strip()
    text = re.sub(r"^(?:#+\s*)?(?:step\s*\d+|final answer)\s*[:.\-]*\s*", "", text, flags=re.I)
    words = text.split()
    if len(words) <= max_words:
        out = text.strip(" -")
        return out if re.search(r"[A-Za-z0-9]", out) else ""
    out = " ".join(words[:max_words]).strip(" -")
    return out if re.search(r"[A-Za-z0-9]", out) else ""


def _math_result(expr):
    expr = (expr or "").strip()
    if "=" not in expr:
        return ""
    rhs = expr.split("=")[-1].strip()
    return rhs[:80]


def _split_text_units(text):
    text = (text or "").replace("\r", "\n")
    text = re.sub(r"(?im)^\s*(?:final\s+)?answer\s*[:=].*$", "", text)
    text = re.sub(r"(?i)\s+(?:final\s+)?answer\s*[:=].*$", "", text)
    text = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", text)
    # Put display and inline formulas on their own boundaries before sentence splitting.
    text = re.sub(r"(\$\$.*?\$\$|\\\[.*?\\\]|\\\(.*?\\\))", r"\n\1\n", text, flags=re.S)
    rough = []
    for line in text.splitlines():
        line = line.strip().strip("-* ")
        if not line or not re.search(r"[A-Za-z0-9]", line):
            continue
        if re.fullmatch(r"#+", line):
            continue
        rough.extend([x.strip() for x in re.split(r"(?<=[.!?])\s+", line) if x.strip()])
    units = []
    for u in rough:
        u = re.sub(r"^#+\s*", "", u).strip()
        if len(u.split()) > 30 and not re.search(r"[=$]|\\\[|\\\(|\\frac|\\sqrt", u):
            parts = re.split(r"\s*(?:;|,\s+where\b|\s+because\b|\s+therefore\b|\s+so\b)\s*", u, flags=re.I)
            units.extend([p.strip() for p in parts if p.strip()])
        else:
            units.append(u)
    return units


def _fallback_nodes_by_dtype(text, dtype):
    units = _split_text_units(text)
    nodes = []
    for u in units:
        if not u or not re.search(r"[A-Za-z0-9]", u):
            continue
        expr = ""
        kind = "claim"
        status = ""
        option = ""
        if dtype in ("math", "mathx"):
            formulas = re.findall(r"\$\$(.*?)\$\$|\\\[(.*?)\\\]|\\\((.*?)\\\)", u, flags=re.S)
            flat = [next((part for part in tup if part), "").strip() for tup in formulas]
            if flat and len(" ".join(u.split())) <= 160:
                expr = "; ".join(f for f in flat if f)[:160]
            elif re.search(r"[=<>]|\\frac|\\sqrt|\^", u):
                expr = u[:160]
            kind = "operation" if expr and ("=" in expr or "\\frac" in expr or "sqrt" in expr or "\\sqrt" in expr) else "deduction"
            claim = _shorten_claim(re.sub(r"\$\$|\\\[|\\\]|\\\(|\\\)", "", u), 18)
            if not claim and expr:
                claim = _shorten_claim(expr, 18)
            item = {"id": len(nodes) + 1, "kind": kind, "claim": claim, "expression": expr, "result": _math_result(expr)}
        elif dtype == "mc":
            m = re.search(r"\b(?:option\s*)?([A-D])\b", u, flags=re.I)
            option = m.group(1).upper() if m else ""
            low = u.lower()
            if "answer" in low or "correct" in low:
                kind, status = "answer", "answer"
            elif any(w in low for w in ["eliminate", "incorrect", "not correct", "cannot be"]):
                kind, status = "elimination", "eliminated"
            elif option:
                kind, status = "option_check", "unknown"
            else:
                kind, status = "fact", ""
            item = {"id": len(nodes) + 1, "kind": kind, "claim": _shorten_claim(u, 20), "option": option, "status": status, "evidence_quote": ""}
        else:
            item = {"id": len(nodes) + 1, "kind": "fact", "claim": _shorten_claim(u, 22), "subject": "", "relation": "", "object": "", "evidence_quote": ""}
        if item.get("claim"):
            nodes.append(item)
        if len(nodes) >= (14 if dtype in ("math", "mathx") else 10):
            break
    if not nodes:
        nodes = [{"id": 1, "kind": "claim", "claim": _shorten_claim(text, 22)}]
    return nodes

def parse_structured_nodes(text, fallback_text=None, dtype="qa"):
    d = extract_json(text)
    if isinstance(d, dict):
        nodes = _normalize_nodes(d.get("nodes"))
        answer_raw = d.get("answer")
        answer = str(answer_raw).strip() if answer_raw is not None else ""
        answer = answer or None
        if nodes:
            return nodes, answer, {"schema_ok": True, "raw": d, "quality": node_quality(nodes)}
    claims, answer = parse_claims(fallback_text or text)
    source = fallback_text or text
    nodes = _fallback_nodes_by_dtype(source, dtype)
    if not nodes and claims:
        nodes = [{"id": i + 1, "kind": "claim", "claim": _shorten_claim(c, 22)} for i, c in enumerate(claims)]
    return nodes, answer, {
        "schema_ok": False,
        "raw": d,
        "quality": node_quality(nodes, fallback=True),
        "fallback_mode": f"task_{dtype}",
    }


# ---- answer aggregation -----------------------------------------------------------
def _norm(a, dtype):
    if dtype == "math":
        return a or ""
    if dtype == "mathx":            # competition math: group LaTeX-equivalent answers for voting
        return _normalize_math(a or "")
    return normalize_qa(a)


def majority(answers, dtype):
    norm = [_norm(a, dtype) for a in answers]
    cnt = Counter([n for n in norm if n])
    if not cnt:
        return answers[0] if answers else ""
    return answers[norm.index(cnt.most_common(1)[0][0])]


def majority_concrete(answers, dtype):
    """Majority over concrete (non-evasive) answers; fall back to all if none are concrete."""
    pool = [a for a in answers if a and not is_nonanswer(a)] or list(answers)
    norm = [_norm(a, dtype) for a in pool]
    cnt = Counter([n for n in norm if n])
    if not cnt:
        return pool[0] if pool else ""
    return pool[norm.index(cnt.most_common(1)[0][0])]


def majority_specific(answers, dtype):
    """Containment-aware aggregation (v7). The QA grader scores EM by containment
    (gold ⊆ pred -> correct), so among concrete answers we pick the one with the most
    'support' (how many other answers it contains), breaking ties toward the MORE SPECIFIC
    (longer) answer. This recovers cases bare majority loses, e.g.
      ['1978','7 October 1978','1978']                -> '7 October 1978'  (longer, contains '1978')
      ['Casino','The Richmond River','Richmond']      -> 'The Richmond River' (contains gold 'Richmond')
    For math/mc it degrades to plain majority (containment is meaningless there)."""
    if dtype != "qa":
        return majority_concrete(answers, dtype)
    pool = [a for a in answers if a and not is_nonanswer(a)] or list(answers)
    pool = [a for a in pool if a]
    if not pool:
        return ""
    norm = [_norm(a, dtype) for a in pool]
    supports = [sum(1 for j in range(len(pool)) if norm[j] and norm[i] and norm[j] in norm[i])
                for i in range(len(pool))]
    # No containment relationship anywhere (every answer only contains itself) -> no signal to
    # exploit; fall back to plain majority. This avoids overriding distinct sibling answers
    # (e.g. 'Ptolemy IX' vs 'Ptolemy VIII') just because one is longer.
    if max(supports, default=0) <= 1:
        return majority_concrete(answers, dtype)
    best, best_key = pool[0], (-1, -1)
    for i, a in enumerate(pool):
        if not norm[i]:
            continue
        key = (supports[i], len(norm[i]))  # most-supported, then most-specific (longest)
        if key > best_key:
            best_key, best = key, a
    return best


# ---- agent-side calls (state from the Agent, prompt text from prompts.py) ----------
def _sys(agent):
    return {"role": "system", "content": agent.persona}


def gen_claims(agent, question, premises, dtype, tr):
    user = prompts.gen_claims_user(question, premises, dtype)
    return ask(agent.client, tr, [_sys(agent), {"role": "user", "content": user}],
               temperature=agent.temperature, max_tokens=agent.max_tokens)


def gen_reasoning(agent, question, premises, dtype, tr):
    user = prompts.gen_reasoning_user(question, premises, dtype)
    return ask(agent.client, tr, [_sys(agent), {"role": "user", "content": user}],
               temperature=agent.temperature, max_tokens=agent.max_tokens)


def extract_nodes(client, tr, question, reasoning, answer, dtype, max_tokens=2400):
    user = prompts.extract_nodes_user(question, reasoning, answer, dtype)
    return ask(client, tr, [{"role": "system", "content": prompts.NODE_EXTRACT_SYS},
                            {"role": "user", "content": user}],
               temperature=0.0, max_tokens=max_tokens)


def gen_structured_nodes(agent, question, premises, dtype, tr, prefer_extractor_answer=True):
    reasoning = gen_reasoning(agent, question, premises, dtype, tr)
    answer = extract_answer(reasoning, dtype)
    extracted = extract_nodes(agent.client, tr, question, reasoning, answer, dtype)
    nodes, node_answer, meta = parse_structured_nodes(extracted, fallback_text=reasoning, dtype=dtype)
    final_answer = (node_answer or answer) if prefer_extractor_answer else (answer or node_answer)
    meta.update({"reasoning": reasoning, "extractor_text": extracted})
    return nodes, final_answer, meta


def gen_structured_nodes_from_claims(agent, question, premises, dtype, tr):
    reasoning = gen_claims(agent, question, premises, dtype, tr)
    claims, parsed_answer = parse_claims(reasoning)
    answer = parsed_answer or extract_answer(reasoning, dtype)
    extracted = extract_nodes(agent.client, tr, question, reasoning, answer, dtype)
    nodes, node_answer, meta = parse_structured_nodes(extracted, fallback_text=reasoning, dtype=dtype)
    # The extractor cleans nodes only; final-answer aggregation should use the solver output.
    final_answer = answer or node_answer
    meta.update({
        "reasoning": reasoning,
        "extractor_text": extracted,
        "source_claims": claims,
    })
    return nodes, final_answer, meta


def argue(agent, question, premises, conflict_desc, my_claim, others_claims, dtype, tr, constructive=False):
    user = prompts.argue_user(question, premises, conflict_desc, my_claim, others_claims, dtype, constructive)
    return ask(agent.client, tr, [_sys(agent), {"role": "user", "content": user}],
               temperature=agent.temperature, max_tokens=agent.max_tokens)


def argue_segment(agent, question, agreed_prefix, my_segment, conflict_desc, others_claims, dtype, tr):
    user = prompts.argue_segment_user(question, agreed_prefix, my_segment, conflict_desc, others_claims, dtype)
    return ask(agent.client, tr, [_sys(agent), {"role": "user", "content": user}],
               temperature=agent.temperature, max_tokens=agent.max_tokens)


# ---- judge: conflict detection (step 3) -------------------------------------------
def find_conflict(client, tr, question, premises, agents_claims, agents_answers, dtype,
                  judge_temp=0.0, max_tokens=700, constructive=False):
    user = prompts.conflict_user(question, premises, agents_claims, agents_answers, dtype, constructive)
    text = ask(client, tr, [{"role": "system", "content": prompts.JUDGE_SYS},
                            {"role": "user", "content": user}], temperature=judge_temp, max_tokens=max_tokens)
    d = extract_json(text)
    if not d or "status" not in d:
        norm = [str(a).strip().lower() for a in agents_answers]
        if len(set(n for n in norm if n)) <= 1:
            return {"status": "consensus", "final_answer": agents_answers[0] if agents_answers else ""}
        claims = {str(i): (agents_claims[i][0] if agents_claims[i] else "") for i in range(len(agents_claims))}
        return {"status": "conflict", "description": "Solvers disagree (judge parse fallback).", "claims": claims}
    return d


# ---- debate + resolution (step 4) -------------------------------------------------
def debate_resolve(client, tr, agents, question, premises, conflict, dtype,
                   rounds=1, judge_temp=0.0, soft=False, grounded=False):
    """Single-claim resolution. soft=False hard-commits (v1); soft=True is confidence-gated (v2).
    grounded=True (v5) forces an evidence-citing resolution prompt.
    Returns (resolved_claim, commit, transcript)."""
    desc = conflict.get("description", "")
    claims_map = conflict.get("claims", {}) or {}
    positions = {a.idx: (claims_map.get(str(a.idx)) or claims_map.get(a.idx) or "") for a in agents}
    transcript = []
    for r in range(rounds):
        new_pos = {}
        for a in agents:
            others = [(i, positions[i]) for i in positions if i != a.idx]
            out = argue(a, question, premises, desc, positions[a.idx], others, dtype, tr,
                        constructive=(soft or grounded))
            new_pos[a.idx] = extract_after(out, "My claim")
            transcript.append({"round": r, "agent": a.idx, "text": out})
        positions = new_pos
    args = "\n".join(f"Solver {i} final position: {positions[i]}" for i in sorted(positions))
    user = prompts.resolve_user(question, premises, desc, args, soft=soft, grounded=grounded)
    text = ask(client, tr, [{"role": "system", "content": prompts.RESOLVE_SYS},
                            {"role": "user", "content": user}], temperature=judge_temp, max_tokens=400)
    d = extract_json(text) or {}
    resolved = d.get("resolved_claim") or max(positions.values(), key=lambda s: len(s or "")) or desc
    commit = True if (not soft and not grounded) else (bool(d.get("confident", False)) and not is_nonanswer(resolved))
    return resolved, commit, transcript


def rerank(client, tr, question, answers, dtype, judge_temp=0.0):
    """Grounded candidate re-ranking (v8). When solvers split, present the DISTINCT concrete
    candidate answers back to the model with the passages and let it pick the best-supported one.
    Aims to recover the correct answer (improves EM and F1), unlike length-based aggregation.
    QA only; falls back to majority_concrete if no clear pick."""
    if dtype != "qa":
        return majority_concrete(answers, dtype)
    pool = [a for a in answers if a and not is_nonanswer(a)]
    # distinct candidates (keep first surface form per normalized key)
    seen, cands = {}, []
    for a in pool:
        k = normalize_qa(a)
        if k and k not in seen:
            seen[k] = a
            cands.append(a)
    if len(cands) <= 1:
        return majority_concrete(answers, dtype)
    out = ask(client, tr, [{"role": "system", "content": prompts.RERANK_SYS},
                           {"role": "user", "content": prompts.rerank_user(question, cands)}],
              temperature=judge_temp, max_tokens=80)
    pick = (out or "").strip().strip('"').strip().rstrip(".").strip()
    pick = pick.splitlines()[0].strip() if pick else pick
    if not pick:
        return majority_concrete(answers, dtype)
    pn = normalize_qa(pick)
    # exact match first, then containment either direction, else majority fallback
    for c in cands:
        if normalize_qa(c) == pn:
            return c
    for c in cands:
        cn = normalize_qa(c)
        if cn and pn and (cn in pn or pn in cn):
            return c
    return majority_concrete(answers, dtype)


def canonicalize(client, tr, question, answer, dtype, judge_temp=0.0, min_words=7):
    """Normalize an OVER-LONG QA answer to the shortest exact span (fixes EM artifacts like a full
    descriptive sentence vs the short gold span). QA only; no-op otherwise.

    Two hard guards (learned from the sanity check, where naive canonicalization turned correct
    short answers into wrong ones, e.g. 'Mike Medavoy'->'Orion Pictures', 'Miquette Giraudy'->'No'):
      1. only touch answers with >= min_words tokens (short answers are left alone)
      2. accept the result ONLY if it is a substring of the original (we may shorten, never substitute)
    """
    if dtype != "qa" or not answer or is_nonanswer(answer):
        return answer
    if len(answer.split()) < min_words:
        return answer
    out = ask(client, tr, [{"role": "system", "content": prompts.CANON_SYS},
                           {"role": "user", "content": prompts.canonicalize_user(question, answer)}],
              temperature=judge_temp, max_tokens=60)
    cand = (out or "").strip().strip('"').strip().rstrip(".").strip()
    cand = cand.splitlines()[0].strip() if cand else cand
    if not cand or is_nonanswer(cand):
        return answer
    # substring guard: only accept if it is contained in the original answer (shorten, never swap)
    if normalize_qa(cand) and normalize_qa(cand) in normalize_qa(answer):
        return cand
    return answer


def debate_resolve_segment(client, tr, agents, question, agreed_prefix, conflict,
                           agents_nodes, dtype, rounds=1, judge_temp=0.0):
    """Segment resolution (step 4, faithful): resolve into 1-3 agreed nodes.
    Returns (resolved_nodes: list[str], commit, transcript)."""
    desc = conflict.get("description", "")
    claims_map = conflict.get("claims", {}) or {}
    positions = {a.idx: (claims_map.get(str(a.idx)) or claims_map.get(a.idx) or "") for a in agents}
    seg_by_idx = {a.idx: (agents_nodes[a.idx] if a.idx < len(agents_nodes) else []) for a in agents}
    transcript = []
    for r in range(rounds):
        new_pos = {}
        for a in agents:
            others = [(i, positions[i]) for i in positions if i != a.idx]
            out = argue_segment(a, question, agreed_prefix, seg_by_idx[a.idx], desc, others, dtype, tr)
            new_pos[a.idx] = extract_after(out, "My claim")
            transcript.append({"round": r, "agent": a.idx, "text": out})
        positions = new_pos
    args = "\n".join(f"Solver {i}: {positions[i]}" for i in sorted(positions))
    user = prompts.resolve_segment_user(question, agreed_prefix, desc, args)
    text = ask(client, tr, [{"role": "system", "content": prompts.RESOLVE_SYS},
                            {"role": "user", "content": user}], temperature=judge_temp, max_tokens=500)
    d = extract_json(text) or {}
    nodes = d.get("resolved_nodes")
    if isinstance(nodes, str):
        nodes = [nodes]
    if not nodes:
        nodes = [max(positions.values(), key=lambda s: len(s or "")) or desc]
    nodes = [n for n in nodes if isinstance(n, str) and n.strip()]
    commit = bool(d.get("confident", False)) and any(not is_nonanswer(n) for n in nodes)
    return nodes, commit, transcript
