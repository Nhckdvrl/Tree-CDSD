"""CDSD v12 — indexed faithful segment debate.

This is the first variant that makes v3's "last agreed node -> first conflict node" claim
operational: the judge must return explicit 1-based claim indices, and each solver debates
only its segment up to that indexed conflict. Resolution also gates commits on whether the
resolved node preserves the exact question relation.

Entrypoints:
  - run:       cdsdfi / v3-fixed, faithful and conservative.
  - run_boost: cdsdx, same process controller plus evidence-aware final rerank.
  - run_bridge: cdsdfb, same as cdsdfi plus a conservative bridge-recovery fallback.
"""
from src.llm.client import Tracker, ask
from src.eval.graders import extract_answer, normalize_qa
from src.util import extract_after, extract_json, is_nonanswer
from src.cdsd import components as C
from src.cdsd import prompts


def _entry(conflict, idx):
    raw = (conflict.get("claims") or {}).get(str(idx))
    if raw is None:
        raw = (conflict.get("claims") or {}).get(idx)
    if isinstance(raw, dict):
        return raw
    return {"index": None, "claim": raw or ""}


def _claim_index(entry, claims):
    """Convert judge-provided 1-based index to a safe 0-based inclusive index."""
    try:
        val = int(entry.get("index"))
    except Exception:
        val = None
    if val is not None and 1 <= val <= len(claims):
        return val - 1

    target = normalize_qa(entry.get("claim") or "")
    if target:
        for i, claim in enumerate(claims):
            cn = normalize_qa(claim)
            if cn and (cn == target or cn in target or target in cn):
                return i
    return len(claims) - 1 if claims else 0


def _segment(claims, entry):
    if not claims:
        return []
    end = _claim_index(entry, claims)
    return claims[: end + 1]


def _boolish(value, default=False):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "yes", "1"):
            return True
        if v in ("false", "no", "0"):
            return False
    return default


def _as_text(value):
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("claim", "node", "text", "resolved_node", "answer"):
            text = value.get(key)
            if isinstance(text, str) and text.strip():
                return text.strip()
        return " ".join(str(v).strip() for v in value.values() if isinstance(v, str) and v.strip())
    return str(value).strip() if value is not None else ""


def _has_passages(question):
    q = question.lower()
    return "==" in question or "following passages" in q or "following facts" in q


def _quote_supported(question, quote):
    q = normalize_qa(quote)
    if len(q) < 8:
        return False
    return q in normalize_qa(question)


def _evidence_supported(question, resolution):
    """For open-book examples, require exact evidence quotes copied from the prompt."""
    if not _has_passages(question):
        return True
    quotes = resolution.get("evidence_quotes") or []
    if isinstance(quotes, str):
        quotes = [quotes]
    quotes = [_as_text(q) for q in quotes]
    quotes = [q for q in quotes if q]
    if not quotes or not all(_quote_supported(question, q) for q in quotes):
        return False
    quote_text = " ".join(quotes).lower()
    nodes = resolution.get("resolved_nodes") or []
    if isinstance(nodes, str):
        nodes = [nodes]
    for node in nodes:
        node_l = _as_text(node).lower()
        if any(w in node_l for w in ("border", "borders", "bordered", "between")):
            if not any(w in quote_text for w in ("border", "borders", "bordered", "between")):
                return False
    return True


def _final_select(client, tr, question, answers, dtype, rerank=False):
    """Final answer selection shared by cdsdfi/cdsdx.

    cdsdfi uses majority_specific to stay close to v9's judge-free selection.
    cdsdx optionally spends one grounded rerank call when concrete answers split.
    """
    concrete = [a for a in answers if a and not is_nonanswer(a)]
    if not concrete:
        return C.majority_specific(answers, dtype)
    distinct = {C._norm(a, dtype) for a in concrete if C._norm(a, dtype)}
    if rerank and dtype == "qa" and len(distinct) > 1:
        return C.rerank(client, tr, question, answers, dtype)
    return C.majority_specific(answers, dtype)


def _find_conflict_indexed(client, tr, question, premises, agents_claims, agents_answers, dtype):
    user = prompts.conflict_indexed_user(question, premises, agents_claims, agents_answers, dtype)
    text = ask(client, tr, [{"role": "system", "content": prompts.JUDGE_SYS},
                            {"role": "user", "content": user}], temperature=0.0, max_tokens=900)
    d = extract_json(text)
    if d and d.get("status"):
        return d
    return C.find_conflict(client, tr, question, premises, agents_claims, agents_answers,
                           dtype, constructive=True)


def _argue_indexed_segment(agent, question, agreed_prefix, my_segment, conflict_desc,
                           others_claims, dtype, tr):
    user = prompts.argue_indexed_segment_user(
        question, agreed_prefix, my_segment, conflict_desc, others_claims, dtype)
    return ask(agent.client, tr, [{"role": "system", "content": agent.persona},
                                  {"role": "user", "content": user}],
               temperature=agent.temperature, max_tokens=agent.max_tokens)


def _debate_resolve_indexed(client, tr, agents, question, agreed_prefix, conflict,
                            agents_nodes, dtype, rounds=1):
    desc = conflict.get("description", "")
    positions = {}
    segments = {}
    for a in agents:
        entry = _entry(conflict, a.idx)
        positions[a.idx] = entry.get("claim") or ""
        segments[a.idx] = _segment(agents_nodes[a.idx] if a.idx < len(agents_nodes) else [], entry)

    transcript = []
    for r in range(rounds):
        new_pos = {}
        for a in agents:
            others = [(i, positions[i]) for i in positions if i != a.idx]
            out = _argue_indexed_segment(a, question, agreed_prefix, segments[a.idx],
                                         desc, others, dtype, tr)
            new_pos[a.idx] = extract_after(out, "My claim")
            transcript.append({"round": r, "agent": a.idx, "text": out})
        positions = new_pos

    args = "\n".join(f"Solver {i}: {positions[i]}" for i in sorted(positions))
    user = prompts.resolve_indexed_segment_user(question, agreed_prefix, desc, args)
    text = ask(client, tr, [{"role": "system", "content": prompts.RESOLVE_SYS},
                            {"role": "user", "content": user}], temperature=0.0, max_tokens=600)
    d = extract_json(text) or {}
    nodes = d.get("resolved_nodes")
    if isinstance(nodes, str):
        nodes = [nodes]
    if not nodes:
        nodes = [max(positions.values(), key=lambda s: len(s or "")) or desc]
    nodes = [_as_text(n) for n in nodes]
    nodes = [n for n in nodes if n]
    relation_preserved = _boolish(d.get("relation_preserved", True), default=True)
    commit = (
        _boolish(d.get("confident", False))
        and relation_preserved
        and _evidence_supported(question, d)
        and any(not is_nonanswer(n) for n in nodes)
    )
    return nodes, commit, transcript, segments, d


def _answer_grounded(question, answer, bridge_node, quotes, dtype):
    if not answer or is_nonanswer(answer):
        return False
    if dtype != "qa":
        return True
    ans = normalize_qa(answer)
    if not ans:
        return False
    hay = normalize_qa(" ".join([bridge_node or "", " ".join(quotes or []), question]))
    return ans in hay


def _bridge_relation_rule(question, bridge_node, quotes):
    q = normalize_qa(question)
    hay = normalize_qa(" ".join([bridge_node or "", " ".join(quotes or [])]))
    rules = [
        (("paternal grandmother",), ("paternal grandmother", "father s mother", "father mother")),
        (("maternal grandmother",), ("maternal grandmother", "mother s mother", "mother mother")),
        (("grandmother",), ("grandmother", "grandparent")),
        (("grandfather",), ("grandfather", "grandparent")),
        (("spouse",), ("spouse", "wife", "husband", "married")),
        (("employer",), ("employer", "professor", "worked at", "worked for", "employed")),
        (("educated",), ("educated", "studied", "attended", "university", "college", "school")),
        (("record label",), ("record label", "label")),
        (("border", "borders", "bordering"), ("border", "borders", "bordering", "adjacent")),
        (("founded",), ("founded", "established")),
    ]
    for triggers, required in rules:
        if any(t in q for t in triggers):
            return any(r in hay for r in required)
    return True


def _bridge_recover(client, tr, question, agreed_prefix, conflict, agents_answers,
                    nodes, dtype):
    """One-shot recovery when a segment resolution is blocked.

    This does not relax the commit gate. It only allows a final answer if a separate
    referee can cite exact prompt evidence for the missing bridge.
    """
    desc = conflict.get("description", "") if isinstance(conflict, dict) else ""
    user = prompts.bridge_recover_user(question, agreed_prefix, desc, agents_answers, nodes)
    text = ask(client, tr, [{"role": "system", "content": prompts.RESOLVE_SYS},
                            {"role": "user", "content": user}], temperature=0.0, max_tokens=700)
    d = extract_json(text) or {}
    answer = _as_text(d.get("answer"))
    bridge_node = _as_text(d.get("bridge_node"))
    quotes = d.get("evidence_quotes") or []
    if isinstance(quotes, str):
        quotes = [quotes]
    quotes = [_as_text(q) for q in quotes if _as_text(q)]
    evidence_payload = {"resolved_nodes": [bridge_node], "evidence_quotes": quotes}
    accepted = (
        _boolish(d.get("confident", False))
        and _boolish(d.get("chain_complete", False), default=False)
        and _boolish(d.get("relation_preserved", True), default=True)
        and bridge_node
        and _evidence_supported(question, evidence_payload)
        and _bridge_relation_rule(question, bridge_node, quotes)
        and _answer_grounded(question, answer, bridge_node, quotes, dtype)
    )
    d["accepted"] = accepted
    return answer if accepted else None, d


def _run_indexed(agents, ex, dtype, max_iters=6, debate_rounds=1, rerank_final=False,
                 bridge_recover=False, **kw):
    client = agents[0].client
    tr = Tracker()
    agreed_prefix, debated, iters_log = [], [], []
    final, agents_answers = None, []

    for it in range(max_iters):
        raws = [C.gen_claims(a, ex.question, agreed_prefix, dtype, tr) for a in agents]
        parsed = [C.parse_claims(r) for r in raws]
        agents_nodes = [p[0] for p in parsed]
        agents_answers = [p[1] or extract_answer(raws[i], dtype) for i, p in enumerate(parsed)]

        concrete = [a for a in agents_answers if a and not is_nonanswer(a)]
        if concrete and len(concrete) == len(agents) and len({C._norm(a, dtype) for a in concrete}) == 1:
            final = concrete[0]
            iters_log.append({"iter": it, "answers": agents_answers, "shortcircuit": True})
            break

        verdict = _find_conflict_indexed(client, tr, ex.question, agreed_prefix,
                                         agents_nodes, agents_answers, dtype)
        log = {"iter": it, "answers": agents_answers, "verdict": verdict}
        if verdict.get("status") == "consensus":
            final = _final_select(client, tr, ex.question, agents_answers, dtype, rerank=rerank_final)
            iters_log.append(log)
            break

        nodes, commit, transcript, segments, resolution = _debate_resolve_indexed(
            client, tr, agents, ex.question, agreed_prefix, verdict, agents_nodes,
            dtype, rounds=debate_rounds)
        log.update({"resolved_nodes": nodes, "commit": commit, "debate": transcript,
                    "segments": segments, "resolution": resolution})
        iters_log.append(log)
        if commit:
            agreed_prefix.extend(nodes)
            debated.extend(nodes)
        else:
            bridge_final = None
            if bridge_recover:
                bridge_final, bridge_log = _bridge_recover(
                    client, tr, ex.question, agreed_prefix, verdict, agents_answers, nodes, dtype)
                log["bridge_recover"] = bridge_log
            final = bridge_final or _final_select(
                client, tr, ex.question, agents_answers, dtype, rerank=rerank_final)
            break

    if final is None:
        final = _final_select(client, tr, ex.question, agents_answers, dtype, rerank=rerank_final)
    if is_nonanswer(final):
        final = C.majority_concrete(agents_answers, dtype) or final
    return {"pred": final, "trace": {"agreed_prefix": agreed_prefix, "iters": iters_log},
            "n_debates": len(debated), "debated_claims": debated, **tr.as_dict()}


def run(agents, ex, dtype, max_iters=6, debate_rounds=1, final_agents=False, **kw):
    """cdsdfi: v3-fixed indexed faithful CDSD."""
    return _run_indexed(agents, ex, dtype, max_iters=max_iters, debate_rounds=debate_rounds,
                        rerank_final=False, **kw)


def run_boost(agents, ex, dtype, max_iters=6, debate_rounds=1, final_agents=False, **kw):
    """cdsdx: indexed faithful CDSD plus evidence-aware rerank for split final answers."""
    return _run_indexed(agents, ex, dtype, max_iters=max_iters, debate_rounds=debate_rounds,
                        rerank_final=True, **kw)


def run_bridge(agents, ex, dtype, max_iters=6, debate_rounds=1, final_agents=False, **kw):
    """cdsdfb: cdsdfi plus conservative bridge recovery after blocked commit."""
    return _run_indexed(agents, ex, dtype, max_iters=max_iters, debate_rounds=debate_rounds,
                        rerank_final=False, bridge_recover=True, **kw)
