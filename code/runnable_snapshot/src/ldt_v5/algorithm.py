from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from src.cdsd import components as C
from src.cdsd.v12_indexed import _as_text, _boolish
from src.llm.client import Tracker
from src.ldt.types import CandidateNode, ReasoningTree
from src.ldt_v3.algorithm import _tree_cdsd_run
from src.util import extract_json, is_nonanswer


def _split_passage_prompt(question: str) -> tuple[str, list[dict], str]:
    """Parse wiki-style full-context prompts into passage blocks plus clean question."""
    text = question or ""
    q_match = re.search(r"\nQuestion:\s*(.+?)\s*$", text, flags=re.S)
    clean_question = q_match.group(1).strip() if q_match else text.strip()
    before_q = text[: q_match.start()] if q_match else text
    matches = list(re.finditer(r"(?m)^==\s*([^=\n]+?)\s*==\s*$", before_q))
    passages: list[dict] = []
    for i, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(before_q)
        body = before_q[start:end].strip()
        if title and body:
            passages.append({"title": title, "body": body})
    intro = before_q[: matches[0].start()].strip() if matches else ""
    return intro, passages, clean_question


def _format_passages(passages: list[dict]) -> str:
    return "\n\n".join(f"== {p['title']} ==\n{p['body']}" for p in passages)


def _augment_titles_from_plan(
    titles: list[str],
    passages: list[dict],
    clean_question: str,
    plan: dict,
    *,
    max_titles: int = 5,
) -> list[str]:
    """Add exact passage-title mentions from the resolved plan without widening too far."""
    selected = []
    seen = set()
    for title in titles:
        text = _as_text(title)
        key = C.normalize_qa(text)
        if text and key and key not in seen:
            seen.add(key)
            selected.append(text)

    plan_text = " ".join(
        _as_text(plan.get(k))
        for k in ("answer_slot", "bridge_chain", "answer", "reason")
    )
    searchable = C.normalize_qa(f"{clean_question} {plan_text}")
    for passage in passages:
        title = _as_text(passage.get("title"))
        key = C.normalize_qa(title)
        if not title or not key or key in seen:
            continue
        if key in searchable:
            seen.add(key)
            selected.append(title)
        if len(selected) >= max_titles:
            break
    return selected[:max_titles]


def _focused_question(
    original: str,
    titles: list[str],
    plan: dict,
    *,
    include_contract: bool = True,
) -> tuple[str, dict]:
    intro, passages, clean = _split_passage_prompt(original)
    if not passages:
        return original, {"focused": False, "reason": "no-passages"}
    title_map = {C.normalize_qa(p["title"]): p for p in passages}
    expanded_titles = _augment_titles_from_plan(titles, passages, clean, plan)
    chosen = []
    seen = set()
    for title in expanded_titles:
        key = C.normalize_qa(title)
        if key in title_map and key not in seen:
            seen.add(key)
            chosen.append(title_map[key])
    if not (1 <= len(chosen) <= 5):
        return original, {"focused": False, "reason": "bad-title-count", "titles": expanded_titles}

    prefix = intro or "Read the following passages and answer the question using only the information in them."
    contract = []
    answer_slot = _as_text(plan.get("answer_slot"))
    bridge_chain = _as_text(plan.get("bridge_chain"))
    if answer_slot:
        contract.append(f"Requested answer slot: {answer_slot}")
    if bridge_chain:
        contract.append(f"Bridge chain to verify: {bridge_chain}")
    if include_contract and contract:
        contract.append("Return the final endpoint for the requested slot, not an intermediate entity.")
        contract_text = "\nReasoning contract:\n" + "\n".join(f"- {line}" for line in contract) + "\n"
    else:
        contract_text = ""
    focused = f"{prefix}\n\n{_format_passages(chosen)}\n{contract_text}\nQuestion: {clean}"
    return focused, {
        "focused": True,
        "selected_titles": [p["title"] for p in chosen],
        "planner_titles": titles,
        "expanded_titles": expanded_titles,
        "contract_applied": bool(include_contract and contract),
        "n_original_passages": len(passages),
        "n_focused_passages": len(chosen),
    }


def _evidence_plan_prompt(question: str) -> str:
    intro, passages, clean = _split_passage_prompt(question)
    if passages:
        listing = "\n".join(f"{i+1}. {p['title']}: {p['body'][:500]}" for i, p in enumerate(passages))
    else:
        listing = question[:6000]
    return (
        "Build an evidence plan for a multi-hop QA reasoning tree. "
        "Select only passages that are needed for the exact bridge chain. "
        "Avoid passages that merely share a surface title/name with the question. "
        "Do not answer from outside the provided passages.\n\n"
        f"Question:\n{clean}\n\n"
        f"Passage candidates:\n{listing}\n\n"
        "Return JSON only with keys: "
        '{"answer_slot":"short description of requested answer type/slot",'
        '"bridge_chain":"entity/relation chain to verify",'
        '"selected_titles":["title1","title2"],'
        '"answer":"candidate short answer or empty",'
        '"confidence":0.0,'
        '"reason":"brief"}'
    )


def _make_plan(agent, question: str, tr: Tracker) -> dict:
    text = C.ask(
        agent.client,
        tr,
        [{"role": "system", "content": agent.persona},
         {"role": "user", "content": _evidence_plan_prompt(question)}],
        temperature=agent.temperature,
        max_tokens=700,
    )
    data = extract_json(text) or {}
    titles = data.get("selected_titles") or []
    if isinstance(titles, str):
        titles = [titles]
    data["selected_titles"] = [_as_text(t) for t in titles if _as_text(t)]
    data["answer"] = _as_text(data.get("answer"))
    data["answer_slot"] = _as_text(data.get("answer_slot"))
    data["bridge_chain"] = _as_text(data.get("bridge_chain"))
    data["confidence"] = float(data.get("confidence") or 0.0)
    data["raw"] = text
    return data


def _resolve_plan_prompt(question: str, plans: list[dict]) -> str:
    clean = _split_passage_prompt(question)[2]
    rendered = []
    for i, plan in enumerate(plans):
        rendered.append(
            f"Plan {i}: titles={plan.get('selected_titles')}; "
            f"slot={plan.get('answer_slot')}; chain={plan.get('bridge_chain')}; "
            f"answer={plan.get('answer')}; confidence={plan.get('confidence')}; "
            f"reason={plan.get('reason')}"
        )
    return (
        "Resolve the evidence plans for a multi-hop QA tree. "
        "Choose the smallest passage set that supports the exact bridge chain. "
        "If plans disagree because of same-name distractors, prefer the plan whose chain "
        "matches the wording of the question, not the majority title. "
        "Return only titles that appear in the plans.\n\n"
        f"Question:\n{clean}\n\n"
        + "\n".join(rendered)
        + "\n\nReturn JSON only with keys: "
        '{"answer_slot":"...","bridge_chain":"...",'
        '"selected_titles":["..."],"answer":"candidate or empty",'
        '"confidence":0.0,"use_focused_context":true|false,"reason":"..."}'
    )


def _resolve_plan(client, tr: Tracker, question: str, plans: list[dict]) -> dict:
    text = C.ask(
        client,
        tr,
        [{"role": "system", "content": "You are a careful evidence planner."},
         {"role": "user", "content": _resolve_plan_prompt(question, plans)}],
        temperature=0.0,
        max_tokens=700,
    )
    data = extract_json(text) or {}
    titles = data.get("selected_titles") or []
    if isinstance(titles, str):
        titles = [titles]
    plan_titles = {C.normalize_qa(t) for p in plans for t in p.get("selected_titles", [])}
    data["selected_titles"] = [
        _as_text(t) for t in titles
        if _as_text(t) and C.normalize_qa(t) in plan_titles
    ][:5]
    data["answer_slot"] = _as_text(data.get("answer_slot"))
    data["bridge_chain"] = _as_text(data.get("bridge_chain"))
    data["answer"] = _as_text(data.get("answer"))
    data["confidence"] = float(data.get("confidence") or 0.0)
    data["use_focused_context"] = _boolish(data.get("use_focused_context", False), default=False)
    data["raw"] = text
    return data


def _self_reported_unsupported(plan: dict) -> str:
    text = C.normalize_qa(
        " ".join(_as_text(plan.get(k)) for k in ("reason", "raw"))
    )
    markers = (
        "not present in the text",
        "not in the text",
        "not provided in the text",
        "outside the provided",
        "real world fact",
        "assuming",
        "assumption",
    )
    for marker in markers:
        if marker in text:
            return marker
    return ""


def _validate_plan_prompt(question: str, focused_question: str, resolved: dict) -> str:
    clean = _split_passage_prompt(question)[2]
    return (
        "Validate an evidence plan before it is allowed to guide a reasoning tree. "
        "Every bridge edge must be explicitly supported by the selected passages. "
        "Reject the plan if it relies on outside knowledge, an assumption, a same-name "
        "surface match, or a direction of relation not stated in the passages.\n\n"
        f"Question:\n{clean}\n\n"
        "Resolved plan:\n"
        f"answer_slot: {resolved.get('answer_slot')}\n"
        f"bridge_chain: {resolved.get('bridge_chain')}\n"
        f"selected_titles: {resolved.get('selected_titles')}\n\n"
        f"Selected passages:\n{focused_question[:9000]}\n\n"
        "Return JSON only with keys: "
        '{"supported":true|false,'
        '"edge_support":[{"edge":"...","supported":true|false,"evidence":"short quote or empty"}],'
        '"confidence":0.0,"reason":"brief"}'
    )


def _validate_plan(client, tr: Tracker, question: str, focused_question: str, resolved: dict) -> dict:
    confessed = _self_reported_unsupported(resolved)
    if confessed:
        return {
            "supported": False,
            "confidence": 1.0,
            "reason": f"planner-self-reported-unsupported:{confessed}",
            "edge_support": [],
        }
    text = C.ask(
        client,
        tr,
        [{"role": "system", "content": "You audit evidence-plan support strictly."},
         {"role": "user", "content": _validate_plan_prompt(question, focused_question, resolved)}],
        temperature=0.0,
        max_tokens=700,
    )
    data = extract_json(text) or {}
    data["supported"] = _boolish(data.get("supported"), default=False)
    data["confidence"] = float(data.get("confidence") or 0.0)
    data["reason"] = _as_text(data.get("reason"))
    data["edge_support"] = data.get("edge_support") if isinstance(data.get("edge_support"), list) else []
    if data["edge_support"] and all(
        _boolish(edge.get("supported"), default=False)
        for edge in data["edge_support"] if isinstance(edge, dict)
    ):
        data["supported"] = True
        data["reason"] = data["reason"] or "all-edges-supported"
    data["raw"] = text
    return data


def _add_plan_branch(tree: ReasoningTree, agent_idx: int, plan: dict) -> list[dict]:
    parent = tree.root_id
    added = []
    pieces = [
        ("answer_slot", plan.get("answer_slot")),
        ("bridge_chain", plan.get("bridge_chain")),
        ("selected_titles", " -> ".join(plan.get("selected_titles") or [])),
        ("candidate_answer", plan.get("answer")),
    ]
    for local_idx, (kind, text) in enumerate(pieces, start=1):
        text = _as_text(text)
        if not text:
            continue
        cand = CandidateNode(
            candidate_id=f"plan_a{agent_idx}_{local_idx}",
            text=text,
            depth=tree.nodes[parent].depth + 1,
            parent_id=parent,
            support_agents=[agent_idx],
            confidence=float(plan.get("confidence") or 0.0),
            score=0.55 + min(float(plan.get("confidence") or 0.0), 1.0) * 0.25,
            status="evidence_plan",
            is_final=(kind == "candidate_answer"),
            answer=text if kind == "candidate_answer" else "",
            history={"role": "evidence_plan", "agent": agent_idx, "kind": kind, "plan": plan},
        )
        node = tree.add_child(parent, cand)
        added.append(node.to_dict())
        parent = node.node_id
    return added


def _plan_tree(question: str, plans: list[dict], resolved: dict) -> dict:
    tree = ReasoningTree(question)
    branches = [_add_plan_branch(tree, i, plan) for i, plan in enumerate(plans)]
    if resolved.get("selected_titles"):
        cand = CandidateNode(
            candidate_id="resolved_plan",
            text=f"Resolved evidence plan: {' -> '.join(resolved.get('selected_titles') or [])}",
            depth=1,
            parent_id=tree.root_id,
            support_agents=[],
            confidence=float(resolved.get("confidence") or 0.0),
            score=0.9,
            status="resolved_evidence_plan",
            is_final=bool(resolved.get("answer")),
            answer=resolved.get("answer") or "",
            history={"role": "resolved_evidence_plan", "plan": resolved},
        )
        tree.add_child(tree.root_id, cand)
    return {"branches": branches, "tree": tree.to_dict()}


def _nonanswerish(text: str) -> bool:
    lower = _as_text(text).lower().strip()
    if is_nonanswer(lower):
        return True
    markers = (
        "do not state", "does not state", "not stated", "not mentioned",
        "cannot be determined", "cannot be identified", "not identifiable",
        "no one", "unknown", "not present",
        "do not contain", "does not contain", "not found",
        "not in the text", "not in text",
        "information is not provided", "information not provided",
        "no award mentioned", "no spouse mentioned", "no answer provided",
    )
    return any(marker in lower for marker in markers)


def _candidate_leaves(
    full: dict,
    focused_evidence: dict | None,
    focused_contract: dict | None,
) -> list[dict]:
    leaves = [
        {
            "candidate_id": "full",
            "source": "full_tree",
            "answer": _as_text(full.get("pred")),
            "default": True,
        }
    ]
    if focused_evidence is not None:
        leaves.append({
            "candidate_id": "focused_evidence",
            "source": "focused_evidence_tree",
            "answer": _as_text(focused_evidence.get("pred")),
            "default": False,
        })
    if focused_contract is not None:
        leaves.append({
            "candidate_id": "focused_contract",
            "source": "focused_contract_tree",
            "answer": _as_text(focused_contract.get("pred")),
            "default": False,
        })

    out = []
    seen = set()
    for leaf in leaves:
        answer = leaf["answer"]
        key = C.normalize_qa(answer)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(leaf)
    return out[:10]


def _contract_prompt(question: str, leaves: list[dict], resolved: dict, plan_validation: dict) -> str:
    clean = _split_passage_prompt(question)[2]
    rendered = "\n".join(
        f"{leaf['candidate_id']} ({leaf['source']}): {leaf['answer']}"
        for leaf in leaves
    )
    plan_lines = []
    if _as_text(resolved.get("answer_slot")):
        plan_lines.append(f"answer_slot: {resolved.get('answer_slot')}")
    if _as_text(resolved.get("bridge_chain")):
        plan_lines.append(f"bridge_chain: {resolved.get('bridge_chain')}")
    if resolved.get("selected_titles"):
        plan_lines.append(f"selected_titles: {resolved.get('selected_titles')}")
    plan_text = "\n".join(plan_lines) or "none"
    validation_text = (
        f"supported: {plan_validation.get('supported')}; "
        f"confidence: {plan_validation.get('confidence')}; "
        f"reason: {plan_validation.get('reason')}"
    )
    return (
        "You are selecting a terminal leaf from a multi-hop reasoning tree. "
        "You must choose only one listed candidate_id; do not write a new answer. "
        "The evidence plan is a constraint on relation and support, not an answer candidate. "
        "If Plan validation says supported=false, treat the plan as a rejected hypothesis: "
        "a plan-constrained leaf must then be chosen only when the original passages "
        "independently support its exact answer better than the full/evidence leaves. "
        "Judge each candidate against the exact question relation, answer type, "
        "passage support, and minimality. Prefer the full_tree candidate when "
        "candidates are similarly valid, but never prefer a non-answer over a "
        "supported concrete terminal leaf.\n\n"
        f"Question:\n{clean}\n\n"
        f"Evidence-plan constraints:\n{plan_text}\n\n"
        f"Plan validation:\n{validation_text}\n\n"
        f"Passages and context:\n{question[:9000]}\n\n"
        f"Candidate leaves:\n{rendered}\n\n"
        "Return JSON only with keys: "
        '{"chosen_id":"candidate_id",'
        '"scores":[{"candidate_id":"...","answers_question":true,'
        '"answer_type_match":true,"supported":true,"minimal":true,'
        '"confidence":0.0,"reason":"brief"}],'
        '"reason":"brief"}'
    )


def _score_value(score: dict, answer: str) -> float:
    if _nonanswerish(answer):
        return -1.0
    confidence = float(score.get("confidence") or 0.0)
    checks = [
        _boolish(score.get("answers_question"), default=False),
        _boolish(score.get("answer_type_match"), default=False),
        _boolish(score.get("supported"), default=False),
        _boolish(score.get("minimal"), default=False),
    ]
    return confidence + 0.12 * sum(checks)


def _answer_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9'’\-]*", _as_text(text))


def _canonicalize_entity_answer(pred: str, selected: dict, dtype: str) -> tuple[str, dict]:
    if dtype != "qa" or _nonanswerish(pred):
        return pred, {"changed": False, "reason": "not-applicable"}
    pred_words = _answer_words(pred)
    if len(pred_words) != 1:
        return pred, {"changed": False, "reason": "not-single-token-answer"}
    final_selection = ((selected.get("trace") or {}).get("final_selection") or {})
    guard = final_selection.get("specificity_guard") or {}
    candidates = guard.get("strict_candidates") or []
    pred_key = C.normalize_qa(pred)
    matches = []
    for candidate in candidates:
        text = _as_text(candidate).strip()
        key = C.normalize_qa(text)
        words = _answer_words(text)
        if not key or key == pred_key or len(words) <= len(pred_words) or len(words) > len(pred_words) + 3:
            continue
        if key.startswith(pred_key + " "):
            matches.append(text)
    if len(matches) == 1:
        return matches[0], {
            "changed": True,
            "mode": "single-token-entity-expansion",
            "from": pred,
            "to": matches[0],
            "candidates": candidates[:8],
        }
    return pred, {
        "changed": False,
        "reason": "no-unique-tree-expansion",
        "candidates": candidates[:8],
    }


def _compound_subanswer_override(
    pred: str,
    selected_by_id: dict[str, dict | None],
    dtype: str,
) -> tuple[str, dict | None, dict]:
    if dtype != "qa" or _nonanswerish(pred):
        return pred, None, {"changed": False, "reason": "not-applicable"}
    pred_key = C.normalize_qa(pred)
    if not re.search(r"\b(and|or)\b", pred_key):
        return pred, None, {"changed": False, "reason": "not-compound-answer"}
    candidates = []
    for source_id in ("focused_contract", "focused_evidence"):
        result = selected_by_id.get(source_id)
        if not result:
            continue
        answer = _as_text(result.get("pred"))
        key = C.normalize_qa(answer)
        if not key or key == pred_key or _nonanswerish(answer):
            continue
        if key in pred_key and len(key.split()) < len(pred_key.split()):
            candidates.append((answer, result, source_id))
    if len(candidates) == 1:
        answer, result, source_id = candidates[0]
        return answer, result, {
            "changed": True,
            "mode": "focused-subanswer-of-compound",
            "from": pred,
            "to": answer,
            "selected_source": source_id,
        }
    return pred, None, {
        "changed": False,
        "reason": "no-unique-focused-subanswer",
        "candidates": [item[0] for item in candidates],
    }


def _canonical_leaf_id(value: str, leaves: list[dict]) -> str:
    raw = _as_text(value).strip()
    if not raw:
        return ""
    by_id = {leaf["candidate_id"]: leaf for leaf in leaves}
    by_source = {leaf["source"]: leaf for leaf in leaves}
    without_suffix = re.sub(r"\s*\([^)]*\)\s*$", "", raw).strip()
    if without_suffix in by_id:
        return without_suffix
    suffix_match = re.search(r"\(([^)]*)\)\s*$", raw)
    if suffix_match and suffix_match.group(1).strip() in by_source:
        return by_source[suffix_match.group(1).strip()]["candidate_id"]
    aliases = {
        "full_tree": "full",
        "full tree": "full",
        "focused_tree": "focused_contract",
        "focused tree": "focused_contract",
        "focused": "focused_contract",
        "focused_evidence_tree": "focused_evidence",
        "focused evidence tree": "focused_evidence",
        "evidence_tree": "focused_evidence",
        "evidence tree": "focused_evidence",
        "focused_contract_tree": "focused_contract",
        "focused contract tree": "focused_contract",
        "contract_tree": "focused_contract",
        "contract tree": "focused_contract",
    }
    if raw in by_id:
        return raw
    if raw in by_source:
        return by_source[raw]["candidate_id"]
    lower = raw.lower()
    if lower in aliases and aliases[lower] in by_id:
        return aliases[lower]
    if lower in ("focused_tree", "focused tree", "focused") and "focused_evidence" in by_id:
        return "focused_evidence"
    raw_key = C.normalize_qa(raw)
    for leaf in leaves:
        if raw_key and raw_key == C.normalize_qa(leaf.get("answer")):
            return leaf["candidate_id"]
    return raw


def _contract_select(
    client,
    tr: Tracker,
    question: str,
    leaves: list[dict],
    resolved: dict,
    plan_validation: dict,
) -> tuple[str, dict]:
    default = next((leaf for leaf in leaves if leaf.get("candidate_id") == "full"), leaves[0])
    if len(leaves) == 1:
        return default["answer"], {"chosen_id": default["candidate_id"], "reason": "single-leaf", "leaves": leaves}

    text = C.ask(
        client,
        tr,
        [
            {"role": "system", "content": "You verify reasoning-tree terminal leaves."},
            {"role": "user", "content": _contract_prompt(question, leaves, resolved, plan_validation)},
        ],
        temperature=0.0,
        max_tokens=900,
    )
    data = extract_json(text) or {}
    scores = data.get("scores") if isinstance(data.get("scores"), list) else []
    score_by_id = {
        _canonical_leaf_id(score.get("candidate_id"), leaves): score
        for score in scores if isinstance(score, dict)
    }
    leaf_by_id = {leaf["candidate_id"]: leaf for leaf in leaves}
    chosen_id = _canonical_leaf_id(data.get("chosen_id"), leaves)
    chosen = leaf_by_id.get(chosen_id, default)
    full_score = score_by_id.get("full", {})
    chosen_score = score_by_id.get(chosen["candidate_id"], {})

    leaf_values = {
        leaf["candidate_id"]: _score_value(score_by_id.get(leaf["candidate_id"], {}), leaf["answer"])
        for leaf in leaves
    }
    full_value = leaf_values.get("full", -1.0)
    chosen_value = leaf_values.get(chosen["candidate_id"], -1.0)
    full_minimal = _boolish(full_score.get("minimal"), default=False)
    answers_question = _boolish(chosen_score.get("answers_question"), default=False)
    answer_type_match = _boolish(chosen_score.get("answer_type_match"), default=False)
    supported = _boolish(chosen_score.get("supported"), default=False)
    minimal = _boolish(chosen_score.get("minimal"), default=False)
    chosen_core = answers_question and answer_type_match and not _nonanswerish(chosen["answer"])
    chosen_valid = chosen_core and (
        supported
        or (minimal and chosen_value >= full_value + 0.24)
    )
    score_sufficient = (
        float(chosen_score.get("confidence") or 0.0) >= 0.72
        or chosen_value >= full_value + (0.10 if minimal and not full_minimal else 0.24)
    )
    required_margin = 0.10 if minimal and not full_minimal else 0.18
    full_nonanswer = _nonanswerish(default["answer"])
    concrete_non_full = [
        leaf for leaf in leaves
        if leaf["candidate_id"] != "full" and not _nonanswerish(leaf["answer"])
    ]
    best_concrete = max(
        concrete_non_full,
        key=lambda leaf: leaf_values.get(leaf["candidate_id"], -1.0),
        default=None,
    )
    nonanswer_replaced = bool(full_nonanswer and best_concrete is not None)
    accept_non_full = (
        chosen["candidate_id"] != "full"
        and chosen_valid
        and score_sufficient
        and (full_nonanswer or chosen_value >= full_value + required_margin)
    )
    if nonanswer_replaced:
        final_leaf = best_concrete
    else:
        final_leaf = chosen if accept_non_full else default
    return final_leaf["answer"], {
        "chosen_id": final_leaf["candidate_id"],
        "model_chosen_id": chosen_id,
        "accepted_non_full": accept_non_full,
        "nonanswer_replaced": nonanswer_replaced,
        "full_value": round(full_value, 4),
        "chosen_value": round(chosen_value, 4),
        "leaf_values": {key: round(value, 4) for key, value in leaf_values.items()},
        "scores": scores,
        "leaves": leaves,
        "raw": text,
        "reason": _as_text(data.get("reason")),
    }


def _projection_agreement_prompt(
    question: str,
    groups: list[dict],
    current_group: str,
    plan_validation: dict,
) -> str:
    clean = _split_passage_prompt(question)[2]
    rendered = "\n".join(
        f"{group['group_id']}: answer={group['answer']}; sources={group['sources']}"
        for group in groups
    )
    return (
        "Audit terminal disagreement among projections of a reasoning tree. "
        "A source is one complete tree projection, not a vote from the same prompt. "
        "Use projection agreement as evidence, but choose the answer that best matches "
        "the exact relation and hop depth in the question. For nested relations, count "
        "hops literally: father of X's father means parent(parent(X)); child of X's child "
        "means child(child(X)). If plan validation is unsupported, do not let the "
        "plan-constrained projection override other projections unless the passages "
        "independently support its exact relation.\n\n"
        f"Question:\n{clean}\n\n"
        f"Full context:\n{question[:9000]}\n\n"
        f"Plan validation: supported={plan_validation.get('supported')}; "
        f"reason={plan_validation.get('reason')}\n\n"
        f"Current selected group: {current_group}\n\n"
        f"Candidate groups:\n{rendered}\n\n"
        "Return JSON only with keys: "
        '{"chosen_group":"group_id","confidence":0.0,"reason":"brief"}'
    )


def _projection_agreement_audit(
    client,
    tr: Tracker,
    question: str,
    selected_by_id: dict[str, dict | None],
    current_id: str,
    plan_validation: dict,
    dtype: str,
) -> tuple[str, dict | None, dict]:
    buckets: dict[str, dict] = {}
    for source_id, result in selected_by_id.items():
        if not result:
            continue
        answer = _as_text(result.get("pred"))
        key = C.normalize_qa(answer)
        if not key or _nonanswerish(answer):
            continue
        bucket = buckets.setdefault(key, {"answer": answer, "sources": [], "source_ids": []})
        bucket["sources"].append(source_id)
        bucket["source_ids"].append(source_id)
    if len(buckets) < 2 or max(len(b["sources"]) for b in buckets.values()) < 2:
        return "", None, {"changed": False, "reason": "no-two-projection-disagreement"}

    groups = []
    current_answer = _as_text((selected_by_id.get(current_id) or {}).get("pred"))
    current_key = C.normalize_qa(current_answer)
    current_group = ""
    for idx, (key, bucket) in enumerate(buckets.items()):
        group_id = f"g{idx}"
        group = {
            "group_id": group_id,
            "key": key,
            "answer": bucket["answer"],
            "sources": bucket["sources"],
            "source_ids": bucket["source_ids"],
        }
        if key == current_key:
            current_group = group_id
        groups.append(group)
    if not current_group:
        return "", None, {"changed": False, "reason": "current-answer-not-in-groups"}

    text = C.ask(
        client,
        tr,
        [{"role": "system", "content": "You audit disagreements between reasoning-tree projections."},
         {"role": "user", "content": _projection_agreement_prompt(
             question, groups, current_group, plan_validation)}],
        temperature=0.0,
        max_tokens=700,
    )
    data = extract_json(text) or C.extract_json(text) or {}
    if not data:
        group_match = re.search(r'"chosen_group"\s*:\s*"([^"]+)"', text)
        conf_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', text)
        if group_match:
            data = {
                "chosen_group": group_match.group(1),
                "confidence": float(conf_match.group(1)) if conf_match else 0.0,
                "reason": "regex-json-recovery",
            }
    chosen_group_id = _as_text(data.get("chosen_group"))
    group_by_id = {group["group_id"]: group for group in groups}
    chosen_group = group_by_id.get(chosen_group_id)
    confidence = float(data.get("confidence") or 0.0)
    current = group_by_id.get(current_group) or {}
    current_answer = current.get("answer") or ""
    if (
        chosen_group
        and chosen_group_id != current_group
        and current.get("source_ids") == ["focused_contract"]
    ):
        reason_text = C.normalize_qa(f"{data.get('reason', '')} {text}")
        allow_contract_override = (
            len(chosen_group.get("source_ids") or []) >= 2
            and "minor textual gap" not in reason_text
        )
        if allow_contract_override and confidence < 0.65:
            confidence = 0.66
        if not allow_contract_override:
            return "", None, {
                "changed": False,
                "reason": "keep-current-plan-constrained-terminal",
                "groups": groups,
                "raw": data,
                "raw_text": text,
            }
    if chosen_group and chosen_group_id != current_group:
        if _nonanswerish(chosen_group.get("answer", "")) and not _nonanswerish(current_answer):
            return "", None, {
                "changed": False,
                "reason": "reject-audit-nonanswer-over-concrete",
                "groups": groups,
                "raw": data,
                "raw_text": text,
            }
        current_key_for_span = C.normalize_qa(current_answer)
        chosen_key_for_span = C.normalize_qa(chosen_group.get("answer", ""))
        if (
            current_key_for_span
            and current_key_for_span in chosen_key_for_span
            and re.search(r"\b(and|or)\b", chosen_key_for_span)
        ):
            return "", None, {
                "changed": False,
                "reason": "reject-audit-compound-expansion",
                "groups": groups,
                "raw": data,
                "raw_text": text,
            }
        if (
            confidence < 0.65
            and len(chosen_group.get("source_ids") or []) >= 2
            and current.get("source_ids") != ["focused_contract"]
        ):
            confidence = 0.66
    if not chosen_group or confidence < 0.65:
        return "", None, {
            "changed": False,
            "reason": "audit-not-confident",
            "groups": groups,
            "raw": data,
            "raw_text": text,
        }
    chosen_source = chosen_group["source_ids"][0]
    changed = chosen_group_id != current_group
    return chosen_group["answer"], selected_by_id.get(chosen_source), {
        "changed": changed,
        "from_group": current_group,
        "to_group": chosen_group_id,
        "answer": chosen_group["answer"],
        "selected_source": chosen_source,
        "confidence": confidence,
        "groups": groups,
        "raw": data,
        "raw_text": text,
    }


def run(
    agents,
    ex,
    dtype,
    max_iters: int = 6,
    debate_rounds: int = 1,
    ldt_final_top_paths: int = 12,
    ldt_no_commit_select: bool = False,
    ldt_verified_final: bool = False,
    ldt_audit_consensus: bool = False,
    ldt_recover_nonanswer: bool = True,
    ldt_max_branch_nodes: int = 8,
    ldt_minimal_final_audit: bool = True,
    **kw,
):
    """LDT-v5: dual reasoning trees with contract-based terminal selection."""
    kw.pop("dataset", None)
    client = agents[0].client
    tr = Tracker()
    plans = [_make_plan(agent, ex.question, tr) for agent in agents]
    resolved = _resolve_plan(client, tr, ex.question, plans)
    plan_trace = _plan_tree(ex.question, plans, resolved)

    full = _tree_cdsd_run(
        agents,
        ex,
        dtype,
        max_iters=max_iters,
        debate_rounds=debate_rounds,
        final_top_paths=ldt_final_top_paths,
        max_branch_nodes=ldt_max_branch_nodes,
        no_commit_select=ldt_no_commit_select,
        verified_final=ldt_verified_final,
        audit_consensus=ldt_audit_consensus,
        recover_nonanswer=ldt_recover_nonanswer,
        specificity_guard=True,
        minimal_final_audit=ldt_minimal_final_audit,
        commit_policy=None,
        **kw,
    )

    focused_evidence_question, focus_log = _focused_question(
        ex.question, resolved.get("selected_titles") or [], resolved, include_contract=False)
    plan_validation = (
        _validate_plan(client, tr, ex.question, focused_evidence_question, resolved)
        if focus_log.get("focused")
        else {"supported": False, "confidence": 0.0, "reason": focus_log.get("reason"), "edge_support": []}
    )
    focused_contract_question, focused_contract_log = _focused_question(
        ex.question, resolved.get("selected_titles") or [], resolved, include_contract=True)
    contract_allowed = focus_log.get("focused") and not _self_reported_unsupported(resolved)
    should_focus = (
        dtype == "qa"
        and resolved.get("use_focused_context")
        and float(resolved.get("confidence") or 0.0) >= 0.55
        and focus_log.get("focused")
    )
    focused_evidence = None
    focused_contract = None
    if should_focus:
        focused_evidence = _tree_cdsd_run(
            agents,
            replace(ex, question=focused_evidence_question),
            dtype,
            max_iters=max_iters,
            debate_rounds=debate_rounds,
            final_top_paths=ldt_final_top_paths,
            max_branch_nodes=ldt_max_branch_nodes,
            no_commit_select=ldt_no_commit_select,
            verified_final=ldt_verified_final,
            audit_consensus=ldt_audit_consensus,
            recover_nonanswer=ldt_recover_nonanswer,
            specificity_guard=True,
            minimal_final_audit=ldt_minimal_final_audit,
            commit_policy=None,
            **kw,
        )
        if contract_allowed and focused_contract_log.get("focused"):
            focused_contract = _tree_cdsd_run(
                agents,
                replace(ex, question=focused_contract_question),
                dtype,
                max_iters=max_iters,
                debate_rounds=debate_rounds,
                final_top_paths=ldt_final_top_paths,
                max_branch_nodes=ldt_max_branch_nodes,
                no_commit_select=ldt_no_commit_select,
                verified_final=ldt_verified_final,
                audit_consensus=ldt_audit_consensus,
                recover_nonanswer=ldt_recover_nonanswer,
                specificity_guard=True,
                minimal_final_audit=ldt_minimal_final_audit,
                commit_policy=None,
                **kw,
            )

    leaves = _candidate_leaves(full, focused_evidence, focused_contract)
    pred, contract = _contract_select(client, tr, ex.question, leaves, resolved, plan_validation)
    selected_by_id = {
        "full": full,
        "focused_evidence": focused_evidence,
        "focused_contract": focused_contract,
    }
    selected = selected_by_id.get(contract.get("chosen_id")) or full
    audited_pred, audited_selected, agreement_audit = _projection_agreement_audit(
        client,
        tr,
        ex.question,
        selected_by_id,
        contract.get("chosen_id") or "full",
        plan_validation,
        dtype,
    )
    if agreement_audit.get("changed") and audited_pred and audited_selected:
        pred = audited_pred
        selected = audited_selected
    pred, subanswer_selected, subanswer_override = _compound_subanswer_override(
        pred, selected_by_id, dtype)
    if subanswer_override.get("changed") and subanswer_selected:
        selected = subanswer_selected
    pred, canonicalization = _canonicalize_entity_answer(pred, selected, dtype)

    trace = {
        "algorithm": "ldtv5",
        "evidence_plans": plans,
        "resolved_evidence_plan": resolved,
        "plan_validation": plan_validation,
        "focused_contract_allowed": bool(contract_allowed),
        "focus": focus_log,
        "focused_contract_focus": focused_contract_log,
        "used_focused_context": bool(should_focus),
        "contract_selection": contract,
        "projection_agreement_audit": agreement_audit,
        "compound_subanswer_override": subanswer_override,
        "answer_canonicalization": canonicalization,
        "plan_tree": plan_trace,
        "full_inner": full.get("trace", {}),
        "focused_inner": selected.get("trace", {}) if contract.get("chosen_id", "").startswith("focused") else {},
        "focused_evidence_inner": focused_evidence.get("trace", {}) if focused_evidence is not None else {},
        "focused_contract_inner": focused_contract.get("trace", {}) if focused_contract is not None else {},
        "inner": selected.get("trace", {}),
        "tree_predictions": {
            "full": _as_text(full.get("pred")),
            "focused_evidence": _as_text(focused_evidence.get("pred")) if focused_evidence is not None else "",
            "focused_contract": _as_text(focused_contract.get("pred")) if focused_contract is not None else "",
        },
    }
    costs = selected.copy()
    costs.update({
        "pred": pred,
        "trace": trace,
        "n_debates": selected.get("n_debates", 0),
        "debated_claims": selected.get("debated_claims", []),
        "n_no_commit": selected.get("n_no_commit", 0),
        "no_commit_trace": selected.get("no_commit_trace", []),
        "calls": (
            int(full.get("calls", 0))
            + int((focused_evidence or {}).get("calls", 0))
            + int((focused_contract or {}).get("calls", 0))
            + tr.calls
        ),
        "prompt_tokens": (
            int(full.get("prompt_tokens", 0))
            + int((focused_evidence or {}).get("prompt_tokens", 0))
            + int((focused_contract or {}).get("prompt_tokens", 0))
            + tr.prompt_tokens
        ),
        "completion_tokens": (
            int(full.get("completion_tokens", 0))
            + int((focused_evidence or {}).get("completion_tokens", 0))
            + int((focused_contract or {}).get("completion_tokens", 0))
            + tr.completion_tokens
        ),
    })
    return costs
