from __future__ import annotations

import re
from typing import Any

from src.eval.graders import extract_answer, normalize_qa
from src.llm.client import Tracker, ask
from src.ldt import prompts as base_prompts
from src.ldt.algorithm import (
    _as_text,
    _boolish,
    _clean_answer,
    _clamp,
    _debate_group,
    _detect_relations,
    _enforce_proposal_contract,
    _merge_proposals,
    _path_payload,
    _parse_next_step,
    _rank_frontier,
    _sample_next_step,
    _score_nonconflict_group,
    _select_children,
)
from src.ldt.types import CandidateNode, ReasoningTree
from src.util import extract_json, is_nonanswer

from . import prompts


ALGORITHM_VERSION = "v8_contested_terminal"


def _clean_terminal_answer(answer: Any, dtype: str) -> str:
    text = _as_text(answer).strip().strip('"').strip("'").strip()
    if dtype == "qa":
        return text.strip()
    return _clean_answer(text, dtype)


def _is_explicit_atomic_quantity(cand: CandidateNode, dtype: str) -> bool:
    if dtype != "qa":
        return False
    answer = _clean_terminal_answer(cand.answer, dtype)
    if not answer or not re.search(r"\d", answer):
        return False
    return normalize_qa(answer) in normalize_qa(cand.text)


def _needs_strict_role_edges(question: str, dtype: str) -> bool:
    if dtype != "qa":
        return False
    text = re.sub(r"\s+", " ", (question or "").lower())
    family_or_direction = re.search(
        r"\b(father-in-law|mother-in-law|paternal|maternal|grandfather|grandmother|"
        r"grandparent|father|mother|parent|spouse|husband|wife|married|son|daughter|"
        r"child|children|brother|sister|sibling)\b",
        text,
    )
    alternative_comparison = re.search(r"\bor\b", text) and re.search(
        r"\b(first|earlier|later|older|younger|before|after)\b",
        text,
    )
    return bool(family_or_direction or alternative_comparison)


def _requested_location_granularity(question: str) -> str:
    text = re.sub(r"\s+", " ", (question or "").lower())
    match = re.search(
        r"\b(?:what|which|in which)\s+"
        r"(borough|county|province|state|country|district|municipality|region)\b",
        text,
    )
    return match.group(1) if match else ""


def _debated_contract(agents, client, tr: Tracker, question: str) -> dict:
    proposals = []
    for agent in agents:
        raw = ask(
            agent.client,
            tr,
            [{"role": "system", "content": f"{prompts.CONTRACT_SYS} {prompts.JSON_ONLY}"},
             {"role": "user", "content": prompts.contract_user(question)}],
            temperature=agent.temperature,
            max_tokens=min(500, agent.max_tokens),
        )
        data = extract_json(raw) or {}
        data["agent"] = agent.idx
        data["raw"] = raw
        proposals.append(data)
    raw = ask(
        client,
        tr,
        [{"role": "system", "content": f"{prompts.CONTRACT_REFEREE_SYS} {prompts.JSON_ONLY}"},
         {"role": "user", "content": prompts.resolve_contract_user(question, proposals)}],
        temperature=0.0,
        max_tokens=500,
    )
    resolved = extract_json(raw) or {}
    resolved["raw"] = raw
    return {"agent_contracts": proposals, "resolved": resolved}


def _question_with_contract(question: str, contract_trace: dict) -> str:
    resolved = contract_trace.get("resolved") or {}
    slot = _as_text(resolved.get("answer_slot"))
    bridge = _as_text(resolved.get("bridge_contract"))
    rejects = resolved.get("reject_if") or []
    if isinstance(rejects, str):
        rejects = [rejects]
    lines = []
    if slot:
        lines.append(f"Answer slot: {slot}")
    if bridge:
        lines.append(f"Bridge contract: {bridge}")
    for item in rejects[:4]:
        text = _as_text(item)
        if text:
            lines.append(f"Reject if: {text}")
    if not lines:
        return question
    return question.rstrip() + "\n\nDebated reasoning contract:\n" + "\n".join(f"- {x}" for x in lines)


def _sample_challenge(agent, question: str, tree: ReasoningTree, parent_id: str,
                      majority: CandidateNode, depth: int, dtype: str,
                      tr: Tracker) -> CandidateNode | None:
    raw = ask(
        agent.client,
        tr,
        [{"role": "system", "content": f"{prompts.CHALLENGE_SYS} {prompts.JSON_ONLY}"},
         {"role": "user", "content": prompts.challenge_user(
             question, tree.prefix_text(parent_id), majority.to_dict(), dtype)}],
        temperature=agent.temperature,
        max_tokens=min(500, agent.max_tokens),
    )
    data = extract_json(raw) or {}
    if not _as_text(data.get("node") or data.get("next_hop") or data.get("claim")):
        return None
    prop = _parse_next_step(raw, proposal_id=depth * 1000 + 900 + agent.idx,
                            agent_idx=agent.idx, dtype=dtype)
    prop = _enforce_proposal_contract(prop, question, dtype)
    if not prop.text:
        return None
    if normalize_qa(prop.text) == normalize_qa(majority.text):
        return None
    cand = CandidateNode(
        candidate_id=f"{parent_id}_d{depth}_challenge_a{agent.idx}",
        text=prop.text,
        depth=depth,
        parent_id=parent_id,
        support_agents=[agent.idx],
        proposals=[prop],
        confidence=prop.confidence,
        score=_clamp(prop.confidence - 0.08, default=0.50),
        status="uncertain",
        is_final=prop.is_final,
        answer=prop.answer,
        merge_notes="adversarial challenger",
        history={
            "role": "adversarial_challenge",
            "majority_candidate": majority.to_dict(),
            "raw": raw,
        },
    )
    return cand


def _answer_keys(candidates: list[CandidateNode], members: list[int]) -> set[str]:
    keys = set()
    for idx in members:
        answer = _clean_answer(candidates[idx].answer, "qa")
        key = normalize_qa(answer)
        if key:
            keys.add(key)
    return keys


def _should_debate(group: dict, candidates: list[CandidateNode]) -> bool:
    members = list(group.get("members") or [])
    if len(members) <= 1:
        return False
    if group.get("type") == "conflict":
        return True
    if group.get("type") == "uncertain":
        return True
    return len(_answer_keys(candidates, members)) > 1


def _absorb_compatible_challenges(group: dict, candidates: list[CandidateNode]) -> None:
    if group.get("type") != "compatible":
        return
    for idx in group.get("members") or []:
        cand = candidates[idx]
        if cand.history.get("role") != "adversarial_challenge":
            continue
        cand.status = "rejected"
        cand.score = min(cand.score, 0.35)
        cand.history.setdefault("challenge_resolution", []).append({
            "action": "absorbed_not_expanded",
            "reason": "relation judge found the challenge compatible rather than competing",
        })


def _synthesize_compatible_group(
    parent_id: str,
    depth: int,
    group: dict,
    candidates: list[CandidateNode],
) -> CandidateNode | None:
    if group.get("type") != "compatible":
        return None
    members = list(group.get("members") or [])
    if len(members) <= 1:
        return None
    unique_texts = []
    seen = set()
    support_agents = set()
    proposals = []
    scores = []
    confidences = []
    answers = []
    for idx in members:
        cand = candidates[idx]
        key = normalize_qa(cand.text)
        if key and key not in seen:
            seen.add(key)
            unique_texts.append(cand.text)
        support_agents.update(cand.support_agents)
        proposals.extend(cand.proposals)
        scores.append(cand.score)
        confidences.append(cand.confidence)
        if cand.answer:
            answers.append(cand.answer)
    if len(unique_texts) <= 1:
        return None
    answer = ""
    answer_keys = {normalize_qa(a) for a in answers if normalize_qa(a)}
    if len(answer_keys) == 1:
        answer = answers[0]
    synth_id = f"{parent_id}_d{depth}_compat_synth_{members[0]}"
    synth = CandidateNode(
        candidate_id=synth_id,
        text="; ".join(unique_texts),
        depth=depth,
        parent_id=parent_id,
        support_agents=sorted(support_agents),
        proposals=proposals,
        confidence=sum(confidences) / max(1, len(confidences)),
        score=max(scores) if scores else 0.62,
        status="accepted",
        is_final=bool(answer),
        answer=answer,
        merge_notes="compatible candidates synthesized after relation judgment",
        history={
            "role": "compatible_synthesis",
            "source_candidate_ids": [candidates[i].candidate_id for i in members],
            "relation_group": group,
        },
    )
    for idx in members:
        cand = candidates[idx]
        cand.status = "rejected"
        cand.score = min(cand.score, 0.35)
        cand.history.setdefault("synthesized_into", synth_id)
    return synth


def _synthesis_allowed(
    client,
    tr: Tracker,
    question: str,
    tree: ReasoningTree,
    parent_id: str,
    group: dict,
    candidates: list[CandidateNode],
) -> tuple[bool, dict]:
    members = list(group.get("members") or [])
    payload = [candidates[i].to_dict() for i in members]
    if any(candidates[i].is_final or candidates[i].answer for i in members):
        return False, {
            "used": False,
            "combine": False,
            "reason": "terminal candidates are rivals, not synthesis evidence",
            "members": members,
        }
    raw = ask(
        client,
        tr,
        [{"role": "system", "content": f"{prompts.SYNTHESIS_SYS} {prompts.JSON_ONLY}"},
         {"role": "user", "content": prompts.synthesis_user(
             question, tree.prefix_text(parent_id), payload)}],
        temperature=0.0,
        max_tokens=260,
    )
    data = extract_json(raw) or {}
    combine = data.get("combine")
    if isinstance(combine, str):
        combine = combine.strip().lower() in ("true", "yes", "1")
    return bool(combine), {
        "used": True,
        "combine": bool(combine),
        "members": members,
        "raw": raw,
        "parsed": data,
    }


def _terminal_answer_from_path(path: dict, dtype: str) -> str:
    hint = _clean_terminal_answer(path.get("answer_hint"), dtype)
    if hint:
        return hint
    nodes = path.get("nodes") or []
    if nodes:
        return _clean_terminal_answer(extract_answer(_as_text(nodes[-1].get("text")), dtype), dtype)
    return ""


def _ranked_terminal_ids(tree: ReasoningTree, *, prefer_deeper_on_tie: bool) -> list[str]:
    terminals = tree.final_ids() or tree.leaf_ids()
    if not prefer_deeper_on_tie:
        return tree.ranked_terminal_ids()
    return sorted(
        terminals,
        key=lambda nid: (
            tree.path_score(nid),
            tree.nodes[nid].score,
            tree.nodes[nid].depth,
        ),
        reverse=True,
    )


def _yes_no_stance_vote(paths: list[dict], dtype: str) -> dict[str, Any]:
    if dtype != "qa":
        return {"used": False}
    weights = {"yes": 0.0, "no": 0.0}
    counts = {"yes": 0, "no": 0}
    voters = []
    for path in paths:
        stance = normalize_qa(_terminal_answer_from_path(path, dtype))
        if stance not in weights:
            continue
        weight = float(path.get("path_score") or 0.0)
        weights[stance] += weight
        counts[stance] += 1
        voters.append({
            "path_id": path.get("path_id"),
            "stance": stance,
            "weight": weight,
        })
    if not voters:
        return {"used": False}
    winner = "yes" if weights["yes"] >= weights["no"] else "no"
    loser = "no" if winner == "yes" else "yes"
    return {
        "used": True,
        "winner": winner,
        "winner_weight": weights[winner],
        "loser_weight": weights[loser],
        "margin": weights[winner] - weights[loser],
        "counts": counts,
        "voters": voters,
    }


def _allow_yes_no_stance_vote(question: str) -> bool:
    text = re.sub(r"\s+", " ", (question or "").lower())
    comparative = re.search(
        r"\b(more|less|fewer|greater|larger|smaller|higher|lower|older|younger|"
        r"earlier|later|longer|shorter)\b.{0,80}\bthan\b",
        text,
    )
    return not bool(comparative)


def _final_path_debate(
    agents,
    client,
    tr: Tracker,
    question: str,
    tree: ReasoningTree,
    dtype: str,
    *,
    top_paths: int,
    debate_rounds: int,
    use_slot_grounding: bool,
    strict_role_edges: bool,
    use_binary_stance_vote: bool,
) -> tuple[str, dict]:
    terminal_ids = _ranked_terminal_ids(
        tree, prefer_deeper_on_tie=True)[:top_paths]
    if not terminal_ids:
        return "", {"reason": "empty-tree", "debated": False}
    paths = [_path_payload(tree, nid, f"p{i}", dtype) for i, nid in enumerate(terminal_ids)]
    transcript = []
    for r in range(max(1, debate_rounds)):
        for agent in agents:
            raw = ask(
                agent.client,
                tr,
                [{"role": "system", "content": f"{prompts.FINAL_DEBATE_SYS} {prompts.JSON_ONLY}"},
                 {"role": "user", "content": prompts.final_path_debate_user(
                     question, paths, dtype, strict_role_edges=strict_role_edges)}],
                temperature=agent.temperature,
                max_tokens=min(700, agent.max_tokens),
            )
            transcript.append({
                "round": r,
                "agent": agent.idx,
                "raw": raw,
                "parsed": extract_json(raw) or {},
            })

    raw = ask(
        client,
        tr,
        [{"role": "system", "content": f"{prompts.FINAL_REFEREE_SYS} {prompts.JSON_ONLY}"},
         {"role": "user", "content": prompts.resolve_final_debate_user(
             question, paths, transcript, dtype, strict_role_edges=strict_role_edges)}],
        temperature=0.0,
        max_tokens=500,
    )
    data = extract_json(raw) or {}
    answer = _clean_terminal_answer(data.get("answer"), dtype)
    chosen_path = _as_text(data.get("chosen_path"))
    format_repair = {"used": False}
    if dtype == "qa" and base_prompts.is_yes_no_question(question):
        chosen_payload = next((p for p in paths if p.get("path_id") == chosen_path), None)
        hinted = _terminal_answer_from_path(chosen_payload or {}, dtype)
        hinted_key = normalize_qa(hinted)
        answer_key = normalize_qa(answer)
        if hinted_key in ("yes", "no") and answer_key in ("yes", "no") and hinted_key != answer_key:
            format_repair = {
                "used": True,
                "reason": "chosen_path_yes_no_hint",
                "chosen_path": chosen_path,
                "resolution_answer": answer,
                "path_answer_hint": hinted,
            }
            answer = hinted
        vote = (
            _yes_no_stance_vote(paths, dtype)
            if use_binary_stance_vote and _allow_yes_no_stance_vote(question)
            else {"used": False}
        )
        if vote.get("used") and vote.get("margin", 0.0) >= 0.25:
            answer_key = normalize_qa(answer)
            if answer_key in ("yes", "no") and answer_key != vote.get("winner"):
                format_repair = {
                    "used": True,
                    "reason": "weighted_terminal_yes_no_vote",
                    "previous_repair": format_repair,
                    "resolution_answer": answer,
                    "vote": vote,
                }
                answer = vote["winner"]
    if not answer:
        answer, bridge_log = _bridge_completion_challenge(
            agents, client, tr, question, paths, dtype, debate_rounds)
    else:
        bridge_log = None
    if use_slot_grounding:
        answer, slot_grounding = _slot_grounding_debate(
            agents, client, tr, question, paths, answer, data, dtype, debate_rounds)
    else:
        slot_grounding = {"used": False}
    absence_log = None
    if dtype == "qa" and is_nonanswer(answer):
        answer, absence_log = _absence_challenge(
            agents, client, tr, question, answer, paths, dtype, debate_rounds)
    return answer, {
        "debated": True,
        "reason": "terminal_path_cross_examination" if len(paths) == 1 else "terminal_path_debate",
        "paths": paths,
        "transcript": transcript,
        "resolution": data,
        "resolution_raw": raw,
        "chosen_path": chosen_path,
        "parse_ok": bool(data),
        "bridge_completion": bridge_log,
        "absence_challenge": absence_log,
        "slot_grounding": slot_grounding,
        "format_repair": format_repair,
    }


def _terminality_debate(
    agents,
    client,
    tr: Tracker,
    question: str,
    tree: ReasoningTree,
    parent_id: str,
    cand: CandidateNode,
    dtype: str,
    debate_rounds: int,
    strict_role_edges: bool,
    use_granularity_guard: bool,
) -> dict[str, Any]:
    critiques = []
    prefix = tree.prefix_text(parent_id)
    candidate_payload = cand.to_dict()
    candidate_payload.pop("history", None)
    for r in range(max(1, debate_rounds)):
        for agent in agents:
            raw = ask(
                agent.client,
                tr,
                [{"role": "system", "content": f"{prompts.TERMINALITY_DEBATE_SYS} {prompts.JSON_ONLY}"},
                 {"role": "user", "content": prompts.terminality_debate_user(
                     question, prefix, candidate_payload, dtype,
                     strict_role_edges=strict_role_edges)}],
                temperature=agent.temperature,
                max_tokens=min(600, agent.max_tokens),
            )
            critiques.append({
                "round": r,
                "agent": agent.idx,
                "raw": raw,
                "parsed": extract_json(raw) or {},
            })
    raw = ask(
        client,
        tr,
        [{"role": "system", "content": f"{prompts.TERMINALITY_REFEREE_SYS} {prompts.JSON_ONLY}"},
         {"role": "user", "content": prompts.resolve_terminality_user(
             question, prefix, candidate_payload, critiques, dtype,
             strict_role_edges=strict_role_edges)}],
        temperature=0.0,
        max_tokens=420,
    )
    data = extract_json(raw) or {}
    is_terminal = _boolish(data.get("is_terminal"), default=cand.is_final)
    slot_match = _as_text(data.get("slot_match")).lower()
    relation_complete = _boolish(data.get("relation_complete"), default=is_terminal)
    role_direction_ok = _boolish(data.get("role_direction_ok"), default=True)
    unsupported_assertion = _boolish(data.get("unsupported_assertion"), default=False)
    answer = _clean_terminal_answer(data.get("answer"), dtype)
    missing_hop = _as_text(data.get("missing_hop"))
    atomic_quantity = _is_explicit_atomic_quantity(cand, dtype)
    requested_granularity = (
        _requested_location_granularity(question)
        if use_granularity_guard and dtype == "qa"
        else ""
    )
    granularity_missing = False
    if slot_match and slot_match != "exact":
        is_terminal = False
    if not relation_complete:
        is_terminal = False
    if strict_role_edges and (not role_direction_ok or unsupported_assertion):
        is_terminal = False
    if strict_role_edges and is_terminal and answer:
        bare_entity_answer = normalize_qa(cand.text) == normalize_qa(answer)
        answer_in_prefix = normalize_qa(answer) in normalize_qa(prefix)
        if bare_entity_answer and not answer_in_prefix:
            is_terminal = False
            missing_hop = missing_hop or "prove the required role edge for this entity"
    if requested_granularity and is_terminal and answer:
        haystack = f"{cand.text} {answer}".lower()
        if requested_granularity not in haystack:
            is_terminal = False
            granularity_missing = True
            missing_hop = missing_hop or f"map the location to the requested {requested_granularity}"
    if atomic_quantity:
        is_terminal = True
        relation_complete = True
        role_direction_ok = True
        unsupported_assertion = False
        granularity_missing = False
        if slot_match and slot_match != "exact":
            slot_match = "exact_quantity"
    if is_terminal and not answer:
        answer = _clean_terminal_answer(cand.answer, dtype)
    log = {
        "used": True,
        "candidate_id": cand.candidate_id,
        "candidate_before": candidate_payload,
        "critiques": critiques,
        "resolution": data,
        "resolution_raw": raw,
        "parse_ok": bool(data),
        "is_terminal": is_terminal,
        "answer": answer,
        "missing_hop": missing_hop,
        "slot_match": slot_match,
        "relation_complete": relation_complete,
        "role_direction_ok": role_direction_ok,
        "unsupported_assertion": unsupported_assertion,
        "strict_role_edges": strict_role_edges,
        "requested_granularity": requested_granularity,
        "granularity_missing": granularity_missing,
        "atomic_quantity": atomic_quantity,
    }
    if is_terminal:
        cand.is_final = True
        cand.answer = answer or cand.answer
        cand.history.setdefault("terminality_debate", []).append({
            **log,
            "action": "keep_terminal",
        })
    else:
        cand.is_final = False
        cand.answer = ""
        cand.history.setdefault("terminality_debate", []).append({
            **log,
            "action": "reopen_as_intermediate",
        })
    return log


def _audit_terminal_candidates(
    agents,
    client,
    tr: Tracker,
    question: str,
    tree: ReasoningTree,
    parent_id: str,
    candidates: list[CandidateNode],
    dtype: str,
    debate_rounds: int,
    keep_contested_terminals: bool,
    strict_role_edges: bool,
    use_granularity_guard: bool,
) -> list[dict[str, Any]]:
    logs = []
    contested = []
    for cand in candidates:
        if not cand.is_final and not cand.answer:
            continue
        before = cand.to_dict()
        before.pop("history", None)
        logs.append(_terminality_debate(
            agents, client, tr, question, tree, parent_id, cand, dtype,
            debate_rounds, strict_role_edges, use_granularity_guard))
        log = logs[-1]
        if (
            keep_contested_terminals
            and not log.get("is_terminal")
            and _clean_terminal_answer(before.get("answer"), dtype)
        ):
            contested.append(CandidateNode(
                candidate_id=f"{before['candidate_id']}_contested_terminal",
                text=before["text"],
                depth=before["depth"],
                parent_id=before["parent_id"],
                support_agents=list(before.get("support_agents") or []),
                proposals=list(cand.proposals),
                confidence=float(before.get("confidence") or cand.confidence),
                score=min(float(before.get("score") or cand.score), 0.54),
                status="uncertain",
                is_final=True,
                answer=_clean_terminal_answer(before.get("answer"), dtype),
                merge_notes="contested terminal retained after terminality debate",
                history={
                    "role": "contested_terminal",
                    "terminality_debate": log,
                    "source_candidate_before_reopen": before,
                },
            ))
    candidates.extend(contested)
    for cand in contested:
        logs.append({
            "used": True,
            "candidate_id": cand.candidate_id,
            "is_terminal": True,
            "answer": cand.answer,
            "action": "retain_contested_terminal_branch",
            "source_candidate_id": cand.history["source_candidate_before_reopen"].get("candidate_id"),
        })
    return logs


def _slot_grounding_debate(
    agents,
    client,
    tr: Tracker,
    question: str,
    paths: list[dict],
    current_answer: str,
    final_resolution: dict,
    dtype: str,
    debate_rounds: int,
) -> tuple[str, dict]:
    arguments = []
    for r in range(max(1, debate_rounds)):
        for agent in agents:
            raw = ask(
                agent.client,
                tr,
                [{"role": "system", "content": f"{prompts.SLOT_GROUNDING_SYS} {prompts.JSON_ONLY}"},
                 {"role": "user", "content": prompts.slot_grounding_user(
                     question, paths, current_answer, final_resolution, dtype)}],
                temperature=agent.temperature,
                max_tokens=min(700, agent.max_tokens),
            )
            arguments.append({
                "round": r,
                "agent": agent.idx,
                "raw": raw,
                "parsed": extract_json(raw) or {},
            })
    raw = ask(
        client,
        tr,
        [{"role": "system", "content": f"{prompts.FINAL_REFEREE_SYS} {prompts.JSON_ONLY}"},
         {"role": "user", "content": prompts.resolve_slot_grounding_user(
             question, paths, current_answer, arguments, dtype)}],
        temperature=0.0,
        max_tokens=400,
    )
    data = extract_json(raw) or {}
    answer = _clean_terminal_answer(data.get("answer"), dtype)
    changed = data.get("changed")
    if isinstance(changed, str):
        changed = changed.strip().lower() in ("true", "yes", "1")
    if not answer and not changed:
        answer = current_answer
    return answer, {
        "used": True,
        "current_answer": current_answer,
        "arguments": arguments,
        "resolution": data,
        "resolution_raw": raw,
        "parse_ok": bool(data),
    }


def _bridge_completion_challenge(
    agents,
    client,
    tr: Tracker,
    question: str,
    paths: list[dict],
    dtype: str,
    debate_rounds: int,
) -> tuple[str, dict]:
    completions = []
    for r in range(max(1, debate_rounds)):
        for agent in agents:
            raw = ask(
                agent.client,
                tr,
                [{"role": "system", "content": f"{prompts.BRIDGE_COMPLETION_SYS} {prompts.JSON_ONLY}"},
                 {"role": "user", "content": prompts.bridge_completion_user(question, paths, dtype)}],
                temperature=agent.temperature,
                max_tokens=min(700, agent.max_tokens),
            )
            completions.append({
                "round": r,
                "agent": agent.idx,
                "raw": raw,
                "parsed": extract_json(raw) or {},
            })
    raw = ask(
        client,
        tr,
        [{"role": "system", "content": f"{prompts.FINAL_REFEREE_SYS} {prompts.JSON_ONLY}"},
         {"role": "user", "content": prompts.resolve_bridge_completion_user(
             question, paths, completions, dtype)}],
        temperature=0.0,
        max_tokens=400,
    )
    data = extract_json(raw) or {}
    answer = _clean_terminal_answer(data.get("answer"), dtype)
    return answer, {
        "used": True,
        "completions": completions,
        "resolution": data,
        "resolution_raw": raw,
        "parse_ok": bool(data),
    }


def _absence_challenge(
    agents,
    client,
    tr: Tracker,
    question: str,
    selected_answer: str,
    paths: list[dict],
    dtype: str,
    debate_rounds: int,
) -> tuple[str, dict]:
    challenges = []
    for r in range(max(1, debate_rounds)):
        for agent in agents:
            raw = ask(
                agent.client,
                tr,
                [{"role": "system", "content": f"{prompts.ABSENCE_CHALLENGE_SYS} {prompts.JSON_ONLY}"},
                 {"role": "user", "content": prompts.absence_challenge_user(
                     question, selected_answer, paths, dtype)}],
                temperature=agent.temperature,
                max_tokens=min(700, agent.max_tokens),
            )
            challenges.append({
                "round": r,
                "agent": agent.idx,
                "raw": raw,
                "parsed": extract_json(raw) or {},
            })
    raw = ask(
        client,
        tr,
        [{"role": "system", "content": f"{prompts.FINAL_REFEREE_SYS} {prompts.JSON_ONLY}"},
         {"role": "user", "content": prompts.resolve_absence_challenge_user(
             question, selected_answer, paths, challenges, dtype)}],
        temperature=0.0,
        max_tokens=400,
    )
    data = extract_json(raw) or {}
    answer = _clean_terminal_answer(data.get("answer"), dtype)
    if not answer:
        answer = selected_answer
    return answer, {
        "used": True,
        "selected_absence_answer": selected_answer,
        "challenges": challenges,
        "resolution": data,
        "resolution_raw": raw,
        "parse_ok": bool(data),
    }


def _run_debate_tree(
    agents,
    ex,
    dtype,
    *,
    max_depth: int,
    per_parent_beam: int,
    global_beam: int,
    debate_budget: int,
    debate_rounds: int,
    uncertain_threshold: float,
    final_confidence: float,
    min_depth_before_final: int,
    final_top_paths: int,
    llm_merge: bool,
    force_keep_best: bool,
    use_contract: bool,
    use_synthesis: bool,
    use_slot_grounding: bool,
    use_terminality_debate: bool,
    keep_contested_terminals: bool,
    open_frontier_patience: bool,
    open_frontier_margin: float,
    strict_role_edges: bool | None,
    use_granularity_guard: bool,
    use_binary_stance_vote: bool,
) -> dict[str, Any]:
    client = agents[0].client
    tr = Tracker()
    contract_trace = (
        _debated_contract(agents, client, tr, ex.question)
        if use_contract
        else {"agent_contracts": [], "resolved": {}, "disabled": True}
    )
    working_question = _question_with_contract(ex.question, contract_trace)
    clean_question = ex.meta.get("question_clean", ex.question)
    resolved_strict_role_edges = (
        _needs_strict_role_edges(clean_question, dtype)
        if strict_role_edges is None
        else bool(strict_role_edges)
    )
    effective_open_frontier_patience = open_frontier_patience and resolved_strict_role_edges
    tree = ReasoningTree(ex.question)
    frontier = [tree.root_id]
    layers = []
    local_debates = 0
    debated_groups = []

    for depth in range(1, max_depth + 1):
        layer_log = {"depth": depth, "frontier_in": list(frontier), "parents": []}
        next_frontier = []
        for parent_id in frontier:
            parent_log: dict[str, Any] = {"parent_id": parent_id, "prefix": tree.prefix_text(parent_id)}
            proposals = [
                _sample_next_step(agent, ex.question, tree, parent_id, depth, dtype, tr)
                for agent in agents
            ]
            parent_log["proposals"] = [p.to_dict() for p in proposals]
            live = [p for p in proposals if p.text]
            candidates, merge_meta = _merge_proposals(
                client, tr, ex.question, tree, parent_id, depth, live, len(agents),
                dtype=dtype, llm_merge=llm_merge)
            parent_log["merge"] = merge_meta
            challenge_log = {"used": False}
            if len(candidates) == 1:
                challenger = agents[depth % len(agents)]
                challenge = _sample_challenge(
                    challenger, ex.question, tree, parent_id, candidates[0],
                    depth, dtype, tr)
                challenge_log = {
                    "used": True,
                    "agent": challenger.idx,
                    "added": bool(challenge),
                    "candidate": challenge.to_dict() if challenge else None,
                }
                if challenge:
                    candidates.append(challenge)
            parent_log["adversarial_challenge"] = challenge_log
            parent_log["candidates_before_debate"] = [c.to_dict() for c in candidates]

            groups, relation_meta = _detect_relations(
                client, tr, working_question, tree, parent_id, candidates, dtype, len(agents))
            parent_log["relations"] = relation_meta
            group_logs = []
            synthesized = []
            synthesis_checks = []
            for group in groups:
                if _should_debate(group, candidates) and local_debates < debate_budget:
                    _, meta = _debate_group(
                        client, tr, agents, working_question, tree, parent_id,
                        candidates, group, dtype, rounds=debate_rounds)
                    local_debates += 1
                    debated_groups.append(meta)
                    group_logs.append({"mode": "local_debate", **meta})
                else:
                    group_logs.append(_score_nonconflict_group(candidates, group, len(agents)))
                if (
                    use_synthesis
                    and group.get("type") == "compatible"
                    and len(group.get("members") or []) > 1
                ):
                    allow, check = _synthesis_allowed(
                        client, tr, working_question, tree, parent_id, group, candidates)
                    synthesis_checks.append(check)
                    if allow:
                        synth = _synthesize_compatible_group(parent_id, depth, group, candidates)
                        if synth:
                            synthesized.append(synth)
                elif not use_synthesis:
                    _absorb_compatible_challenges(group, candidates)
            parent_log["group_processing"] = group_logs
            parent_log["synthesis_checks"] = synthesis_checks
            if synthesized:
                candidates.extend(synthesized)
            parent_log["compatible_synthesis"] = [c.to_dict() for c in synthesized]

            if depth < min_depth_before_final:
                for cand in candidates:
                    if cand.is_final:
                        cand.history.setdefault("depth_contract", []).append({
                            "action": "delay_final_until_min_depth",
                            "depth": depth,
                            "min_depth": min_depth_before_final,
                        })
                        cand.is_final = False
                        cand.answer = ""
            terminality_logs = (
                _audit_terminal_candidates(
                    agents, client, tr, working_question, tree, parent_id,
                    candidates, dtype, debate_rounds, keep_contested_terminals,
                    resolved_strict_role_edges, use_granularity_guard)
                if use_terminality_debate
                else []
            )
            parent_log["terminality_debate"] = terminality_logs
            parent_log["candidates_after_debate"] = [c.to_dict() for c in candidates]
            kept, prune_meta = _select_children(
                candidates,
                per_parent_beam=per_parent_beam,
                uncertain_threshold=uncertain_threshold,
                force_keep_best=force_keep_best,
            )
            parent_log["pruning"] = prune_meta
            added = []
            for cand in kept:
                node = tree.add_child(parent_id, cand)
                added.append(node.to_dict())
                if not node.is_final:
                    next_frontier.append(node.node_id)
            parent_log["added_children"] = added
            layer_log["parents"].append(parent_log)

        ranked = _rank_frontier(tree, next_frontier)
        frontier = ranked[:global_beam]
        layer_log["frontier_out_before_global_prune"] = ranked
        layer_log["frontier_out"] = list(frontier)
        layers.append(layer_log)

        final_ids = tree.final_ids()
        if final_ids and depth >= min_depth_before_final:
            best_final = max(final_ids, key=lambda nid: tree.path_score(nid))
            best_final_score = tree.path_score(best_final)
            open_rivals = []
            if effective_open_frontier_patience and frontier:
                rival_threshold = max(uncertain_threshold, best_final_score - open_frontier_margin)
                open_rivals = [
                    {
                        "node_id": nid,
                        "path_score": tree.path_score(nid),
                        "node_score": tree.nodes[nid].score,
                        "text": tree.nodes[nid].text,
                    }
                    for nid in frontier
                    if tree.path_score(nid) >= rival_threshold
                ]
            if best_final_score >= final_confidence and not open_rivals:
                layers[-1]["stop"] = {
                    "reason": "confident_debated_path",
                    "node_id": best_final,
                    "path_score": best_final_score,
                }
                break
            if best_final_score >= final_confidence and open_rivals:
                layers[-1]["deferred_stop"] = {
                    "reason": "open_frontier_patience",
                    "best_final": best_final,
                    "best_final_score": best_final_score,
                    "margin": open_frontier_margin,
                    "open_rivals": open_rivals,
                }
        if not frontier:
            layers[-1]["stop"] = {"reason": "empty_frontier"}
            break

    answer, final_debate = _final_path_debate(
        agents, client, tr, working_question, tree, dtype,
        top_paths=final_top_paths,
        debate_rounds=debate_rounds,
        use_slot_grounding=use_slot_grounding,
        strict_role_edges=resolved_strict_role_edges,
        use_binary_stance_vote=use_binary_stance_vote,
    )
    trace = {
        "algorithm": "ldtd",
        "algorithm_version": ALGORITHM_VERSION,
        "design": "layerwise multi-agent debate tree",
        "debated_contract": contract_trace,
        "original_question": ex.question,
        "config": {
            "max_depth": max_depth,
            "per_parent_beam": per_parent_beam,
            "global_beam": global_beam,
            "debate_budget": debate_budget,
            "debate_rounds": debate_rounds,
            "uncertain_threshold": uncertain_threshold,
            "final_confidence": final_confidence,
            "min_depth_before_final": min_depth_before_final,
            "final_top_paths": final_top_paths,
            "llm_merge": llm_merge,
            "force_keep_best": force_keep_best,
            "use_contract": use_contract,
            "use_synthesis": use_synthesis,
            "use_slot_grounding": use_slot_grounding,
            "use_terminality_debate": use_terminality_debate,
            "keep_contested_terminals": keep_contested_terminals,
            "open_frontier_patience": open_frontier_patience,
            "effective_open_frontier_patience": effective_open_frontier_patience,
            "open_frontier_margin": open_frontier_margin,
            "strict_role_edges": resolved_strict_role_edges,
            "strict_role_edges_mode": "auto" if strict_role_edges is None else "forced",
            "use_granularity_guard": use_granularity_guard,
            "use_binary_stance_vote": use_binary_stance_vote,
        },
        "layers": layers,
        "tree": tree.to_dict(),
        "final_path_debate": final_debate,
    }
    return {
        "pred": answer,
        "trace": trace,
        "n_debates": local_debates + int(bool(final_debate.get("debated"))),
        "n_local_debates": local_debates,
        "n_final_debates": int(bool(final_debate.get("debated"))),
        "debated_claims": [
            {"group": item.get("group"), "resolution": item.get("resolution")}
            for item in debated_groups
        ],
        **tr.as_dict(),
    }


def run(
    agents,
    ex,
    dtype,
    max_iters: int = 5,
    debate_rounds: int = 1,
    ldt_max_depth: int | None = None,
    ldt_per_parent_beam: int = 2,
    ldt_global_beam: int = 6,
    ldt_debate_budget: int = 24,
    ldt_uncertain_threshold: float = 0.50,
    ldt_final_confidence: float = 0.88,
    ldt_min_depth_before_final: int = 2,
    ldt_final_top_paths: int = 8,
    ldt_no_llm_merge: bool = False,
    ldt_use_contract: bool = False,
    ldt_use_synthesis: bool = False,
    ldt_use_slot_grounding: bool = False,
    ldt_use_terminality_debate: bool = True,
    ldt_keep_contested_terminals: bool = True,
    ldt_open_frontier_patience: bool = False,
    ldt_open_frontier_margin: float = 0.12,
    ldt_strict_role_edges: bool | None = False,
    ldt_use_granularity_guard: bool = False,
    ldt_use_binary_stance_vote: bool = False,
    **kw,
):
    """v8 Contested Terminal Debate Tree."""
    kw.pop("dataset", None)
    max_depth = ldt_max_depth if ldt_max_depth is not None else max_iters
    return _run_debate_tree(
        agents,
        ex,
        dtype,
        max_depth=max_depth,
        per_parent_beam=ldt_per_parent_beam,
        global_beam=ldt_global_beam,
        debate_budget=ldt_debate_budget,
        debate_rounds=debate_rounds,
        uncertain_threshold=ldt_uncertain_threshold,
        final_confidence=ldt_final_confidence,
        min_depth_before_final=ldt_min_depth_before_final,
        final_top_paths=ldt_final_top_paths,
        llm_merge=not ldt_no_llm_merge,
        force_keep_best=True,
        use_contract=ldt_use_contract,
        use_synthesis=ldt_use_synthesis,
        use_slot_grounding=ldt_use_slot_grounding,
        use_terminality_debate=ldt_use_terminality_debate,
        keep_contested_terminals=ldt_keep_contested_terminals,
        open_frontier_patience=ldt_open_frontier_patience,
        open_frontier_margin=ldt_open_frontier_margin,
        strict_role_edges=ldt_strict_role_edges,
        use_granularity_guard=ldt_use_granularity_guard,
        use_binary_stance_vote=ldt_use_binary_stance_vote,
    )
