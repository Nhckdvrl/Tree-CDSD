from __future__ import annotations

import itertools
import json
import re
from collections import defaultdict
from typing import Any

from src.eval.graders import extract_answer, normalize_qa
from src.llm.client import Tracker, ask
from src.util import extract_json, is_nonanswer
from src.ldt import prompts
from src.ldt.types import CandidateNode, ReasoningTree, StepProposal


def _clamp(x: Any, lo: float = 0.0, hi: float = 1.0, default: float = 0.5) -> float:
    try:
        val = float(x)
    except Exception:
        return default
    return max(lo, min(hi, val))


def _boolish(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "yes", "1", "final"):
            return True
        if v in ("false", "no", "0", "not_final"):
            return False
    return default


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("node", "text", "claim", "answer", "statement"):
            val = value.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return str(value).strip() if value is not None else ""


def _extract_loose_json(text: str) -> dict | None:
    d = extract_json(text)
    if isinstance(d, dict):
        return d
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    start, end = t.find("{"), t.rfind("}")
    if start >= 0 and end > start:
        chunk = t[start:end + 1]
        chunk = re.sub(r":\s*False\b", ": false", chunk)
        chunk = re.sub(r":\s*True\b", ": true", chunk)
        chunk = re.sub(r":\s*None\b", ": null", chunk)
        try:
            parsed = json.loads(chunk)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None
    return None


def _norm_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    text = re.sub(r"^(?:step|hop|node)\s*\d+\s*[:.)-]\s*", "", text, flags=re.I)
    text = re.sub(r"^\d+\s*[:.)-]\s*", "", text)
    text = re.sub(r"^(?:final\s+)?answer\s*[:=]\s*", "", text, flags=re.I)
    return normalize_qa(text)


def _token_jaccard(a: str, b: str) -> float:
    aa = set(_norm_text(a).split())
    bb = set(_norm_text(b).split())
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / len(aa | bb)


def _strip_numbering(text: str) -> str:
    return re.sub(r"^\s*\d+\s*[:.)-]\s*", "", text or "").strip()


def _safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except Exception:
        return default


def _line_field(text: str, key: str) -> str:
    m = re.search(rf"(?im)^[ \t,]*\"?{re.escape(key)}\"?[ \t]*:[ \t]*(.*)$", text or "")
    if not m:
        return ""
    return m.group(1).strip().strip('"').strip("'").strip()


def _clean_answer(answer: str, dtype: str) -> str:
    answer = (answer or "").strip().strip('"').strip("'").strip()
    if dtype == "qa":
        answer = re.sub(r"^\d+\s*[:.)-]\s*", "", answer).strip()
        if not answer or answer.lower().rstrip(":") == "answer":
            return ""
        if re.fullmatch(r"\d+\.?", answer):
            return ""
    return answer


def _violates_answer_contract(question: str, dtype: str, answer: str) -> bool:
    if dtype != "qa" or not prompts.is_yes_no_question(question):
        return False
    return bool(answer) and normalize_qa(answer) not in ("yes", "no")


def _enforce_proposal_contract(prop: StepProposal, question: str, dtype: str) -> StepProposal:
    if _violates_answer_contract(question, dtype, prop.answer):
        prop.answer = ""
        prop.is_final = False
    if dtype == "qa" and prompts.is_yes_no_question(question) and prop.is_final:
        if normalize_qa(prop.answer) not in ("yes", "no"):
            prop.answer = ""
            prop.is_final = False
    return prop


def _enforce_candidate_contract(cand: CandidateNode, question: str, dtype: str) -> None:
    if cand.is_final and not cand.answer:
        extracted = extract_answer(cand.text, dtype)
        cleaned = _clean_answer(extracted, dtype) if extracted and len(extracted.split()) <= 16 else ""
        if cleaned:
            cand.answer = cleaned
        else:
            cand.history.setdefault("answer_contract", []).append({
                "action": "unset_final_without_answer",
                "text": cand.text,
            })
            cand.is_final = False
    if _violates_answer_contract(question, dtype, cand.answer):
        cand.history.setdefault("answer_contract", []).append({
            "action": "cleared_non_yes_no_answer",
            "answer": cand.answer,
        })
        cand.answer = ""
        cand.is_final = False
    if dtype == "qa" and prompts.is_yes_no_question(question) and cand.is_final:
        if normalize_qa(cand.answer) not in ("yes", "no"):
            cand.history.setdefault("answer_contract", []).append({
                "action": "unset_final_without_yes_no_answer",
                "answer": cand.answer,
            })
            cand.answer = ""
            cand.is_final = False


def _candidate_index(value: Any, candidates: list[CandidateNode]) -> int | None:
    if isinstance(value, int):
        return value if 0 <= value < len(candidates) else None
    if isinstance(value, str):
        val = value.strip()
        if val.isdigit():
            i = int(val)
            return i if 0 <= i < len(candidates) else None
        for i, cand in enumerate(candidates):
            if val == cand.candidate_id:
                return i
    return None


def _path_nodes(tree: ReasoningTree, parent_id: str):
    return tree.path_to(parent_id, include_root=False)


def _parse_next_step(raw: str, proposal_id: int, agent_idx: int, dtype: str) -> StepProposal:
    d = _extract_loose_json(raw)
    if isinstance(d, dict):
        text = _as_text(d.get("node") or d.get("next_hop") or d.get("text") or d.get("claim"))
        answer = _clean_answer(_as_text(d.get("answer")), dtype)
        is_final = _boolish(d.get("is_final"), default=False) or bool(answer)
        if not answer and is_final:
            answer = _clean_answer(extract_answer(text, dtype), dtype)
        if not text and answer:
            text = f"Final answer: {answer}"
        if text:
            return StepProposal(
                proposal_id=proposal_id,
                agent_idx=agent_idx,
                text=text,
                answer=answer,
                is_final=is_final,
                confidence=_clamp(d.get("confidence"), default=0.55),
                raw=raw,
                parse_ok=True,
            )
    node_line = _line_field(raw, "node")
    if node_line:
        answer = _clean_answer(_line_field(raw, "answer"), dtype)
        is_final_raw = _line_field(raw, "is_final")
        conf_raw = _line_field(raw, "confidence")
        return StepProposal(
            proposal_id=proposal_id,
            agent_idx=agent_idx,
            text=node_line,
            answer=answer,
            is_final=_boolish(is_final_raw, default=False) or bool(answer),
            confidence=_clamp(conf_raw, default=0.5),
            raw=raw,
            parse_ok=False,
        )
    stripped_lines = [line.strip().rstrip(",") for line in (raw or "").splitlines() if line.strip()]
    quoted = [line.strip().strip('"').strip("'") for line in stripped_lines]
    if quoted and not stripped_lines[0].startswith("{") and ":" not in stripped_lines[0]:
        node = quoted[0].strip()
        tail = "\n".join(stripped_lines[1:])
        if re.search(r'"?(?:is_final|answer|confidence)"?[ \t]*:', tail):
            final_match = re.search(r'"?is_final"?[ \t]*:[ \t]*(true|false|True|False)', tail)
            answer_match = re.search(r'"?answer"?[ \t]*:[ \t]*"?([^"\n]*)', tail, flags=re.S)
            conf_match = re.search(r'"?confidence"?[ \t]*:[ \t]*([0-9.]+)', tail)
            is_final = _boolish(final_match.group(1), default=False) if final_match else False
            answer = _clean_answer(answer_match.group(1).strip() if answer_match else "", dtype)
            conf = _clamp(conf_match.group(1), default=0.5) if conf_match else 0.5
        else:
            is_final = _boolish(quoted[1], default=False) if len(quoted) > 1 else False
            answer = _clean_answer(quoted[2].strip() if len(quoted) > 2 else "", dtype)
            conf = _clamp(quoted[3], default=0.5) if len(quoted) > 3 else 0.5
        if node:
            return StepProposal(
                proposal_id=proposal_id,
                agent_idx=agent_idx,
                text=node,
                answer=answer,
                is_final=is_final or bool(answer),
                confidence=conf,
                raw=raw,
                parse_ok=False,
            )

    node_match = re.search(r'"node"\s*:\s*"([^"]+)"', raw or "", flags=re.S)
    if node_match:
        node = node_match.group(1).strip()
        answer_match = re.search(r'"answer"\s*:\s*"([^"]*)"', raw or "", flags=re.S)
        final_match = re.search(r'"is_final"\s*:\s*(true|false|True|False)', raw or "")
        conf_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', raw or "")
        answer = _clean_answer(answer_match.group(1).strip() if answer_match else "", dtype)
        return StepProposal(
            proposal_id=proposal_id,
            agent_idx=agent_idx,
            text=node,
            answer=answer,
            is_final=_boolish(final_match.group(1), default=False) if final_match else bool(answer),
            confidence=_clamp(conf_match.group(1), default=0.5) if conf_match else 0.5,
            raw=raw,
            parse_ok=False,
        )

    answer = extract_answer(raw, dtype)
    text = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.I | re.S).strip()
    lines = [line.strip(" -*") for line in text.splitlines() if line.strip()]
    fallback = lines[0] if lines else text[:300].strip()
    if len(fallback.split()) > 35:
        fallback = " ".join(fallback.split()[:35])
    if not fallback and answer:
        fallback = f"Final answer: {answer}"
    return StepProposal(
        proposal_id=proposal_id,
        agent_idx=agent_idx,
        text=fallback,
        answer=_clean_answer(answer, dtype) if answer and len(answer.split()) <= 16 else "",
        is_final=bool(answer and len(answer.split()) <= 16),
        confidence=0.45,
        raw=raw,
        parse_ok=False,
    )


def _sample_next_step(agent, question: str, tree: ReasoningTree, parent_id: str,
                      depth: int, dtype: str, tr: Tracker, max_tokens: int = 700) -> StepProposal:
    user = prompts.next_step_user(question, _path_nodes(tree, parent_id), dtype)
    raw = ask(
        agent.client,
        tr,
        [{"role": "system", "content": prompts.NEXT_STEP_SYS},
         {"role": "user", "content": user}],
        temperature=agent.temperature,
        max_tokens=min(max_tokens, agent.max_tokens),
    )
    prop = _parse_next_step(raw, proposal_id=depth * 1000 + agent.idx, agent_idx=agent.idx, dtype=dtype)
    return _enforce_proposal_contract(prop, question, dtype)


def _deterministic_clusters(proposals: list[StepProposal], dtype: str) -> list[list[int]]:
    clusters: list[list[int]] = []
    for i, prop in enumerate(proposals):
        assigned = False
        for cluster in clusters:
            rep = proposals[cluster[0]]
            same_answer = bool(prop.answer and rep.answer and _norm_text(prop.answer) == _norm_text(rep.answer))
            exact = _norm_text(prop.text) == _norm_text(rep.text)
            near = _token_jaccard(prop.text, rep.text) >= 0.86
            final_same = prop.is_final and rep.is_final and same_answer
            if exact or near or final_same:
                cluster.append(i)
                assigned = True
                break
        if not assigned:
            clusters.append([i])
    return clusters


def _valid_cluster_members(raw_members: Any, n: int) -> list[int]:
    if not isinstance(raw_members, list):
        return []
    out = []
    for val in raw_members:
        idx = _safe_int(val)
        if idx is not None and 0 <= idx < n and idx not in out:
            out.append(idx)
    return out


def _llm_clusters(client, tr: Tracker, question: str, path_nodes, proposals: list[StepProposal],
                  max_tokens: int = 700) -> tuple[list[dict], dict]:
    if len(proposals) <= 1:
        return [], {"used": False, "reason": "single-proposal"}
    payload = [
        {
            "id": i,
            "agent": p.agent_idx,
            "node": p.text,
            "answer": p.answer,
            "is_final": p.is_final,
        }
        for i, p in enumerate(proposals)
    ]
    raw = ask(
        client,
        tr,
        [{"role": "system", "content": f"{prompts.MERGE_SYS} {prompts.JSON_ONLY}"},
         {"role": "user", "content": prompts.merge_user(question, path_nodes, payload)}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    d = extract_json(raw) or {}
    clusters = d.get("clusters") if isinstance(d, dict) else None
    if not isinstance(clusters, list):
        return [], {"used": True, "parse_ok": False, "raw": raw}
    cleaned = []
    used = set()
    for item in clusters:
        if not isinstance(item, dict):
            continue
        members = _valid_cluster_members(item.get("members"), len(proposals))
        members = [m for m in members if m not in used]
        if not members:
            continue
        used.update(members)
        cleaned.append({
            "members": members,
            "canonical_text": _as_text(item.get("canonical_text")),
            "reason": _as_text(item.get("reason")),
        })
    for i in range(len(proposals)):
        if i not in used:
            cleaned.append({"members": [i], "canonical_text": proposals[i].text, "reason": "merge fallback singleton"})
    return cleaned, {"used": True, "parse_ok": bool(cleaned), "raw": raw, "notes": d.get("notes", "")}


def _build_candidates(parent_id: str, depth: int, proposals: list[StepProposal],
                      clusters: list[dict], merge_meta: dict, n_agents: int,
                      question: str, dtype: str) -> list[CandidateNode]:
    candidates = []
    for j, cluster in enumerate(clusters):
        members = cluster.get("members") or []
        props = [proposals[i] for i in members]
        if not props:
            continue
        canonical = _as_text(cluster.get("canonical_text"))
        if not canonical:
            canonical = max((p.text for p in props), key=lambda s: len(s or ""))
        answers = [p.answer for p in props if p.answer]
        answer = _clean_answer(answers[0], dtype) if answers else ""
        is_final = any(p.is_final for p in props) or bool(answer)
        support_agents = sorted({p.agent_idx for p in props})
        avg_conf = sum(p.confidence for p in props) / len(props)
        support_bonus = min(0.18, 0.06 * max(0, len(support_agents) - 1))
        score = _clamp(avg_conf + support_bonus, default=0.55)
        cand = CandidateNode(
            candidate_id=f"{parent_id}_d{depth}_c{j}",
            text=canonical,
            depth=depth,
            parent_id=parent_id,
            support_agents=support_agents,
            proposals=props,
            confidence=avg_conf,
            score=score,
            status="accepted" if len(support_agents) > max(1, n_agents // 2) else "uncertain",
            is_final=is_final,
            answer=answer,
            merge_notes=_as_text(cluster.get("reason")),
            history={"merge": merge_meta},
        )
        _enforce_candidate_contract(cand, question, dtype)
        candidates.append(cand)
    return candidates


def _coalesce_equivalent_candidates(candidates: list[CandidateNode], question: str, dtype: str) -> list[CandidateNode]:
    out: list[CandidateNode] = []
    for cand in candidates:
        key = _norm_text(cand.text)
        merged = False
        for existing in out:
            same_text = key and key == _norm_text(existing.text)
            answers_conflict = bool(
                cand.answer and existing.answer
                and normalize_qa(cand.answer) != normalize_qa(existing.answer)
            )
            if same_text and not answers_conflict:
                existing.proposals.extend(cand.proposals)
                existing.support_agents = sorted(set(existing.support_agents) | set(cand.support_agents))
                existing.confidence = max(existing.confidence, cand.confidence)
                existing.score = max(existing.score, cand.score)
                if len(_strip_numbering(cand.text)) < len(_strip_numbering(existing.text)) or existing.text.startswith(("1.", "2.", "3.")):
                    existing.text = _strip_numbering(cand.text)
                if cand.answer and not existing.answer:
                    existing.answer = cand.answer
                existing.is_final = existing.is_final or cand.is_final
                existing.merge_notes = "; ".join(x for x in [existing.merge_notes, cand.merge_notes, "post-llm exact coalesce"] if x)
                existing.history.setdefault("coalesced_candidates", []).append(cand.to_dict())
                _enforce_candidate_contract(existing, question, dtype)
                merged = True
                break
        if not merged:
            cand.text = _strip_numbering(cand.text) or cand.text
            out.append(cand)
    for i, cand in enumerate(out):
        cand.candidate_id = f"{cand.parent_id}_d{cand.depth}_c{i}"
    return out


def _merge_proposals(client, tr: Tracker, question: str, tree: ReasoningTree, parent_id: str,
                     depth: int, proposals: list[StepProposal], n_agents: int, dtype: str,
                     llm_merge: bool = True) -> tuple[list[CandidateNode], dict]:
    det = _deterministic_clusters(proposals, dtype=dtype)
    path_nodes = _path_nodes(tree, parent_id)
    merge_meta: dict[str, Any] = {"deterministic_clusters": det, "llm_merge": llm_merge}
    clusters = [{"members": c, "canonical_text": "", "reason": "deterministic merge"} for c in det]
    if llm_merge and len(proposals) > 1:
        llm, llm_meta = _llm_clusters(client, tr, question, path_nodes, proposals)
        merge_meta["llm"] = llm_meta
        if llm:
            clusters = llm
    candidates = _build_candidates(parent_id, depth, proposals, clusters, merge_meta,
                                   n_agents, question, dtype)
    candidates = _coalesce_equivalent_candidates(candidates, question, dtype)
    return candidates, merge_meta


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _base_candidate_score(cand: CandidateNode, n_agents: int) -> float:
    support = cand.support / max(1, n_agents)
    final_bonus = 0.05 if cand.is_final and cand.answer else 0.0
    return _clamp(0.55 * cand.confidence + 0.35 * support + 0.10 + final_bonus, default=0.55)


def _detect_relations(client, tr: Tracker, question: str, tree: ReasoningTree, parent_id: str,
                      candidates: list[CandidateNode], dtype: str, n_agents: int,
                      max_tokens: int = 1000) -> tuple[list[dict], dict]:
    if not candidates:
        return [], {"used": False, "reason": "no-candidates"}
    if len(candidates) == 1:
        cand = candidates[0]
        cand.score = max(cand.score, _base_candidate_score(cand, n_agents))
        cand.status = "accepted"
        return [{"type": "compatible", "members": [0], "reason": "single candidate"}], {
            "used": False,
            "reason": "single-candidate",
        }

    payload = [
        {
            "id": i,
            "candidate_id": c.candidate_id,
            "node": c.text,
            "support_agents": c.support_agents,
            "support": c.support,
            "answer": c.answer,
            "is_final": c.is_final,
        }
        for i, c in enumerate(candidates)
    ]
    raw = ask(
        client,
        tr,
        [{"role": "system", "content": f"{prompts.RELATION_SYS} {prompts.JSON_ONLY}"},
         {"role": "user", "content": prompts.relation_user(question, _path_nodes(tree, parent_id), payload, dtype)}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    d = extract_json(raw) or {}
    scores = d.get("scores") if isinstance(d, dict) else None
    if isinstance(scores, list):
        for item in scores:
            if not isinstance(item, dict):
                continue
            idx = _candidate_index(item.get("candidate"), candidates)
            if idx is None:
                continue
            cand = candidates[idx]
            cand.score = _clamp(item.get("score"), default=_base_candidate_score(cand, n_agents))
            status = str(item.get("status") or cand.status).strip().lower()
            cand.status = status if status in ("accepted", "uncertain", "rejected") else cand.status
            cand.is_final = cand.is_final or _boolish(item.get("is_final"), default=False)
            answer = _clean_answer(_as_text(item.get("answer")), dtype)
            if answer:
                cand.answer = answer
                cand.is_final = True
            cand.history.setdefault("relation_score", item)
    for cand in candidates:
        if "relation_score" not in cand.history:
            cand.score = max(cand.score, _base_candidate_score(cand, n_agents))
        _enforce_candidate_contract(cand, question, dtype)

    relations = d.get("relations") if isinstance(d, dict) else None
    relation_log = []
    uf_conflict = _UnionFind(len(candidates))
    uncertain_edges = []
    if isinstance(relations, list):
        for item in relations:
            if not isinstance(item, dict):
                continue
            a = _candidate_index(item.get("a"), candidates)
            b = _candidate_index(item.get("b"), candidates)
            rel = str(item.get("relation") or "uncertain").strip().lower()
            if a is None or b is None or a == b:
                continue
            if rel not in ("compatible", "conflict", "uncertain"):
                rel = "uncertain"
            relation_log.append({"a": a, "b": b, "relation": rel, "reason": _as_text(item.get("reason"))})
            if rel == "conflict":
                uf_conflict.union(a, b)
            elif rel == "uncertain":
                uncertain_edges.append((a, b))

    conflict_components: dict[int, list[int]] = defaultdict(list)
    conflict_touched = set()
    for item in relation_log:
        if item["relation"] == "conflict":
            conflict_touched.add(item["a"])
            conflict_touched.add(item["b"])
    for idx in conflict_touched:
        conflict_components[uf_conflict.find(idx)].append(idx)

    groups = []
    assigned = set()
    for members in conflict_components.values():
        members = sorted(set(members))
        if len(members) > 1:
            groups.append({"type": "conflict", "members": members, "reason": "pairwise conflict detected"})
            assigned.update(members)

    uncertain_uf = _UnionFind(len(candidates))
    uncertain_touched = set()
    for a, b in uncertain_edges:
        if a not in assigned and b not in assigned:
            uncertain_uf.union(a, b)
            uncertain_touched.update([a, b])
    uncertain_components: dict[int, list[int]] = defaultdict(list)
    for idx in uncertain_touched:
        if idx not in assigned:
            uncertain_components[uncertain_uf.find(idx)].append(idx)
    for members in uncertain_components.values():
        members = sorted(set(members))
        if members:
            groups.append({"type": "uncertain", "members": members, "reason": "uncertain pairwise relation"})
            assigned.update(members)

    compatible = [i for i in range(len(candidates)) if i not in assigned]
    if compatible:
        groups.append({"type": "compatible", "members": compatible, "reason": "no local conflict detected"})
    if not groups:
        groups.append({"type": "uncertain", "members": list(range(len(candidates))), "reason": "relation parse fallback"})

    meta = {
        "used": True,
        "parse_ok": bool(d),
        "raw": raw,
        "relations": relation_log,
        "rationale": d.get("rationale", "") if isinstance(d, dict) else "",
        "groups": groups,
    }
    return groups, meta


def _candidate_payload(candidates: list[CandidateNode], indices: list[int]) -> list[dict]:
    return [
        {
            "id": i,
            "candidate_id": candidates[i].candidate_id,
            "node": candidates[i].text,
            "support_agents": candidates[i].support_agents,
            "support": candidates[i].support,
            "initial_status": candidates[i].status,
            "initial_score": round(candidates[i].score, 4),
            "answer": candidates[i].answer,
            "is_final": candidates[i].is_final,
        }
        for i in indices
    ]


def _decision_map(raw_decisions: Any, candidates: list[CandidateNode]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    if not isinstance(raw_decisions, list):
        return out
    for item in raw_decisions:
        if not isinstance(item, dict):
            continue
        idx = _candidate_index(item.get("candidate"), candidates)
        if idx is None:
            continue
        out[idx] = item
    return out


def _debate_group(client, tr: Tracker, agents, question: str, tree: ReasoningTree,
                  parent_id: str, candidates: list[CandidateNode], group: dict,
                  dtype: str, rounds: int = 1, max_tokens: int = 900) -> tuple[list[int], dict]:
    members = list(group.get("members") or [])
    payload = _candidate_payload(candidates, members)
    path_nodes = _path_nodes(tree, parent_id)
    transcript = []
    for r in range(rounds):
        for agent in agents:
            own_idx = next((i for i in members if agent.idx in candidates[i].support_agents), None)
            own = payload[[x["id"] for x in payload].index(own_idx)] if own_idx is not None else None
            raw = ask(
                agent.client,
                tr,
                [{"role": "system", "content": prompts.DEBATE_SYS},
                 {"role": "user", "content": prompts.debate_user(question, path_nodes, payload, own, dtype)}],
                temperature=agent.temperature,
                max_tokens=min(max_tokens, agent.max_tokens),
            )
            d = extract_json(raw) or {}
            transcript.append({
                "round": r,
                "agent": agent.idx,
                "own_candidate": own_idx,
                "raw": raw,
                "parsed": d,
            })

    raw = ask(
        client,
        tr,
        [{"role": "system", "content": f"{prompts.RESOLVE_SYS} {prompts.JSON_ONLY}"},
         {"role": "user", "content": prompts.resolve_debate_user(question, path_nodes, payload, transcript, dtype)}],
        temperature=0.0,
        max_tokens=900,
    )
    d = extract_json(raw) or {}
    decisions = _decision_map(d.get("decisions"), candidates)
    for idx in members:
        cand = candidates[idx]
        decision = decisions.get(idx, {})
        status = str(decision.get("status") or cand.status).strip().lower()
        cand.status = status if status in ("accepted", "uncertain", "rejected") else cand.status
        cand.score = _clamp(decision.get("score"), default=cand.score)
        cand.is_final = cand.is_final or _boolish(decision.get("is_final"), default=False)
        answer = _clean_answer(_as_text(decision.get("answer")), dtype)
        if answer:
            cand.answer = answer
            cand.is_final = True
        _enforce_candidate_contract(cand, question, dtype)
        cand.history["debate"] = {
            "group": group,
            "transcript": transcript,
            "resolution": d,
            "resolution_raw": raw,
        }
    meta = {
        "group": group,
        "transcript": transcript,
        "resolution": d,
        "resolution_raw": raw,
        "parse_ok": bool(d),
    }
    return members, meta


def _score_nonconflict_group(candidates: list[CandidateNode], group: dict,
                             n_agents: int, uncertain_penalty: float = 0.08) -> dict:
    members = list(group.get("members") or [])
    gtype = group.get("type")
    for idx in members:
        cand = candidates[idx]
        base = max(cand.score, _base_candidate_score(cand, n_agents))
        if gtype == "compatible":
            cand.score = _clamp(base + 0.04, default=base)
            if cand.status != "rejected":
                cand.status = "accepted" if cand.score >= 0.58 or cand.support > 1 else "uncertain"
        else:
            cand.score = _clamp(base - uncertain_penalty, default=base)
            if cand.status == "accepted" and cand.score < 0.68:
                cand.status = "uncertain"
        cand.history.setdefault("light_score", {"group": group, "score": cand.score, "status": cand.status})
    return {"group": group, "members": members, "mode": "light_score"}


def _suppress_early_final(candidates: list[CandidateNode], depth: int, min_depth: int) -> None:
    if depth >= min_depth:
        return
    for cand in candidates:
        if not cand.is_final:
            continue
        cand.history.setdefault("min_depth_guard", []).append({
            "action": "suppressed_early_final",
            "depth": depth,
            "min_depth_before_final": min_depth,
            "suppressed_answer": cand.answer,
        })
        cand.is_final = False
        cand.answer = ""


def _rank_candidates(candidates: list[CandidateNode]) -> list[CandidateNode]:
    status_rank = {"accepted": 2, "uncertain": 1, "rejected": 0}
    return sorted(
        candidates,
        key=lambda c: (
            status_rank.get(c.status, 1),
            c.score,
            c.support,
            c.confidence,
            1 if c.is_final else 0,
        ),
        reverse=True,
    )


def _select_children(candidates: list[CandidateNode], per_parent_beam: int,
                     uncertain_threshold: float, force_keep_best: bool = True) -> tuple[list[CandidateNode], dict]:
    eligible = [
        c for c in candidates
        if c.status == "accepted" or (c.status == "uncertain" and c.score >= uncertain_threshold)
    ]
    ranked = _rank_candidates(eligible)
    fallback = None
    if not ranked and force_keep_best and candidates:
        fallback = _rank_candidates(candidates)[0]
        fallback.status = "uncertain"
        fallback.score = min(fallback.score, uncertain_threshold)
        fallback.history["fallback_keep_best"] = True
        ranked = [fallback]
    kept = ranked[:per_parent_beam]
    return kept, {
        "eligible": [c.candidate_id for c in eligible],
        "kept": [c.candidate_id for c in kept],
        "fallback_keep_best": fallback.candidate_id if fallback else "",
    }


def _rank_frontier(tree: ReasoningTree, frontier: list[str]) -> list[str]:
    return sorted(frontier, key=lambda nid: (tree.path_score(nid), tree.nodes[nid].score), reverse=True)


def _path_payload(tree: ReasoningTree, node_id: str, path_id: str, dtype: str) -> dict:
    node = tree.nodes[node_id]
    path = tree.path_to(node_id, include_root=False)
    return {
        "path_id": path_id,
        "terminal_node_id": node_id,
        "path_score": round(tree.path_score(node_id), 4),
        "terminal_score": round(node.score, 4),
        "is_final": node.is_final,
        "answer_hint": _clean_answer(node.answer or extract_answer(node.text, dtype), dtype),
        "nodes": [
            {
                "node_id": n.node_id,
                "text": n.text,
                "score": round(n.score, 4),
                "status": n.status,
                "answer": n.answer,
                "is_final": n.is_final,
            }
            for n in path
        ],
    }


def _fallback_answer_from_node(node, dtype: str) -> str:
    if node.answer:
        return _clean_answer(node.answer, dtype)
    answer = _clean_answer(extract_answer(node.text, dtype), dtype)
    if answer and not is_nonanswer(answer):
        return answer
    return node.text


def _needs_yes_no_repair(question: str, answer: str, dtype: str) -> bool:
    if dtype != "qa" or not prompts.is_yes_no_question(question):
        return False
    return normalize_qa(answer) not in ("yes", "no")


def _repair_yes_no_answer(client, tr: Tracker, question: str, paths: list[dict],
                          selected_answer: str, selected_payload: dict | None) -> tuple[str, dict]:
    payload = {
        "question": question,
        "selected_answer": selected_answer,
        "selected_path": selected_payload or {},
        "candidate_paths": paths,
        "task": (
            "The question requires a yes/no answer. Convert the selected path "
            "into exactly 'yes' or 'no'. Do not return an entity, property, date, "
            "or explanation."
        ),
        "output_schema": {
            "answer": "yes | no",
            "rationale": "short reason",
        },
    }
    raw = ask(
        client,
        tr,
        [{"role": "system", "content": f"{prompts.FINAL_SYS} {prompts.JSON_ONLY}"},
         {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        temperature=0.0,
        max_tokens=220,
    )
    d = extract_json(raw) or {}
    answer = normalize_qa(_clean_answer(_as_text(d.get("answer")), "qa"))
    if answer in ("yes", "no"):
        return answer, {"used": True, "raw": raw, "parsed": d, "accepted": True}
    return selected_answer, {"used": True, "raw": raw, "parsed": d, "accepted": False}


def _select_final_answer(client, tr: Tracker, question: str, tree: ReasoningTree, dtype: str,
                         top_paths: int = 6) -> tuple[str, dict]:
    terminal_ids = tree.ranked_terminal_ids()[:top_paths]
    if not terminal_ids:
        return "", {"reason": "empty-tree"}
    path_payloads = [_path_payload(tree, nid, f"p{i}", dtype) for i, nid in enumerate(terminal_ids)]
    raw = ask(
        client,
        tr,
        [{"role": "system", "content": f"{prompts.FINAL_SYS} {prompts.JSON_ONLY}"},
         {"role": "user", "content": prompts.final_select_user(question, path_payloads, dtype)}],
        temperature=0.0,
        max_tokens=500,
    )
    d = extract_json(raw) or {}
    answer = _clean_answer(_as_text(d.get("answer")), dtype)
    chosen_path = _as_text(d.get("chosen_path"))
    chosen_idx = None
    for i, payload in enumerate(path_payloads):
        if payload["path_id"] == chosen_path:
            chosen_idx = i
            break
    if not answer:
        chosen_idx = 0 if chosen_idx is None else chosen_idx
        answer = _clean_answer(_fallback_answer_from_node(tree.nodes[terminal_ids[chosen_idx]], dtype), dtype)
    repair_log = None
    if _needs_yes_no_repair(question, answer, dtype):
        selected_payload = path_payloads[chosen_idx] if chosen_idx is not None else path_payloads[0]
        answer, repair_log = _repair_yes_no_answer(client, tr, question, path_payloads, answer, selected_payload)
    return answer, {
        "paths": path_payloads,
        "raw": raw,
        "parsed": d,
        "chosen_path": chosen_path or (path_payloads[0]["path_id"] if path_payloads else ""),
        "parse_ok": bool(d),
        **({"format_repair": repair_log} if repair_log else {}),
    }


def _run_ldt(
    agents,
    ex,
    dtype,
    max_depth: int = 5,
    per_parent_beam: int = 2,
    global_beam: int = 6,
    debate_budget: int = 20,
    debate_rounds: int = 1,
    uncertain_threshold: float = 0.52,
    final_confidence: float = 0.82,
    min_depth_before_final: int = 2,
    llm_merge: bool = True,
    force_keep_best: bool = True,
    **kw,
):
    client = agents[0].client
    tr = Tracker()
    tree = ReasoningTree(ex.question)
    frontier = [tree.root_id]
    layers = []
    debates_used = 0
    debated_groups = []

    for depth in range(1, max_depth + 1):
        layer_log = {"depth": depth, "parents": [], "frontier_in": list(frontier)}
        next_frontier = []
        for parent_id in frontier:
            parent_log: dict[str, Any] = {"parent_id": parent_id, "prefix": tree.prefix_text(parent_id)}
            proposals = [
                _sample_next_step(agent, ex.question, tree, parent_id, depth, dtype, tr)
                for agent in agents
            ]
            parent_log["proposals"] = [p.to_dict() for p in proposals]

            live_props = [p for p in proposals if p.text]
            candidates, merge_meta = _merge_proposals(
                client, tr, ex.question, tree, parent_id, depth, live_props, len(agents),
                dtype=dtype, llm_merge=llm_merge)
            parent_log["merge"] = merge_meta
            parent_log["candidates_before_judge"] = [c.to_dict() for c in candidates]

            groups, relation_meta = _detect_relations(
                client, tr, ex.question, tree, parent_id, candidates, dtype, len(agents))
            parent_log["relations"] = relation_meta
            group_logs = []
            for group in groups:
                if group.get("type") == "conflict" and debates_used < debate_budget:
                    _, meta = _debate_group(
                        client, tr, agents, ex.question, tree, parent_id,
                        candidates, group, dtype, rounds=debate_rounds)
                    debates_used += 1
                    debated_groups.append(meta)
                    group_logs.append({"mode": "debate", **meta})
                else:
                    meta = _score_nonconflict_group(candidates, group, len(agents))
                    if group.get("type") == "conflict":
                        meta["mode"] = "budget_fallback_score"
                    group_logs.append(meta)
            parent_log["group_processing"] = group_logs
            _suppress_early_final(candidates, depth, min_depth_before_final)
            parent_log["candidates_after_debate"] = [c.to_dict() for c in candidates]

            kept, prune_meta = _select_children(
                candidates, per_parent_beam=per_parent_beam,
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

        ranked_frontier = _rank_frontier(tree, next_frontier)
        frontier = ranked_frontier[:global_beam]
        layer_log["frontier_out_before_global_prune"] = ranked_frontier
        layer_log["frontier_out"] = list(frontier)
        layers.append(layer_log)

        final_ids = tree.final_ids()
        if final_ids and depth >= min_depth_before_final:
            best_final = max(final_ids, key=lambda nid: tree.path_score(nid))
            if tree.path_score(best_final) >= final_confidence:
                layers[-1]["stop"] = {
                    "reason": "confident_final_answer",
                    "node_id": best_final,
                    "path_score": tree.path_score(best_final),
                }
                break
        if not frontier:
            layers[-1]["stop"] = {"reason": "empty_frontier"}
            break

    final_answer, final_log = _select_final_answer(client, tr, ex.question, tree, dtype)
    trace = {
        "algorithm": "ldt",
        "config": {
            "max_depth": max_depth,
            "per_parent_beam": per_parent_beam,
            "global_beam": global_beam,
            "debate_budget": debate_budget,
            "debate_rounds": debate_rounds,
            "uncertain_threshold": uncertain_threshold,
            "final_confidence": final_confidence,
            "min_depth_before_final": min_depth_before_final,
            "llm_merge": llm_merge,
            "force_keep_best": force_keep_best,
        },
        "layers": layers,
        "tree": tree.to_dict(),
        "final_selection": final_log,
    }
    return {
        "pred": final_answer,
        "trace": trace,
        "n_debates": debates_used,
        "debated_claims": [
            {
                "group": item.get("group"),
                "resolution": item.get("resolution"),
            }
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
    ldt_debate_budget: int = 20,
    ldt_uncertain_threshold: float = 0.52,
    ldt_final_confidence: float = 0.82,
    ldt_min_depth_before_final: int = 2,
    ldt_no_llm_merge: bool = False,
    **kw,
):
    """Layerwise Debate Tree entrypoint for src.run."""
    max_depth = ldt_max_depth if ldt_max_depth is not None else max_iters
    return _run_ldt(
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
        llm_merge=not ldt_no_llm_merge,
        **kw,
    )
