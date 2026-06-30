"""CDSD v15 — structured node decomposition.

This variant keeps the v12 indexed faithful debate controller, but replaces the
old "ask solvers to directly emit Claim 1/2/3" step with two stages:

1. Each solver writes a normal step-by-step solution.
2. A deterministic extractor converts that solution into typed reasoning nodes.

The goal is to make the claim/node layer more robust for small models and more
portable across QA, math, and multiple-choice tasks.
"""

import re

from src.llm.client import Tracker
from src.util import is_nonanswer
from src.cdsd import components as C
from src.cdsd import prompts
from src.cdsd.v12_indexed import (
    _answer_grounded,
    _as_text,
    _boolish,
    _bridge_recover,
    _bridge_relation_rule,
    _debate_resolve_indexed,
    _evidence_supported,
    _final_select,
    _find_conflict_indexed,
)


def _commit_diagnostics(question, resolution, nodes):
    resolution = resolution if isinstance(resolution, dict) else {}
    return {
        "confident": _boolish(resolution.get("confident", False), default=False),
        "relation_preserved": _boolish(resolution.get("relation_preserved", True), default=True),
        "evidence_supported": _evidence_supported(question, resolution),
        "concrete_node": any(not is_nonanswer(n) for n in nodes),
    }


def _apply_commit_policy(question, strict_commit, resolution, nodes, policy=None):
    diag = _commit_diagnostics(question, resolution, nodes)
    log = {"policy": policy or "strict", "strict_commit": bool(strict_commit), **diag}
    if strict_commit or policy in (None, "strict"):
        log["commit"] = bool(strict_commit)
        return bool(strict_commit), log
    if policy == "conflict_soft":
        soft_commit = diag["concrete_node"] and (diag["confident"] or diag["relation_preserved"])
        log["commit"] = bool(soft_commit)
        log["relaxed_reason"] = (
            "concrete-node-and-confident-or-relation-preserved"
            if soft_commit else "soft-gate-failed"
        )
        return bool(soft_commit), log
    if policy == "concrete_only":
        # Near-removal of the gate: commit any concrete resolved node, ignoring
        # confident / relation_preserved / evidence_supported entirely.
        c = diag["concrete_node"]
        log["commit"] = bool(c)
        log["relaxed_reason"] = "concrete-node-only" if c else "no-concrete-node"
        return bool(c), log
    if policy == "always":
        # Full removal of the gate: always commit the resolver's nodes on conflict.
        log["commit"] = True
        log["relaxed_reason"] = "always-commit"
        return True, log
    log["commit"] = bool(strict_commit)
    log["relaxed_reason"] = "unknown-policy"
    return bool(strict_commit), log


def _run_structured(agents, ex, dtype, max_iters=6, debate_rounds=1,
                    rerank_final=False, prefer_extractor_answer=True,
                    bridge_recover=False, node_source="reasoning",
                    recover_nonanswer=False, specificity_guard=False,
                    audit_consensus=False, verified_final=False,
                    commit_policy=None, no_commit_select=False, **kw):
    client = agents[0].client
    tr = Tracker()
    agreed_prefix, debated, iters_log = [], [], []
    final, agents_answers = None, []
    agents_node_payloads = []

    for it in range(max_iters):
        if node_source == "claims":
            generated = [
                C.gen_structured_nodes_from_claims(a, ex.question, agreed_prefix, dtype, tr)
                for a in agents
            ]
        else:
            generated = [
                C.gen_structured_nodes(
                    a, ex.question, agreed_prefix, dtype, tr,
                    prefer_extractor_answer=prefer_extractor_answer)
                for a in agents
            ]
        agents_node_payloads = [g[0] for g in generated]
        agents_nodes = [C.node_claims(nodes) for nodes in agents_node_payloads]
        agents_answers = [g[1] for g in generated]
        node_meta = [g[2] for g in generated]

        log = {
            "iter": it,
            "answers": agents_answers,
            "node_payloads": agents_node_payloads,
            "node_quality": [m.get("quality", {}) for m in node_meta],
            "node_schema_ok": [bool(m.get("schema_ok")) for m in node_meta],
            "node_generation": node_meta,
        }

        concrete = [a for a in agents_answers if a and not is_nonanswer(a)]
        if concrete and len(concrete) == len(agents) and len({C._norm(a, dtype) for a in concrete}) == 1:
            final = concrete[0]
            log["shortcircuit"] = "answer_agreement_after_node_extraction"
            iters_log.append(log)
            break

        verdict = _find_conflict_indexed(client, tr, ex.question, agreed_prefix,
                                         agents_nodes, agents_answers, dtype)
        log["verdict"] = verdict

        if verdict.get("status") == "consensus":
            if verified_final:
                final, select_log = _verified_select(client, tr, ex.question, agents_answers, dtype)
                log["verified_select"] = select_log
            else:
                final = _final_select(client, tr, ex.question, agents_answers, dtype,
                                      rerank=rerank_final)
            if audit_consensus and _needs_consensus_audit(final, agents_answers, dtype):
                audited, audit_log = _audit_consensus(
                    client, tr, ex.question, agreed_prefix, agents_answers,
                    agents_nodes, final, dtype)
                log["consensus_audit"] = audit_log
                final = audited or final
            if recover_nonanswer and is_nonanswer(final):
                recovered, recover_log = _recover_nonanswer(
                    client, tr, ex.question, agreed_prefix, agents_answers,
                    agents_nodes, dtype)
                log["nonanswer_recover"] = recover_log
                final = recovered or final
            iters_log.append(log)
            break

        nodes, commit, transcript, segments, resolution = _debate_resolve_indexed(
            client, tr, agents, ex.question, agreed_prefix, verdict, agents_nodes,
            dtype, rounds=debate_rounds)
        commit, commit_log = _apply_commit_policy(
            ex.question, commit, resolution, nodes, policy=commit_policy)
        log.update({
            "resolved_nodes": nodes,
            "commit": commit,
            "commit_check": commit_log,
            "debate": transcript,
            "segments": segments,
            "resolution": resolution,
        })
        iters_log.append(log)

        if commit:
            agreed_prefix.extend(nodes)
            debated.extend(nodes)
        else:
            bridge_final = None
            if bridge_recover:
                bridge_final, bridge_log = _bridge_recover(
                    client, tr, ex.question, agreed_prefix, verdict,
                    agents_answers, nodes, dtype)
                log["bridge_recover"] = bridge_log
            selected_final = None
            if no_commit_select:
                selected_final, select_log = _no_commit_select(
                    client, tr, ex.question, agreed_prefix, agents_answers,
                    verdict, nodes, transcript, dtype,
                    strict=(no_commit_select == "strict"))
                log["no_commit_select"] = select_log
            if bridge_final:
                final = bridge_final
            elif selected_final:
                final = selected_final
            elif verified_final:
                final, select_log = _verified_select(client, tr, ex.question, agents_answers, dtype)
                log["verified_select"] = select_log
            else:
                final = _final_select(
                    client, tr, ex.question, agents_answers, dtype,
                    rerank=rerank_final)
            break

    if final is None:
        if verified_final:
            final, select_log = _verified_select(client, tr, ex.question, agents_answers, dtype)
            iters_log.append({"iter": "final_verified_select", "verified_select": select_log})
        else:
            final = _final_select(client, tr, ex.question, agents_answers, dtype,
                                  rerank=rerank_final)
        if recover_nonanswer and is_nonanswer(final):
            recovered, recover_log = _recover_nonanswer(
                client, tr, ex.question, agreed_prefix, agents_answers,
                [], dtype)
            iters_log.append({"iter": "final_nonanswer_recover", "nonanswer_recover": recover_log})
            final = recovered or final
    if is_nonanswer(final):
        final = C.majority_concrete(agents_answers, dtype) or final
    guard_log = None
    if specificity_guard:
        if specificity_guard == "strict":
            guarded, guard_log = _specificity_guard_strict(ex.question, final, agents_node_payloads, dtype)
        elif specificity_guard == "answer_type":
            guarded, guard_log = _specificity_guard_answer_type(ex.question, final, agents_node_payloads, dtype)
        else:
            guarded, guard_log = _specificity_guard(ex.question, final, agents_node_payloads, dtype)
        final = guarded or final
    # Forward no-commit metrics. A debate that fails the commit gate appends
    # nothing to `debated`, so n_debates/debated_claims (committed nodes only)
    # never reflect it. Count those iters here and keep a compact trace so the
    # no-commit path is queryable straight from the result file rather than
    # re-derived post-hoc. Each no-commit debate breaks the loop, so this is the
    # iter that directly produced `pred`.
    no_commit_trace = [
        {
            "iter": it.get("iter"),
            "verdict": it.get("verdict"),
            "segments": it.get("segments"),
            "resolved_nodes": it.get("resolved_nodes"),
            "resolution": it.get("resolution"),
            "commit_check": it.get("commit_check"),
            # which selector produced the final answer after the blocked commit
            "final_path": (
                "bridge_recover" if (it.get("bridge_recover") or {}).get("accepted")
                else "no_commit_select" if (it.get("no_commit_select") or {}).get("accepted")
                else "verified_select" if "verified_select" in it
                else "final_select"
            ),
        }
        for it in iters_log
        if ("debate" in it) and not it.get("commit")
    ]
    return {
        "pred": final,
        "trace": {"agreed_prefix": agreed_prefix, "iters": iters_log,
                  **({"specificity_guard": guard_log} if guard_log else {})},
        "n_debates": len(debated),
        "debated_claims": debated,
        "n_no_commit": len(no_commit_trace),
        "no_commit_trace": no_commit_trace,
        **tr.as_dict(),
    }


def _candidate_answers(answers, dtype):
    candidates, seen = [], set()
    for answer in answers or []:
        text = _as_text(answer)
        key = C._norm(text, dtype)
        if not text or is_nonanswer(text) or not key or key in seen:
            continue
        seen.add(key)
        candidates.append(text)
    return candidates


def _no_commit_select(client, tr, question, agreed_prefix, agents_answers,
                      verdict, resolved_nodes, transcript, dtype, strict=True):
    fallback = _final_select(client, tr, question, agents_answers, dtype, rerank=False)
    if dtype != "qa":
        return None, {"accepted": False, "reason": "non-qa", "fallback": fallback}
    candidates = _candidate_answers(agents_answers, dtype)
    if len(candidates) <= 1:
        return None, {"accepted": False, "reason": "not-enough-candidates",
                      "fallback": fallback, "candidates": candidates}
    user = prompts.no_commit_select_user(
        question, agreed_prefix, agents_answers,
        (verdict or {}).get("description", ""), resolved_nodes, transcript, candidates)
    text = C.ask(client, tr, [{"role": "system", "content": prompts.RESOLVE_SYS},
                              {"role": "user", "content": user}],
                 temperature=0.0, max_tokens=800)
    d = C.extract_json(text) or {}
    answer = _as_text(d.get("answer"))
    chosen = None
    for cand in candidates:
        if C._norm(cand, dtype) == C._norm(answer, dtype):
            chosen = cand
            break
    quotes = d.get("evidence_quotes") or []
    if isinstance(quotes, str):
        quotes = [quotes]
    quotes = [_as_text(q) for q in quotes if _as_text(q)]
    evidence_payload = {"resolved_nodes": [chosen or answer], "evidence_quotes": quotes}
    accepted = (
        chosen is not None
        and not is_nonanswer(chosen)
        and _boolish(d.get("confident", False))
        and _boolish(d.get("relation_preserved", True), default=True)
        and _evidence_supported(question, evidence_payload)
        and _bridge_relation_rule(question, chosen, quotes)
        and _answer_grounded(question, chosen, chosen, quotes, dtype)
    )
    if strict and accepted:
        # In strict mode, use the extra selector only when it changes the normal fallback.
        # This keeps the method from spending an extra call merely to restate majority.
        accepted = C._norm(chosen, dtype) != C._norm(fallback, dtype)
        if not accepted:
            d["reason"] = "same-as-fallback"
    d.update({"accepted": accepted, "fallback": fallback,
              "chosen": chosen, "candidates": candidates})
    return (chosen if accepted else None), d


def _recover_nonanswer(client, tr, question, agreed_prefix, agents_answers,
                       agents_nodes, dtype):
    if dtype != "qa":
        return None, {"accepted": False, "reason": "non-qa"}
    user = prompts.nonanswer_recover_user(question, agreed_prefix, agents_answers, agents_nodes)
    text = C.ask(client, tr, [{"role": "system", "content": prompts.RESOLVE_SYS},
                              {"role": "user", "content": user}],
                 temperature=0.0, max_tokens=700)
    d = C.extract_json(text) or {}
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


def _needs_consensus_audit(final, agents_answers, dtype):
    if dtype != "qa":
        return False
    if not final or is_nonanswer(final):
        return True
    concrete = [a for a in agents_answers if a and not is_nonanswer(a)]
    norms = {C._norm(a, dtype) for a in concrete if C._norm(a, dtype)}
    if len(norms) > 1:
        return True
    words = C.normalize_qa(final).split()
    if len(words) >= 9:
        return True
    return False


def _audit_consensus(client, tr, question, agreed_prefix, agents_answers,
                     agents_nodes, draft_answer, dtype):
    if dtype != "qa":
        return None, {"accepted": False, "reason": "non-qa"}
    user = prompts.consensus_audit_user(
        question, agreed_prefix, agents_answers, agents_nodes, draft_answer)
    text = C.ask(client, tr, [{"role": "system", "content": prompts.RESOLVE_SYS},
                              {"role": "user", "content": user}],
                 temperature=0.0, max_tokens=800)
    d = C.extract_json(text) or {}
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
    # Do not replace a concrete consensus with another equivalent spelling; this
    # audit is meant to repair bad consensus, not to churn answers.
    if accepted and answer and C._norm(answer, dtype) == C._norm(draft_answer, dtype):
        accepted = False
        d["reason"] = "same-as-draft"
    d["accepted"] = accepted
    return answer if accepted else None, d


def _verified_select(client, tr, question, answers, dtype):
    fallback = _final_select(client, tr, question, answers, dtype, rerank=False)
    concrete = [a for a in answers if a and not is_nonanswer(a)]
    distinct = {C._norm(a, dtype) for a in concrete if C._norm(a, dtype)}
    if dtype != "qa" or len(distinct) <= 1:
        return fallback, {"accepted": False, "reason": "not-needed", "fallback": fallback}
    candidates = []
    seen = set()
    for a in concrete:
        key = C._norm(a, dtype)
        if key and key not in seen:
            seen.add(key)
            candidates.append(a)
    user = prompts.candidate_verify_user(question, candidates)
    text = C.ask(client, tr, [{"role": "system", "content": prompts.RESOLVE_SYS},
                              {"role": "user", "content": user}],
                 temperature=0.0, max_tokens=700)
    d = C.extract_json(text) or {}
    answer = _as_text(d.get("answer"))
    chosen = None
    for cand in candidates:
        if C._norm(cand, dtype) == C._norm(answer, dtype):
            chosen = cand
            break
    quotes = d.get("evidence_quotes") or []
    if isinstance(quotes, str):
        quotes = [quotes]
    quotes = [_as_text(q) for q in quotes if _as_text(q)]
    evidence_payload = {"resolved_nodes": [chosen or answer], "evidence_quotes": quotes}
    accepted = (
        chosen is not None
        and _boolish(d.get("confident", False))
        and _boolish(d.get("relation_preserved", True), default=True)
        and _evidence_supported(question, evidence_payload)
        and _bridge_relation_rule(question, chosen, quotes)
        and _answer_grounded(question, chosen, chosen, quotes, dtype)
    )
    d.update({"accepted": accepted, "fallback": fallback, "candidates": candidates})
    return (chosen if accepted else fallback), d


def _candidate_texts_from_nodes(node_payloads):
    texts = []
    for nodes in node_payloads or []:
        for node in nodes or []:
            if not isinstance(node, dict):
                continue
            for key in ("subject", "object", "claim", "evidence_quote"):
                value = node.get(key)
                if isinstance(value, str) and value.strip():
                    texts.append(value.strip())
    return texts


def _title_candidates(question):
    return [m.group(1).strip() for m in re.finditer(r"==\s*([^=\n]{2,120})\s*==", question or "")]


def _mention_candidates(text, answer):
    if not text or not answer:
        return []
    out = []
    escaped = re.escape(answer.strip())
    suffix = r"(?:,\s*[^.\n;:()]{2,90}|(?:\s+[A-Z][\w'’\-]+){1,6})"
    for m in re.finditer(escaped + suffix, text, flags=re.IGNORECASE):
        cand = m.group(0).strip(" ,.;:\n\t")
        if cand:
            out.append(cand)
    return out


def _bad_entity_expansion(answer, cand):
    cn = C.normalize_qa(cand)
    an = C.normalize_qa(answer)
    if not cn or cn == an:
        return True
    if not (cn.startswith(an + " ") or cn.startswith(an + ",")):
        return True
    words = cn.split()
    if len(words) > 12 or len(cand) > 110:
        return True
    if an in ("yes", "no", "true", "false", "unknown"):
        return True
    suffix = cn[len(an):].strip(" ,")
    if not suffix:
        return True
    banned_prefixes = (
        "officially", "commonly", "also", "known", "born", "died", "was",
        "were", "is", "are", "has", "had", "and", "or", "the", "a", "an",
    )
    if suffix.split()[0] in banned_prefixes:
        return True
    return False


def _specificity_guard(question, answer, node_payloads, dtype):
    """Recover a more complete entity string that is already present in passages/nodes.

    This is deliberately conservative: it only expands a short QA entity when the
    longer form strictly contains the selected answer as its prefix. It never invents
    a new answer and it does not run on yes/no, math, or MC tasks.
    """
    if dtype != "qa" or not answer or is_nonanswer(answer):
        return answer, {"changed": False, "reason": "not-applicable"}
    if len(C.normalize_qa(answer).split()) > 5:
        return answer, {"changed": False, "reason": "answer-not-short"}

    raw_sources = []
    raw_sources.extend(_title_candidates(question))
    raw_sources.extend(_candidate_texts_from_nodes(node_payloads))
    candidates = []
    for src in raw_sources:
        candidates.append(src)
        candidates.extend(_mention_candidates(src, answer))
    candidates.extend(_mention_candidates(question, answer))

    clean = []
    seen = set()
    for cand in candidates:
        cand = re.sub(r"\s+", " ", str(cand or "")).strip(" ,.;:")
        key = C.normalize_qa(cand)
        if key in seen or _bad_entity_expansion(answer, cand):
            continue
        seen.add(key)
        clean.append(cand)
    if not clean:
        return answer, {"changed": False, "reason": "no-grounded-expansion"}

    best = max(clean, key=lambda c: ("," in c, len(C.normalize_qa(c).split()), len(c)))
    return best, {"changed": True, "from": answer, "to": best, "candidates": clean[:8]}


def _node_entity_texts(node_payloads):
    texts = []
    for nodes in node_payloads or []:
        for node in nodes or []:
            if not isinstance(node, dict):
                continue
            for key in ("subject", "object"):
                value = node.get(key)
                if isinstance(value, str) and value.strip():
                    texts.append(value.strip())
    return texts


def _looks_like_short_entity(cand):
    text = str(cand or "").strip()
    if not text or len(text) > 90:
        return False
    norm = C.normalize_qa(text)
    words = norm.split()
    if not (2 <= len(words) <= 9):
        return False
    bad = {
        "is", "are", "was", "were", "has", "had", "have", "because", "while",
        "when", "where", "who", "which", "that", "this", "these", "those",
        "meets", "criteria", "director", "winner", "twice", "first", "before",
        "after", "came", "out", "founded", "opened", "located", "plays",
    }
    if any(w in bad for w in words):
        return False
    if any(ch in text for ch in "\"“”"):
        return False
    return True


def _strict_expansion_ok(answer, cand):
    if _bad_entity_expansion(answer, cand) or not _looks_like_short_entity(cand):
        return False
    an = C.normalize_qa(answer)
    cn = C.normalize_qa(cand)
    suffix = cn[len(an):].strip(" ,")
    if not suffix:
        return False
    # Prefer appositive/location/title completions; allow all-title-case suffixes such
    # as "Prima Divisione" for league names, but reject prose-like continuations.
    if "," in cand:
        return True
    if answer.strip().replace(",", "").isdigit() and len(suffix.split()) == 1:
        return True
    tail = cand[len(answer):].strip()
    return bool(re.fullmatch(r"(?:[A-Z][\w'’\-]+(?:\s+|$)){1,5}", tail))


def _specificity_guard_strict(question, answer, node_payloads, dtype):
    if dtype != "qa" or not answer or is_nonanswer(answer):
        return answer, {"changed": False, "reason": "not-applicable"}
    if len(C.normalize_qa(answer).split()) > 5:
        return answer, {"changed": False, "reason": "answer-not-short"}

    candidates = []
    candidates.extend(_node_entity_texts(node_payloads))
    for title in _title_candidates(question):
        tn = C.normalize_qa(title)
        an = C.normalize_qa(answer)
        if tn.startswith(an + " ") or tn.startswith(an + ","):
            candidates.append(title)

    clean, seen = [], set()
    for cand in candidates:
        cand = re.sub(r"\s+", " ", str(cand or "")).strip(" ,.;:")
        key = C.normalize_qa(cand)
        if key in seen or not _strict_expansion_ok(answer, cand):
            continue
        seen.add(key)
        clean.append(cand)
    if not clean:
        return answer, {"changed": False, "reason": "no-strict-expansion"}
    best = max(clean, key=lambda c: ("," in c, len(C.normalize_qa(c).split()), len(c)))
    return best, {"changed": True, "mode": "strict", "from": answer, "to": best, "candidates": clean[:8]}


def _answer_type_expansion_ok(question, answer, cand):
    qn = C.normalize_qa(question)
    an = C.normalize_qa(answer)
    cn = C.normalize_qa(cand)
    suffix = cn[len(an):].strip(" ,")
    if not suffix:
        return False
    suffix_words = suffix.split()
    if not suffix_words:
        return False
    first = suffix_words[0]

    if "county" in qn:
        return "county" in cn and ("county" not in an or cn.startswith(an + " county"))
    if "australian city" in qn or ("city" in qn and "australia" in qn):
        return "," in cand and any(w in cn for w in ("australia", "south australia", "new south wales", "victoria"))
    if any(w in qn for w in ("inhabitant", "population")):
        return first in ("inhabitants", "people", "residents")
    if any(w in qn for w in ("seated", "capacity", "seating")):
        return first in ("seated", "seats")
    if first in ("inc", "incorporated", "ltd", "limited", "corp", "corporation", "plc", "gmbh"):
        return True
    # League names often need their official full tier/title; keep this to title-case
    # suffixes so we do not attach explanatory prose.
    if "league" in qn and "," not in cand:
        tail = cand[len(answer):].strip()
        return bool(re.fullmatch(r"(?:[A-Z][\w'’\-]+(?:\s+|$)){1,4}", tail))
    return False


def _specificity_guard_answer_type(question, answer, node_payloads, dtype):
    guarded, log = _specificity_guard_strict(question, answer, node_payloads, dtype)
    if not (log or {}).get("changed"):
        return guarded, log
    filtered = [c for c in log.get("candidates", []) if _answer_type_expansion_ok(question, answer, c)]
    if not filtered:
        return answer, {"changed": False, "mode": "answer-type", "reason": "no-type-safe-expansion",
                        "strict_candidates": log.get("candidates", [])[:8]}
    best = max(filtered, key=lambda c: ("," in c, len(C.normalize_qa(c).split()), len(c)))
    return best, {"changed": True, "mode": "answer-type", "from": answer, "to": best,
                  "candidates": filtered[:8], "strict_candidates": log.get("candidates", [])[:8]}


def run(agents, ex, dtype, max_iters=6, debate_rounds=1, final_agents=False, **kw):
    """cdsdn: structured-node CDSD."""
    return _run_structured(agents, ex, dtype, max_iters=max_iters,
                           debate_rounds=debate_rounds, rerank_final=False, **kw)


def run_select(agents, ex, dtype, max_iters=6, debate_rounds=1, final_agents=False, **kw):
    """cdsdns: structured-node CDSD with final evidence-aware rerank."""
    return _run_structured(agents, ex, dtype, max_iters=max_iters,
                           debate_rounds=debate_rounds, rerank_final=True, **kw)


def run_preserve_answer(agents, ex, dtype, max_iters=6, debate_rounds=1,
                        final_agents=False, **kw):
    """cdsdnp: structured-node CDSD, but preserve each solver's original final answer."""
    return _run_structured(
        agents, ex, dtype, max_iters=max_iters, debate_rounds=debate_rounds,
        rerank_final=False, prefer_extractor_answer=False, **kw)


def run_bridge(agents, ex, dtype, max_iters=6, debate_rounds=1,
               final_agents=False, **kw):
    """cdsdnb: structured-node CDSD plus conservative bridge recovery on failed commits."""
    return _run_structured(
        agents, ex, dtype, max_iters=max_iters, debate_rounds=debate_rounds,
        rerank_final=False, bridge_recover=True, **kw)


def run_claim_clean(agents, ex, dtype, max_iters=6, debate_rounds=1,
                    final_agents=False, **kw):
    """cdsdnc: old CDSD answer prompt plus structured node cleanup."""
    return _run_structured(
        agents, ex, dtype, max_iters=max_iters, debate_rounds=debate_rounds,
        rerank_final=False, node_source="claims", **kw)


def run_nonanswer_recover(agents, ex, dtype, max_iters=6, debate_rounds=1,
                          final_agents=False, **kw):
    """cdsdnr: structured-node CDSD plus a guarded recovery for non-answer consensus."""
    return _run_structured(
        agents, ex, dtype, max_iters=max_iters, debate_rounds=debate_rounds,
        rerank_final=False, recover_nonanswer=True, **kw)


def run_granularity_guard(agents, ex, dtype, max_iters=6, debate_rounds=1,
                          final_agents=False, **kw):
    """cdsdng: structured-node CDSD plus conservative entity granularity guard."""
    return _run_structured(
        agents, ex, dtype, max_iters=max_iters, debate_rounds=debate_rounds,
        rerank_final=False, specificity_guard=True, **kw)


def run_granularity_guard_strict(agents, ex, dtype, max_iters=6, debate_rounds=1,
                                 final_agents=False, **kw):
    """cdsdngs: structured-node CDSD plus strict entity granularity guard."""
    return _run_structured(
        agents, ex, dtype, max_iters=max_iters, debate_rounds=debate_rounds,
        rerank_final=False, specificity_guard="strict", **kw)


def run_granularity_guard_answer_type(agents, ex, dtype, max_iters=6, debate_rounds=1,
                                      final_agents=False, **kw):
    """cdsdnga: structured-node CDSD plus answer-type-aware entity granularity guard."""
    return _run_structured(
        agents, ex, dtype, max_iters=max_iters, debate_rounds=debate_rounds,
        rerank_final=False, specificity_guard="answer_type", **kw)


def run_granularity_guard_answer_type_soft_commit(agents, ex, dtype, max_iters=6,
                                                  debate_rounds=1,
                                                  final_agents=False, **kw):
    """cdsdngac: cdsdnga plus conflict-only soft commit ablation."""
    return _run_structured(
        agents, ex, dtype, max_iters=max_iters, debate_rounds=debate_rounds,
        rerank_final=False, specificity_guard="answer_type",
        commit_policy="conflict_soft", **kw)


def run_granularity_guard_answer_type_concrete_commit(agents, ex, dtype, max_iters=6,
                                                      debate_rounds=1,
                                                      final_agents=False, **kw):
    """cdsdngax: cdsdnga but commit any concrete resolved node (near gate removal)."""
    return _run_structured(
        agents, ex, dtype, max_iters=max_iters, debate_rounds=debate_rounds,
        rerank_final=False, specificity_guard="answer_type",
        commit_policy="concrete_only", **kw)


def run_granularity_guard_answer_type_always_commit(agents, ex, dtype, max_iters=6,
                                                    debate_rounds=1,
                                                    final_agents=False, **kw):
    """cdsdngaa: cdsdnga but always commit the resolver nodes (full gate removal)."""
    return _run_structured(
        agents, ex, dtype, max_iters=max_iters, debate_rounds=debate_rounds,
        rerank_final=False, specificity_guard="answer_type",
        commit_policy="always", **kw)


def run_granularity_guard_answer_type_debate_select(agents, ex, dtype, max_iters=6,
                                                    debate_rounds=1,
                                                    final_agents=False, **kw):
    """cdsdngad: cdsdnga plus debate-aware final selection after blocked commit."""
    return _run_structured(
        agents, ex, dtype, max_iters=max_iters, debate_rounds=debate_rounds,
        rerank_final=False, specificity_guard="answer_type",
        no_commit_select=True, **kw)


def run_audited_consensus(agents, ex, dtype, max_iters=6, debate_rounds=1,
                          final_agents=False, **kw):
    """cdsdna: structured-node CDSD with answer-type guard and consensus audit."""
    return _run_structured(
        agents, ex, dtype, max_iters=max_iters, debate_rounds=debate_rounds,
        rerank_final=False, specificity_guard="answer_type",
        audit_consensus=True, **kw)


def run_verified_select(agents, ex, dtype, max_iters=6, debate_rounds=1,
                        final_agents=False, **kw):
    """cdsdnv: structured-node CDSD with quote-verified final candidate selection."""
    return _run_structured(
        agents, ex, dtype, max_iters=max_iters, debate_rounds=debate_rounds,
        rerank_final=False, specificity_guard="answer_type",
        verified_final=True, **kw)
