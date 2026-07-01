from __future__ import annotations

import json
import re


JSON_ONLY = "Output only valid JSON. Do not wrap it in Markdown."

NEXT_STEP_SYS = (
    "You are a precise reasoning-tree agent. Your job is to add exactly one "
    "atomic next reasoning hop, not a full solution unless the answer is now "
    "fully determined."
)

MERGE_SYS = (
    "You are a conservative node-merging judge. Merge only candidates that say "
    "the same factual or mathematical step. Do not merge compatible but distinct "
    "hops, and do not merge contradictions."
)

RELATION_SYS = (
    "You are a strict local reasoning judge. Compare candidate next hops under "
    "the same parent prefix, identify contradictions, and score whether each "
    "candidate is useful and reliable."
)

DEBATE_SYS = (
    "You are a local debate participant. Focus only on the disputed next-hop "
    "candidates. Use the question and prefix to find the most reliable branch."
)

RESOLVE_SYS = (
    "You are a debate referee. Resolve only the local conflict among the listed "
    "next-hop candidates. Keep multiple branches only when they can coexist or "
    "the evidence is genuinely insufficient."
)

FINAL_SYS = (
    "You are a final answer selector for a reasoning tree. Prefer a path whose "
    "weakest node is strong. Return the shortest correct final answer."
)


def _question_stem(question: str) -> str:
    parts = re.split(r"\bQuestion\s*:\s*", question or "", flags=re.I)
    return parts[-1].strip() if parts else (question or "").strip()


def is_yes_no_question(question: str) -> bool:
    stem = _question_stem(question).lower()
    if "answer yes or no" in stem:
        return True
    if re.match(r"^(is|are|was|were|do|does|did|has|have|had)\b", stem):
        if re.search(r"\bor\b", stem) and re.search(
            r"\b(first|earlier|later|older|younger|larger|smaller|higher|lower|"
            r"longer|shorter|closer|farther|more|less|before|after)\b",
            stem,
        ):
            return False
    return bool(re.match(r"^(is|are|was|were|do|does|did|can|could|would|should|has|have|had)\b", stem))


def answer_format(dtype: str, question: str = "") -> str:
    if dtype == "mathx":
        return "If final, put the mathematical final answer in answer; LaTeX is allowed."
    if dtype == "math":
        return "If final, put only the final number in answer."
    if dtype == "mc":
        return "If final, put only the option letter A/B/C/D in answer."
    if is_yes_no_question(question):
        return "This is a yes/no question. If final, answer must be exactly 'yes' or 'no'."
    return "If final, put only the short answer span in answer."


def _path_lines(path_nodes) -> str:
    if not path_nodes:
        return "(no prior reasoning hops yet)"
    return "\n".join(f"{i}. {node.text}" for i, node in enumerate(path_nodes, start=1))


def next_step_user(question: str, path_nodes, dtype: str) -> str:
    payload = {
        "question": question,
        "current_prefix": _path_lines(path_nodes),
        "task": (
            "Propose exactly one next reasoning hop from the current prefix. "
            "The hop should be atomic: one factual bridge, one arithmetic step, "
            "one option elimination, or one final-answer step. Do not repeat a "
            "previous prefix node. Do not write a full chain. Write the hop as "
            "a declarative statement, not an instruction such as 'identify', "
            "'determine', or 'find'."
        ),
        "answer_format": answer_format(dtype, question),
        "output_schema": {
            "node": "one atomic next-hop statement",
            "is_final": "boolean; true only if this node directly determines the final answer",
            "answer": "short final answer if is_final, otherwise empty string",
            "confidence": "number from 0.0 to 1.0",
            "why": "one short reason",
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def merge_user(question: str, path_nodes, proposals: list[dict]) -> str:
    payload = {
        "question": question,
        "current_prefix": _path_lines(path_nodes),
        "candidate_proposals": proposals,
        "task": (
            "Cluster semantically equivalent proposals. Equivalent means the "
            "same next reasoning hop under this prefix, not merely related."
        ),
        "output_schema": {
            "clusters": [
                {
                    "members": [0],
                    "canonical_text": "best concise wording for this hop",
                    "reason": "why these members are equivalent",
                }
            ],
            "notes": "short merge rationale",
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def relation_user(question: str, path_nodes, candidates: list[dict], dtype: str) -> str:
    payload = {
        "question": question,
        "current_prefix": _path_lines(path_nodes),
        "candidate_next_hops": candidates,
        "task": (
            "For candidates under the same parent, judge pairwise relations and "
            "score each candidate. A conflict means both cannot be true/useful in "
            "the same reasoning path. Compatible means they can coexist as "
            "separate branches or complementary facts."
        ),
        "answer_format": answer_format(dtype, question),
        "output_schema": {
            "relations": [
                {
                    "a": 0,
                    "b": 1,
                    "relation": "compatible | conflict | uncertain",
                    "reason": "short reason",
                }
            ],
            "scores": [
                {
                    "candidate": 0,
                    "status": "accepted | uncertain | rejected",
                    "score": "0.0 to 1.0",
                    "is_final": "boolean",
                    "answer": "short answer if final, else empty",
                    "reason": "short reason",
                }
            ],
            "rationale": "short global rationale",
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def debate_user(question: str, path_nodes, group: list[dict], own_candidate: dict | None, dtype: str) -> str:
    payload = {
        "question": question,
        "current_prefix": _path_lines(path_nodes),
        "conflict_group": group,
        "your_original_candidate": own_candidate or {},
        "task": (
            "Debate this local group. Identify which candidate(s) should survive, "
            "which should be rejected, and whether multiple branches should remain."
        ),
        "answer_format": answer_format(dtype, question),
        "output_schema": {
            "preferred_candidates": [0],
            "reject_candidates": [1],
            "argument": "concise evidence-based argument",
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def resolve_debate_user(question: str, path_nodes, group: list[dict],
                        debate_statements: list[dict], dtype: str) -> str:
    payload = {
        "question": question,
        "current_prefix": _path_lines(path_nodes),
        "conflict_group": group,
        "debate_statements": debate_statements,
        "task": (
            "Resolve the local debate. Accept reliable candidates, reject wrong "
            "or redundant conflicting candidates, and keep uncertain candidates "
            "only when pruning them would be unsafe."
        ),
        "answer_format": answer_format(dtype, question),
        "output_schema": {
            "decisions": [
                {
                    "candidate": 0,
                    "status": "accepted | uncertain | rejected",
                    "score": "0.0 to 1.0",
                    "is_final": "boolean",
                    "answer": "short answer if final, else empty",
                    "reason": "short reason",
                }
            ],
            "rationale": "short resolution rationale",
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def final_select_user(question: str, paths: list[dict], dtype: str) -> str:
    payload = {
        "question": question,
        "candidate_paths": paths,
        "task": (
            "Select the best final answer from the reasoning-tree paths. Prefer "
            "the path with the highest minimum node score unless its final answer "
            "is not supported by the path."
        ),
        "answer_format": answer_format(dtype, question),
        "important": (
            "Answer the user's question directly. If the path establishes an "
            "intermediate property, convert it into the requested final form. "
            "A candidate path's answer_hint may be intermediate or wrong; use "
            "it only if it satisfies the exact question relation and answer type. "
            "If a node lists multiple facts or roles, choose the one requested "
            "by the question rather than the whole list or a distractor. "
            "For yes/no questions, do not answer with the shared property; "
            "answer exactly yes or no."
        ),
        "output_schema": {
            "answer": "short final answer",
            "chosen_path": "path id from candidate_paths",
            "confidence": "0.0 to 1.0",
            "rationale": "short reason",
        },
    }
    return json.dumps(payload, ensure_ascii=False)
