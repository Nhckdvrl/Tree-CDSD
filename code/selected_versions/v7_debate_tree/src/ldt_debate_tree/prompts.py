from __future__ import annotations

import json

from src.ldt import prompts as base_prompts


JSON_ONLY = base_prompts.JSON_ONLY

FINAL_DEBATE_SYS = (
    "You are a path-level debate participant. Compare complete reasoning-tree "
    "paths, identify unsupported hops and wrong final relations, and argue for "
    "the path that best answers the question."
)

FINAL_REFEREE_SYS = (
    "You are a debate-tree referee. Use the path debate to choose the final "
    "answer. Prefer paths whose debated weaknesses have been resolved by explicit "
    "evidence, not by majority vote alone."
)

ABSENCE_CHALLENGE_SYS = (
    "You are an adversarial evidence finder. A debate-tree path claims that the "
    "answer is missing. Search the provided context and debated paths for a "
    "concrete bridge answer before accepting absence."
)

BRIDGE_COMPLETION_SYS = (
    "You are a bridge-completion debate participant. The current terminal path "
    "was judged incomplete or relation-drifted. Complete the missing bridge from "
    "the provided context if possible."
)

TERMINALITY_DEBATE_SYS = (
    "You are a debate-tree terminality critic. Decide whether a proposed final "
    "node really satisfies the exact question relation, or whether it is only an "
    "intermediate entity that must remain open for another tree hop."
)

TERMINALITY_REFEREE_SYS = (
    "You are a debate-tree terminality referee. Resolve agents' critiques and "
    "decide whether the candidate node should be committed as a terminal answer "
    "or reopened as an expandable intermediate node."
)

SLOT_GROUNDING_SYS = (
    "You are an answer-slot grounding debater. Given the debated reasoning-tree "
    "paths and the current answer, identify the exact answer span that satisfies "
    "the question's requested relation."
)

CHALLENGE_SYS = (
    "You are an adversarial debate-tree challenger. Your job is to test whether "
    "the current majority next-hop has missed a better local branch, confused an "
    "entity, or answered the wrong relation."
)

SYNTHESIS_SYS = (
    "You are a debate-tree synthesis judge. Decide whether compatible local "
    "candidates should be fused into one reasoning state or kept separate."
)

CONTRACT_SYS = (
    "You are a debate-tree planning agent. Define the local reasoning contract "
    "that later tree nodes must satisfy, without solving from outside the given "
    "context."
)

CONTRACT_REFEREE_SYS = (
    "You are a debate-tree contract referee. Resolve multiple agents' contracts "
    "into one concise reasoning contract for tree expansion."
)


def answer_role_contract(question: str, *, strict_role_edges: bool = False) -> str:
    text = (
        "Before accepting any final answer, infer the answer role from the exact "
        "question wording. If the question compares alternatives such as 'A or B' "
        "with first/earlier/later/older/younger/higher/lower, return the selected "
        "alternative entity, not yes/no. For kinship or role questions, return the "
        "entity occupying the requested role and keep relation direction exact "
        "(father-in-law is the spouse's father, not the spouse; mother-in-law is "
        "the spouse's mother, not the spouse; paternal grandparent is the "
        "father's parent, not the father; maternal grandparent is the mother's "
        "parent, not the mother; do not confuse spouse, parent, child, or "
        "grandparent). "
        "For 'what/which [type]' questions, return an instance of the requested "
        "type; reject related entities of a different type. "
        "If the question asks for a named person, company, place, network, league, "
        "or work, return a concrete name rather than a descriptive placeholder. "
        "For dates, return the most specific date supported by the path."
    )
    if strict_role_edges:
        text += (
            " For relation-direction questions, a terminal path must instantiate "
            "the required role edges; do not accept a final node that merely "
            "repeats the target relation without showing the bridge edge that "
            "makes the answer occupy that role."
        )
    return text


def contract_user(question: str) -> str:
    payload = {
        "question": question,
        "task": (
            "State the answer slot and bridge relation that the reasoning tree "
            "must satisfy. Do not provide a long solution. Focus on what would "
            "count as answering the exact question rather than an intermediate "
            "entity or a merely compatible fact. Do not name the unknown final "
            "answer or unresolved bridge endpoint; write placeholders such as "
            "[producer], [spouse], or [answer] instead."
        ),
        "output_schema": {
            "answer_slot": "requested answer type or entity slot",
            "bridge_contract": "minimal relation chain to verify",
            "reject_if": ["relation drift or premature stopping condition"],
            "reason": "short",
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def resolve_contract_user(question: str, contracts: list[dict]) -> str:
    payload = {
        "question": question,
        "agent_contracts": contracts,
        "task": (
            "Resolve the contracts into one tree-level contract. Keep it concise "
            "and general. It should guide debate and synthesis, but it must not "
            "directly force a final answer. Do not include a guessed final answer "
            "or unresolved endpoint; keep unknowns as placeholders."
        ),
        "output_schema": {
            "answer_slot": "requested answer type or entity slot",
            "bridge_contract": "minimal relation chain to verify",
            "reject_if": ["relation drift or premature stopping condition"],
            "confidence": "0.0 to 1.0",
            "reason": "short",
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def challenge_user(question: str, prefix: str, majority_candidate: dict, dtype: str) -> str:
    payload = {
        "question": question,
        "current_prefix": prefix,
        "majority_candidate": majority_candidate,
        "task": (
            "Propose one competing next-hop only if it is a plausible local "
            "alternative to the majority candidate. The alternative may be a "
            "different entity, relation, answer slot, or missing bridge hop. If "
            "the majority candidate is already the best local hop, return an "
            "empty node."
        ),
        "answer_format": base_prompts.answer_format(dtype, question),
        "output_schema": {
            "node": "one competing atomic next-hop, or empty string",
            "is_final": "boolean",
            "answer": "short final answer if final, otherwise empty",
            "confidence": "0.0 to 1.0",
            "challenge_reason": "why this challenges the majority candidate",
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def synthesis_user(question: str, prefix: str, candidates: list[dict]) -> str:
    payload = {
        "question": question,
        "current_prefix": prefix,
        "compatible_candidates": candidates,
        "task": (
            "Return combine=true only if the candidates are complementary facts "
            "on the same reasoning chain and should both be carried forward in "
            "one tree state. Return combine=false if they are alternative "
            "entities, alternative answers, relation drift, a claim that evidence "
            "is missing, or facts that can coexist but do not advance the same "
            "bridge chain."
        ),
        "output_schema": {
            "combine": "boolean",
            "reason": "short reason",
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def final_path_debate_user(
    question: str,
    paths: list[dict],
    dtype: str,
    *,
    strict_role_edges: bool = False,
) -> str:
    payload = {
        "question": question,
        "answer_role_contract": answer_role_contract(question, strict_role_edges=strict_role_edges),
        "candidate_paths": paths,
        "task": (
            "Debate the terminal paths. For each serious rival, state whether "
            "its final answer follows from its path, where the path drifts from "
            "the requested relation, whether the answer has the required role "
            "and form, and which path should survive."
        ),
        "answer_format": base_prompts.answer_format(dtype, question),
        "output_schema": {
            "preferred_path": "path id",
            "reject_paths": ["path id"],
            "argument": "concise evidence-based argument",
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def terminality_debate_user(
    question: str,
    prefix: str,
    candidate: dict,
    dtype: str,
    *,
    strict_role_edges: bool = False,
) -> str:
    task = (
        "Debate only terminality. Check whether this node's answer satisfies "
        "the exact requested relation in the question. If it merely names an "
        "intermediate bridge entity, a broader/narrower location, a related "
        "person, or an attribute from the wrong hop, mark it non-terminal and "
        "state the missing next relation. First identify the target answer "
        "slot and the candidate answer slot. Be strict about type and "
        "granularity: a town is not a borough, a country is not a province, "
        "a parent is not a grandparent, and an entity description is not the "
        "named company/person/place requested. For questions asking a "
        "quantity, date, year, count, or population, the answer slot is the "
        "quantity/date itself; do not force another entity hop unless the "
        "quantity clearly belongs to the wrong subject. Do not guess a new "
        "final answer unless the candidate itself already supports it. Apply "
        "the answer_role_contract strictly when deciding terminality."
    )
    schema = {
        "target_slot": "the exact answer type/relation requested by the question",
        "candidate_slot": "what type/relation the candidate answer actually gives",
        "slot_match": "exact | too_broad | too_narrow | wrong_type | unsupported | unknown",
        "relation_complete": "boolean",
        "is_terminal": "boolean",
        "answer": "short final answer if terminal, otherwise empty",
        "missing_hop": "relation still needed if non-terminal, otherwise empty",
        "critique": "short reason grounded in the question and prefix",
    }
    if strict_role_edges:
        task += (
            " Write the required_edges from the question and the observed_edges "
            "actually proved by the prefix plus candidate. If the candidate only "
            "restates the requested relation, or adds a new role edge not entailed "
            "by the observed_edges, mark unsupported_assertion=true and "
            "is_terminal=false."
        )
        schema.update({
            "required_edges": ["minimal role/relation edges required by the question"],
            "observed_edges": ["role/relation edges actually supported by the prefix and candidate"],
            "role_direction_ok": "boolean",
            "unsupported_assertion": "boolean",
        })
    payload = {
        "question": question,
        "answer_role_contract": answer_role_contract(question, strict_role_edges=strict_role_edges),
        "current_prefix": prefix,
        "candidate_final_node": candidate,
        "task": task,
        "answer_format": base_prompts.answer_format(dtype, question),
        "output_schema": schema,
    }
    return json.dumps(payload, ensure_ascii=False)


def resolve_terminality_user(
    question: str,
    prefix: str,
    candidate: dict,
    critiques: list[dict],
    dtype: str,
    *,
    strict_role_edges: bool = False,
) -> str:
    task = (
        "Resolve whether the candidate should close this tree path. Return "
        "is_terminal=false if the candidate is an intermediate hop or relation "
        "drift, even when the entity is mentioned in the context. Return "
        "is_terminal=true only when the answer directly satisfies the exact "
        "question relation. Require slot_match=exact and relation_complete=true. "
        "If agents disagree, prefer reopening the node unless the exact answer "
        "slot is explicitly supported by the prefix or candidate node. Apply "
        "the answer_role_contract strictly."
    )
    schema = {
        "target_slot": "the exact answer type/relation requested by the question",
        "candidate_slot": "what type/relation the candidate answer actually gives",
        "slot_match": "exact | too_broad | too_narrow | wrong_type | unsupported | unknown",
        "relation_complete": "boolean",
        "is_terminal": "boolean",
        "answer": "short final answer if terminal, otherwise empty",
        "missing_hop": "relation still needed if non-terminal, otherwise empty",
        "confidence": "0.0 to 1.0",
        "rationale": "short reason",
    }
    if strict_role_edges:
        task = (
            "Resolve whether the candidate should close this tree path. Return "
            "is_terminal=false if the candidate is an intermediate hop or relation "
            "drift, even when the entity is mentioned in the context. Return "
            "is_terminal=true only when the answer directly satisfies the exact "
            "question relation. Require slot_match=exact and relation_complete=true. "
            "Also require role_direction_ok=true and unsupported_assertion=false: "
            "the terminal path must prove the required role edges, not merely "
            "repeat the question's relation label. "
            "If agents disagree, prefer reopening the node unless the exact answer "
            "slot is explicitly supported by the prefix or candidate node. Apply "
            "the answer_role_contract strictly."
        )
        schema.update({
            "required_edges": ["minimal role/relation edges required by the question"],
            "observed_edges": ["role/relation edges actually supported by the prefix and candidate"],
            "role_direction_ok": "boolean",
            "unsupported_assertion": "boolean",
        })
    payload = {
        "question": question,
        "answer_role_contract": answer_role_contract(question, strict_role_edges=strict_role_edges),
        "current_prefix": prefix,
        "candidate_final_node": candidate,
        "terminality_critiques": critiques,
        "task": task,
        "answer_format": base_prompts.answer_format(dtype, question),
        "output_schema": schema,
    }
    return json.dumps(payload, ensure_ascii=False)


def resolve_final_debate_user(
    question: str,
    paths: list[dict],
    debate_statements: list[dict],
    dtype: str,
    *,
    strict_role_edges: bool = False,
) -> str:
    task = (
        "Resolve the terminal path debate. Select one path or synthesize the "
        "short final answer only when the debate identifies a path-supported "
        "correction. Reject answers that violate the answer_role_contract, "
        "even if their path contains related true facts. Do not use outside "
        "knowledge."
    )
    schema = {
        "answer": "short final answer",
        "chosen_path": "path id",
        "confidence": "0.0 to 1.0",
        "rationale": "short reason grounded in the debated paths",
    }
    if strict_role_edges:
        task = (
            "Resolve the terminal path debate. Select one path or synthesize the "
            "short final answer only when the debate identifies a path-supported "
            "correction. Reject answers that violate the answer_role_contract, "
            "or whose path only asserts the target relation without proving the "
            "required role edges. Prefer a path that explicitly instantiates the "
            "question's bridge relation over a path that only paraphrases it, "
            "and prefer a longer completed bridge over a shorter path that stops "
            "at a related intermediate entity with the wrong answer granularity, "
            "even if their path contains related true facts. For in-law or "
            "grandparent questions, treat spouse/parent as bridge entities unless "
            "the requested role is exactly spouse/parent. Do not use outside "
            "knowledge."
        )
        schema["role_edge_audit"] = (
            "short comparison of the required role edge and the chosen path's observed edge"
        )
    payload = {
        "question": question,
        "answer_role_contract": answer_role_contract(question, strict_role_edges=strict_role_edges),
        "candidate_paths": paths,
        "debate_statements": debate_statements,
        "task": task,
        "answer_format": base_prompts.answer_format(dtype, question),
        "output_schema": schema,
    }
    return json.dumps(payload, ensure_ascii=False)


def slot_grounding_user(
    question: str,
    paths: list[dict],
    current_answer: str,
    final_resolution: dict,
    dtype: str,
) -> str:
    payload = {
        "question": question,
        "candidate_paths": paths,
        "current_answer": current_answer,
        "final_path_resolution": final_resolution,
        "task": (
            "Ground the final answer to the exact requested slot. If the path or "
            "context mentions several related entities, roles, locations, dates, "
            "or attributes, choose only the one that satisfies the question's "
            "relation. If the current_answer is already exact, keep it."
        ),
        "answer_format": base_prompts.answer_format(dtype, question),
        "output_schema": {
            "answer": "short exact answer span",
            "keep_current": "boolean",
            "slot": "brief description of requested slot",
            "argument": "short evidence-based reason",
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def resolve_slot_grounding_user(
    question: str,
    paths: list[dict],
    current_answer: str,
    grounding_arguments: list[dict],
    dtype: str,
) -> str:
    payload = {
        "question": question,
        "candidate_paths": paths,
        "current_answer": current_answer,
        "grounding_arguments": grounding_arguments,
        "task": (
            "Resolve the answer-slot grounding debate. Return the shortest exact "
            "answer that satisfies the requested relation. Prefer a candidate "
            "with explicit support in the paths or context. If no candidate is "
            "better grounded than current_answer, keep current_answer."
        ),
        "answer_format": base_prompts.answer_format(dtype, question),
        "output_schema": {
            "answer": "short exact final answer",
            "changed": "boolean",
            "confidence": "0.0 to 1.0",
            "rationale": "short reason",
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def absence_challenge_user(question: str, selected_answer: str, paths: list[dict], dtype: str) -> str:
    payload = {
        "question": question,
        "selected_absence_answer": selected_answer,
        "candidate_paths": paths,
        "task": (
            "The current answer says the information is missing or cannot be "
            "determined. Try to refute that by finding a concrete answer supported "
            "by the passages or paths. If no concrete answer is supported, keep "
            "the absence answer."
        ),
        "answer_format": base_prompts.answer_format(dtype, question),
        "output_schema": {
            "concrete_answer": "short answer if found, otherwise empty",
            "keep_absence": "boolean",
            "argument": "short evidence-based reason",
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def resolve_absence_challenge_user(
    question: str,
    selected_answer: str,
    paths: list[dict],
    challenges: list[dict],
    dtype: str,
) -> str:
    payload = {
        "question": question,
        "selected_absence_answer": selected_answer,
        "candidate_paths": paths,
        "absence_challenges": challenges,
        "task": (
            "Resolve whether the absence answer survives adversarial challenge. "
            "Return a concrete answer only if at least one challenge gives a "
            "path-supported bridge answer; otherwise keep the absence answer."
        ),
        "answer_format": base_prompts.answer_format(dtype, question),
        "output_schema": {
            "answer": "short final answer",
            "absence_survives": "boolean",
            "confidence": "0.0 to 1.0",
            "rationale": "short reason",
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def bridge_completion_user(question: str, paths: list[dict], dtype: str) -> str:
    payload = {
        "question": question,
        "candidate_paths": paths,
        "task": (
            "The terminal debate did not accept the path's answer. Complete the "
            "missing bridge relation from the question using the provided context "
            "and paths. Return a concrete answer only if supported."
        ),
        "answer_format": base_prompts.answer_format(dtype, question),
        "output_schema": {
            "answer": "short answer if the missing bridge can be completed, otherwise empty",
            "argument": "short evidence-based reason",
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def resolve_bridge_completion_user(
    question: str,
    paths: list[dict],
    completions: list[dict],
    dtype: str,
) -> str:
    payload = {
        "question": question,
        "candidate_paths": paths,
        "bridge_completion_arguments": completions,
        "task": (
            "Resolve the missing-bridge debate. Return the concrete answer only "
            "if it satisfies the exact question relation and is supported by the "
            "context; otherwise return an empty answer."
        ),
        "answer_format": base_prompts.answer_format(dtype, question),
        "output_schema": {
            "answer": "short final answer or empty",
            "confidence": "0.0 to 1.0",
            "rationale": "short reason",
        },
    }
    return json.dumps(payload, ensure_ascii=False)
