"""All CDSD prompt templates in one place — tweak these to iterate on the algorithm.

Keep strings stable if you want disk-cache reuse across runs (the cache is keyed on the
exact messages). Change a template and the affected calls will re-hit the model.
"""

JSON_ONLY_SUFFIX = " /no_think"

JUDGE_SYS = (
    "You are a meticulous reasoning referee. You output only valid JSON, nothing else."
    + JSON_ONLY_SUFFIX
)
RESOLVE_SYS = (
    "You are a decisive reasoning referee. You output only valid JSON, nothing else."
    + JSON_ONLY_SUFFIX
)

MATH_INSTR = "Solve the problem step by step. End with a line exactly: 'Answer: <number>'."
QA_INSTR = ("Answer the question. Reason step by step using intermediate facts, "
            "then end with a line exactly: 'Answer: <short answer>'.")
MC_INSTR = ("Reason step by step, then choose the correct option. "
            "End with a line exactly: 'Answer: <letter A/B/C/D>'.")


MATHX_INSTR = ("Solve the competition math problem step by step. "
               "End with your final answer in \\boxed{...}.")


def instr(dtype):
    if dtype == "mathx":
        return MATHX_INSTR
    if dtype == "math":
        return MATH_INSTR
    if dtype == "mc":
        return MC_INSTR
    return QA_INSTR


NODE_EXTRACT_SYS = (
    "You are a strict reasoning-node extractor. You convert a solver's solution "
    "into small ordered checkable nodes and output only valid JSON."
    + JSON_ONLY_SUFFIX
)


def gen_claims_user(question, premises, dtype):
    prem = "\n".join(f"- {p}" for p in premises) if premises else "None yet."
    kind = {"math": "number", "mc": "letter A/B/C/D"}.get(dtype, "short answer")
    return (
        f"You are solving the following problem together with other independent solvers.\n"
        f"Problem: {question}\n\n"
        f"Established facts already agreed upon (treat as TRUE and build on them):\n{prem}\n\n"
        f"Continue the reasoning to the final answer. Output your reasoning as a numbered list of "
        f"atomic claims, ONE per line, each a single verifiable step:\n"
        f"Claim 1: <step>\nClaim 2: <step>\n...\n"
        f"Then a final line exactly: 'Answer: <{kind}>'."
    )


def gen_reasoning_user(question, premises, dtype):
    prem = "\n".join(f"- {p}" for p in premises) if premises else "None yet."
    return (
        f"{instr(dtype)}\n\n"
        f"Problem: {question}\n\n"
        f"Established facts already agreed upon (treat as TRUE and build on them):\n{prem}\n\n"
        f"Give a complete solution. Keep intermediate facts explicit enough that another "
        f"solver can later compare them step by step."
    )


def extract_nodes_user(question, reasoning, answer, dtype):
    if dtype in ("math", "mathx"):
        task = (
            "For math, each node should be one arithmetic operation, algebraic "
            "transformation, case split, or deduction. Use formula-level nodes, not prose summaries. "
            "Do not include subject/relation/object/evidence fields unless they are essential."
        )
        schema = (
            f"Return ONLY this compact JSON object:\n"
            f'{{"nodes":[{{"id":1,"kind":"setup|operation|deduction|case|check|answer",'
            f'"claim":"one short mathematical step",'
            f'"expression":"optional formula being transformed",'
            f'"result":"optional result"}}],"answer":"..."}}'
        )
    elif dtype == "mc":
        task = (
            "For multiple choice, each node should be one evidence fact, option check, "
            "or option elimination. Preserve option letters and whether an option is supported, "
            "contradicted, eliminated, or unknown."
        )
        schema = (
            f"Return ONLY this compact JSON object:\n"
            f'{{"nodes":[{{"id":1,"kind":"fact|option_check|elimination|comparison|answer",'
            f'"claim":"one short checkable step",'
            f'"option":"A|B|C|D or empty",'
            f'"status":"supported|contradicted|eliminated|unknown|answer",'
            f'"evidence_quote":"short quote when available"}}],"answer":"..."}}'
        )
    else:
        task = (
            "For QA, each node should be one evidence-backed entity/relation/value claim "
            "or one bridge connecting two facts. If passages/facts are present, copy a "
            "short exact quote into evidence_quote when the node uses that evidence."
        )
        schema = (
            f"Return ONLY this JSON object:\n"
            f'{{"nodes":[{{"id":1,"kind":"fact|bridge|deduction|answer",'
            f'"claim":"one atomic node as a sentence",'
            f'"subject":"","relation":"","object":"",'
            f'"evidence_quote":""}}],"answer":"..."}}'
        )
    return (
        f"Problem:\n{question}\n\n"
        f"Solver solution text:\n{reasoning}\n\n"
        f"Detected final answer: {answer or '(none)'}\n\n"
        f"Extract the solver's reasoning into an ordered node chain.\n"
        f"{task}\n\n"
        f"Rules:\n"
        f"- Preserve the solver's actual reasoning; do not invent new facts.\n"
        f"- Each node must contain exactly one checkable claim, operation, or bridge.\n"
        f"- Split broad summary sentences into smaller nodes when they contain multiple facts.\n"
        f"- Keep nodes aligned to the original question's subject and relation.\n"
        f"- Keep JSON compact: no markdown headings, no display-math delimiters, no repeated formulas.\n"
        f"- Keep each claim under 20 words when possible; use terse expression/result fields.\n"
        f"- Use 2-8 nodes for normal QA/MC and 2-12 nodes for math unless the solution is truly shorter.\n"
        f"- The final node may state the final answer if it follows from prior nodes.\n\n"
        f"{schema}"
    )


def argue_user(question, premises, conflict_desc, my_claim, others_claims, dtype, constructive):
    prem = "\n".join(f"- {p}" for p in premises) if premises else "None."
    others = "\n".join(f"- Solver {i}: {c}" for i, c in others_claims)
    if constructive:
        guide = ("Critically check this step, but your goal is to COMPLETE the reasoning, not to refuse "
                 "it. If the facts/passages support a conclusion by combining information across them, "
                 "state it concretely. Only say information is missing if it is genuinely absent.")
    else:
        guide = ("Critically examine all positions. Check for arithmetic mistakes, misread details, or "
                 "wrong facts. Defend your position if it is correct, or revise it if you find an error.")
    return (
        f"Problem: {question}\n"
        f"Established facts: {prem}\n\n"
        f"There is a disagreement about this reasoning step:\n{conflict_desc}\n\n"
        f"Your current position: {my_claim}\n"
        f"Other solvers' positions:\n{others}\n\n"
        f"{guide} Give a brief justification (2-4 sentences), then end with a line exactly: "
        f"'My claim: <claim>'."
    )


def argue_segment_user(question, agreed_prefix, my_segment, conflict_desc, others_claims, dtype):
    prem = "\n".join(f"- {p}" for p in agreed_prefix) if agreed_prefix else "None."
    seg = "\n".join(f"  {i + 1}. {c}" for i, c in enumerate(my_segment)) if my_segment else "(none)"
    others = "\n".join(f"- Solver {i}: {c}" for i, c in others_claims)
    return (
        f"Problem: {question}\n"
        f"Agreed facts so far (treat as TRUE): {prem}\n\n"
        f"Your reasoning for the current segment (the steps leading to a disputed point):\n{seg}\n\n"
        f"Disputed point: {conflict_desc}\n"
        f"Other solvers' positions at the disputed point:\n{others}\n\n"
        f"Re-examine your segment. Goal: COMPLETE the reasoning using the agreed facts and any passages "
        f"in the problem (combine information across passages when needed); only say information is "
        f"missing if it is genuinely absent. Give a brief justification (2-4 sentences), then end with a "
        f"line exactly: 'My claim: <your corrected conclusion for this segment>'."
    )


def conflict_user(question, premises, agents_claims, agents_answers, dtype, constructive):
    prem = "\n".join(f"- {p}" for p in premises) if premises else "None yet."
    blocks = []
    for i, claims in enumerate(agents_claims):
        cl = "\n".join(f"  Claim {j + 1}: {c}" for j, c in enumerate(claims)) or "  (no claims)"
        blocks.append(f"Solver {i} (final answer: {agents_answers[i]}):\n{cl}")
    body = "\n\n".join(blocks)
    extra = ("Do NOT flag a conflict merely because solvers phrase things differently or one is more "
             "cautious; only flag a GENUINE contradiction in a value, fact, or logical step.\n"
             if constructive else "")
    return (
        f"Compare these independent solvers working on the same problem.\n"
        f"Problem: {question}\n\n"
        f"Established facts (already agreed, treat as TRUE):\n{prem}\n\n"
        f"Each solver's new reasoning:\n{body}\n\n"
        f"Decide:\n"
        f"- If all solvers reach the SAME final answer and there is no material disagreement in the "
        f"reasoning, respond consensus with that final answer.\n"
        f"- Otherwise find the EARLIEST claim where the solvers materially disagree (they assert a "
        f"different value, fact, or logical step). Summarize the disagreement and quote each solver's "
        f"conflicting claim.\n"
        + extra
        + "\n"
        f"Respond with ONLY one JSON object:\n"
        f'{{"status":"consensus","final_answer":"..."}}\n'
        f"OR\n"
        f'{{"status":"conflict","description":"...","claims":{{"0":"...","1":"...","2":"..."}}}}'
    )


def conflict_indexed_user(question, premises, agents_claims, agents_answers, dtype):
    """Indexed conflict schema for faithful segment slicing.

    The older conflict prompt quotes conflicting claims but does not reliably expose claim
    indices, so v3 cannot know which prefix segment to debate. This prompt asks the judge to
    return a 1-based claim index per solver.
    """
    prem = "\n".join(f"- {p}" for p in premises) if premises else "None yet."
    blocks = []
    for i, claims in enumerate(agents_claims):
        cl = "\n".join(f"  Claim {j + 1}: {c}" for j, c in enumerate(claims)) or "  (no claims)"
        blocks.append(f"Solver {i} (final answer: {agents_answers[i]}):\n{cl}")
    body = "\n\n".join(blocks)
    return (
        f"Compare these independent solvers working on the same problem.\n"
        f"Problem: {question}\n\n"
        f"Established facts (already agreed, treat as TRUE):\n{prem}\n\n"
        f"Each solver's new reasoning:\n{body}\n\n"
        f"Your job is to find the EARLIEST material conflict in the reasoning, not merely the "
        f"final-answer disagreement. A material conflict is a different entity, relation, value, "
        f"or logical bridge needed to answer the exact question. Do not broaden the question's "
        f"subject or relation.\n\n"
        f"If all solvers reach the same final answer and there is no material disagreement, return "
        f'{{"status":"consensus","final_answer":"..."}}.\n'
        f"Otherwise return ONLY JSON with the 1-based conflicting claim index for each solver. "
        f"The claims object MUST contain one entry for every solver id shown above. "
        f"If a solver omits the exact conflicting fact, use the nearest claim that creates its "
        f"final answer and copy that claim text exactly:\n"
        f'{{"status":"conflict","description":"...",'
        f'"claims":{{"0":{{"index":1,"claim":"..."}},'
        f'"1":{{"index":1,"claim":"..."}},'
        f'"2":{{"index":1,"claim":"..."}}}}}}'
    )


def argue_indexed_segment_user(question, agreed_prefix, my_segment, conflict_desc, others_claims, dtype):
    prem = "\n".join(f"- {p}" for p in agreed_prefix) if agreed_prefix else "None."
    seg = "\n".join(f"  {i + 1}. {c}" for i, c in enumerate(my_segment)) if my_segment else "(none)"
    others = "\n".join(f"- Solver {i}: {c}" for i, c in others_claims)
    return (
        f"Problem: {question}\n"
        f"Agreed facts so far (treat as TRUE):\n{prem}\n\n"
        f"Your reasoning segment up to the first disputed claim:\n{seg}\n\n"
        f"Disputed point:\n{conflict_desc}\n\n"
        f"Other solvers' disputed claims:\n{others}\n\n"
        f"Re-examine ONLY this segment. Preserve the exact subject and relation asked by the "
        f"question; do not replace a local entity relation with a broader background fact. Use the "
        f"given passages/facts when present. Give a brief justification (2-4 sentences), then end "
        f"with a line exactly: 'My claim: <corrected segment conclusion>'."
    )


def resolve_indexed_segment_user(question, agreed_prefix, desc, args):
    prem = "\n".join(f"- {p}" for p in agreed_prefix) if agreed_prefix else "None."
    return (
        f"Problem: {question}\n"
        f"Agreed facts so far:\n{prem}\n\n"
        f"The solvers debated the segment leading to this first disputed point:\n{desc}\n\n"
        f"Their final positions:\n{args}\n\n"
        f"Produce 1-3 atomic resolved nodes for this segment. The nodes must preserve the exact "
        f"subject/relation requested by the question and must be supported by the passages/facts "
        f"when passages are present. Put exact supporting quotes from the problem/passages in "
        f"evidence_quotes; each quote must be copied verbatim, not paraphrased. Set "
        f"relation_preserved=false if the best node broadens, "
        f"changes, or dodges the question relation. Set confident=false if the evidence is absent "
        f"or the positions remain genuinely contradictory.\n"
        f'Respond ONLY JSON: {{"resolved_nodes":["..."],"evidence_quotes":["..."],'
        f'"relation_preserved":true|false,"confident":true|false}}'
    )


def bridge_recover_user(question, agreed_prefix, conflict_desc, agents_answers, resolved_nodes):
    prem = "\n".join(f"- {p}" for p in agreed_prefix) if agreed_prefix else "None."
    answers = "\n".join(f"- Solver {i}: {a}" for i, a in enumerate(agents_answers))
    nodes = "\n".join(f"- {n}" for n in resolved_nodes) if resolved_nodes else "None."
    return (
        f"Problem: {question}\n\n"
        f"Agreed facts so far:\n{prem}\n\n"
        f"The segment debate could not be safely committed because evidence or relation "
        f"preservation was uncertain.\n"
        f"Disputed point:\n{conflict_desc}\n\n"
        f"Uncommitted tentative nodes:\n{nodes}\n\n"
        f"Agent final answers:\n{answers}\n\n"
        f"Using ONLY the passages/facts inside the problem, try to recover the minimal missing "
        f"bridge needed to answer the exact question. Multi-hop bridges are allowed, but then "
        f"you must provide enough separate exact quotes to verify every hop in the bridge "
        f"(for example: entity -> county, then county -> bordering county). Preserve relation direction exactly "
        f"(for example paternal vs maternal, parent vs child, spouse of X vs performer named X, "
        f"bordering county/town direction, founded era vs development era). Do not answer from "
        f"outside knowledge. If the quotes do not establish the full chain, set chain_complete=false "
        f"and confident=false.\n\n"
        f"Return ONLY JSON with:\n"
        f'- "answer": the shortest answer to the original question;\n'
        f'- "bridge_node": one atomic bridge fact that directly supports that answer;\n'
        f'- "evidence_quotes": exact verbatim quote(s) copied from the problem/passages;\n'
        f'- "relation_preserved": false if the node changes or dodges the question relation;\n'
        f'- "chain_complete": true only if the quote set verifies the full bridge, not just one hop;\n'
        f'- "confident": true only if the quote(s) support the bridge.\n'
        f'Respond ONLY JSON: {{"answer":"...","bridge_node":"...",'
        f'"evidence_quotes":["..."],"relation_preserved":true|false,'
        f'"chain_complete":true|false,"confident":true|false}}'
    )


def nonanswer_recover_user(question, agreed_prefix, agents_answers, agents_nodes):
    prem = "\n".join(f"- {p}" for p in agreed_prefix) if agreed_prefix else "None."
    answers = "\n".join(f"- Solver {i}: {a}" for i, a in enumerate(agents_answers))
    blocks = []
    for i, nodes in enumerate(agents_nodes):
        body = "\n".join(f"  {j + 1}. {n}" for j, n in enumerate(nodes[:8])) or "  (none)"
        blocks.append(f"Solver {i} nodes:\n{body}")
    node_text = "\n\n".join(blocks)
    return (
        f"Problem:\n{question}\n\n"
        f"Agreed facts so far:\n{prem}\n\n"
        f"All solvers converged to a non-answer or insufficient-information answer:\n{answers}\n\n"
        f"Their extracted reasoning nodes were:\n{node_text}\n\n"
        f"Using ONLY the passages/facts inside the problem, decide whether the original question "
        f"actually has a supported answer. If it does, recover the shortest answer and one atomic "
        f"bridge node that directly supports it. Provide exact verbatim quote(s) copied from the "
        f"problem/passages. If the evidence does not establish the answer, set confident=false.\n\n"
        f"Return ONLY JSON: "
        f'{{"answer":"...","bridge_node":"...",'
        f'"evidence_quotes":["..."],"relation_preserved":true|false,'
        f'"chain_complete":true|false,"confident":true|false}}'
    )


def consensus_audit_user(question, agreed_prefix, agents_answers, agents_nodes, draft_answer):
    prem = "\n".join(f"- {p}" for p in agreed_prefix) if agreed_prefix else "None."
    answers = "\n".join(f"- Solver {i}: {a}" for i, a in enumerate(agents_answers))
    blocks = []
    for i, nodes in enumerate(agents_nodes):
        body = "\n".join(f"  {j + 1}. {n}" for j, n in enumerate(nodes[:8])) or "  (none)"
        blocks.append(f"Solver {i} nodes:\n{body}")
    node_text = "\n\n".join(blocks)
    return (
        f"Problem:\n{question}\n\n"
        f"Agreed facts so far:\n{prem}\n\n"
        f"The conflict judge marked the solvers as consensus, but this consensus may be a "
        f"non-answer, an underspecified answer, or a relation-drift answer.\n\n"
        f"Draft consensus answer: {draft_answer}\n\n"
        f"Solver answers:\n{answers}\n\n"
        f"Extracted reasoning nodes:\n{node_text}\n\n"
        f"Audit the consensus against the exact original question. Using ONLY the passages/facts "
        f"inside the problem, either verify the draft answer or replace it with the shortest answer "
        f"that directly satisfies the asked subject and relation. You must provide exact verbatim "
        f"quote(s) copied from the problem/passages that establish every hop. If the passages do not "
        f"establish the answer, set confident=false. Set relation_preserved=false if the answer "
        f"changes the asked relation, e.g. birthplace vs country, cause vs event, person vs office, "
        f"award vs nominee, or location of an entity vs location of a related entity.\n\n"
        f"Return ONLY JSON: "
        f'{{"answer":"...","bridge_node":"...",'
        f'"evidence_quotes":["..."],"relation_preserved":true|false,'
        f'"chain_complete":true|false,"confident":true|false}}'
    )


def candidate_verify_user(question, candidates):
    cand = "\n".join(f"- Candidate {i}: {a}" for i, a in enumerate(candidates))
    return (
        f"Problem:\n{question}\n\n"
        f"Candidate final answers from the solvers:\n{cand}\n\n"
        f"Choose the single candidate that is best supported by the passages/facts and directly "
        f"answers the exact question relation. You may only choose one of the candidate answer "
        f"strings above, copied exactly. Do not invent a new answer. Provide exact verbatim quote(s) "
        f"from the problem/passages that establish the chosen candidate and relation. If no candidate "
        f"is clearly supported, set confident=false.\n\n"
        f"Return ONLY JSON: "
        f'{{"answer":"<one exact candidate string>",'
        f'"evidence_quotes":["..."],"relation_preserved":true|false,'
        f'"confident":true|false}}'
    )


def no_commit_select_user(question, agreed_prefix, agents_answers, conflict_desc,
                          resolved_nodes, debate_transcript, candidates):
    prem = "\n".join(f"- {p}" for p in agreed_prefix) if agreed_prefix else "None."
    answers = "\n".join(f"- Solver {i}: {a}" for i, a in enumerate(agents_answers))
    nodes = "\n".join(f"- {n}" for n in resolved_nodes) if resolved_nodes else "None."
    debate = []
    for item in debate_transcript or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if len(text) > 1200:
            text = text[:1200] + " ..."
        debate.append(f"- round {item.get('round')} solver {item.get('agent')}: {text}")
    debate_text = "\n".join(debate) if debate else "None."
    cand = "\n".join(f"- Candidate {i}: {a}" for i, a in enumerate(candidates))
    return (
        f"Problem:\n{question}\n\n"
        f"Agreed facts so far:\n{prem}\n\n"
        f"The conflict-localized debate found a dispute but the resolved node was NOT safe "
        f"to commit into the reasoning chain. Do not blindly trust the resolved node.\n\n"
        f"Disputed point:\n{conflict_desc}\n\n"
        f"Uncommitted resolved node(s):\n{nodes}\n\n"
        f"Solver final answers:\n{answers}\n\n"
        f"Debate transcript:\n{debate_text}\n\n"
        f"Allowed final-answer candidates:\n{cand}\n\n"
        f"Use the debate only as diagnostic evidence. Choose the single candidate answer that is "
        f"best supported by exact passages/facts in the problem and directly preserves the relation "
        f"asked by the original question. You may only choose one candidate string above, copied "
        f"exactly. If the debate shows every candidate is unsupported or relation-drifted, set "
        f"confident=false. Provide exact verbatim evidence quote(s) copied from the problem/passages "
        f"that support the chosen candidate.\n\n"
        f"Return ONLY JSON: "
        f'{{"answer":"<one exact candidate string>",'
        f'"evidence_quotes":["..."],"relation_preserved":true|false,'
        f'"debate_helpful":true|false,"confident":true|false}}'
    )


def resolve_user(question, premises, desc, args, soft, grounded=False):
    prem = "\n".join(f"- {p}" for p in premises) if premises else "None."
    if grounded:
        return (
            f"Problem: {question}\n"
            f"Established facts: {prem}\n\n"
            f"Disagreement: {desc}\n\n"
            f"After debate, the solvers' positions are:\n{args}\n\n"
            f"Decide the single most likely correct claim for this step. You MUST ground it in "
            f"evidence: quote the specific passage span or established fact that supports it inside "
            f"the claim. Prefer a concrete, affirmative claim that lets the reasoning continue "
            f"(combine facts across passages when needed). Set \"confident\": false ONLY if no passage "
            f"or fact supports any position, or the positions are genuinely contradictory.\n"
            f'Respond with ONLY JSON: {{"resolved_claim":"...","evidence":"...","confident":true|false}}'
        )
    if soft:
        return (
            f"Problem: {question}\n"
            f"Established facts: {prem}\n\n"
            f"Disagreement: {desc}\n\n"
            f"After debate, the solvers' positions are:\n{args}\n\n"
            f"Using the established facts and any passages in the problem, decide the single most likely "
            f"correct claim for this step. Prefer a concrete, affirmative claim that lets the reasoning "
            f"continue (combine facts across passages when needed). Set \"confident\": false ONLY if the "
            f"positions are genuinely contradictory or the needed fact is truly absent.\n"
            f'Respond with ONLY JSON: {{"resolved_claim":"...","confident":true|false}}'
        )
    return (
        f"Problem: {question}\n"
        f"Established facts: {prem}\n\n"
        f"Disagreement: {desc}\n\n"
        f"After debate, the solvers' positions are:\n{args}\n\n"
        f"Decide the single correct claim for this step, stated as an established fact. "
        f'Respond with ONLY JSON: {{"resolved_claim":"..."}}'
    )


RERANK_SYS = ("You select the single best-supported answer from candidates, using only the given "
              "passages. Output only the exact chosen answer text, copied verbatim."
              + JSON_ONLY_SUFFIX)


def rerank_user(question, candidates):
    cand = "\n".join(f"- {c}" for c in candidates)
    return (
        f"{question}\n\n"
        f"Independent solvers proposed these different candidate answers:\n{cand}\n\n"
        f"Using ONLY the information in the problem/passages above, decide which candidate is best "
        f"supported by the evidence and most directly answers the question (the correct level of "
        f"detail, not over- or under-specified). Reply with ONLY the exact text of the single best "
        f"candidate, copied verbatim — no extra words, no explanation."
    )


CANON_SYS = "You extract the shortest exact answer phrase. Output only the phrase, nothing else."


def canonicalize_user(question, answer):
    return (
        f"Question: {question}\n\n"
        f"A solver produced this answer (which may be a full sentence or over/under-specified):\n"
        f"{answer}\n\n"
        f"Output ONLY the shortest exact phrase that directly answers the question — a name, "
        f"entity, date, place, or yes/no — with no surrounding sentence, no explanation, and no "
        f"trailing punctuation. Keep the level of detail the question asks for (e.g. include the "
        f"day if the answer is a full date; keep the specific district/city, not the country)."
    )


def resolve_segment_user(question, agreed_prefix, desc, args):
    prem = "\n".join(f"- {p}" for p in agreed_prefix) if agreed_prefix else "None."
    return (
        f"Problem: {question}\n"
        f"Agreed facts so far: {prem}\n\n"
        f"The solvers debated the reasoning segment leading to this disputed point: {desc}\n\n"
        f"Their final positions:\n{args}\n\n"
        f"Produce the AGREED reasoning for this segment as 1-3 atomic claim nodes that follow from the "
        f"agreed facts / passages and resolve the dispute. Prefer concrete, chain-completing claims; set "
        f'"confident": false ONLY if the positions are genuinely contradictory or the needed fact is truly '
        f'absent.\nRespond ONLY JSON: {{"resolved_nodes":["...","..."],"confident":true|false}}'
    )
