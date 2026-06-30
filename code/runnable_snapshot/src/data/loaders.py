import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen


@dataclass
class Example:
    id: str
    question: str
    answer: str
    supporting_facts: list = field(default_factory=list)  # gold entity/title list (multi-hop)
    meta: dict = field(default_factory=dict)


def _clip_text(text, max_chars=12000):
    text = str(text or "")
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return text[:head].rstrip() + "\n\n[... truncated ...]\n\n" + text[-tail:].lstrip()


def _choice_prompt(stem, choices):
    return stem.strip() + "\n" + "\n".join(f"{letter}. {text}" for letter, text in choices)


def _iter_json_url(url):
    with urlopen(url) as f:
        return json.loads(f.read().decode("utf-8"))


def _wiki_summary(title, cache_dir="data/cache/wiki_summary"):
    """Fetch a short Wikipedia summary for evidence-only benchmark prompts."""
    title = str(title or "").strip()
    if not title:
        return ""
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    cache_key = "".join(c if c.isalnum() or c in "._-" else "_" for c in title)[:160]
    path = root / f"{cache_key}.json"
    if path.exists():
        try:
            return json.loads(path.read_text()).get("extract", "")
        except Exception:
            pass
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title.replace(' ', '_'), safe='')}"
    try:
        req = Request(url, headers={"User-Agent": "research-3-cdsd/0.1"})
        with urlopen(req, timeout=10) as f:
            data = json.loads(f.read().decode("utf-8"))
        path.write_text(json.dumps({"title": title, "extract": data.get("extract", "")}, ensure_ascii=False))
        return data.get("extract", "")
    except Exception:
        return ""


def load_gsm8k(n=None, offset=0, split="test"):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split)
    out = []
    for i in range(offset, len(ds)):
        ex = ds[i]
        gold = ex["answer"].split("####")[-1].strip().replace(",", "")
        out.append(Example(id=f"gsm8k-{i}", question=ex["question"], answer=gold,
                           meta={"full_answer": ex["answer"]}))
        if n and len(out) >= n:
            break
    return out


# 2WikiMultiHopQA is not consistently mirrored on HF; try a few ids, else a local dev.json.
TWOWIKI_HF_CANDIDATES = ["voidful/2WikiMultihopQA", "xanhho/2WikiMultihopQA"]
TWOWIKI_LOCAL_PATHS = ["~/data/2wikimultihop/dev.json", "data/2wiki/dev.json", "data/2wikimultihop/dev.json"]


def _format_context(context, gold_titles=None):
    """context = list of [title, [sentences]]. If gold_titles given, keep only those paragraphs."""
    blocks = []
    for item in context or []:
        if not item:
            continue
        title = item[0]
        if gold_titles is not None and title not in gold_titles:
            continue
        sents = item[1] if len(item) > 1 else []
        text = " ".join(sents) if isinstance(sents, list) else str(sents)
        blocks.append(f"== {title} ==\n{text}")
    return "\n\n".join(blocks)


def _twowiki_record(rec, i, context_mode="none"):
    qid = rec.get("_id") or rec.get("id") or f"2wiki-{i}"
    sf = rec.get("supporting_facts")
    titles = []
    if isinstance(sf, list):
        for item in sf:
            if isinstance(item, (list, tuple)) and item:
                titles.append(str(item[0]))
            elif isinstance(item, str):
                titles.append(item)
    elif isinstance(sf, dict) and "title" in sf:
        titles = [str(t) for t in sf["title"]]
    q = rec.get("question", "")
    ex = Example(id=str(qid), question=q, answer=str(rec.get("answer", "")),
                 supporting_facts=sorted(set(titles)), meta={"type": rec.get("type")})
    if context_mode and context_mode != "none":
        gold = set(titles) if context_mode == "gold" else None
        ctx = _format_context(rec.get("context"), gold)
        ex.meta["question_clean"] = q
        ex.question = ("Read the following passages and answer the question using only the "
                       f"information in them.\n\n{ctx}\n\nQuestion: {q}")
    return ex


def load_twowiki(n=None, offset=0, split="dev", local_path=None, context_mode="none"):
    candidates = ([local_path] if local_path else []) + [os.path.expanduser(p) for p in TWOWIKI_LOCAL_PATHS]
    for p in candidates:
        if p and os.path.exists(p):
            with open(p) as f:
                data = json.load(f)
            data = data[offset:]
            out = [_twowiki_record(r, offset + i, context_mode) for i, r in enumerate(data)]
            return out[:n] if n else out
    from datasets import load_dataset
    last = None
    for hid in TWOWIKI_HF_CANDIDATES:
        for sp in [split, "validation", "dev", "test", "train"]:
            try:
                ds = load_dataset(hid, split=sp)
            except Exception as e:
                last = e
                continue
            out = []
            for i in range(offset, len(ds)):
                out.append(_twowiki_record(ds[i], i, context_mode))
                if n and len(out) >= n:
                    break
            return out
    raise RuntimeError(
        "Could not load 2WikiMultiHopQA from HF candidates "
        f"{TWOWIKI_HF_CANDIDATES}. Provide --twowiki_path to an official dev.json. Last error: {last}")


def _hotpot_context_block(context, gold_titles=None):
    """HF hotpot context is {title: [...], sentences: [[...], ...]}."""
    blocks = []
    for t, sents in zip(context.get("title", []), context.get("sentences", [])):
        if gold_titles is not None and t not in gold_titles:
            continue
        text = " ".join(sents) if isinstance(sents, list) else str(sents)
        blocks.append(f"== {t} ==\n{text}")
    return "\n\n".join(blocks)


def load_hotpotqa(n=None, offset=0, split="validation", context_mode="none"):
    from datasets import load_dataset
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    out = []
    for i in range(offset, len(ds)):
        ex = ds[i]
        sf = ex.get("supporting_facts") or {}
        titles = list(dict.fromkeys(sf.get("title", [])))
        q = ex["question"]
        e = Example(id=str(ex.get("id", f"hotpot-{i}")), question=q, answer=str(ex["answer"]),
                    supporting_facts=titles, meta={"type": ex.get("type"), "level": ex.get("level")})
        if context_mode and context_mode != "none":
            gold = set(titles) if context_mode == "gold" else None
            ctx = _hotpot_context_block(ex.get("context") or {}, gold)
            e.meta["question_clean"] = q
            e.question = ("Read the following passages and answer the question using only the "
                          f"information in them.\n\n{ctx}\n\nQuestion: {q}")
        out.append(e)
        if n and len(out) >= n:
            break
    return out


def load_mmlu(n=None, offset=0, split="validation"):
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split=split)
    letters = "ABCD"
    out = []
    for i in range(offset, len(ds)):
        ex = ds[i]
        choices = ex["choices"]
        qtext = ex["question"].strip() + "\n" + "\n".join(
            f"{letters[j]}. {c}" for j, c in enumerate(choices))
        out.append(Example(id=f"mmlu-{i}", question=qtext, answer=letters[int(ex["answer"])],
                           supporting_facts=[], meta={"subject": ex.get("subject")}))
        if n and len(out) >= n:
            break
    return out


def load_musique(n=None, offset=0, split="validation", context_mode="none"):
    from datasets import load_dataset
    ds = load_dataset("dgslibisey/MuSiQue", split=split)
    out = []
    for i in range(offset, len(ds)):
        ex = ds[i]
        paras = ex.get("paragraphs") or []
        sup = [p for p in paras if str(p.get("is_supporting")).lower() == "true"]
        titles = sorted({p["title"] for p in sup})
        q = ex["question"]
        e = Example(id=str(ex.get("id", f"musique-{i}")), question=q, answer=str(ex["answer"]),
                    supporting_facts=titles, meta={"aliases": list(ex.get("answer_aliases") or [])})
        if context_mode and context_mode != "none":
            chosen = sup if context_mode == "gold" else paras
            ctx = "\n\n".join(f"== {p['title']} ==\n{p['paragraph_text']}" for p in chosen)
            e.meta["question_clean"] = q
            e.question = ("Read the following passages and answer the question using only the "
                          f"information in them.\n\n{ctx}\n\nQuestion: {q}")
        out.append(e)
        if n and len(out) >= n:
            break
    return out


def load_gpqa(n=None, offset=0, split="train", config="gpqa_diamond"):
    """GPQA-Diamond: graduate-level science MCQ. Build a 4-option question, deterministically
    shuffling the (correct + 3 incorrect) answers per item so the gold letter is stable."""
    import hashlib
    from datasets import load_dataset
    ds = load_dataset("Idavidrein/gpqa", config, split=split)
    letters = "ABCD"
    out = []
    for i in range(offset, len(ds)):
        ex = ds[i]
        correct = str(ex["Correct Answer"]).strip()
        opts = [correct, str(ex["Incorrect Answer 1"]).strip(),
                str(ex["Incorrect Answer 2"]).strip(), str(ex["Incorrect Answer 3"]).strip()]
        # deterministic per-question permutation
        seed = int(hashlib.md5(ex["Question"].encode()).hexdigest(), 16)
        order = list(range(4))
        for j in range(3, 0, -1):
            seed, k = divmod(seed, j + 1)
            order[j], order[k] = order[k], order[j]
        shuffled = [opts[o] for o in order]
        gold_letter = letters[shuffled.index(correct)]
        qtext = ex["Question"].strip() + "\n" + "\n".join(
            f"{letters[j]}. {c}" for j, c in enumerate(shuffled))
        out.append(Example(id=str(ex.get("Record ID", f"gpqa-{i}")), question=qtext,
                           answer=gold_letter, supporting_facts=[],
                           meta={"subdomain": ex.get("Subdomain"),
                                 "domain": ex.get("High-level domain")}))
        if n and len(out) >= n:
            break
    return out


def load_strategyqa(n=None, offset=0, split="test", context_mode="none"):
    from datasets import load_dataset
    ds = load_dataset("ChilleD/StrategyQA", split=split)
    out = []
    for i in range(offset, len(ds)):
        ex = ds[i]
        q = ex["question"]
        ans = "yes" if ex["answer"] else "no"
        e = Example(id=str(ex.get("qid", f"strategyqa-{i}")), question=q, answer=ans,
                    supporting_facts=[], meta={})
        if context_mode and context_mode != "none":
            e.meta["question_clean"] = q
            e.question = ("Use the following facts to answer the question.\n\n"
                          f"{ex.get('facts') or ''}\n\nQuestion: {q}\nAnswer yes or no.")
        else:
            e.question = f"{q}\nAnswer yes or no."
        out.append(e)
        if n and len(out) >= n:
            break
    return out


def load_math500(n=None, offset=0, split="test"):
    """MATH-500 (Hendrycks et al. competition math, the HF eval subset). Harder than GSM8K;
    answers are LaTeX (fractions/expressions), graded by the `mathx` equivalence path."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split=split)
    out = []
    for i in range(offset, len(ds)):
        ex = ds[i]
        out.append(Example(id=str(ex.get("unique_id", f"math500-{i}")), question=ex["problem"],
                           answer=str(ex["answer"]), supporting_facts=[],
                           meta={"level": ex.get("level"), "subject": ex.get("subject")}))
        if n and len(out) >= n:
            break
    return out


def load_gsm_hard(n=None, offset=0, split="train"):
    """GSM-Hard (Gao et al. PAL): GSM8K questions with large/perturbed numbers, integer answers.
    Works with the existing numeric `math` grader; just harder arithmetic than GSM8K."""
    from datasets import load_dataset
    ds = load_dataset("reasoning-machines/gsm-hard", split=split)
    out = []
    for i in range(offset, len(ds)):
        ex = ds[i]
        tgt = ex["target"]
        gold = str(int(tgt)) if float(tgt).is_integer() else str(tgt)
        out.append(Example(id=f"gsmhard-{i}", question=ex["input"], answer=gold,
                           supporting_facts=[], meta={}))
        if n and len(out) >= n:
            break
    return out


def load_musr(n=None, offset=0, domain="murder_mysteries"):
    """MuSR (Sprague et al. 2024): multi-step soft-reasoning over a synthetic narrative, MC format.
    The HF dataset TAUR-Lab/MuSR exposes its three domains as splits. Each row has a narrative,
    a question, a stringified `choices` list, and `answer_index` (0-based)."""
    import ast
    from datasets import load_dataset
    ds = load_dataset("TAUR-Lab/MuSR", split=domain)
    letters = "ABCD"
    out = []
    for i in range(offset, len(ds)):
        ex = ds[i]
        choices = ex["choices"]
        if isinstance(choices, str):
            choices = ast.literal_eval(choices)
        gold_idx = int(ex["answer_index"])
        qtext = (f"{ex['narrative']}\n\n{ex['question']}\n"
                 + "\n".join(f"{letters[j]}. {c}" for j, c in enumerate(choices)))
        out.append(Example(id=f"musr-{domain}-{i}", question=qtext, answer=letters[gold_idx],
                           supporting_facts=[], meta={"domain": domain}))
        if n and len(out) >= n:
            break
    return out


def load_squad2(n=None, offset=0, split="validation", context_mode="full"):
    """SQuAD 2.0 (Rajpurkar et al. 2018): extractive open-book QA with ~half unanswerable items.
    Context is always embedded (the task is inherently reading comprehension); unanswerable
    questions get gold 'unanswerable' and the model is told to emit that when the span is absent."""
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad_v2", split=split)
    out = []
    for i in range(offset, len(ds)):
        ex = ds[i]
        texts = (ex.get("answers") or {}).get("text") or []
        gold = texts[0] if texts else "unanswerable"
        aliases = list(dict.fromkeys(texts[1:])) if len(texts) > 1 else []
        q = ex["question"]
        e = Example(id=str(ex.get("id", f"squad2-{i}")), question=q, answer=gold,
                    supporting_facts=[ex.get("title")] if ex.get("title") else [],
                    meta={"aliases": aliases, "answerable": bool(texts)})
        e.meta["question_clean"] = q
        e.question = ("Read the passage and answer the question using only the information in it. "
                      "If the passage does not contain the answer, reply exactly 'unanswerable'.\n\n"
                      f"== {ex.get('title', 'Passage')} ==\n{ex['context']}\n\nQuestion: {q}")
        out.append(e)
        if n and len(out) >= n:
            break
    return out


def load_fever(n=None, offset=0, split="dev"):
    from datasets import load_dataset
    ds = load_dataset("pminervini/hl-fever", "default", split=split)
    mapping = {"SUPPORTS": "A", "supported": "A", "REFUTES": "B", "refuted": "B",
               "NOT ENOUGH INFO": "C", "not enough info": "C"}
    choices = [("A", "SUPPORTS"), ("B", "REFUTES"), ("C", "NOT ENOUGH INFO")]
    out = []
    for i in range(offset, len(ds)):
        ex = ds[i]
        label = str(ex["label"])
        q = _choice_prompt(f"Verify the claim.\n\nClaim: {ex['claim']}\n\nChoose the best label.", choices)
        out.append(Example(id=f"fever-{ex.get('id', i)}", question=q,
                           answer=mapping.get(label, label[:1].upper()),
                           meta={"label": label}))
        if n and len(out) >= n:
            break
    return out


def load_feverous(n=None, offset=0, split="validation"):
    from datasets import load_dataset
    ds = load_dataset("Dzeniks/feverous_3way", split=split)
    # Mirror convention observed in Dzeniks/feverous_3way: 0 supports, 1 refutes, 2 not enough info.
    mapping = {0: "A", 1: "B", 2: "C"}
    labels = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT ENOUGH INFO"}
    choices = [("A", "SUPPORTS"), ("B", "REFUTES"), ("C", "NOT ENOUGH INFO")]
    out = []
    for i in range(offset, len(ds)):
        ex = ds[i]
        evidence = _clip_text(ex.get("evidence") or "")
        stem = ("Verify the claim using the evidence.\n\n"
                f"Evidence:\n{evidence}\n\nClaim: {ex['claim']}\n\nChoose the best label.")
        label = int(ex["label"])
        out.append(Example(id=f"feverous-{ex.get('id', i)}",
                           question=_choice_prompt(stem, choices),
                           answer=mapping[label],
                           meta={"label": labels.get(label, str(label))}))
        if n and len(out) >= n:
            break
    return out


def load_hover(n=None, offset=0, split="validation", context_mode="full"):
    urls = {
        "train": "https://raw.githubusercontent.com/hover-nlp/hover/main/data/hover/hover_train_release_v1.1.json",
        "validation": "https://raw.githubusercontent.com/hover-nlp/hover/main/data/hover/hover_dev_release_v1.1.json",
        "dev": "https://raw.githubusercontent.com/hover-nlp/hover/main/data/hover/hover_dev_release_v1.1.json",
        "test": "https://raw.githubusercontent.com/hover-nlp/hover/main/data/hover/hover_test_release_v1.1.json",
    }
    data = _iter_json_url(urls.get(split, urls["validation"]))
    choices = [("A", "SUPPORTED"), ("B", "NOT_SUPPORTED")]
    out = []
    for i, ex in enumerate(data[offset:], start=offset):
        if "label" not in ex:
            continue
        label = str(ex["label"])
        facts = [sf[0] for sf in ex.get("supporting_facts") or [] if sf]
        evidence_blocks = []
        if context_mode and context_mode != "none":
            for title in facts:
                summary = _wiki_summary(title)
                if summary:
                    evidence_blocks.append(f"== {title} ==\n{summary}")
                else:
                    evidence_blocks.append(f"== {title} ==")
        evidence = "\n\n".join(evidence_blocks)
        stem = "Verify the multi-hop claim."
        if evidence:
            stem += f"\n\nEvidence:\n{evidence}"
        stem += f"\n\nClaim: {ex['claim']}\n\nChoose the best label."
        q = _choice_prompt(stem, choices)
        out.append(Example(id=str(ex.get("uid", f"hover-{i}")), question=q,
                           answer="A" if label == "SUPPORTED" else "B",
                           supporting_facts=facts, meta={"label": label, "num_hops": ex.get("num_hops")}))
        if n and len(out) >= n:
            break
    return out


def load_qasc(n=None, offset=0, split="validation"):
    from datasets import load_dataset
    ds = load_dataset("allenai/qasc", split=split)
    out = []
    for i in range(offset, len(ds)):
        ex = ds[i]
        labels = ex["choices"]["label"]
        texts = ex["choices"]["text"]
        fact1 = ex.get("fact1") or ""
        fact2 = ex.get("fact2") or ""
        evidence = ""
        if fact1 or fact2:
            evidence = f"Facts:\n1. {fact1}\n2. {fact2}\n\n"
        qtext = evidence + ex["question"].strip() + "\n" + "\n".join(
            f"{labels[j]}. {texts[j]}" for j in range(len(texts)))
        out.append(Example(id=str(ex.get("id", f"qasc-{i}")), question=qtext,
                           answer=str(ex["answerKey"]).strip().upper(),
                           supporting_facts=[fact1, fact2],
                           meta={"combinedfact": ex.get("combinedfact")}))
        if n and len(out) >= n:
            break
    return out


def load_multihop_rag(n=None, offset=0, split="train"):
    from datasets import load_dataset
    ds = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG", split=split)
    out = []
    for i in range(offset, len(ds)):
        ex = ds[i]
        blocks, titles = [], []
        for ev in ex.get("evidence_list") or []:
            title = ev.get("title") or ev.get("source") or "Evidence"
            titles.append(title)
            fact = ev.get("fact") or ""
            source = ev.get("source") or ""
            date = ev.get("published_at") or ""
            blocks.append(f"== {title} ==\nSource: {source}\nDate: {date}\n{fact}")
        ctx = _clip_text("\n\n".join(blocks))
        q = ex["query"]
        prompt = ("Read the evidence and answer the question using only the evidence.\n\n"
                  f"{ctx}\n\nQuestion: {q}")
        out.append(Example(id=f"multihoprag-{i}", question=prompt, answer=str(ex["answer"]),
                           supporting_facts=titles, meta={"question_clean": q, "type": ex.get("question_type")}))
        if n and len(out) >= n:
            break
    return out


def load_webqsp(n=None, offset=0, split="test"):
    from datasets import load_dataset
    ds = load_dataset("ml1996/webqsp", split=split)
    out = []
    for i in range(offset, len(ds)):
        ex = ds[i]
        triples = ex.get("graph") or []
        graph = "\n".join(" | ".join(map(str, t)) for t in triples[:120])
        q = ex["question"]
        prompt = ("Answer the question. The following knowledge graph triples may be useful.\n\n"
                  f"{_clip_text(graph, 10000)}\n\nQuestion: {q}")
        answers = ex.get("answer") or []
        gold = str(answers[0]) if answers else ""
        out.append(Example(id=str(ex.get("id", f"webqsp-{i}")), question=prompt, answer=gold,
                           meta={"question_clean": q, "aliases": [str(a) for a in answers[1:]]}))
        if n and len(out) >= n:
            break
    return out


def load_complex_webquestions(n=None, offset=0, split="dev"):
    urls = {
        "train": "https://www.dropbox.com/sh/7pkwkrfnwqhsnpo/AAAIHeWX0cPpbpwK6w06BCxva/ComplexWebQuestions_train.json?dl=1",
        "dev": "https://www.dropbox.com/sh/7pkwkrfnwqhsnpo/AADH8beLbOUWxwvY_K38E3ADa/ComplexWebQuestions_dev.json?dl=1",
        "validation": "https://www.dropbox.com/sh/7pkwkrfnwqhsnpo/AADH8beLbOUWxwvY_K38E3ADa/ComplexWebQuestions_dev.json?dl=1",
        "test": "https://www.dropbox.com/sh/7pkwkrfnwqhsnpo/AABr4ysSy_Tg8Wfxww4i_UWda/ComplexWebQuestions_test.json?dl=1",
    }
    data = _iter_json_url(urls.get(split, urls["dev"]))
    out = []
    for i, ex in enumerate(data[offset:], start=offset):
        answers = ex.get("answers") or []
        if not answers:
            continue
        gold = str(answers[0].get("answer") or "")
        aliases = []
        for ans in answers:
            aliases.append(str(ans.get("answer") or ""))
            aliases.extend(str(a) for a in ans.get("aliases") or [])
        aliases = [a for a in dict.fromkeys(aliases) if a and a != gold]
        q = ex.get("question") or ex.get("webqsp_question") or ""
        out.append(Example(id=str(ex.get("ID", f"cwq-{i}")), question=q, answer=gold,
                           meta={"aliases": aliases, "question_clean": q}))
        if n and len(out) >= n:
            break
    return out


def load_longbench(n=None, offset=0, task="qasper"):
    from datasets import load_dataset
    try:
        ds = load_dataset("THUDM/LongBench", task, split="test")
    except RuntimeError:
        from huggingface_hub import hf_hub_download
        import zipfile
        zip_path = hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset")
        root = os.path.join(os.path.dirname(zip_path), "data")
        if not os.path.exists(root):
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(os.path.dirname(zip_path))
        path = os.path.join(root, f"{task}.jsonl")
        data = [json.loads(line) for line in open(path, encoding="utf-8")]
        ds = data
    out = []
    for i in range(offset, len(ds)):
        ex = ds[i]
        q = ex.get("input") or ex.get("question") or ""
        answers = ex.get("answers") or ex.get("answer") or []
        if isinstance(answers, str):
            answers = [answers]
        ctx = _clip_text(ex.get("context") or "", 12000)
        prompt = ("Read the context and answer the question concisely.\n\n"
                  f"{ctx}\n\nQuestion: {q}")
        out.append(Example(id=str(ex.get("_id", f"longbench-{task}-{i}")), question=prompt,
                           answer=str(answers[0]) if answers else "",
                           meta={"question_clean": q, "aliases": [str(a) for a in answers[1:]],
                                 "task": task, "length": ex.get("length")}))
        if n and len(out) >= n:
            break
    return out


def load_examples(dataset, n=None, offset=0, twowiki_path=None, context_mode="none"):
    d = dataset.lower()
    if d == "gsm8k":
        return load_gsm8k(n=n, offset=offset)
    if d in ("math500", "math_500", "math"):
        return load_math500(n=n, offset=offset)
    if d in ("gsm_hard", "gsmhard", "gsm-hard", "gsm_plus", "gsmplus"):
        return load_gsm_hard(n=n, offset=offset)
    if d == "mmlu":
        return load_mmlu(n=n, offset=offset)
    if d in ("gpqa", "gpqa_diamond", "gpqa-diamond"):
        return load_gpqa(n=n, offset=offset)
    if d == "musique":
        return load_musique(n=n, offset=offset, context_mode=context_mode)
    if d in ("strategyqa", "strategy_qa"):
        return load_strategyqa(n=n, offset=offset, context_mode=context_mode)
    if d in ("musr", "musr_mm", "musr_murder"):
        return load_musr(n=n, offset=offset, domain="murder_mysteries")
    if d in ("musr_op", "musr_object"):
        return load_musr(n=n, offset=offset, domain="object_placements")
    if d in ("musr_ta", "musr_team"):
        return load_musr(n=n, offset=offset, domain="team_allocation")
    if d in ("squad2", "squad_v2", "squadv2"):
        return load_squad2(n=n, offset=offset, context_mode=context_mode)
    if d == "fever":
        return load_fever(n=n, offset=offset)
    if d == "feverous":
        return load_feverous(n=n, offset=offset)
    if d == "hover":
        return load_hover(n=n, offset=offset)
    if d == "qasc":
        return load_qasc(n=n, offset=offset)
    if d in ("multihop_rag", "multihoprag", "mhrag"):
        return load_multihop_rag(n=n, offset=offset)
    if d in ("webqsp", "web_questions", "webquestions"):
        return load_webqsp(n=n, offset=offset)
    if d in ("cwq", "complex_webquestions", "complex_web_questions"):
        return load_complex_webquestions(n=n, offset=offset)
    if d in ("longbench_qasper", "qasper"):
        return load_longbench(n=n, offset=offset, task="qasper")
    if d in ("longbench_hotpotqa", "lb_hotpotqa"):
        return load_longbench(n=n, offset=offset, task="hotpotqa")
    if d in ("longbench_2wikimqa", "lb_2wiki"):
        return load_longbench(n=n, offset=offset, task="2wikimqa")
    if d in ("longbench_musique", "lb_musique"):
        return load_longbench(n=n, offset=offset, task="musique")
    if d in ("2wiki", "twowiki", "2wikimultihop", "2wikimultihopqa"):
        return load_twowiki(n=n, offset=offset, local_path=twowiki_path, context_mode=context_mode)
    if d in ("hotpot", "hotpotqa"):
        return load_hotpotqa(n=n, offset=offset, context_mode=context_mode)
    raise ValueError(f"unknown dataset: {dataset}")
