import argparse
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

from src.llm.client import LLMClient
from src.agents.agent import Agent
from src.data.loaders import load_examples
from src.eval.graders import extract_answer, grade
from src.methods.cot import run_cot
from src.methods.self_consistency import run_self_consistency, run_self_consistency_budget
from src.methods.self_refine import run_self_refine
from src.methods.standard_mad import run_standard_mad, run_standard_mad_self
from src.methods.divergent_mad import run_divergent_mad
from src.methods.consensus import run_consensus, run_consensus_self
from src.ldt import run as run_ldt
from src.ldt_v2 import run as run_ldtv2
from src.ldt_v3 import run as run_ldtv3
from src.ldt_v4 import run as run_ldtv4
from src.ldt_v5 import run as run_ldtv5
from src.ldt_debate_tree import run as run_ldtd
from src.cdsd import (run_cdsd, run_cdsd_soft, run_cdsd_faithful, run_cdsd_anchored, run_cdsd_grounded,
                      run_cdsd_canonical, run_cdsd_aggregate, run_cdsd_rerank,
                      run_cdsd_judgefix, run_cdsd_judgegate, run_cdsd_judgeverbatim,
                      run_cdsd_faithful_indexed, run_cdsd_x, run_cdsd_bridge,
                      run_cdsd_structured_nodes, run_cdsd_structured_select,
                      run_cdsd_structured_preserve, run_cdsd_structured_bridge,
                      run_cdsd_structured_claim_clean,
                      run_cdsd_structured_nonanswer_recover,
                      run_cdsd_structured_granularity_guard,
                      run_cdsd_structured_granularity_guard_strict,
                      run_cdsd_structured_granularity_guard_answer_type,
                      run_cdsd_structured_granularity_guard_answer_type_soft_commit,
                      run_cdsd_structured_granularity_guard_answer_type_concrete_commit,
                      run_cdsd_structured_granularity_guard_answer_type_always_commit,
                      run_cdsd_structured_granularity_guard_answer_type_debate_select,
                      run_cdsd_structured_audited_consensus,
                      run_cdsd_structured_verified_select)

METHODS = {"cot": run_cot, "sc": run_self_consistency,
           "scb": run_self_consistency_budget, "sc_budget": run_self_consistency_budget,
           "srefine": run_self_refine, "selfrefine": run_self_refine,
           # Du et al. keep each agent's own conversation memory across rounds, so the faithful
           # MAD baseline is the self-memory variant; the memoryless version stays as mad_noself.
           "mad": run_standard_mad_self, "mad_noself": run_standard_mad,
           "dmad": run_divergent_mad, "consensus": run_consensus,
           "mad_self": run_standard_mad_self, "consensus_self": run_consensus_self,
           "ldt": run_ldt, "ldtv2": run_ldtv2, "ldtv3": run_ldtv3, "ldtv4": run_ldtv4,
           "ldtv5": run_ldtv5, "ldtd": run_ldtd,
           "cdsd": run_cdsd, "cdsds": run_cdsd_soft, "cdsdf": run_cdsd_faithful,
           "cdsda": run_cdsd_anchored, "cdsde": run_cdsd_grounded,
           "cdsdc": run_cdsd_canonical, "cdsdg": run_cdsd_aggregate, "cdsdr": run_cdsd_rerank,
           "cdsdj": run_cdsd_judgefix, "cdsdk": run_cdsd_judgegate, "cdsdv": run_cdsd_judgeverbatim,
           "cdsdi": run_cdsd_faithful_indexed, "cdsdfi": run_cdsd_faithful_indexed,
           "cdsdx": run_cdsd_x, "cdsdfb": run_cdsd_bridge,
           "cdsdn": run_cdsd_structured_nodes, "cdsdns": run_cdsd_structured_select,
           "cdsdnp": run_cdsd_structured_preserve, "cdsdnb": run_cdsd_structured_bridge,
           "cdsdnc": run_cdsd_structured_claim_clean,
           "cdsdnr": run_cdsd_structured_nonanswer_recover,
           "cdsdng": run_cdsd_structured_granularity_guard,
           "cdsdngs": run_cdsd_structured_granularity_guard_strict,
           "cdsdnga": run_cdsd_structured_granularity_guard_answer_type,
           "cdsdngac": run_cdsd_structured_granularity_guard_answer_type_soft_commit,
           "cdsdngax": run_cdsd_structured_granularity_guard_answer_type_concrete_commit,
           "cdsdngaa": run_cdsd_structured_granularity_guard_answer_type_always_commit,
           "cdsdngad": run_cdsd_structured_granularity_guard_answer_type_debate_select,
           "cdsdna": run_cdsd_structured_audited_consensus,
           "cdsdnv": run_cdsd_structured_verified_select}


def dtype_of(dataset):
    d = dataset.lower()
    if d in ("math500", "math_500", "math"):
        return "mathx"
    if d == "gsm8k" or d.startswith("gsm"):
        return "math"
    if d in ("mmlu", "gpqa", "gpqa_diamond", "gpqa-diamond", "qasc", "fever", "feverous", "hover"):
        return "mc"
    if d.startswith("musr"):
        return "mc"
    return "qa"


def build_agents(cfg, client):
    personas = cfg["personas"]
    temps = cfg.get("temperatures", [0.7] * len(personas))
    n = cfg.get("n_agents", len(personas))
    return [Agent(i, client, personas[i % len(personas)],
                  temperature=temps[i % len(temps)], max_tokens=cfg.get("max_tokens", 1024))
            for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(METHODS))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--sample_seed", type=int, default=None,
                    help="shuffle loaded examples with this seed before taking n examples")
    ap.add_argument("--sample_pool", type=int, default=None,
                    help="when --sample_seed is set, load this many examples before shuffling")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--endpoint", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_iters", type=int, default=None)
    ap.add_argument("--debate_rounds", type=int, default=None,
                    help="CDSD: how many argue rounds per conflict (default 1)")
    ap.add_argument("--ldt_max_depth", type=int, default=None,
                    help="LDT: maximum tree depth; defaults to config ldt.max_depth")
    ap.add_argument("--ldt_per_parent_beam", type=int, default=None,
                    help="LDT: max kept children per parent")
    ap.add_argument("--ldt_global_beam", type=int, default=None,
                    help="LDT: max frontier nodes kept per layer")
    ap.add_argument("--ldt_debate_budget", type=int, default=None,
                    help="LDT: max conflict groups debated per question")
    ap.add_argument("--ldt_uncertain_threshold", type=float, default=None,
                    help="LDT: min score for keeping uncertain nodes")
    ap.add_argument("--ldt_final_confidence", type=float, default=None,
                    help="LDT: path-score threshold for early confident final stop")
    ap.add_argument("--ldt_min_depth_before_final", type=int, default=None,
                    help="LDT: minimum depth before early final-answer stopping")
    ap.add_argument("--ldt_no_llm_merge", action="store_true",
                    help="LDT: disable LLM merge refinement and use deterministic merge only")
    ap.add_argument("--ldt_final_debate_rounds", type=int, default=None,
                    help="LDT-v2: rounds of final cross-path debate")
    ap.add_argument("--ldt_final_top_paths", type=int, default=None,
                    help="LDT-v2/v3: number of terminal paths audited at final selection")
    ap.add_argument("--ldt_no_commit_select", action="store_true",
                    help="LDT-v3: use debate-aware candidate selection after a blocked commit")
    ap.add_argument("--ldt_verified_final", action="store_true",
                    help="LDT-v3: verify split final candidates with quoted evidence")
    ap.add_argument("--ldt_audit_consensus", action="store_true",
                    help="LDT-v3: audit suspicious consensus/non-answer final selections")
    ap.add_argument("--ldt_recover_nonanswer", action="store_true",
                    help="LDT-v3: attempt exact-quote recovery for non-answer finals")
    ap.add_argument("--ldt_max_branch_nodes", type=int, default=None,
                    help="LDT-v3: max structured nodes copied into each agent branch")
    ap.add_argument("--ldt_minimal_final_audit", action="store_true",
                    help="LDT-v3: audit long/compound QA finals against tree-grounded short candidates")
    ap.add_argument("--ldt_use_contract", action="store_true",
                    help="LDTD: enable root-level debated reasoning contract")
    ap.add_argument("--ldt_use_synthesis", action="store_true",
                    help="LDTD: enable compatible node synthesis")
    ap.add_argument("--ldt_use_slot_grounding", action="store_true",
                    help="LDTD: enable terminal answer-slot grounding debate")
    ap.add_argument("--ldt_no_terminality_debate", action="store_true",
                    help="LDTD: disable in-tree terminality debate for ablation")
    ap.add_argument("--ldt_no_contested_terminals", action="store_true",
                    help="LDTD: do not keep contested terminal branches after terminality debate")
    ap.add_argument("--ldt_no_open_frontier_patience", action="store_true",
                    help="LDTD: allow early stopping even when a strong open rival frontier remains")
    ap.add_argument("--ldt_open_frontier_margin", type=float, default=None,
                    help="LDTD: keep expanding open rivals within this path-score margin of the best final")
    ap.add_argument("--ldt_strict_role_edges", action="store_true",
                    help="LDTD: force strict role-edge terminality/final debate")
    ap.add_argument("--ldt_no_strict_role_edges", action="store_true",
                    help="LDTD: disable auto strict role-edge terminality/final debate")
    ap.add_argument("--mad_rounds", type=int, default=None)
    ap.add_argument("--sc_samples", type=int, default=None)
    ap.add_argument("--twowiki_path", default=None)
    ap.add_argument("--context_mode", default="none", choices=["none", "gold", "full"])
    ap.add_argument("--final_agents", action="store_true",
                    help="CDSD v2/v3: take final answer from agent majority instead of the judge")
    ap.add_argument("--no_cache", action="store_true")
    ap.add_argument("--skip_existing", action="store_true",
                    help="skip if the output jsonl already has >= n rows (resume a suite)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    client = LLMClient(model=args.model or cfg["model"],
                       endpoint=args.endpoint or cfg["endpoint"],
                       api_key=cfg.get("api_key", "EMPTY"), use_cache=not args.no_cache,
                       extra_body=cfg.get("extra_body"))
    acfg = cfg
    if args.method in ("ldt", "ldtv2", "ldtv3", "ldtv4", "ldtv5", "ldtd",
                       "cdsds", "cdsdf", "cdsda", "cdsde", "cdsdc", "cdsdg", "cdsdr", "cdsdi",
                       "cdsdfi", "cdsdx", "cdsdfb", "cdsdn", "cdsdns",
                       "cdsdnp", "cdsdnb", "cdsdnc", "cdsdnr", "cdsdng", "cdsdngs", "cdsdnga", "cdsdngac", "cdsdngad", "cdsdna",
                       "cdsdnv",
                       "cdsdj", "cdsdk", "cdsdv") and cfg.get("personas_soft"):
        acfg = {**cfg, "personas": cfg["personas_soft"],
                "temperatures": cfg.get("temperatures_soft", cfg.get("temperatures"))}
    agents = build_agents(acfg, client)
    dtype = dtype_of(args.dataset)
    load_n = args.n
    if args.sample_seed is not None:
        load_n = args.sample_pool or max(args.n * 5, args.n)
    examples = load_examples(args.dataset, n=load_n, offset=args.offset,
                             twowiki_path=args.twowiki_path, context_mode=args.context_mode)
    if args.sample_seed is not None:
        rng = random.Random(args.sample_seed)
        rng.shuffle(examples)
        examples = examples[:args.n]

    ccfg, scfg, mcfg = cfg.get("cdsd", {}), cfg.get("sc", {}), cfg.get("mad", {})
    lcfg = cfg.get(args.method if args.method in ("ldtv2", "ldtv3", "ldtv4", "ldtv5", "ldtd") else "ldt", cfg.get("ldt", {}))
    ldt_defaults = {
        "max_depth": 5,
        "per_parent_beam": 2,
        "global_beam": 6,
        "debate_budget": 20,
        "uncertain_threshold": 0.52,
        "final_confidence": 0.96 if args.method == "ldtv2" else 0.82,
        "min_depth_before_final": 3 if args.method == "ldtv2" else 1,
        "no_llm_merge": False,
        "final_debate_rounds": 1,
        "final_top_paths": 8,
        "no_commit_select": True,
        "verified_final": False,
        "audit_consensus": False,
        "recover_nonanswer": True,
        "max_branch_nodes": 8,
        "use_terminality_debate": True,
        "keep_contested_terminals": True,
        "open_frontier_patience": True,
        "open_frontier_margin": 0.12,
        "strict_role_edges": None,
    }
    mkw = dict(
        dataset=args.dataset,
        max_iters=args.max_iters if args.max_iters is not None else ccfg.get("max_iters", 6),
        debate_rounds=args.debate_rounds if args.debate_rounds is not None else ccfg.get("debate_rounds", 1),
        mad_rounds=args.mad_rounds if args.mad_rounds is not None else mcfg.get("rounds", 2),
        sc_samples=args.sc_samples if args.sc_samples is not None else scfg.get("samples", 3),
        sc_temperature=scfg.get("temperature", 0.8),
        cot_temperature=cfg.get("cot", {}).get("temperature", 0.0),
        final_agents=args.final_agents,
        ldt_max_depth=args.ldt_max_depth if args.ldt_max_depth is not None else lcfg.get("max_depth", ldt_defaults["max_depth"]),
        ldt_per_parent_beam=(
            args.ldt_per_parent_beam if args.ldt_per_parent_beam is not None
            else lcfg.get("per_parent_beam", ldt_defaults["per_parent_beam"])
        ),
        ldt_global_beam=args.ldt_global_beam if args.ldt_global_beam is not None else lcfg.get("global_beam", ldt_defaults["global_beam"]),
        ldt_debate_budget=(
            args.ldt_debate_budget if args.ldt_debate_budget is not None
            else lcfg.get("debate_budget", ldt_defaults["debate_budget"])
        ),
        ldt_uncertain_threshold=(
            args.ldt_uncertain_threshold if args.ldt_uncertain_threshold is not None
            else lcfg.get("uncertain_threshold", ldt_defaults["uncertain_threshold"])
        ),
        ldt_final_confidence=(
            args.ldt_final_confidence if args.ldt_final_confidence is not None
            else lcfg.get("final_confidence", ldt_defaults["final_confidence"])
        ),
        ldt_min_depth_before_final=(
            args.ldt_min_depth_before_final if args.ldt_min_depth_before_final is not None
            else lcfg.get("min_depth_before_final", ldt_defaults["min_depth_before_final"])
        ),
        ldt_no_llm_merge=args.ldt_no_llm_merge or bool(lcfg.get("no_llm_merge", ldt_defaults["no_llm_merge"])),
        ldt_final_debate_rounds=(
            args.ldt_final_debate_rounds if args.ldt_final_debate_rounds is not None
            else lcfg.get("final_debate_rounds", ldt_defaults["final_debate_rounds"])
        ),
        ldt_final_top_paths=(
            args.ldt_final_top_paths if args.ldt_final_top_paths is not None
            else lcfg.get("final_top_paths", ldt_defaults["final_top_paths"])
        ),
        ldt_no_commit_select=args.ldt_no_commit_select or bool(lcfg.get("no_commit_select", ldt_defaults["no_commit_select"])),
        ldt_verified_final=args.ldt_verified_final or bool(lcfg.get("verified_final", ldt_defaults["verified_final"])),
        ldt_audit_consensus=args.ldt_audit_consensus or bool(lcfg.get("audit_consensus", ldt_defaults["audit_consensus"])),
        ldt_recover_nonanswer=args.ldt_recover_nonanswer or bool(lcfg.get("recover_nonanswer", ldt_defaults["recover_nonanswer"])),
        ldt_max_branch_nodes=(
            args.ldt_max_branch_nodes if args.ldt_max_branch_nodes is not None
            else lcfg.get("max_branch_nodes", ldt_defaults["max_branch_nodes"])
        ),
        ldt_minimal_final_audit=args.ldt_minimal_final_audit or bool(lcfg.get("minimal_final_audit", ldt_defaults.get("minimal_final_audit", False))),
        ldt_use_contract=args.ldt_use_contract or bool(lcfg.get("use_contract", False)),
        ldt_use_synthesis=args.ldt_use_synthesis or bool(lcfg.get("use_synthesis", False)),
        ldt_use_slot_grounding=args.ldt_use_slot_grounding or bool(lcfg.get("use_slot_grounding", False)),
        ldt_use_terminality_debate=(
            (not args.ldt_no_terminality_debate)
            and bool(lcfg.get("use_terminality_debate", ldt_defaults["use_terminality_debate"]))
        ),
        ldt_keep_contested_terminals=(
            (not args.ldt_no_contested_terminals)
            and bool(lcfg.get("keep_contested_terminals", ldt_defaults["keep_contested_terminals"]))
        ),
        ldt_open_frontier_patience=(
            (not args.ldt_no_open_frontier_patience)
            and bool(lcfg.get("open_frontier_patience", ldt_defaults["open_frontier_patience"]))
        ),
        ldt_open_frontier_margin=(
            args.ldt_open_frontier_margin if args.ldt_open_frontier_margin is not None
            else lcfg.get("open_frontier_margin", ldt_defaults["open_frontier_margin"])
        ),
        ldt_strict_role_edges=(
            True if args.ldt_strict_role_edges
            else False if args.ldt_no_strict_role_edges
            else lcfg.get("strict_role_edges", ldt_defaults["strict_role_edges"])
        ),
    )
    fn = METHODS[args.method]
    os.makedirs("results", exist_ok=True)
    if args.out:
        out = args.out
    elif args.method in ("ldt", "ldtv2", "ldtv3", "ldtv4", "ldtv5", "ldtd"):
        m = re.search(r"(?:^|_)(v\d+)(?:_|$)", args.tag or "")
        version_dir = m.group(1) if m else "v0"
        tag_suffix = f"_{args.tag}" if args.tag else ""
        out = f"results/ldt/{version_dir}/{args.dataset}_{args.method}{tag_suffix}.jsonl"
    else:
        out = f"results/{args.dataset}_{args.method}{('_' + args.tag) if args.tag else ''}.jsonl"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    if args.skip_existing and os.path.exists(out):
        try:
            nrows = sum(1 for _ in open(out))
        except Exception:
            nrows = 0
        if nrows >= len(examples):
            print(f"[skip] {out} already has {nrows} rows")
            return

    def work(idx_ex):
        idx, ex = idx_ex
        t0 = time.time()
        try:
            res = fn(agents, ex, dtype, **mkw)
            pa = extract_answer(res["pred"], dtype)
            golds = [ex.answer] + (ex.meta.get("aliases") or [])
            g = max((grade(pa, gg, dtype) for gg in golds), key=lambda r: (r["correct"], r["f1"]))
            return {"order": idx, "id": ex.id, "dataset": args.dataset, "method": args.method,
                    "question": ex.meta.get("question_clean", ex.question), "gold": ex.answer,
                    "pred_raw": res["pred"], "pred": pa,
                    "correct": g["correct"], "em": g["em"], "f1": g["f1"], "calls": res["calls"],
                    "prompt_tokens": res["prompt_tokens"], "completion_tokens": res["completion_tokens"],
                    "time": round(time.time() - t0, 2), "supporting_facts": ex.supporting_facts,
                    "n_debates": res.get("n_debates"), "debated_claims": res.get("debated_claims"),
                    "n_no_commit": res.get("n_no_commit"), "no_commit_trace": res.get("no_commit_trace"),
                    "trace": res["trace"]}
        except Exception as e:
            return {"order": idx, "id": ex.id, "dataset": args.dataset, "method": args.method,
                    "question": ex.question, "gold": ex.answer, "pred_raw": "", "pred": "",
                    "correct": 0.0, "em": 0.0, "f1": 0.0, "calls": 0, "prompt_tokens": 0,
                    "completion_tokens": 0, "time": round(time.time() - t0, 2),
                    "supporting_facts": ex.supporting_facts,
                    "error": f"{type(e).__name__}: {e}", "trace": {}}

    t0 = time.time()
    records, done, total = [], 0, len(examples)
    with ThreadPoolExecutor(max_workers=args.workers) as exe:
        futs = [exe.submit(work, (i, ex)) for i, ex in enumerate(examples)]
        for fu in as_completed(futs):
            records.append(fu.result())
            done += 1
            if done % 10 == 0 or done == total:
                acc = sum(r["correct"] for r in records) / len(records)
                print(f"  [{done}/{total}] running EM={acc:.3f}", flush=True)
    records.sort(key=lambda r: r["order"])

    with open(out, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(records)
    acc = sum(r["correct"] for r in records) / n
    f1 = sum(r["f1"] for r in records) / n
    calls = sum(r["calls"] for r in records) / n
    ptok = sum(r["prompt_tokens"] for r in records) / n
    ctok = sum(r["completion_tokens"] for r in records) / n
    nerr = sum(1 for r in records if r.get("error"))
    print(f"\n=== {args.method} on {args.dataset} (n={n}) ===")
    print(f"EM/acc: {acc:.3f}   F1: {f1:.3f}   errors: {nerr}")
    print(f"avg calls/q: {calls:.2f}   avg prompt_tok: {ptok:.0f}   avg compl_tok: {ctok:.0f}")
    print(f"total wall: {time.time() - t0:.1f}s   -> {out}")


if __name__ == "__main__":
    main()
