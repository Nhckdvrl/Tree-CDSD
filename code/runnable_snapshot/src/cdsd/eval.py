"""One-command test/iteration entrypoint for the CDSD family.

Runs the chosen variants (plus cot/sc/mad as reference) over the chosen datasets and prints
the comparison + flip-analysis table. Re-runs are cheap thanks to the on-disk cache.

Examples:
  python -m src.cdsd.eval --variants cdsds,cdsdf --datasets gsm8k,2wiki --n 150
  python -m src.cdsd.eval --variants cdsdf --datasets 2wiki --context gold --n 150
"""
import argparse
import subprocess
import sys

ROOT = "/home/xiang/research-3"
PY = sys.executable


def run(method, dataset, n, context, workers, extra_tag):
    tag = extra_tag or ("ob_" + context if context != "none" else "")
    cmd = [PY, "-m", "src.run", "--method", method, "--dataset", dataset,
           "--n", str(n), "--workers", str(workers)]
    if context != "none":
        cmd += ["--context_mode", context]
    if tag:
        cmd += ["--tag", tag]
    print(f"  >>> {method} {dataset} ctx={context} n={n}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=False)
    return f"results/{dataset}_{method}{('_' + tag) if tag else ''}.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="cdsds,cdsdf", help="comma list from cdsd,cdsds,cdsdf")
    ap.add_argument("--datasets", default="gsm8k,2wiki")
    ap.add_argument("--context", default="none", choices=["none", "gold", "full"],
                    help="open-book context for QA datasets")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--baselines", default="cot,sc,mad")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    variants = [v for v in args.variants.split(",") if v]
    baselines = [b for b in args.baselines.split(",") if b]
    for ds in [d for d in args.datasets.split(",") if d]:
        ctx = args.context if ds != "gsm8k" else "none"  # gsm8k has no context
        files = {}
        for m in baselines + variants:
            files[m] = run(m, ds, args.n, ctx, args.workers, args.tag)
        ref = files.get("cot")
        others = [files[m] for m in (baselines[1:] + variants) if m in files]
        print(f"\n================= {ds} (ctx={ctx}) =================", flush=True)
        subprocess.run([PY, "-m", "src.analyze", "--ref", ref, "--files", *others], cwd=ROOT, check=False)


if __name__ == "__main__":
    main()
