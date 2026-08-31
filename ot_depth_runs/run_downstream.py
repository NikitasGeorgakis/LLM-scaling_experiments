#!/usr/bin/env python3
"""E6: downstream evaluation of locked candidates with lm-eval-harness.

Requires:  pip install "lm-eval>=0.4"

  python run_downstream.py --model-key gpt2-large \
      --candidates runs/locked_gpt2/*.pt \
      --tasks lambada_openai,hellaswag,arc_easy,arc_challenge,piqa,winogrande,wikitext \
      --out runs/e6_downstream

For accuracy tasks the script pairs per-sample correctness between base and
candidate and reports a two-sided exact McNemar (binomial) p-value; for
wikitext it reports the aggregate word-perplexity change.
"""
import argparse
import glob
import itertools
import json
import os
import time

import otdepth as od


def per_sample_acc(samples):
    out = {}
    for s in samples:
        key = s.get("doc_id", s.get("idx"))
        acc = s.get("acc", s.get("acc_norm"))
        if key is not None and acc is not None:
            out[key] = float(acc)
    return out


def evaluate(model, tok, tasks, bs, limit):
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=bs)
    res = simple_evaluate(model=lm, tasks=tasks, limit=limit,
                          log_samples=True)
    metrics = {t: {k: v for k, v in res["results"][t].items()
                   if isinstance(v, (int, float))} for t in res["results"]}
    samples = {t: per_sample_acc(res.get("samples", {}).get(t, []))
               for t in res.get("samples", {})}
    return metrics, samples


def mcnemar(b, c):
    """Two-sided exact binomial McNemar p for discordant counts b, c."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return min(1.0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--tasks", default="lambada_openai,hellaswag,arc_easy,"
                    "arc_challenge,piqa,winogrande,wikitext")
    ap.add_argument("--batch-size", default="8")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    tasks = a.tasks.split(",")
    paths = sorted(itertools.chain.from_iterable(glob.glob(p)
                                                 for p in a.candidates))
    model, tok = od.load_model(a.model_key, a.revision, a.dtype, a.device)

    print("evaluating BASE ...")
    base_metrics, base_samples = evaluate(model, tok, tasks, a.batch_size,
                                          a.limit)
    report = {"base": base_metrics, "candidates": []}

    for path in paths:
        meta, cand = od.load_locked(path, model)
        name, g, i = meta["construction"], meta["gamma"], meta["i"]
        print(f"evaluating {name} (i={i}, gamma={g}) ...")
        with od.inserted(model, i, cand, g):
            m, s = evaluate(model, tok, tasks, a.batch_size, a.limit)
        del cand
        paired = {}
        for t in s:
            common = set(s[t]) & set(base_samples.get(t, {}))
            if not common:
                continue
            b = sum(1 for k in common
                    if base_samples[t][k] > s[t][k])   # base right, cand wrong
            c = sum(1 for k in common
                    if base_samples[t][k] < s[t][k])   # cand right, base wrong
            paired[t] = dict(n=len(common), base_only_correct=b,
                             cand_only_correct=c, p_mcnemar=mcnemar(b, c))
        report["candidates"].append(dict(candidate=name, gamma=g, i=i,
                                         file=path, metrics=m,
                                         paired_vs_base=paired))
        for t, pr in paired.items():
            print(f"  {t}: +{pr['cand_only_correct']}/-"
                  f"{pr['base_only_correct']} of {pr['n']} "
                  f"(McNemar p={pr['p_mcnemar']:.3f})")

    report["wallclock_s"] = time.time() - t0
    json.dump(report, open(os.path.join(a.out, "downstream.json"), "w"),
              indent=1)
    print("->", os.path.join(a.out, "downstream.json"))


if __name__ == "__main__":
    main()
