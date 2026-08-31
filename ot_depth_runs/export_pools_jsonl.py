#!/usr/bin/env python3
"""Bridge from the otbli pipeline to this package.

The otbli runs cached their evaluation pools as TOKENIZED tensors
(results/pools_<model>_seed<seed>.pt with keys "selection"/"confirmation"),
whereas this package's pack_pool() reads TEXT from .jsonl with a "text" field
and re-tokenizes per model. That difference is deliberate on both sides -- the
runbook needs text so the same pool can be fed to models with different
tokenizers -- but it means POOL_A/POOL_B do not exist as files yet.

This script produces them, and answers the SKIP_DOCS question exactly instead
of by guesswork:

  1. decode the cached selection/confirmation blocks back to text and write
     pool_A.jsonl / pool_B.jsonl (round-trips through the same tokenizer, so
     the evaluation content is the content the original runs actually saw);
  2. optionally replay the document stream with the same dataset and seed and
     count how many documents were consumed to fill those pools, printing the
     SKIP_DOCS value that makes the new pools C/D/E/G provably disjoint.

Usage:
    python3 export_pools_jsonl.py --pt ~/otbli/results/pools_gpt2-large_seed1234.pt \\
        --model openai-community/gpt2-large --out pools/

    # add --count-docs to also compute SKIP_DOCS (re-streams the dataset once)
    python3 export_pools_jsonl.py --pt ... --model ... --out pools/ \\
        --count-docs --dataset monology/pile-uncopyrighted --seed 1234
"""
import argparse
import hashlib
import json
import os


def write_jsonl(path, blocks, tok, batch=64):
    """Decode [n_batches, batch_seqs, seq_len] token blocks to text lines."""
    flat = blocks.reshape(-1, blocks.shape[-1])
    n, h = 0, hashlib.sha256()
    with open(path, "w") as f:
        for s in range(0, flat.shape[0], batch):
            for text in tok.batch_decode(flat[s:s + batch],
                                         skip_special_tokens=True):
                if not text.strip():
                    continue
                line = json.dumps({"text": text}) + "\n"
                f.write(line)
                h.update(line.encode())
                n += 1
    return n, h.hexdigest()


def count_docs_consumed(dataset, seed, tok, n_tokens):
    """Replay the otbli document stream and count how many documents were
    needed to reach n_tokens -- i.e. how many the original pools consumed."""
    import sys
    sys.path.insert(0, os.path.expanduser("~/otbli"))
    from otbli.data import _doc_stream
    eos, total, docs = tok.eos_token_id, 0, 0
    for text in _doc_stream(dataset, seed):
        total += len(tok(text, add_special_tokens=False).input_ids) + 1
        docs += 1
        if total >= n_tokens:
            break
    return docs, total


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pt", required=True, help="otbli pools_*.pt cache")
    ap.add_argument("--model", required=True, help="HF name whose tokenizer made it")
    ap.add_argument("--out", default="pools")
    ap.add_argument("--name-a", default="pool_A")
    ap.add_argument("--name-b", default="pool_B")
    ap.add_argument("--count-docs", action="store_true")
    ap.add_argument("--dataset", default="monology/pile-uncopyrighted")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--margin", type=float, default=2.0,
                    help="safety multiplier applied to the counted docs")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer

    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    d = torch.load(args.pt, map_location="cpu")
    sel, conf = d["selection"], d["confirmation"]
    print(f"loaded {args.pt}\n  selection {tuple(sel.shape)}  "
          f"confirmation {tuple(conf.shape)}")

    for name, blocks in ((args.name_a, sel), (args.name_b, conf)):
        path = os.path.join(args.out, f"{name}.jsonl")
        n, sha = write_jsonl(path, blocks, tok)
        print(f"pool {name}: {n} docs -> {path}  sha256={sha[:16]}")

    n_tokens = int(sel.numel() + conf.numel())
    print(f"\ntokens in A+B: {n_tokens:,}")
    if args.count_docs:
        docs, total = count_docs_consumed(args.dataset, args.seed, tok, n_tokens)
        skip = int(args.margin * docs)
        print(f"documents consumed by A+B: {docs:,} ({total:,} tokens)")
        print(f"\n  export SKIP_DOCS={skip}      # {args.margin:g}x margin over "
              f"{docs:,} — new pools are then provably disjoint from A and B")
    else:
        print("re-run with --count-docs (and --dataset/--seed matching the "
              "original build) to get the exact SKIP_DOCS instead of a guess")


if __name__ == "__main__":
    main()
