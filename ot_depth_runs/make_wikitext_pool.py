#!/usr/bin/env python3
"""Build a WikiText-103 pool for the E7-style OOD check."""
import argparse
import hashlib
import json
import os


def pseudo_documents(split, target_chars, min_chars):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split,
                      streaming=True)
    buf = []
    buf_len = 0
    for ex in ds:
        line = ex["text"]
        if not line.strip():
            continue
        buf.append(line)
        buf_len += len(line)
        if buf_len >= target_chars:
            doc = "".join(buf)
            if len(doc) >= min_chars:
                yield doc
            buf, buf_len = [], 0
    if buf:
        doc = "".join(buf)
        if len(doc) >= min_chars:
            yield doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--count", type=int, default=3000)
    ap.add_argument("--target-chars", type=int, default=5000)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--out", default="pools")
    ap.add_argument("--name", default="G_wikitext")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.name}.jsonl")
    n = 0
    with open(path, "w") as f:
        for doc in pseudo_documents(args.split, args.target_chars, args.min_chars):
            f.write(json.dumps({"text": doc}) + "\n")
            n += 1
            if n >= args.count:
                break

    if n < args.count:
        print(f"WARNING: only got {n}/{args.count} docs from split={args.split} "
              f"-- the split may be smaller than requested; consider a larger "
              f"split (e.g. --split train) or a smaller --target-chars.")

    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    print(f"pool {args.name}: {n} docs -> {path}  sha256={sha}  "
          f"(split={args.split}, target_chars={args.target_chars}, "
          f"min_chars={args.min_chars})")


if __name__ == "__main__":
    main()
