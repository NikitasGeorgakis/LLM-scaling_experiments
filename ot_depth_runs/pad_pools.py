#!/usr/bin/env python3
import argparse
import json


def stream_docs(source, skip, take, min_chars=200):
    import sys, os
    sys.path.insert(0, os.path.expanduser("~/otbli"))
    from otbli.data import _doc_stream
    n, out = 0, []
    for text in _doc_stream(source, seed=1234):
        if n < skip:
            n += 1
            continue
        if len(text) >= min_chars:
            out.append(text)
        n += 1
        if len(out) >= take:
            break
    return out


def append_jsonl(path, texts):
    with open(path, "a") as f:
        for t in texts:
            f.write(json.dumps({"text": t}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-a", default="pools/pool_A.jsonl")
    ap.add_argument("--pool-b", default="pools/pool_B.jsonl")
    ap.add_argument("--already-consumed", type=int, required=True)
    ap.add_argument("--extra-a", type=int, default=150)
    ap.add_argument("--extra-b", type=int, default=150)
    ap.add_argument("--dataset", default="monology/pile-uncopyrighted")
    args = ap.parse_args()

    start = args.already_consumed
    docs_a = stream_docs(args.dataset, start, args.extra_a)
    append_jsonl(args.pool_a, docs_a)
    print(f"pool_A: +{len(docs_a)} docs (stream positions [{start}, {start + args.extra_a}))")

    start2 = start + args.extra_a
    docs_b = stream_docs(args.dataset, start2, args.extra_b)
    append_jsonl(args.pool_b, docs_b)
    print(f"pool_B: +{len(docs_b)} docs (stream positions [{start2}, {start2 + args.extra_b}))")

    print(f"next free stream position: {start2 + args.extra_b}")


if __name__ == "__main__":
    main()
