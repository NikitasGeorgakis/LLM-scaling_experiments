#!/usr/bin/env python3
"""Carve disjoint document-level pools from a held-out text source.

Documents are consumed IN ORDER after --skip-docs; each pool takes its count
sequentially, so pools are disjoint by construction and reproducible from the
manifest. Use the SAME source you used for the original selection (A) and
confirmation (B) pools, with --skip-docs set beyond what A and B consumed.

Examples:
  # final third pool C plus two extra selection pools D, E for E5:
  python make_pools.py --source file:pile_holdout.jsonl --skip-docs 6000 \
      --pool C_final=3000 --pool D_sel=1500 --pool E_sel=1500 --out pools/
  # OpenWebText pool for E7:
  python make_pools.py --source hf:Skylion007/openwebtext:train \
      --pool F_openwebtext=3000 --out pools/
Source formats: file:<path>.jsonl (field "text"), file:<path>.txt (one doc per
line), hf:<dataset>[:<split>[:<field>]] (streaming, in order).
"""
import argparse
import hashlib
import json
import os


def iter_docs(source):
    if source.startswith("file:"):
        path = source[5:]
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                if path.endswith(".jsonl"):
                    yield json.loads(line)["text"]
                else:
                    yield line.rstrip("\n")
    elif source.startswith("hf:"):
        parts = source[3:].split(":")
        name, split = parts[0], (parts[1] if len(parts) > 1 else "train")
        field = parts[2] if len(parts) > 2 else "text"
        from datasets import load_dataset
        for ex in load_dataset(name, split=split, streaming=True):
            yield ex[field]
    else:
        raise ValueError("source must start with file: or hf:")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--skip-docs", type=int, default=0)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--pool", action="append", required=True,
                    help="name=doc_count, may repeat; carved sequentially")
    ap.add_argument("--out", default="pools")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    specs = [(p.split("=")[0], int(p.split("=")[1])) for p in args.pool]
    it = iter_docs(args.source)
    consumed = skipped = 0
    while skipped < args.skip_docs:
        next(it)
        skipped += 1
    manifest = {"source": args.source, "skip_docs": args.skip_docs,
                "min_chars": args.min_chars, "pools": []}
    for name, count in specs:
        path = os.path.join(args.out, f"{name}.jsonl")
        h = hashlib.sha256()
        taken = 0
        start = args.skip_docs + consumed
        with open(path, "w") as f:
            while taken < count:
                doc = next(it)
                consumed += 1
                if len(doc) < args.min_chars:
                    continue
                line = json.dumps({"text": doc}) + "\n"
                f.write(line)
                h.update(line.encode())
                taken += 1
        manifest["pools"].append({"name": name, "path": path, "docs": taken,
                                  "source_index_start": start,
                                  "source_index_end": args.skip_docs + consumed,
                                  "sha256": h.hexdigest()})
        print(f"pool {name}: {taken} docs -> {path}  sha256={h.hexdigest()[:16]}")
    mpath = os.path.join(args.out, "manifest.json")
    prev = []
    if os.path.exists(mpath):
        prev = json.load(open(mpath)).get("history", [])
    json.dump({"latest": manifest, "history": prev + [manifest]},
              open(mpath, "w"), indent=1)
    print("manifest ->", mpath)


if __name__ == "__main__":
    main()
