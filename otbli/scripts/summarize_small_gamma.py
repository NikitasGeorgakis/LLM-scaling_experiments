#!/usr/bin/env python3
"""Combine results/small_gamma/small_gamma_*.json into one ranked CSV."""
import argparse
import csv
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="results/small_gamma")
    ap.add_argument("--out", default="results/small_gamma/summary_small_gamma.csv")
    args = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.in_dir, "small_gamma_*.json"))):
        with open(path) as f:
            j = json.load(f)
        if not j.get("complete", False):
            continue
        best_i = j.get("best_position")
        best = next((r for r in j.get("results", []) if r["i"] == best_i), None)
        rows.append({
            "model": j["model"],
            "arch": j["arch"],
            "layers": j["layers"],
            "params": j["params"],
            "base_loss": j["base_loss"],
            "best_i": best_i,
            "best_pair": "" if best is None else f"F{best['pair'][0]}-F{best['pair'][1]}",
            "best_gamma": j.get("best_gamma", 0.0),
            "best_dL": j.get("best_dL", 0.0),
            "ci_lo": "" if best is None else best["best_ci95"][0],
            "ci_hi": "" if best is None else best["best_ci95"][1],
            "best_KL": "" if best is None else best["best_KL"],
            "best_rep": "" if best is None else best["best_rep"],
            "wallclock_s": j.get("wallclock_s", ""),
        })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if not rows:
        print("No completed small-gamma result JSONs found.")
        return
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")
    for r in rows:
        print(f"{r['model']:<35s} {r['best_pair']:<10s} gamma={r['best_gamma']:<8g} dL={r['best_dL']:+.3e}")


if __name__ == "__main__":
    main()
