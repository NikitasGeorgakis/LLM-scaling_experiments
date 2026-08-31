#!/usr/bin/env python3
"""Run the COMPLETE experiment suite unattended: full-scale evaluation on every
model in the list (one subprocess per model, so a failure or OOM on one cannot
kill the rest and GPU memory is fully released between models), then an
optional checkpoint-trajectory run, then a combined summary CSV.

Resume-safe: models whose results/full_scale_<model>.json already exists are
skipped, so a killed job can simply be resubmitted.

Typical uses:
  # everything, locally, surviving logout (machine must stay powered on):
  nohup python scripts/run_all.py --device cuda > logs/run_all.log 2>&1 &

  # on a SLURM cluster (survives closing your own computer):
  sbatch slurm/kuma_run_all.sbatch
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_MODELS = [
    "EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b", "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
]


def sh(cmd, log_path):
    print(f"        $ {' '.join(cmd)}\n        log: {log_path}", flush=True)
    with open(log_path, "w") as lf:
        p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=REPO)
    return p.returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--extra-models", nargs="*", default=[],
                    help="appended to --models, e.g. EleutherAI/pythia-12b "
                         "(48 GB fp32 weights; fits a 94 GB H100)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--force", action="store_true",
                    help="re-run models even if their results JSON exists")
    ap.add_argument("--trajectory-model", default=None,
                    help="optionally also run the Section-6.5 trajectory for "
                         "this model, e.g. EleutherAI/pythia-6.9b")
    ap.add_argument("--trajectory-steps", nargs="+", type=int,
                    default=[512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 143000])
    args = ap.parse_args()

    os.chdir(REPO)
    os.makedirs("logs", exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    models = args.models + args.extra_models
    extra = ["--dataset", args.dataset] if args.dataset else []

    status = []
    for name in models:
        short = name.split("/")[-1]
        res = os.path.join(args.out, f"full_scale_{short}.json")
        if os.path.exists(res) and not args.force:
            print(f"[skip ] {name}  (found {res}; resume-safe)", flush=True)
            status.append((name, "skipped (already done)"))
            continue
        print(f"[run  ] {name}", flush=True)
        t = time.time()
        rc = sh([sys.executable, "scripts/run_full_scale.py",
                 "--models", name, "--device", args.device, "--out", args.out]
                + extra, f"logs/full_scale_{short}.log")
        ok = rc == 0 and os.path.exists(res)
        status.append((name, "ok" if ok else f"FAILED (rc={rc}, see log)"))
        print(f"[done ] {name}: {status[-1][1]}  ({time.time() - t:.0f}s)",
              flush=True)

    # ------------------------------------------------- combined summary CSV
    rows = []
    for name in models:
        res = os.path.join(args.out, f"full_scale_{name.split('/')[-1]}.json")
        if not os.path.exists(res):
            continue
        j = json.load(open(res))
        cand = j.get("loss_only_candidate") or {}
        pair = j["stage_b"][0]["pair"]
        rows.append({
            "model": name, "blocks": j["blocks"], "params": j["params"],
            "L0_sel": f"{j['L0_sel']:.4f}",
            "screened_pair": f"(F{pair[0]},F{pair[1]})",
            "gamma_star": j["gamma_star"],
            "lossonly_gamma": cand.get("gamma", ""),
            "lossonly_dL": cand.get("dL", ""),
            "lossonly_ci_lo": (cand.get("ci") or ["", ""])[0],
            "lossonly_ci_hi": (cand.get("ci") or ["", ""])[1],
            "wallclock_s": f"{j['wallclock_s']:.0f}",
        })
    if rows:
        path = os.path.join(args.out, "summary_all.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\ncombined summary -> {path}")
        for r in rows:
            print(f"  {r['model']:<28s} L(0)={r['L0_sel']}  "
                  f"{r['screened_pair']:<10s} gamma* = {r['gamma_star']}")

    # ------------------------------------------------- optional trajectory
    if args.trajectory_model:
        short = args.trajectory_model.split("/")[-1]
        print(f"[run  ] trajectory {args.trajectory_model}", flush=True)
        rc = sh([sys.executable, "scripts/run_trajectory.py",
                 "--model", args.trajectory_model,
                 "--steps", *map(str, args.trajectory_steps),
                 "--device", args.device, "--out", args.out] + extra,
                f"logs/trajectory_{short}.log")
        status.append((f"trajectory:{args.trajectory_model}",
                       "ok" if rc == 0 else f"FAILED (rc={rc}, see log)"))

    print("\n" + "=" * 78)
    for name, st in status:
        print(f"  {name:<40s} {st}")
    failed = [s for s in status if s[1].startswith("FAILED")]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
