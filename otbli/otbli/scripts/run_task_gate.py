#!/usr/bin/env python3
"""Gated duplication of an early block, judged on DOWNSTREAM TASK ACCURACY.

The one published training-free depth edit with a measurable gain on converged
LLMs is duplicating a single early layer, evaluated on tasks rather than loss.
This script runs that edit through the unchanged two-stage protocol:

  Stage A   screen the early positions (default blocks 1-3) at gamma = 1
            on the selection half of the questions
  Stage B   full gate grid at the top position(s): every grid point records
            the task gain AND the held-out loss cost
  confirm   the single pre-declared test on the disjoint half of the questions,
            rule (3.48) in accuracy form (--confirm, run it once)

The trade-off table is the point: in the loss-targeted runs duplication is the
worst arm (dL = +0.03..+0.08 nats), so if accuracy rises while loss degrades,
the gate locates where -- and whether any gamma buys the task gain at a
negligible loss cost.

Example (the Pythia sizes of the published result):
    python scripts/run_task_gate.py --models EleutherAI/pythia-410m \\
        EleutherAI/pythia-1b --device cuda --limit 200 --out results

Then, ONCE, for the surviving candidate:
    python scripts/run_task_gate.py --models EleutherAI/pythia-1b \\
        --confirm --position 0 --gamma 0.5 --device cuda --limit 200
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from otbli import load_model
from otbli.arch import get_layers
from otbli.config import (DataConfig, GateConfig, ProtocolConfig, TaskConfig,
                          print_registered)
from otbli.data import load_or_build_pools
from otbli.metrics import batch_losses
from otbli.tasks import build_lm, eval_tasks_per_doc, flatten, split_questions
from otbli.task_protocol import (run_task_confirmation, stage_a_screen_tasks,
                                 stage_b_gate_tasks)


def run_one_model(name, args, gate_cfg, data_cfg, proto_cfg, task_cfg):
    short = name.split("/")[-1]
    print(f"\n{'#' * 78}\n# {name}   [target metric: task accuracy]\n{'#' * 78}",
          flush=True)
    t0 = time.time()

    model, tok = load_model(name, device=args.device)
    L = len(get_layers(model))
    print(f"blocks L = {L}, parameters = "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")

    lm = build_lm(model, tok, batch_size=args.batch_size, device=args.device)

    # --- baseline pass at gamma = 0 defines the question set and the pairing
    base = eval_tasks_per_doc(lm, task_cfg.tasks, task_cfg.limit,
                              task_cfg.num_fewshot, task_cfg.seed)
    task_cfg.metrics = {t: d["metric"] for t, d in base.items()}
    keys, acc0 = flatten(base)
    sel_idx, conf_idx = split_questions(keys, seed=task_cfg.seed)
    print(f"questions: {len(keys)} total over {len(base)} tasks "
          f"({len(sel_idx)} selection / {len(conf_idx)} confirmation, disjoint)")
    for t in sorted(base):
        print(f"    {t:<52s} n={len(base[t]['doc_ids']):<5d} "
              f"metric={base[t]['metric']}  acc={base[t]['scores'].mean():.4f}")
    acc0_sel, acc0_conf = acc0[sel_idx], acc0[conf_idx]
    print(f"baseline accuracy: selection {acc0_sel.mean():.4f}   "
          f"confirmation {acc0_conf.mean():.4f} (held back)")

    # --- loss pool, so every gate point also reports its loss cost
    pools_path = os.path.join(args.out, f"pools_{short}_seed{data_cfg.seed}.pt")
    loss_pool, _ = load_or_build_pools(tok, data_cfg, pools_path)
    base_loss = batch_losses(model, loss_pool, args.device)
    print(f"baseline loss L(0) = {base_loss.mean():.4f} nats/token")

    # ------------------------------------------------------------- confirm only
    if args.confirm:
        if args.position is None or args.gamma is None:
            sys.exit("--confirm needs --position and --gamma "
                     "(decide them from the selection records first)")
        print("\nWARNING: the confirmation questions may be used ONCE per "
              "candidate. Report the outcome whatever it is.")
        res = run_task_confirmation(model, lm, keys, conf_idx, acc0_conf,
                                    args.device, args.position, args.gamma,
                                    task_cfg, proto_cfg.n_boot_conf)
        print(f"\nduplicate F{res['dup_block']}, gamma = {res['gamma']}, "
              f"n = {res['n_questions']} questions")
        print(f"  accuracy   {res['acc0']:.4f} -> {res['acc_gamma']:.4f}")
        print(f"  mean d     = {res['d_mean']*100:+.2f} accuracy points")
        print(f"  CI95       = [{res['ci'][0]*100:+.2f}, {res['ci'][1]*100:+.2f}] pt")
        print(f"  McNemar    = {res['mcnemar_gained']} gained / "
              f"{res['mcnemar_lost']} lost, p = {res['mcnemar_p']:.4f}")
        print(f"  rule (3.48, accuracy form, delta = "
              f"{res['delta_acc']*100:.1f} pt): "
              f"{'ACCEPTED' if res['accepted'] else 'NOT CONFIRMED'}")
        path = os.path.join(args.out,
                            f"task_confirmation_{short}_i{args.position}_g{args.gamma}.json")
        with open(path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"  written to {path}")
        return res

    # ------------------------------------------------------------ Stage A + B
    positions = args.positions if args.positions else list(task_cfg.positions)
    positions = [i for i in positions if 0 <= i < L - 1]
    screen = stage_a_screen_tasks(model, lm, keys, sel_idx, acc0_sel,
                                  args.device, positions, task_cfg)
    top = screen[:proto_cfg.top_k_positions]
    stage_b = [stage_b_gate_tasks(model, lm, keys, sel_idx, acc0_sel, loss_pool,
                                  base_loss, args.device, r["i"], gate_cfg,
                                  proto_cfg, task_cfg) for r in top]

    best = min(stage_b, key=lambda r: r["records"][r["gamma_hat"]]["J"])
    print(f"\n>>> VERDICT [{short}]  gamma* = {best['gamma_star']}"
          + ("  ->  no gate retained (M^+_0 = M)" if best["gamma_star"] == 0.0
             else f"  ->  duplicate F{best['dup_block']} at gamma = "
                  f"{best['gamma_star']}, dacc = "
                  f"{best['records'][best['gamma_star']]['dacc']*100:+.2f} pt"))
    if best["tradeoff"]:
        print("    accuracy/loss trade-off (gammas with a task gain, cheapest first):")
        for r in best["tradeoff"][:6]:
            print(f"      gamma={r['gamma']:<8g} dacc={r['dacc']*100:+.2f} pt   "
                  f"dL={r['dL']:+.4f} nats   KL={r['KL']:.3e}")
        cheap = best["tradeoff"][0]
        print(f"    -> cheapest gaining gamma = {cheap['gamma']}: "
              f"{cheap['dacc']*100:+.2f} pt for {cheap['dL']:+.4f} nats")
    else:
        print("    no gamma on the grid improved task accuracy at this position")
    print("    (nothing above is confirmed: re-run with --confirm --position "
          "<i> --gamma <g> exactly once)")

    out = {"model": name, "blocks": L, "target_metric": "task_accuracy",
           "tasks": list(task_cfg.tasks), "limit": task_cfg.limit,
           "num_fewshot": task_cfg.num_fewshot,
           "per_task_baseline": {t: {"n": len(d["doc_ids"]),
                                     "metric": d["metric"],
                                     "acc": float(d["scores"].mean())}
                                 for t, d in base.items()},
           "n_sel": int(len(sel_idx)), "n_conf": int(len(conf_idx)),
           "A0_sel": float(acc0_sel.mean()), "L0": float(base_loss.mean()),
           "stage_a": screen, "stage_b": stage_b,
           "gamma_star": best["gamma_star"], "wallclock_s": time.time() - t0}
    path = os.path.join(args.out, f"task_gate_{short}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"    results written to {path}  ({out['wallclock_s']:.0f}s)")

    del model, lm
    if str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+",
                    default=["EleutherAI/pythia-410m", "EleutherAI/pythia-1b"])
    ap.add_argument("--tasks", nargs="+", default=None,
                    help="lm-eval task names (default: the TaskConfig set)")
    ap.add_argument("--list-tasks", metavar="PATTERN", default=None,
                    help="print matching task names of the installed harness and exit")
    ap.add_argument("--with-controls", action="store_true",
                    help="append TaskConfig.control_tasks (positive control: "
                         "tasks where the model is demonstrably above chance, so "
                         "a null on BigBench can be told apart from no signal)")
    ap.add_argument("--limit", type=int, default=200,
                    help="questions per task (None = all); keep it fixed across runs")
    ap.add_argument("--num-fewshot", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--positions", nargs="*", type=int, default=None,
                    help="0-indexed positions to screen (default 0 1 2 = blocks 1-3)")
    ap.add_argument("--delta-acc", type=float, default=0.01,
                    help="materiality margin in accuracy FRACTION (0.01 = 1 point)")
    ap.add_argument("--kl-free", action="store_true",
                    help="ablation: drop the output-KL feasibility constraint "
                         "(a task-improving edit is expected to move the output "
                         "distribution, so the stability screen may be the "
                         "binding constraint rather than the metric)")
    ap.add_argument("--confirm", action="store_true",
                    help="run ONLY the single out-of-sample confirmation")
    ap.add_argument("--position", type=int, default=None)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--kl-batches", type=int, default=8)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    if args.list_tasks is not None:
        from otbli.tasks import list_tasks
        for n in list_tasks(args.list_tasks):
            print(n)
        return

    os.makedirs(args.out, exist_ok=True)
    gate_cfg, data_cfg, proto_cfg = GateConfig(), DataConfig(), ProtocolConfig()
    task_cfg = TaskConfig()
    if args.tasks:
        task_cfg.tasks = tuple(args.tasks)
    if args.with_controls:
        task_cfg.tasks = tuple(dict.fromkeys(task_cfg.tasks + task_cfg.control_tasks))
    task_cfg.limit = args.limit
    task_cfg.num_fewshot = args.num_fewshot
    task_cfg.delta_acc = args.delta_acc
    task_cfg.kl_free = args.kl_free
    task_cfg.n_boot_sel = proto_cfg.n_boot_sel
    if args.dataset:
        data_cfg.dataset = args.dataset
    proto_cfg.kl_batches = args.kl_batches
    print_registered(gate_cfg, data_cfg, proto_cfg, task_cfg)

    torch.manual_seed(data_cfg.seed)
    for name in args.models:
        run_one_model(name, args, gate_cfg, data_cfg, proto_cfg, task_cfg)


if __name__ == "__main__":
    main()
