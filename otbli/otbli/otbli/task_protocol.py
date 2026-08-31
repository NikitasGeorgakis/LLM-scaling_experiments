"""The two-stage protocol of Section 3.4 with the target metric swapped from
held-out loss to DOWNSTREAM TASK ACCURACY, and G_bar swapped from the
barycentric layer to a verbatim duplicate of an early block.

Why this configuration. Duplicating a single early layer is the one published
training-free depth edit with a measurable gain on converged LLMs, and it is
measured on tasks, not on loss. In the loss-targeted runs of Section 6 the
duplicate baseline is the WORST arm (dL = +0.03..+0.08 nats), so the two
metrics are expected to disagree. The gate turns that disagreement into a
measurable curve: gamma sweeps continuously between the intact model and full
duplication, and every point on the grid records the task gain AND the loss
cost, so a favourable trade-off region (if one exists) is located rather than
assumed.

Everything else is the pre-registered machinery, unchanged: the same grid
(3.29), the same feasibility screen (3.37), the same objective shape (3.41)
with the loss term replaced by the normalized accuracy term, the same
smallest-minimizer tie-break (3.43), the same refinement pass (3.44), the same
materiality rule (3.47) with delta now in accuracy points, and the same
confirmation rule (3.48) on a disjoint half of the questions — with the
inequality directions flipped, because accuracy is better when larger.
"""
import numpy as np
import torch

from .atomize import build_duplicate_block
from .arch import get_layers
from .insertion import GatedInsertion
from .metrics import batch_kl, batch_losses, bootstrap_ci, efficiency_cost
from .protocol import _free, _model_dtype, measure_drift1
from .tasks import align, eval_tasks_per_doc, mcnemar_exact


def make_duplicate_insertion(model, i: int, device, source: int = None):
    """G_bar = verbatim copy of block `source` (default: block i), gated in
    immediately before block i+1. At gamma = 1 this is exactly 'run block i
    twice'; at gamma = 0 the hook is not registered and M^+_0 == M."""
    layers = get_layers(model)
    src = i if source is None else source
    gbar = build_duplicate_block(layers[src]).to(device=device,
                                                 dtype=_model_dtype(model))
    ins = GatedInsertion(model, i + 1, gbar)
    ins.source_block = src
    return ins


# --------------------------------------------------------------------- Stage A
def stage_a_screen_tasks(model, lm, keys_ref, sel_idx, acc0_sel, device,
                         positions, task_cfg, verbose: bool = True):
    """Screen the candidate early positions at gamma = 1 (full duplication) on
    the SELECTION questions only; rank by accuracy gain."""
    out = []
    for i in positions:
        ins = make_duplicate_insertion(model, i, device)
        ins.set_gamma(1.0)
        per_task = eval_tasks_per_doc(lm, task_cfg.tasks, task_cfg.limit,
                                      task_cfg.num_fewshot, task_cfg.seed,
                                      task_cfg.metrics)
        ins.set_gamma(0.0)
        ins.remove()
        del ins
        _free(device)
        acc = align(keys_ref, per_task)[sel_idx]
        d = acc - acc0_sel
        rec = {"i": i, "pair": (i + 1, i + 2), "dup_block": i + 1,
               "acc_gamma1": float(acc.mean()),
               "dacc_gamma1": float(d.mean()),
               "ci": bootstrap_ci(d, task_cfg.n_boot_sel, seed=task_cfg.seed)}
        if verbose:
            print(f"[Stage A/task] duplicate F{i+1} -> gamma=1: "
                  f"acc {acc0_sel.mean():.4f} -> {rec['acc_gamma1']:.4f}  "
                  f"dacc = {rec['dacc_gamma1']:+.4f} "
                  f"({rec['dacc_gamma1']*100:+.2f} pt)  "
                  f"CI95 = [{rec['ci'][0]*100:+.2f}, {rec['ci'][1]*100:+.2f}] pt",
                  flush=True)
        out.append(rec)
    return sorted(out, key=lambda r: -r["dacc_gamma1"])


# --------------------------------------------------------------------- Stage B
def stage_b_gate_tasks(model, lm, keys_ref, sel_idx, acc0_sel, loss_pool,
                       base_loss, device, i: int, gate_cfg, proto_cfg,
                       task_cfg, verbose: bool = True):
    """Full gate machinery at position i with accuracy as the target metric.

    Each grid point also records the held-out LOSS cost, so the output is the
    accuracy/loss trade-off curve the loss-only runs cannot produce.
    """
    ins = make_duplicate_insertion(model, i, device)
    A0 = float(acc0_sel.mean())
    L0 = float(base_loss.mean())

    drift1 = measure_drift1(model, ins, loss_pool, device, proto_cfg.drift_batches)
    eff = efficiency_cost(model, ins.gbar, ins, loss_pool, device, gate_cfg.omega,
                          proto_cfg.latency_batches, proto_cfg.latency_repeats)
    if verbose:
        print(f"[Stage B/task] duplicate F{i+1} before F{i+2}   "
              f"A(0) = {A0:.4f}   L(0) = {L0:.4f}   "
              f"D_rep(1) = {drift1:.5f}   C_eff = {eff['C_eff']:.4f}", flush=True)

    records, paired = {}, {}

    def evaluate(gamma: float) -> dict:
        gamma = float(gamma)
        if gamma in records:
            return records[gamma]
        if gamma == 0.0:
            rec = {"gamma": 0.0, "acc": A0, "dacc": 0.0, "dacc_norm": 0.0,
                   "loss": L0, "dL_raw": 0.0, "KL": 0.0, "rep": 0.0, "Ceff": 0.0}
        else:
            ins.set_gamma(gamma)
            per_task = eval_tasks_per_doc(lm, task_cfg.tasks, task_cfg.limit,
                                          task_cfg.num_fewshot, task_cfg.seed,
                                          task_cfg.metrics)
            losses = batch_losses(model, loss_pool, device)
            kl = batch_kl(model, ins, loss_pool[:proto_cfg.kl_batches], device)
            ins.set_gamma(0.0)

            acc = align(keys_ref, per_task)[sel_idx]
            paired[gamma] = acc - acc0_sel
            A, Lm = float(acc.mean()), float(losses.mean())
            rec = {"gamma": gamma, "acc": A, "dacc": A - A0,
                   # eq. (3.31) with the sign flipped: accuracy is better larger
                   "dacc_norm": -(A - A0) / (abs(A0) + gate_cfg.eps_L),
                   "loss": Lm, "dL_raw": Lm - L0, "KL": kl,
                   "rep": gamma * gamma * drift1, "Ceff": eff["C_eff"]}
        feas = (rec["acc"] >= A0                                   # eq. (3.37)
                and rec["rep"] <= gate_cfg.eps_rep
                and rec["Ceff"] <= gate_cfg.eps_eff)
        if not task_cfg.kl_free:
            feas = feas and rec["KL"] <= gate_cfg.eps_KL
        rec["feasible"] = bool(feas)
        rec["J"] = (gate_cfg.lam_L * rec["dacc_norm"]                # eq. (3.41)
                    + gate_cfg.lam_KL * rec["KL"] / gate_cfg.eps_KL
                    + gate_cfg.lam_rep * rec["rep"] / gate_cfg.eps_rep
                    + gate_cfg.lam_eff * rec["Ceff"] / gate_cfg.eps_eff
                    + gate_cfg.lam_gamma * gamma * gamma)
        records[gamma] = rec
        if verbose:
            print(f"  gamma={gamma:<10g} dacc={rec['dacc']*100:+.2f} pt  "
                  f"dL={rec['dL_raw']:+.4f} nats  KL={rec['KL']:.3e}  "
                  f"J={rec['J']:+.5e}  "
                  f"{'feasible' if rec['feasible'] else 'infeasible'}", flush=True)
        return rec

    for g in sorted(set(float(g) for g in gate_cfg.grid)):
        evaluate(g)

    def argmin_feasible() -> float:
        feas = [g for g, r in records.items() if r["feasible"]]
        if not feas:
            return 0.0                                              # safe fallback
        jmin = min(records[g]["J"] for g in feas)
        return min(g for g in feas if records[g]["J"] == jmin)       # eq. (3.43)

    g_hat = argmin_feasible()
    refined = False
    if g_hat > 0.0:                                                 # eq. (3.44)
        s = sorted(records)
        k = s.index(g_hat)
        g_lo = s[k - 1]
        g_hi = s[k + 1] if k + 1 < len(s) else s[k]
        for q in range(gate_cfg.refine_Q + 1):
            evaluate(round(g_lo + q * (g_hi - g_lo) / gate_cfg.refine_Q, 12))
        g_hat = argmin_feasible()
        refined = True
    elif verbose:
        print("  refinement pass skipped: coarse best is the boundary gamma = 0")

    # eq. (3.47) with delta in accuracy points
    material = g_hat > 0.0 and records[g_hat]["acc"] > A0 + task_cfg.delta_acc
    gamma_star = g_hat if material else 0.0

    # the trade-off summary: gammas that gain accuracy, ranked by loss cost
    tradeoff = sorted(
        ({"gamma": g, "dacc": r["dacc"], "dL": r["dL_raw"], "KL": r["KL"]}
         for g, r in records.items() if g > 0.0 and r["dacc"] > 0.0),
        key=lambda r: r["dL"])

    ins.remove()
    del ins
    _free(device)
    return {"i": i, "pair": (i + 1, i + 2), "dup_block": i + 1,
            "A0": A0, "L0": L0, "drift1": drift1, "eff": eff,
            "gamma_hat": g_hat, "gamma_star": gamma_star,
            "retained": bool(material), "refined": refined,
            "records": records, "tradeoff": tradeoff,
            "paired": {g: d.tolist() for g, d in paired.items()}}


# ---------------------------------------------------------------- confirmation
def run_task_confirmation(model, lm, keys_ref, conf_idx, acc0_conf, device,
                          i: int, gamma: float, task_cfg, n_boot: int = 10000):
    """The single pre-declared out-of-sample test on the held-back questions.

    Rule (3.48), accuracy form: accept iff mean(d) > +delta AND CI_lower_95 > 0,
    with d the per-question paired accuracy difference. The exact McNemar test
    on the discordant pairs is reported alongside as a secondary check.
    """
    ins = make_duplicate_insertion(model, i, device)
    ins.set_gamma(gamma)
    per_task = eval_tasks_per_doc(lm, task_cfg.tasks, task_cfg.limit,
                                  task_cfg.num_fewshot, task_cfg.seed,
                                  task_cfg.metrics)
    ins.set_gamma(0.0)
    ins.remove()
    del ins
    _free(device)

    acc = align(keys_ref, per_task)[conf_idx]
    d = acc - acc0_conf
    lo, hi = bootstrap_ci(d, n_boot, seed=99)
    b, c, p = mcnemar_exact(d)
    return {"i": i, "pair": (i + 1, i + 2), "dup_block": i + 1,
            "gamma": float(gamma), "n_questions": int(len(d)),
            "acc0": float(acc0_conf.mean()), "acc_gamma": float(acc.mean()),
            "d_mean": float(d.mean()), "ci": (lo, hi),
            "mcnemar_lost": b, "mcnemar_gained": c, "mcnemar_p": p,
            "delta_acc": task_cfg.delta_acc,
            "accepted": bool(d.mean() > task_cfg.delta_acc and lo > 0.0)}
