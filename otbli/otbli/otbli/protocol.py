"""The pre-registered two-stage selection-confirmation protocol (Sections 3.4, 6).

Stage A  screens every valid insertion position (L-1 per model) at gamma = 1
         on the SELECTION pool; the top-k positions advance.
Stage B  runs the complete gate machinery at each surviving position: the grid
         of eq. (3.29), all four criteria, the feasibility set (3.37), the
         objective (3.41), the smallest-minimizer tie-break (3.43), one
         refinement pass (3.44) (skipped when the coarse best is the boundary
         gamma = 0), and the materiality retention rule (3.47).
Secondary: the pre-declared loss-only reading (Section 6.3) — efficiency term
         dropped, stability constraints retained — over the records already
         computed; the single strongest cross-size candidate goes to ONE
         confirmation under rule (3.48).
Confirmation: paired evaluation on the untouched CONFIRMATION pool; accept iff
         mean d < -delta  AND  upper 95% bootstrap CI < 0.
"""
import gc
import numpy as np
import torch

from .arch import get_layers
from .atomize import (build_barycentric_block, build_interpolated_block,
                      build_duplicate_block)
from .insertion import GatedInsertion
from .metrics import (batch_losses, batch_kl, efficiency_cost, bootstrap_ci,
                      paired_t)


def _model_dtype(model):
    return next(model.parameters()).dtype


def _free(device):
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()


@torch.no_grad()
def make_insertion(model, i: int, ot_cfg, device, verbose: bool = False):
    """Build G_bar for the 0-indexed pair (layers[i], layers[i+1]) — the
    paper's 1-indexed pair (F_{i+1}, F_{i+2}) — and wire it before layers[i+1]."""
    layers = get_layers(model)
    gbar = build_barycentric_block(
        layers[i], layers[i + 1],
        ot_cfg.tau, ot_cfg.eta, ot_cfg.n_alt_rounds, ot_cfg.n_sinkhorn_iters,
        device, ot_dtype=getattr(torch, ot_cfg.ot_dtype),
        gauge_fix=getattr(ot_cfg, "gauge_fix", True), verbose=verbose,
    ).to(device=device, dtype=_model_dtype(model))
    return GatedInsertion(model, i + 1, gbar)


@torch.no_grad()
def measure_drift1(model, ins, pool, device, n_batches: int) -> float:
    """D_rep(1) on a fixed pool of insertion-point representations; extended to
    all gamma by the exact identity D_rep(gamma) = gamma^2 D_rep(1) (Sec. 3.6)."""
    ins.reset_drift()
    ins.measure_drift = True
    ins.drift_gated = False
    ins.set_gamma(0.0)                       # hook active for measurement only
    for b in pool[:n_batches]:
        model(input_ids=b.to(device), use_cache=False)
    ins.measure_drift = False
    ins.set_gamma(0.0)                       # hook removed
    return ins.drift_value()


# --------------------------------------------------------------------- Stage A
@torch.no_grad()
def stage_a_screen(model, sel_pool, base_sel, device, ot_cfg, proto_cfg,
                   positions=None, verbose: bool = True):
    L = len(get_layers(model))
    positions = list(range(L - 1)) if positions is None else list(positions)
    rows = []
    for i in positions:
        ins = make_insertion(model, i, ot_cfg, device)
        ins.set_gamma(1.0)
        losses = batch_losses(model, sel_pool, device)
        ins.remove()
        d = losses - base_sel
        row = {"i": i, "pair": (i + 1, i + 2), "dL_gamma1": float(d.mean()),
               "ci": bootstrap_ci(d, proto_cfg.n_boot_sel, seed=i)}
        rows.append(row)
        if verbose:
            print(f"[Stage A] pair (F{i+1:>2d},F{i+2:>2d})  "
                  f"paired dL(gamma=1) = {row['dL_gamma1']:+.6f} nats  "
                  f"CI95 = [{row['ci'][0]:+.5f}, {row['ci'][1]:+.5f}]", flush=True)
        del ins
        _free(device)
    rows.sort(key=lambda r: r["dL_gamma1"])
    return rows


# --------------------------------------------------------------------- Stage B
@torch.no_grad()
def stage_b_gate(model, sel_pool, base_sel, device, i: int, ot_cfg, gate_cfg,
                 proto_cfg, verbose: bool = True):
    ins = make_insertion(model, i, ot_cfg, device)
    L0 = float(base_sel.mean())

    drift1 = measure_drift1(model, ins, sel_pool, device, proto_cfg.drift_batches)
    eff = efficiency_cost(model, ins.gbar, ins, sel_pool, device, gate_cfg.omega,
                          proto_cfg.latency_batches, proto_cfg.latency_repeats)
    if verbose:
        print(f"[Stage B] pair (F{i+1},F{i+2})  D_rep(1) = {drift1:.5f}   "
              f"C_eff = {eff['C_eff']:.4f} (FLOPs {eff['flops']:+.1%}, "
              f"lat {eff['latency']:+.1%}, mem {eff['memory']:+.1%}, "
              f"par {eff['params']:+.1%})", flush=True)

    records, paired = {}, {}

    def evaluate(gamma: float) -> dict:
        gamma = float(gamma)
        if gamma in records:
            return records[gamma]
        if gamma == 0.0:
            rec = {"gamma": 0.0, "loss": L0, "dL_raw": 0.0, "dL_norm": 0.0,
                   "KL": 0.0, "rep": 0.0, "Ceff": 0.0}
        else:
            ins.set_gamma(gamma)
            losses = batch_losses(model, sel_pool, device)
            kl = batch_kl(model, ins, sel_pool[:proto_cfg.kl_batches], device)
            ins.set_gamma(0.0)
            paired[gamma] = losses - base_sel
            m = float(losses.mean())
            rec = {"gamma": gamma, "loss": m, "dL_raw": m - L0,
                   "dL_norm": (m - L0) / (abs(L0) + gate_cfg.eps_L),   # eq. (3.31)
                   "KL": kl,
                   "rep": gamma * gamma * drift1,                       # exact identity
                   "Ceff": eff["C_eff"]}                                # block runs in full
        rec["feasible"] = (rec["loss"] <= L0                            # eq. (3.37)
                           and rec["KL"] <= gate_cfg.eps_KL
                           and rec["rep"] <= gate_cfg.eps_rep
                           and rec["Ceff"] <= gate_cfg.eps_eff)
        rec["J"] = (gate_cfg.lam_L * rec["dL_norm"]                     # eq. (3.41)
                    + gate_cfg.lam_KL * rec["KL"] / gate_cfg.eps_KL
                    + gate_cfg.lam_rep * rec["rep"] / gate_cfg.eps_rep
                    + gate_cfg.lam_eff * rec["Ceff"] / gate_cfg.eps_eff
                    + gate_cfg.lam_gamma * gamma * gamma)
        records[gamma] = rec
        if verbose:
            print(f"  gamma={gamma:<10g} dL={rec['dL_raw']:+.3e}  "
                  f"KL={rec['KL']:.3e}  rep={rec['rep']:.3e}  "
                  f"J={rec['J']:+.5e}  "
                  f"{'feasible' if rec['feasible'] else 'infeasible'}", flush=True)
        return rec

    for g in sorted(set(float(g) for g in gate_cfg.grid)):
        evaluate(g)

    def argmin_feasible() -> float:
        feas = [g for g, r in records.items() if r["feasible"]]
        if not feas:
            return 0.0                                                  # safe fallback
        jmin = min(records[g]["J"] for g in feas)
        return min(g for g in feas if records[g]["J"] == jmin)          # eq. (3.43)

    g_hat = argmin_feasible()
    refined = False
    if g_hat > 0.0:                                                     # eq. (3.44)
        s = sorted(records)
        k = s.index(g_hat)
        g_lo = s[k - 1]
        g_hi = s[k + 1] if k + 1 < len(s) else s[k]
        Q = gate_cfg.refine_Q
        for q in range(Q + 1):
            evaluate(round(g_lo + q * (g_hi - g_lo) / Q, 12))
        g_hat = argmin_feasible()
        refined = True
    elif verbose:
        print("  refinement pass skipped: coarse best is the boundary gamma = 0")

    material = g_hat > 0.0 and records[g_hat]["loss"] < L0 - gate_cfg.delta  # eq. (3.47)
    gamma_star = g_hat if material else 0.0
    ins.remove()
    del ins
    _free(device)
    return {"i": i, "pair": (i + 1, i + 2), "L0": L0, "drift1": drift1,
            "eff": eff, "gamma_hat": g_hat, "gamma_star": gamma_star,
            "retained": bool(material), "refined": refined,
            "records": records,
            "paired": {g: d.tolist() for g, d in paired.items()}}


# ------------------------------------------------------------------- baselines
@torch.no_grad()
def baseline_deltas(model, sel_pool, base_sel, device, i: int, tau: float,
                    n_boot: int = 2000) -> dict:
    """Naive averaging and verbatim duplication at gamma = 1 (Table 2 / Sec 6.4 (iv))."""
    dt = _model_dtype(model)
    out = {}
    layers = get_layers(model)
    builders = {
        "naive_average": lambda: build_interpolated_block(
            layers[i], layers[i + 1], tau),
        "duplicate": lambda: build_duplicate_block(layers[i]),
    }
    for name, make in builders.items():
        blk = make().to(device=device, dtype=dt)
        ins = GatedInsertion(model, i + 1, blk)
        ins.set_gamma(1.0)
        losses = batch_losses(model, sel_pool, device)
        ins.remove()
        d = losses - base_sel
        out[name] = {"dL_gamma1": float(d.mean()),
                     "ci": bootstrap_ci(d, n_boot, seed=7)}
        del ins, blk
        _free(device)
    return out


# --------------------------------------------------- secondary loss-only rule
def loss_only_candidate(stage_b_results, gate_cfg, n_boot: int = 2000):
    """Pre-declared secondary rule (Section 6.3): among the gate records already
    computed, the stability-feasible gamma > 0 (KL and rep tolerances only —
    efficiency dropped) with the most negative selection-set paired dL.
    'strength' = |mean| / CI half-width, used for the cross-size pick."""
    best = None
    for res in stage_b_results:
        for g, rec in res["records"].items():
            if g <= 0.0:
                continue
            if rec["KL"] > gate_cfg.eps_KL or rec["rep"] > gate_cfg.eps_rep:
                continue
            d = res["paired"].get(g, res["paired"].get(str(g)))
            if d is None:
                continue
            d = np.asarray(d)
            m = float(d.mean())
            if m >= 0.0:
                continue
            lo, hi = bootstrap_ci(d, n_boot, seed=17)
            half = max((hi - lo) / 2.0, 1e-30)
            cand = {"i": res["i"], "pair": res["pair"], "gamma": g, "dL": m,
                    "ci": (lo, hi), "strength": abs(m) / half,
                    "significant_in_sample": hi < 0.0}
            if best is None or m < best["dL"]:
                best = cand
    return best


# ---------------------------------------------------------------- confirmation
@torch.no_grad()
def run_confirmation(model, conf_pool, device, i: int, gamma: float, ot_cfg,
                     gate_cfg, n_boot: int = 10000) -> dict:
    """The single out-of-sample test of rule (3.48), on the untouched
    confirmation pool: accept iff mean d < -delta AND CI_upper_95 < 0."""
    ins = make_insertion(model, i, ot_cfg, device)
    base = batch_losses(model, conf_pool, device)      # hook absent at gamma=0
    ins.set_gamma(float(gamma))
    cand = batch_losses(model, conf_pool, device)
    ins.remove()
    d = cand - base
    lo, hi = bootstrap_ci(d, n_boot, seed=99)
    t, p1 = paired_t(d)
    accepted = bool(d.mean() < -gate_cfg.delta and hi < 0.0)
    del ins
    _free(device)
    return {"i": i, "pair": (i + 1, i + 2), "gamma": float(gamma),
            "d_mean": float(d.mean()), "ci": (lo, hi), "t": t,
            "p_one_sided": p1, "accepted": accepted, "d": d.tolist()}
