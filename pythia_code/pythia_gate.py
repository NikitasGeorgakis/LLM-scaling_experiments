"""
Full gate-selection machinery of the paper, Section 3.4:

  - initial grid Gamma^(0), denser near zero               (eq. 3.28-3.29)
  - normalized loss change Delta_L                          (eq. 3.30)
  - output-distribution deviation D_KL^out                  (eq. 3.31)
  - representation drift D_rep, via D_rep(g) = g^2 D_rep(1) (eq. 3.32 + Sec 3.6)
  - efficiency functional C_eff with weights omega          (eq. 3.33-3.35)
  - feasible set Gamma_feas                                 (eq. 3.36)
  - normalized criteria and objective J_gamma               (eq. 3.37-3.40)
  - smallest minimizer on ties                              (eq. 3.42)
  - one grid-refinement pass around the best candidate      (eq. 3.43)

All tolerances and weights are pre-registered defaults, overridable from the
CLI; every run prints them before any result is computed.
"""
import numpy as np
import torch

# ---- pre-registered constants (print before use; override via CLI) ---------
DEFAULTS = dict(
    eps_L=1e-8,      # eq. 3.30 denominator guard
    eps_KL=0.05,     # nats/token   tolerance for D_KL^out   (eq. 3.36)
    eps_rep=0.05,    # relative Frobenius drift tolerance    (eq. 3.36)
    eps_eff=0.10,    # efficiency tolerance                  (eq. 3.36)
    lam_L=1.0, lam_KL=0.1, lam_rep=0.05, lam_eff=0.05, lam_gamma=0.01,  # eq. 3.40
    omega_F=0.25, omega_T=0.25, omega_M=0.25, omega_P=0.25,             # eq. 3.34
    Q_refine=8,      # eq. 3.43
)


def paper_grid():
    """Gamma^(0) of eq. (3.29), verbatim."""
    return [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 0.2, 0.5, 1.0]


def print_gate_config(cfg):
    print("GATE-SELECTION CONFIG (pre-registered before evaluation):")
    print(f"  grid Gamma^(0) = {paper_grid()}")
    print(f"  eps_KL={cfg['eps_KL']}  eps_rep={cfg['eps_rep']}  eps_eff={cfg['eps_eff']}")
    print(f"  lambda: L={cfg['lam_L']} KL={cfg['lam_KL']} rep={cfg['lam_rep']} "
          f"eff={cfg['lam_eff']} gamma={cfg['lam_gamma']}")
    print(f"  omega (C_eff): F={cfg['omega_F']} T={cfg['omega_T']} "
          f"M={cfg['omega_M']} P={cfg['omega_P']}   refinement Q={cfg['Q_refine']}")


# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_gammas(W, layer, position, gammas, pool):
    """For each batch: ONE base forward (logits cached in-loop), then one
    candidate forward per gamma; returns per-gamma paired loss arrays and mean
    KL. This ordering makes D_KL^out essentially free relative to the loss
    evaluations."""
    gammas = [g for g in gammas if g > 0.0]
    losses = {g: [] for g in gammas}
    kls = {g: [] for g in gammas}
    base_losses = []
    for b in pool:
        base_logits, base_l = W.logits_and_loss(b)
        base_losses.append(base_l)
        for g in gammas:
            with W.inserted(layer, position, g):
                cand_logits, cand_l = W.logits_and_loss(b)
            losses[g].append(cand_l)
            kls[g].append(W.kl_to_base(base_logits, cand_logits))
        del base_logits
    return (np.array(base_losses),
            {g: np.array(v) for g, v in losses.items()},
            {g: float(np.mean(v)) for g, v in kls.items()})


def compute_ceff(W, layer, position, pool, cfg):
    """C_eff of eq. (3.33): FLOPs (matmul-param proxy), latency (measured),
    memory (param bytes ~ param ratio), parameters. C_eff(0)=0 by bypass
    (eq. 3.35)."""
    ps = W.param_stats(layer)
    par = ps["layer_params"] / ps["model_params"]
    flops = ps["layer_matmul"] / ps["model_matmul"]
    mem = par
    t_base, t_ext = W.measure_latency(layer, position, pool[:2])
    lat = max(0.0, (t_ext - t_base) / t_base)
    ceff = (cfg["omega_F"] * flops + cfg["omega_T"] * lat
            + cfg["omega_M"] * mem + cfg["omega_P"] * par)
    return dict(ceff=float(ceff), flops=float(flops), latency=float(lat),
                mem=float(mem), par=float(par),
                t_base_s=t_base, t_ext_s=t_ext)


def build_records(base_losses, losses, kls, drift_unit, ceff, cfg, paired_stats):
    """One record per gamma (including gamma=0) with every criterion of
    Section 3.4, the feasibility flag (eq. 3.36) and the objective (3.40)."""
    L0 = float(base_losses.mean())
    recs = [dict(gamma=0.0, mean=0.0, ci95=[0.0, 0.0], dL_norm=0.0, kl=0.0,
                 drift=0.0, ceff=0.0, feasible=True, J=0.0)]
    for g in sorted(losses):
        st = paired_stats(base_losses, losses[g], n_boot=2000)
        dLn = st["mean"] / (abs(L0) + cfg["eps_L"])                  # eq. 3.30
        kl = kls[g]
        drift = (g ** 2) * drift_unit                                 # Sec 3.6 identity
        ce = ceff["ceff"]                                             # gamma>0
        feas = (st["mean"] <= 0.0 and kl <= cfg["eps_KL"]
                and drift <= cfg["eps_rep"] and ce <= cfg["eps_eff"])  # eq. 3.36
        J = (cfg["lam_L"] * dLn + cfg["lam_KL"] * kl / cfg["eps_KL"]
             + cfg["lam_rep"] * drift / cfg["eps_rep"]
             + cfg["lam_eff"] * ce / cfg["eps_eff"]
             + cfg["lam_gamma"] * g ** 2)                              # eq. 3.40
        recs.append(dict(gamma=float(g), mean=float(st["mean"]),
                         ci95=list(st["ci95"]), dL_norm=float(dLn),
                         kl=float(kl), drift=float(drift), ceff=float(ce),
                         feasible=bool(feas), J=float(J)))
    return recs


def pick(records, tol=1e-12):
    """arg min J over the feasible set; ties broken by the SMALLEST gamma
    (eq. 3.41-3.42). gamma=0 is always feasible with J=0, so a gamma>0 is
    selected only if it strictly beats doing nothing."""
    feas = [r for r in records if r["feasible"]]
    Jmin = min(r["J"] for r in feas)
    winners = [r for r in feas if r["J"] <= Jmin + tol * max(1.0, abs(Jmin))]
    return min(winners, key=lambda r: r["gamma"])


def refine_grid(records, best_gamma, Q):
    """New points between the best candidate's grid neighbours (eq. 3.43)."""
    gs = sorted(r["gamma"] for r in records)
    if best_gamma not in gs:
        return []
    k = gs.index(best_gamma)
    lo = gs[k - 1] if k > 0 else best_gamma
    hi = gs[k + 1] if k < len(gs) - 1 else best_gamma
    if hi <= lo:
        return []
    pts = [lo + (q / Q) * (hi - lo) for q in range(Q + 1)]
    return sorted({round(p, 10) for p in pts} - set(gs) - {0.0})


@torch.no_grad()
def select_gate(W, layer, position, pool, cfg, paired_stats, verbose=True):
    """Coarse grid -> pick -> one refinement pass -> final pick. Returns
    (best_record, all_records, ceff_info, drift_unit)."""
    drift_unit = W.measure_drift_unit(layer, position, pool[:8])
    ceff = compute_ceff(W, layer, position, pool, cfg)
    if verbose:
        print(f"    D_rep(1) = {drift_unit:.5f}   C_eff = {ceff['ceff']:.5f} "
              f"(flops {ceff['flops']:.4f}, lat {ceff['latency']:.4f}, "
              f"par {ceff['par']:.4f})")

    base, losses, kls = evaluate_gammas(W, layer, position, paper_grid(), pool)
    recs = build_records(base, losses, kls, drift_unit, ceff, cfg, paired_stats)
    best = pick(recs)
    if verbose:
        for r in recs:
            print(f"    g={r['gamma']:<8.4g} dL={r['mean']:+.3e} "
                  f"KL={r['kl']:.2e} Drep={r['drift']:.2e} "
                  f"J={r['J']:+.3e} {'feas' if r['feasible'] else 'INFEAS'}"
                  + ("  <-- coarse best" if r is best else ""))

    new_gs = [] if best["gamma"] == 0.0 else refine_grid(recs, best["gamma"], cfg["Q_refine"])
    if new_gs:
        if verbose:
            print(f"    refinement around g={best['gamma']}: {new_gs}")
        _, l2, k2 = evaluate_gammas(W, layer, position, new_gs, pool)
        losses.update(l2)
        kls.update(k2)
        recs = build_records(base, losses, kls, drift_unit, ceff, cfg, paired_stats)
        best = pick(recs)
        if verbose:
            print(f"    refined best: g={best['gamma']}  J={best['J']:+.3e}  "
                  f"dL={best['mean']:+.3e}")
    return best, recs, ceff, drift_unit, base
