"""Mechanism diagnostics (Section 6.4): a null result is informative only if
the pipeline demonstrably did what Section 3 specifies.

  (i)   exact recovery      : M^+_0 == M bit-exactly (logits + weight fingerprint)
  (ii)  drift identity      : D_rep(gamma) == gamma^2 D_rep(1) to numerical precision
  (iii) matching diagnostic : exact LAP on unit descriptors, eq. (3.49) —
        fraction of units re-matched away from identity, pairing-cost reduction.
        For SwiGLU layers the diagnostic is additionally run WITHOUT the gauge
        fix, turning Section 3.6 remark (i) into a measurable ablation: the
        gauge directions should inflate the raw matched cost.
"""
import numpy as np
import torch

from .atomize import mlp_atoms, _detect_mlp_type


def _lap(X: torch.Tensor, Y: torch.Tensor) -> dict:
    from scipy.optimize import linear_sum_assignment
    C = torch.cdist(X, Y).pow(2).double().numpy()
    r, c = linear_sum_assignment(C)
    cost_id = float(np.mean(np.diag(C)))
    cost_opt = float(C[r, c].mean())
    return {"frac_rematched": float(np.mean(c != r)),
            "mean_cost_identity": cost_id,
            "mean_cost_optimal": cost_opt,
            "reduction": float(1.0 - cost_opt / max(cost_id, 1e-30))}


@torch.no_grad()
def matching_diagnostic(layer_a, layer_b, max_units: int = 4096, seed: int = 0) -> dict:
    """Exact-LAP matching diagnostic, eq. (3.49) / Corollary 3.1."""
    kind = _detect_mlp_type(layer_a)
    Xg = mlp_atoms(layer_a, gauge_fix=True).float().cpu()
    Yg = mlp_atoms(layer_b, gauge_fix=True).float().cpu()
    m = Xg.shape[0]
    idx = None
    if m > max_units:
        g = torch.Generator().manual_seed(seed)
        idx = torch.sort(torch.randperm(m, generator=g)[:max_units]).values
        Xg, Yg = Xg[idx], Yg[idx]
    out = {"mlp_type": kind, "units": int(Xg.shape[0])}
    out.update(_lap(Xg, Yg))
    if kind == "gated":                     # gauge ablation, Sec. 3.6 remark (i)
        Xr = mlp_atoms(layer_a, gauge_fix=False).float().cpu()
        Yr = mlp_atoms(layer_b, gauge_fix=False).float().cpu()
        if idx is not None:
            Xr, Yr = Xr[idx], Yr[idx]
        out["no_gauge"] = _lap(Xr, Yr)
    return out


@torch.no_grad()
def state_fingerprint(model):
    """Cheap but sensitive fingerprint of every parameter tensor."""
    return [(n, float(p.double().abs().sum().item()), tuple(p.shape))
            for n, p in model.named_parameters()]


def fingerprints_equal(f1, f2) -> bool:
    return all(a == b for a, b in zip(f1, f2)) and len(f1) == len(f2)


@torch.no_grad()
def exact_recovery_check(model, batch, device, reference_logits=None):
    """Run the (hook-free) model and compare logits bit-exactly against a
    reference captured before any insertion machinery was constructed."""
    logits = model(input_ids=batch.to(device), use_cache=False).logits
    if reference_logits is None:
        return logits.detach().cpu()
    return bool(torch.equal(logits.detach().cpu(), reference_logits))


@torch.no_grad()
def drift_identity_check(model, ins, pool, device, drift1: float,
                         gamma: float = 0.1, n_batches: int = 2) -> dict:
    """Measure D_rep at `gamma` directly (on the gated output) and compare to
    the exact identity gamma^2 * D_rep(1)."""
    ins.reset_drift()
    ins.measure_drift = True
    ins.drift_gated = True
    ins.set_gamma(gamma)
    for b in pool[:n_batches]:
        model(input_ids=b.to(device), use_cache=False)
    measured = ins.drift_value()
    ins.measure_drift = False
    ins.drift_gated = False
    ins.set_gamma(0.0)
    predicted = gamma * gamma * drift1
    rel_err = abs(measured - predicted) / max(abs(predicted), 1e-30)
    return {"gamma": gamma, "measured": measured, "predicted": predicted,
            "rel_err": rel_err}
