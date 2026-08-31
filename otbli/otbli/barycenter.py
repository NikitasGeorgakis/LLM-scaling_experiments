"""Free-support entropic Wasserstein-2 barycenter of two empirical measures.

Solves eq. (3.17) by alternating between transport-plan updates (log-domain
Sinkhorn) and support updates (eq. 3.18), exactly as used in Sections 5-6:
tau = 1/2, eta_b = 0.05, 25 alternating rounds, 80 Sinkhorn iterations/round.
"""
import torch

from .sinkhorn import pairwise_sq_dists, sinkhorn_log


@torch.no_grad()
def free_support_barycenter(X: torch.Tensor, Y: torch.Tensor, tau: float,
                            eta: float, n_rounds: int = 25,
                            n_sinkhorn: int = 80, tol: float = 0.0,
                            verbose: bool = False) -> torch.Tensor:
    """Barycenter atoms Z of (1-tau) * mu_i + tau * mu_{i+1} in the W2 sense.

    X : [m, d] atoms of mu_i     (uniform masses a = 1/m)
    Y : [n, d] atoms of mu_{i+1} (uniform masses b = 1/n)
    Support size n_b = m (the Pythia pairs have equal widths).

    Initialization: identity-pairing McCann interpolation
    Z0 = (1-tau) X + tau Y — i.e. exactly the 'naive average' baseline —
    which the alternating scheme then improves (Corollary 3.1: the barycenter
    pairing cost never exceeds the identity-pairing cost).
    """
    m, n = X.shape[0], Y.shape[0]
    dev, dt = X.device, X.dtype
    a = torch.full((m,), 1.0 / m, device=dev, dtype=dt)
    b = torch.full((n,), 1.0 / n, device=dev, dtype=dt)
    k = m                                             # support size n_b
    if n == m:
        Z = ((1.0 - tau) * X + tau * Y).clone()
    else:  # unequal widths: seed from a random subset of Y (not hit for Pythia)
        idx = torch.randperm(n, device=dev)[:k]
        Z = ((1.0 - tau) * X + tau * Y[idx]).clone()

    prev_obj = None
    for r in range(n_rounds):
        Cm = pairwise_sq_dists(X, Z)                  # [m, k]
        Cp = pairwise_sq_dists(Y, Z)                  # [n, k]
        Pm = sinkhorn_log(a, torch.full((k,), 1.0 / k, device=dev, dtype=dt), Cm, eta, n_sinkhorn)
        Pp = sinkhorn_log(b, torch.full((k,), 1.0 / k, device=dev, dtype=dt), Cp, eta, n_sinkhorn)
        obj = ((1.0 - tau) * (Pm * Cm).sum() + tau * (Pp * Cp).sum()).item()
        # support update, eq. (3.18) (actual column masses used for stability)
        col = ((1.0 - tau) * Pm.sum(dim=0) + tau * Pp.sum(dim=0)).clamp_min(1e-30)
        Z = ((1.0 - tau) * (Pm.T @ X) + tau * (Pp.T @ Y)) / col[:, None]
        if verbose:
            print(f"    [bary] round {r + 1:02d}/{n_rounds}  transport obj = {obj:.6e}")
        if prev_obj is not None and tol > 0.0 and abs(prev_obj - obj) <= tol * max(1.0, abs(prev_obj)):
            break
        prev_obj = obj
    return Z
