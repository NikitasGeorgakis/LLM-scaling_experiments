"""Entropy-regularized optimal transport in the log domain.

Implementation note (i) of Section 6.1: at LLM widths the naive Gibbs kernel
exp(-C/eta) underflows to exact zero for typical inter-unit descriptor
distances, so a log-domain Sinkhorn with cost normalization is required
(Peyre & Cuturi, 2019). The entropic term is taken relative to the product
coupling a (x) b, matching eq. (3.17).
"""
import torch


@torch.no_grad()
def pairwise_sq_dists(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Squared-Euclidean ground cost c(x, y) = ||x - y||^2 (L_b = I), eq. (3.12)."""
    x2 = (X * X).sum(dim=1, keepdim=True)          # [m, 1]
    y2 = (Y * Y).sum(dim=1, keepdim=True).T        # [1, n]
    C = x2 + y2 - 2.0 * (X @ Y.T)
    return C.clamp_min_(0.0)


@torch.no_grad()
def sinkhorn_log(a: torch.Tensor, b: torch.Tensor, C: torch.Tensor,
                 eta: float, n_iters: int) -> torch.Tensor:
    """Solve  min <P, C~> + eta * KL(P || a b^T)  s.t.  P 1 = a, P^T 1 = b,
    where C~ = C / mean(C) (cost normalization keeps eta scale-free across
    model widths; the barycenter support update (3.18) uses only the plan and
    is unaffected by the rescaling).

    Returns the (dense) transport plan P.
    """
    scale = C.mean().clamp_min(1e-30)
    M = -(C / scale) / eta                          # log Gibbs kernel
    log_a, log_b = a.log(), b.log()
    f = torch.zeros_like(a)
    g = torch.zeros_like(b)
    for _ in range(n_iters):
        # f_i = -eta * logsumexp_j[ (g_j - C_ij)/eta + log b_j ]
        f = -eta * torch.logsumexp(M + (g / eta + log_b)[None, :], dim=1)
        # g_j = -eta * logsumexp_i[ (f_i - C_ij)/eta + log a_i ]
        g = -eta * torch.logsumexp(M.T + (f / eta + log_a)[None, :], dim=1)
    logP = M + (f / eta + log_a)[:, None] + (g / eta + log_b)[None, :]
    return logP.exp()
