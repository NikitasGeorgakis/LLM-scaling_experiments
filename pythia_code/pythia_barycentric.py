"""
Barycentric construction (paper Sections 3.1-3.2) for a GPTNeoX / Pythia layer.

Parameter families B for a GPTNeoXLayer:

  b = 'mlp'   : the d_ff MLP hidden units are the ATOMS. Atom r's descriptor is
                the coupled triple
                    x_r = [ dense_h_to_4h.weight[r, :]   (d_model)
                          ; dense_4h_to_h.weight[:, r]   (d_model)
                          ; dense_h_to_4h.bias[r]        (1)      ]  in R^{2d+1}
                This is exactly the descriptor justified by Prop. 3.1 in the
                paper: the function-preserving symmetry permutes incoming
                weights, outgoing weights and bias JOINTLY, so they must live
                in a single atom. Aligned via a genuine free-support W2
                barycenter (alternating Sinkhorn OT + atom update, eq. 3.16-3.17).

  all others  : input_layernorm, post_attention_layernorm,
                attention.query_key_value, attention.dense,
                mlp.dense_4h_to_h.bias
                -> single-atom families (m = n = 1). The barycenter of two
                Diracs is the McCann interpolant, i.e. plain interpolation
                (1-tau)*theta_i + tau*theta_{i+1}. This is a SPECIAL CASE of
                eq. (3.14), not a separate mechanism.

NUMERICAL NOTE (differs from the toy implementation, and matters at LLM scale):
the toy code used a direct-kernel Sinkhorn, exp(-C/eta), which is fine when the
costs are O(0.2) but UNDERFLOWS TO ZERO for the much larger squared distances
between real Pythia MLP descriptors. Here we (i) normalize the cost matrix by
its mean, so that eta is a scale-free regularization strength, and (ii) run
Sinkhorn in the LOG DOMAIN. Both are numerical fixes only -- the optimization
problem being solved is unchanged.
"""
import copy
import torch


# --------------------------------------------------------------------------- #
#  optimal transport
# --------------------------------------------------------------------------- #
def sinkhorn_log(C, a, b, eta, n_iters=80):
    """Entropic OT plan in the log domain. C is assumed already normalized
    (see free_support_barycenter). Returns P with P@1 = a, P^T@1 = b."""
    loga, logb = a.log(), b.log()
    f = torch.zeros_like(a)
    g = torch.zeros_like(b)
    for _ in range(n_iters):
        f = -eta * torch.logsumexp((g.unsqueeze(0) - C) / eta + logb.unsqueeze(0), dim=1)
        g = -eta * torch.logsumexp((f.unsqueeze(1) - C) / eta + loga.unsqueeze(1), dim=0)
    logP = ((f.unsqueeze(1) + g.unsqueeze(0) - C) / eta
            + loga.unsqueeze(1) + logb.unsqueeze(0))
    return logP.exp()


def free_support_barycenter(X_i, X_j, tau, eta=0.05, n_alt_rounds=25,
                            sinkhorn_iters=80, n_b=None, verbose=False):
    """Solves eq. (3.14)/(3.16)-(3.17): free-support W2 barycenter of the
    empirical measures on the rows of X_i and X_j, uniform masses, squared
    Euclidean ground cost (L_b = I). Returns barycenter atoms Z (n_b x d)."""
    m_i, d = X_i.shape
    m_j, _ = X_j.shape
    n_b = n_b or m_i
    dev = X_i.device
    a  = torch.full((m_i,), 1.0 / m_i, device=dev, dtype=X_i.dtype)
    a2 = torch.full((m_j,), 1.0 / m_j, device=dev, dtype=X_i.dtype)
    w  = torch.full((n_b,), 1.0 / n_b, device=dev, dtype=X_i.dtype)

    # natural-index initialization (requires m_i == m_j == n_b, true here since
    # both neighbouring layers share d_ff)
    Z = (1 - tau) * X_i[:n_b] + tau * X_j[:n_b]

    prev_obj = None
    for r in range(n_alt_rounds):
        C_i = torch.cdist(X_i, Z) ** 2
        C_j = torch.cdist(X_j, Z) ** 2
        # scale-free entropic regularization
        s_i = C_i.mean().clamp_min(1e-12)
        s_j = C_j.mean().clamp_min(1e-12)
        P_minus = sinkhorn_log(C_i / s_i, a,  w, eta, sinkhorn_iters)
        P_plus  = sinkhorn_log(C_j / s_j, a2, w, eta, sinkhorn_iters)

        obj = float((1 - tau) * (P_minus * C_i).sum() + tau * (P_plus * C_j).sum())
        num = (1 - tau) * (P_minus.t() @ X_i) + tau * (P_plus.t() @ X_j)
        Z = num / w.unsqueeze(1)

        if verbose and (r % 5 == 0 or r == n_alt_rounds - 1):
            print(f"      alt round {r:2d}  barycenter objective = {obj:.6e}")
        if prev_obj is not None and abs(prev_obj - obj) < 1e-9 * max(1.0, abs(prev_obj)):
            break
        prev_obj = obj

    return Z, float(obj)


# --------------------------------------------------------------------------- #
#  layer construction
# --------------------------------------------------------------------------- #
def _interp_(dst_mod, mod_i, mod_j, tau):
    """In-place plain interpolation of every parameter of a submodule
    (single-atom family -> McCann interpolant)."""
    with torch.no_grad():
        for p_n, p_i, p_j in zip(dst_mod.parameters(),
                                 mod_i.parameters(), mod_j.parameters()):
            p_n.copy_((1 - tau) * p_i + tau * p_j)


@torch.no_grad()
def build_barycentric_layer(layer_i, layer_j, tau=0.5, eta=0.05,
                            n_alt_rounds=25, sinkhorn_iters=80,
                            work_dtype=torch.float32, verbose=False):
    """theta_bar for a new GPTNeoXLayer from two neighbouring pretrained layers.
    Returns (new_layer, barycenter_objective)."""
    new_layer = copy.deepcopy(layer_i)

    d = layer_i.mlp.dense_h_to_4h.in_features

    # ---- MLP hidden-unit family: genuine OT barycenter --------------------
    W1_i = layer_i.mlp.dense_h_to_4h.weight.to(work_dtype)   # (d_ff, d)
    W2_i = layer_i.mlp.dense_4h_to_h.weight.to(work_dtype)   # (d, d_ff)
    b1_i = layer_i.mlp.dense_h_to_4h.bias.to(work_dtype)     # (d_ff,)
    W1_j = layer_j.mlp.dense_h_to_4h.weight.to(work_dtype)
    W2_j = layer_j.mlp.dense_4h_to_h.weight.to(work_dtype)
    b1_j = layer_j.mlp.dense_h_to_4h.bias.to(work_dtype)

    X_i = torch.cat([W1_i, W2_i.t(), b1_i.unsqueeze(1)], dim=1)   # (d_ff, 2d+1)
    X_j = torch.cat([W1_j, W2_j.t(), b1_j.unsqueeze(1)], dim=1)

    Z, obj = free_support_barycenter(X_i, X_j, tau, eta, n_alt_rounds,
                                     sinkhorn_iters, n_b=X_i.shape[0],
                                     verbose=verbose)

    new_layer.mlp.dense_h_to_4h.weight.copy_(Z[:, :d].to(layer_i.mlp.dense_h_to_4h.weight.dtype))
    new_layer.mlp.dense_4h_to_h.weight.copy_(Z[:, d:2*d].t().contiguous().to(layer_i.mlp.dense_4h_to_h.weight.dtype))
    new_layer.mlp.dense_h_to_4h.bias.copy_(Z[:, 2*d].to(layer_i.mlp.dense_h_to_4h.bias.dtype))

    # ---- remaining families: single atom -> plain interpolation -----------
    new_layer.mlp.dense_4h_to_h.bias.copy_(
        (1 - tau) * layer_i.mlp.dense_4h_to_h.bias + tau * layer_j.mlp.dense_4h_to_h.bias)
    _interp_(new_layer.attention, layer_i.attention, layer_j.attention, tau)
    _interp_(new_layer.input_layernorm, layer_i.input_layernorm, layer_j.input_layernorm, tau)
    _interp_(new_layer.post_attention_layernorm, layer_i.post_attention_layernorm,
             layer_j.post_attention_layernorm, tau)

    return new_layer, obj


@torch.no_grad()
def build_naive_average_layer(layer_i, layer_j, tau=0.5):
    """Baseline 2 of the Evaluation Plan: elementwise averaging with NO OT
    alignment. By Cor. 3.1 this is the barycenter with sigma = identity, so it
    isolates the contribution of the alignment step alone."""
    new_layer = copy.deepcopy(layer_i)
    _interp_(new_layer, layer_i, layer_j, tau)
    return new_layer


@torch.no_grad()
def build_duplicate_layer(layer_i, layer_j=None):
    """Baseline 1: verbatim duplication of F_i (the tau -> 0 endpoint)."""
    return copy.deepcopy(layer_i)


@torch.no_grad()
def matching_diagnostics(layer_i, layer_j, work_dtype=torch.float32, max_units=4096):
    """Section 3.6 diagnostic: exact optimal matching cost vs identity pairing.
    Subsamples units if d_ff is very large (Hungarian is O(m^3))."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    d = layer_i.mlp.dense_h_to_4h.in_features
    X_i = torch.cat([layer_i.mlp.dense_h_to_4h.weight.to(work_dtype),
                     layer_i.mlp.dense_4h_to_h.weight.to(work_dtype).t(),
                     layer_i.mlp.dense_h_to_4h.bias.to(work_dtype).unsqueeze(1)], dim=1)
    X_j = torch.cat([layer_j.mlp.dense_h_to_4h.weight.to(work_dtype),
                     layer_j.mlp.dense_4h_to_h.weight.to(work_dtype).t(),
                     layer_j.mlp.dense_h_to_4h.bias.to(work_dtype).unsqueeze(1)], dim=1)
    m = X_i.shape[0]
    if m > max_units:
        idx = torch.arange(0, m, m // max_units)[:max_units]
        X_i, X_j, m = X_i[idx], X_j[idx], len(idx)
    C = (torch.cdist(X_i, X_j) ** 2).double().cpu().numpy()
    ident = float(np.mean(np.diag(C)))
    r, c = linear_sum_assignment(C)
    opt = float(C[r, c].mean())
    return dict(units=m, identity_cost=ident, optimal_cost=opt,
                reduction_pct=100 * (1 - opt / ident),
                rematched=int((c != np.arange(m)).sum()))
