"""Atomization A_b and reconstruction Q_b for transformer blocks.

The design principle of Section 3.6: B partitions the parameters into orbits
of the layer's function-preserving symmetry group — full OT matching where
there is unit exchangeability, plain interpolation where there is none, and a
ground cost that absorbs any residual continuous gauge.

Supported MLP families (auto-detected):

  gpt_neox (Pythia)  — simple 2-layer GELU MLP. Exchangeable unit r has the
    coupled descriptor x_r = (u_r, v_r, c_r) in R^{2d+1}:
      u_r = row r of  mlp.dense_h_to_4h.weight
      v_r = col r of  mlp.dense_4h_to_h.weight
      c_r = element r of mlp.dense_h_to_4h.bias
    Proposition 3.2 applies verbatim (GELU: permutations are generically the
    only exact unit symmetries, so the plain Euclidean cost behaves well).

  gpt2 — same simple 2-layer GELU MLP, but HF stores Conv1D weights
    transposed ([in, out]); descriptors are identical in R^{2d+1}.

  gated (Llama / Mistral / TinyLlama / Qwen2-style SwiGLU) — unit r computes
      psi(x_r; h) = down_r * SiLU(gate_r . h) * (up_r . h),
    so the layer is STILL a linear functional of the discrete measure over the
    coupled triples x_r = (gate_r, up_r, down_r) in R^{3d}: Propositions 3.1
    and 3.2 extend with this descriptor. However the up-path is linear, so
      (up_r, down_r) -> (lam * up_r, down_r / lam),  lam > 0,
    is an EXACT function-preserving gauge — precisely the failure mode of
    Section 3.6, remark (i). The theoretically principled atomization
    therefore gauge-fixes before measuring distance: normalize ||up_r|| = 1
    and fold the scale into down_r (function-identical reparameterization).
    Set OTConfig.gauge_fix = False to ablate this and test the remark.

  Single-atom families (all archs): attention projections, LayerNorms, output
  biases — the symmetry group acts trivially, so the barycenter reduces to
  linear interpolation (the McCann interpolant of two Diracs). Non-float
  buffers (causal masks etc.) are copied verbatim.
"""
import copy
import torch

from .barycenter import free_support_barycenter


def _detect_mlp_type(layer) -> str:
    if hasattr(layer.mlp, "dense_h_to_4h"):
        return "gpt_neox"
    if hasattr(layer.mlp, "c_fc") and hasattr(layer.mlp, "c_proj"):
        return "gpt2"
    if hasattr(layer.mlp, "gate_proj"):
        return "gated"
    raise ValueError(f"Unknown MLP architecture in layer {type(layer).__name__}; "
                     "see ARCHITECTURES.md for how to add one")


def _mlp_unit_keys(layer) -> set:
    """state_dict keys of the exchangeable unit family (handled by OT, not by
    the single-atom linear interpolation)."""
    return {
        "gpt_neox": {"mlp.dense_h_to_4h.weight", "mlp.dense_h_to_4h.bias",
                     "mlp.dense_4h_to_h.weight"},
        "gpt2":     {"mlp.c_fc.weight", "mlp.c_fc.bias", "mlp.c_proj.weight"},
        "gated":    {"mlp.gate_proj.weight", "mlp.up_proj.weight",
                     "mlp.down_proj.weight"},
    }[_detect_mlp_type(layer)]


# ------------------------------------------------------------------ A_b / Q_b
@torch.no_grad()
def mlp_atoms(layer, gauge_fix: bool = True) -> torch.Tensor:
    """A_b: [m, D] descriptor matrix of the exchangeable MLP unit family.
    D = 2d+1 for simple GELU MLPs, D = 3d for SwiGLU (gauge-fixed by default)."""
    kind = _detect_mlp_type(layer)
    if kind == "gpt_neox":
        Wi = layer.mlp.dense_h_to_4h.weight            # [m, d]
        bi = layer.mlp.dense_h_to_4h.bias              # [m]
        Wo = layer.mlp.dense_4h_to_h.weight            # [d, m]
        return torch.cat([Wi, Wo.T, bi[:, None]], dim=1).contiguous()
    if kind == "gpt2":
        Wi = layer.mlp.c_fc.weight                     # Conv1D: [d, m]
        bi = layer.mlp.c_fc.bias                       # [m]
        Wo = layer.mlp.c_proj.weight                   # Conv1D: [m, d]
        return torch.cat([Wi.T, Wo, bi[:, None]], dim=1).contiguous()
    # gated (SwiGLU)
    G = layer.mlp.gate_proj.weight                     # [m, d]
    U = layer.mlp.up_proj.weight                       # [m, d]
    Dt = layer.mlp.down_proj.weight.T                  # [m, d]
    if gauge_fix:                                      # Sec. 3.6 remark (i)
        s = U.norm(dim=1, keepdim=True).clamp_min(1e-12)
        U = U / s
        Dt = Dt * s
    return torch.cat([G, U, Dt], dim=1).contiguous()


@torch.no_grad()
def write_mlp_atoms(layer, Z: torch.Tensor) -> None:
    """Q_b: write barycenter atoms back as a valid parameter family. For the
    gated family, gauge-fixed coordinates are themselves a valid (function-
    equivalent) parameterization, so they are stored as-is."""
    kind = _detect_mlp_type(layer)
    if kind == "gpt_neox":
        W = layer.mlp.dense_h_to_4h.weight
        d = W.shape[1]
        Z = Z.to(device=W.device, dtype=W.dtype)
        layer.mlp.dense_h_to_4h.weight.copy_(Z[:, :d])
        layer.mlp.dense_4h_to_h.weight.copy_(Z[:, d:2 * d].T.contiguous())
        layer.mlp.dense_h_to_4h.bias.copy_(Z[:, 2 * d])
    elif kind == "gpt2":
        W = layer.mlp.c_fc.weight
        d = W.shape[0]
        Z = Z.to(device=W.device, dtype=W.dtype)
        layer.mlp.c_fc.weight.copy_(Z[:, :d].T.contiguous())
        layer.mlp.c_proj.weight.copy_(Z[:, d:2 * d].contiguous())
        layer.mlp.c_fc.bias.copy_(Z[:, 2 * d])
    else:  # gated
        W = layer.mlp.gate_proj.weight
        d = W.shape[1]
        Z = Z.to(device=W.device, dtype=W.dtype)
        layer.mlp.gate_proj.weight.copy_(Z[:, :d])
        layer.mlp.up_proj.weight.copy_(Z[:, d:2 * d])
        layer.mlp.down_proj.weight.copy_(Z[:, 2 * d:3 * d].T.contiguous())


def _freeze(block):
    block.eval()
    for p in block.parameters():
        p.requires_grad_(False)
    return block


# ------------------------------------------------------------------- builders
@torch.no_grad()
def build_interpolated_block(layer_a, layer_b, tau: float, mlp_Z: torch.Tensor = None):
    """Reconstruct a new transformer block of the same class as layer_a.

    mlp_Z is None  -> plain parameter interpolation (1-tau) theta_i + tau
                      theta_{i+1} of EVERY float tensor: this is exactly the
                      'naive averaging, no OT alignment' baseline of Section 4.
    mlp_Z given    -> single-atom families interpolated linearly (their trivial
                      barycenter); the exchangeable unit family overwritten
                      with the supplied barycenter atoms Z.
    """
    g = copy.deepcopy(layer_a)
    unit_keys = _mlp_unit_keys(layer_a) if mlp_Z is not None else set()
    sa, sb = layer_a.state_dict(), layer_b.state_dict()
    new = {}
    for k in sa:
        if k in unit_keys:
            new[k] = sa[k]                    # placeholder, overwritten below
        elif torch.is_floating_point(sa[k]):
            new[k] = (1.0 - tau) * sa[k] + tau * sb[k]
        else:
            new[k] = sa[k]                    # integer/bool buffers: copy
    g.load_state_dict(new)
    if mlp_Z is not None:
        write_mlp_atoms(g, mlp_Z)
    return _freeze(g)


@torch.no_grad()
def build_barycentric_block(layer_a, layer_b, tau: float, eta: float,
                            n_rounds: int, n_sinkhorn: int, device: str,
                            ot_dtype=None, gauge_fix: bool = True,
                            verbose: bool = False):
    """Full barycentric layer G_bar (Section 3.2): free-support W2 barycenter
    of the exchangeable-unit measures of the two neighboring layers — for ALL
    supported architectures — plus trivial barycenters for the single-atom
    families. A closed-form function of the pretrained weights; no gradient
    step anywhere."""
    if ot_dtype is None:
        ot_dtype = torch.float32
    X = mlp_atoms(layer_a, gauge_fix=gauge_fix).to(device=device, dtype=ot_dtype)
    Y = mlp_atoms(layer_b, gauge_fix=gauge_fix).to(device=device, dtype=ot_dtype)
    Z = free_support_barycenter(X, Y, tau, eta, n_rounds, n_sinkhorn, verbose=verbose)
    return build_interpolated_block(layer_a, layer_b, tau, mlp_Z=Z)


@torch.no_grad()
def build_duplicate_block(layer_a):
    """Verbatim layer duplication — the tau -> 0 endpoint of the barycenter
    path (Corollary 3.1) and the 'frankenmerge' baseline."""
    return _freeze(copy.deepcopy(layer_a))
