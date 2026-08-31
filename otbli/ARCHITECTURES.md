# Architectures: how the theory maps onto each MLP family

The protocol layer (Sinkhorn, barycenter, gate grid, feasibility, rules 3.47/3.48) is
architecture-agnostic. Architecture enters only through the atomization maps `A_b` /
`Q_b` (`otbli/atomize.py`) and the layer-stack path (`otbli/arch.py`). The design
principle (Section 3.6): **partition parameters into orbits of the layer's
function-preserving symmetry group** — OT matching where units are exchangeable,
plain interpolation where the group acts trivially, and a metric that absorbs any
residual continuous gauge.

## Supported families (auto-detected)

**`gpt_neox` (Pythia)** — simple 2-layer GELU MLP. Unit descriptor
`(u_r, v_r, c_r) ∈ R^{2d+1}`. GELU is neither odd nor homogeneous, so permutations
are generically the only exact unit symmetries and the plain Euclidean cost is the
right metric (this is the setting of Propositions 3.1–3.2 as written).

**`gpt2`** — the same GELU MLP; HF stores Conv1D weights transposed (`[in, out]`),
which only changes the bookkeeping in `mlp_atoms` / `write_mlp_atoms`. Theory
transfers verbatim; different data/tokenizer make it the cleanest external control.

**`gated` (Llama / Mistral / TinyLlama / Qwen2-style SwiGLU)** — unit r computes
`ψ(x_r; h) = down_r · SiLU(gate_r·h) · (up_r·h)`, so the layer is still a linear
functional of the measure over coupled triples `x_r = (gate_r, up_r, down_r) ∈ R^{3d}`
and Propositions 3.1–3.2 extend with this descriptor. Because the up-path is linear,
`(up_r, down_r) → (λ·up_r, down_r/λ)` is an **exact continuous gauge** — the failure
mode named in Section 3.6, remark (i): the Euclidean ground cost registers spurious
distance along it. The atomization therefore **gauge-fixes**: `‖up_r‖ = 1` with the
scale folded into `down_r` (a function-identical reparameterization, verified
numerically to machine precision). `OTConfig.gauge_fix = False` ablates this;
`matching_diagnostic` reports gauge-fixed and raw numbers side by side. On synthetic
permuted-and-regauged units the raw matched cost is inflated by orders of magnitude
while the gauge-fixed one recovers the permutation geometry — the measurable form of
remark (i).

Single-atom families everywhere (attention projections, LayerNorms, output biases):
trivial barycenter = linear interpolation. Non-float buffers copied verbatim.

## Adding a new architecture (three edits in `otbli/atomize.py`)

1. Teach `_detect_mlp_type()` to recognize the layer (an `hasattr` check on its MLP).
2. Decide the symmetry: if the MLP is a sum of exchangeable units, add its branch to
   `mlp_atoms` / `write_mlp_atoms` / `_mlp_unit_keys` with a **coupled** descriptor
   (never atomize incoming/outgoing weights as separate families — Section 3.6), and
   gauge-fix any exact rescaling directions; if there is no exchangeable structure,
   return no unit keys and the block is combined by plain interpolation.
3. If the model wrapper stores its blocks somewhere new, add the path to
   `otbli/arch.py::_LAYER_PATHS`.

`tests/smoke_test.py` runs the full pipeline end to end on tiny random GPT-NeoX,
GPT-2 and Llama models (no downloads) and checks, per architecture: detection, the
`Q_b ∘ A_b = id` round-trip, bit-exact recovery at γ = 0, the drift identity, and the
matching diagnostic (including the gauge ablation for SwiGLU).
