"""Pre-registered constants for the Pythia experiments (paper Section 6.1).

Every constant below was fixed and printed BEFORE any evaluation was run:
  - epsilon_KL = epsilon_rep = 0.05, epsilon_eff = 0.10
  - lambda_L = 1, lambda_KL = 0.1, lambda_rep = lambda_eff = 0.05, lambda_gamma = 0.01
  - omega_F = omega_T = omega_M = omega_P = 1/4
  - materiality margin delta = 1e-3 nats            (eq. 3.47)
  - gate grid Gamma^(0)                             (eq. 3.29)
  - tau = 1/2, eta_b = 0.05, 25 alternating rounds, 80 Sinkhorn iters/round
  - selection pool: 25 paired batches of 8 x 1024 tokens; confirmation: 40
  - bootstrap: 2,000 resamples (selection), 10,000 (confirmation)
"""
from dataclasses import dataclass, asdict, field
import json

# eq. (3.29): {0,1} union {gamma_min * rho^j}, denser near zero
GATE_GRID = (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 0.2, 0.5, 1.0)


@dataclass
class OTConfig:
    """Barycentric construction (Sections 3.1-3.2, 6.1)."""
    tau: float = 0.5              # interpolation weight, eq. (3.15)
    eta: float = 0.05             # entropic regularization (on the mean-normalized cost)
    n_alt_rounds: int = 25        # alternating plan/support rounds
    n_sinkhorn_iters: int = 80    # log-domain Sinkhorn iterations per round
    tol: float = 0.0
    gauge_fix: bool = True   # SwiGLU up/down rescaling gauge (Sec. 3.6 rem. (i))
    ot_dtype: str = "float32"     # dtype for descriptors / Sinkhorn


@dataclass
class GateConfig:
    """Gate-selection machinery (Section 3.4) with the Section 6.1 constants."""
    grid: tuple = GATE_GRID
    eps_KL: float = 0.05          # tolerance on D_KL^out, eq. (3.37)
    eps_rep: float = 0.05         # tolerance on D_rep
    eps_eff: float = 0.10         # tolerance on C_eff
    lam_L: float = 1.0            # eq. (3.41)
    lam_KL: float = 0.1
    lam_rep: float = 0.05
    lam_eff: float = 0.05
    lam_gamma: float = 0.01
    omega: tuple = (0.25, 0.25, 0.25, 0.25)  # (FLOPs, latency, memory, params), eq. (3.34)
    delta: float = 1e-3           # materiality margin in nats, eq. (3.47)
    eps_L: float = 1e-8           # denominator guard in eq. (3.31)
    eps_h: float = 1e-8           # denominator guard in eq. (3.33)
    refine_Q: int = 8             # refinement resolution, eq. (3.44)


@dataclass
class DataConfig:
    """Held-out Pile pools (Section 6.1): document-shuffled with a fixed seed,
    packed into 1,024-token blocks; 25 selection + 40 disjoint confirmation
    batches of 8 blocks each."""
    dataset: str = "monology/pile-uncopyrighted"  # HF mirror of held-out Pile text;
                                                  # pass a local .jsonl path to use your own shards
    seed: int = 1234
    block_len: int = 1024
    batch_size: int = 8
    n_sel_batches: int = 25
    n_conf_batches: int = 40


@dataclass
class ProtocolConfig:
    """Two-stage protocol bookkeeping (Sections 3.4, 6.1)."""
    top_k_positions: int = 2      # Stage-A survivors that enter Stage B
    n_boot_sel: int = 2000        # bootstrap resamples, selection set
    n_boot_conf: int = 10000      # bootstrap resamples, confirmation set
    kl_batches: int = 25          # KL pool = selection pool (reduce to speed up)
    drift_batches: int = 4        # batches used to measure D_rep(1)
    latency_batches: int = 5      # batches per latency timing pass
    latency_repeats: int = 3
    match_max_units: int = 4096   # subsample for the exact-LAP matching diagnostic


def print_registered(*cfgs) -> None:
    """Print the pre-registered constants (done before any evaluation)."""
    print("=" * 78)
    print("PRE-REGISTERED CONSTANTS (fixed and printed before any evaluation)")
    print("=" * 78)
    for c in cfgs:
        print(f"[{type(c).__name__}]")
        print(json.dumps(asdict(c), indent=2, default=list))
    print("=" * 78, flush=True)
