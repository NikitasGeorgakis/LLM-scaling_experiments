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


@dataclass
class TaskConfig:
    """Downstream-task target metric (the gated-duplication experiment).

    Pre-registered before any task evaluation:
      - delta_acc = 0.01, i.e. 1 accuracy point, the materiality margin of
        eq. (3.47) restated in task units
      - the candidate positions are the early blocks 1-3, where the published
        training-free duplication gain is reported
      - questions are split 50/50 into a selection and a disjoint confirmation
        half, stratified per task, with the same seed as the Pile pools
    """
    tasks: tuple = (
        "bigbench_causal_judgment_multiple_choice",
        "bigbench_date_understanding_multiple_choice",
        "bigbench_disambiguation_qa_multiple_choice",
        "bigbench_logical_deduction_multiple_choice",
        "bigbench_movie_recommendation_multiple_choice",
        "bigbench_navigate_multiple_choice",
        "bigbench_reasoning_about_colored_objects_multiple_choice",
        "bigbench_ruin_names_multiple_choice",
        "bigbench_snarks_multiple_choice",
        "bigbench_temporal_sequences_multiple_choice",
    )
    control_tasks: tuple = (
        # POSITIVE CONTROL. Small Pythia models sit near chance on most BigBench
        # tasks, and a gain cannot be detected in a metric that is pure noise --
        # the same logic as the mechanism diagnostics of Section 6.4: a null is
        # informative only if the measurement demonstrably has signal. These are
        # likelihood-scored tasks where a 410M-1B model is clearly above chance
        # (sciq ~.84, piqa ~.70, arc_easy ~.57, lambada ~.56), so they show
        # whether the pipeline could have seen an effect at all.
        "sciq", "piqa", "arc_easy", "lambada_openai",
    )
    limit: int = 200              # questions per task; fixed across all gammas
    num_fewshot: int = 0
    seed: int = 1234              # pins doc order AND few-shot contexts (pairing)
    delta_acc: float = 0.01       # materiality margin, eq. (3.47) in accuracy points
    positions: tuple = (0, 1, 2)  # 0-indexed: duplicate block 1, 2 or 3
    kl_free: bool = False         # ablation: drop the output-KL feasibility screen
    n_boot_sel: int = 2000
    metrics: dict = field(default_factory=dict)   # task -> metric key, pinned at gamma=0


def print_registered(*cfgs) -> None:
    """Print the pre-registered constants (done before any evaluation)."""
    print("=" * 78)
    print("PRE-REGISTERED CONSTANTS (fixed and printed before any evaluation)")
    print("=" * 78)
    for c in cfgs:
        print(f"[{type(c).__name__}]")
        print(json.dumps(asdict(c), indent=2, default=list))
    print("=" * 78, flush=True)
