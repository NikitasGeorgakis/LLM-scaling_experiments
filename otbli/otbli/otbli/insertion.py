"""Gated insertion of G_bar into the residual stream.

Implementation note (ii) of Section 6.1: the inserted layer is realized as a
forward PRE-hook that writes G_gamma(h) = (1 - gamma) h + gamma G_bar(h)
(eq. 3.24) into the residual stream immediately before block i+1. The base
model's weights are never touched, and at gamma = 0 the hook is not even
registered, so M^+_0 == M holds bit-exactly.

Evaluation is teacher-forced with use_cache=False; generation with an active
gate is out of scope for the experiments.
"""
import torch

from .arch import get_layers


class GatedInsertion:
    def __init__(self, model, insert_before: int, gbar, eps_h: float = 1e-8):
        """
        model         : any supported CausalLM (weights untouched throughout)
        insert_before : 0-indexed j; G_gamma is applied to the hidden state
                        entering get_layers(model)[j]. For the paper's
                        1-indexed pair (F_i, F_{i+1}) this is j = i (0-indexed
                        i-1 + 1), i.e. immediately after F_i.
        gbar          : the fully-active barycentric layer G_bar (same block class)
        """
        self.model = model
        self.j = insert_before
        self.gbar = gbar
        self.eps_h = eps_h
        self.gamma = 0.0
        self._handle = None
        # drift measurement state, eq. (3.33)
        self.measure_drift = False
        self.drift_gated = False   # False: use G_bar(h)-h (gamma=1 semantics);
                                   # True : use G_gamma(h)-h (identity check)
        self._drift_sum = 0.0
        self._drift_cnt = 0

    # ------------------------------------------------------------------ hook
    def _hook(self, module, args, kwargs):
        if args:
            h, rest = args[0], args[1:]
            extra = {k: v for k, v in kwargs.items() if k != "hidden_states"}
        else:
            h, rest = kwargs["hidden_states"], ()
            extra = {k: v for k, v in kwargs.items() if k != "hidden_states"}
        out = self.gbar(h, *rest, **extra)
        g = out[0] if isinstance(out, tuple) else out
        h_new = (1.0 - self.gamma) * h + self.gamma * g

        if self.measure_drift:
            diff = (h_new - h) if self.drift_gated else (g - h)
            num = diff.float().pow(2).sum(dim=-1)
            den = h.float().pow(2).sum(dim=-1)
            self._drift_sum += (num / (den + self.eps_h)).sum().item()
            self._drift_cnt += num.numel()

        if args:
            return ((h_new,) + rest, kwargs)
        kw = dict(kwargs)
        kw["hidden_states"] = h_new
        return (args, kw)

    # ------------------------------------------------------------ gate state
    def set_gamma(self, gamma: float) -> None:
        self.gamma = float(gamma)
        want = (self.gamma > 0.0) or self.measure_drift
        if want and self._handle is None:
            self._handle = get_layers(self.model)[self.j].register_forward_pre_hook(
                self._hook, with_kwargs=True)
        elif not want and self._handle is not None:
            self._handle.remove()
            self._handle = None

    def remove(self) -> None:
        self.measure_drift = False
        self.gamma = 0.0
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    # ------------------------------------------------------------ drift API
    def reset_drift(self) -> None:
        self._drift_sum = 0.0
        self._drift_cnt = 0

    def drift_value(self) -> float:
        """Mean per-token ratio ||.||^2 / (||h||^2 + eps), eq. (3.33)."""
        return self._drift_sum / max(self._drift_cnt, 1)
