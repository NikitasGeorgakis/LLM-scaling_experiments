"""
Wrapper around a HuggingFace Pythia (GPTNeoX) checkpoint supporting gated
layer insertion and every measurement the paper's gate selection needs:
validation loss, output-distribution KL (eq. 3.31), representation drift
(eq. 3.32), and deployment-efficiency components (eq. 3.33).

Insertion mechanism: a *forward pre-hook* on layer i+1. The hook receives the
exact arguments layer i+1 is about to be called with (hidden states, attention
mask, rotary position embeddings, ...), calls the barycentric layer with those
same arguments, and swaps the hidden state for
    G_gamma(h) = h + gamma * (G_bar(h) - h)          [eq. 3.23]
This is signature-agnostic across transformers versions and guarantees
M+_0 == M bit-exactly (gamma = 0 leaves h untouched).

Indexing: insert_after = i (0-based) puts G_gamma between layers i and i+1,
i.e. between the paper's F_{i+1} and F_{i+2} (1-based). Valid: 0..n_layer-2.
"""
import contextlib
import time
import numpy as np
import torch
import torch.nn.functional as F


class PythiaWrapper:
    def __init__(self, model_name="EleutherAI/pythia-1b", dtype=torch.float32,
                 device="cuda", cache_dir=None, revision=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.revision = revision
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype,
                                                     cache_dir=cache_dir, revision=revision)
        self._finish_init(model)

    @classmethod
    def from_model(cls, model, tokenizer=None, device="cpu"):
        """Wrap an already-constructed GPTNeoX model (used by the integration
        test with random weights, and for custom local checkpoints)."""
        self = cls.__new__(cls)
        self.model_name = getattr(model.config, "name_or_path", "local-gptneox")
        self.device = device
        self.dtype = next(model.parameters()).dtype
        self.tokenizer = tokenizer
        self._finish_init(model)
        return self

    def _finish_init(self, model):
        model.config.use_cache = False
        model.eval()
        model.to(self.device)
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model

    # ---- structure ---------------------------------------------------------
    @property
    def layers(self):
        return self.model.gpt_neox.layers

    @property
    def n_layer(self):
        return len(self.layers)

    @property
    def d_model(self):
        return self.model.config.hidden_size

    @property
    def d_ff(self):
        return self.model.config.intermediate_size

    def n_params(self):
        return sum(p.numel() for p in self.model.parameters())

    def describe(self):
        return (f"{self.model_name}@{getattr(self,'revision',None) or 'main'}: n_layer={self.n_layer}, d_model={self.d_model}, "
                f"d_ff={self.d_ff}, dtype={self.dtype}, params={self.n_params()/1e6:.1f}M")

    def param_stats(self, layer):
        """Parameter/FLOPs accounting for C_eff (eq. 3.33). FLOPs proxy:
        matmul parameters (every nn.Linear weight, incl. embed_out; the
        embedding lookup is excluded), since per-token matmul FLOPs ~ 2 x
        matmul params."""
        def matmul_params(mod):
            return sum(m.weight.numel() for m in mod.modules()
                       if isinstance(m, torch.nn.Linear))
        lp = sum(p.numel() for p in layer.parameters())
        return dict(layer_params=lp, model_params=self.n_params(),
                    layer_matmul=matmul_params(layer),
                    model_matmul=matmul_params(self.model))

    # ---- gated insertion ---------------------------------------------------
    @contextlib.contextmanager
    def inserted(self, bary_layer, insert_after, gamma):
        if not (0 <= insert_after <= self.n_layer - 2):
            raise ValueError(f"insert_after must be in [0, {self.n_layer-2}]")
        target = self.layers[insert_after + 1]

        def hook(module, args, kwargs):
            if len(args) > 0:
                h, rest = args[0], args[1:]
            else:
                h, rest = kwargs.pop("hidden_states"), ()
            if gamma == 0.0:
                h_new = h
            else:
                out = bary_layer(h, *rest, **kwargs)
                h_bar = out[0] if isinstance(out, (tuple, list)) else out
                h_new = h + gamma * (h_bar - h)
            if len(args) > 0:
                return (h_new,) + rest, kwargs
            kwargs["hidden_states"] = h_new
            return args, kwargs

        handle = target.register_forward_pre_hook(hook, with_kwargs=True)
        try:
            yield
        finally:
            handle.remove()

    # ---- loss & KL ---------------------------------------------------------
    @torch.no_grad()
    def logits_and_loss(self, input_ids):
        """Causal-LM logits and mean loss (nats/token). Loss is computed
        manually from the shifted logits so the SAME logits tensor can feed
        the KL computation."""
        input_ids = input_ids.to(self.device)
        logits = self.model(input_ids=input_ids).logits.float()
        pred, tgt = logits[:, :-1], input_ids[:, 1:]
        loss = F.cross_entropy(pred.reshape(-1, pred.size(-1)), tgt.reshape(-1))
        return logits, float(loss)

    @staticmethod
    def kl_to_base(base_logits, cand_logits):
        """Mean_{contexts} KL(p_0 || p_gamma) in nats/token (eq. 3.31), over
        the same predictive positions as the loss."""
        p0 = F.log_softmax(base_logits[:, :-1].float(), dim=-1)
        pg = F.log_softmax(cand_logits[:, :-1].float(), dim=-1)
        return float((p0.exp() * (p0 - pg)).sum(-1).mean())

    @torch.no_grad()
    def per_batch_losses(self, batches, bary_layer=None, insert_after=None, gamma=0.0):
        if bary_layer is None or gamma == 0.0:
            return np.array([self.logits_and_loss(b)[1] for b in batches])
        with self.inserted(bary_layer, insert_after, gamma):
            return np.array([self.logits_and_loss(b)[1] for b in batches])

    # ---- representation drift (eq. 3.32) ----------------------------------
    @torch.no_grad()
    def measure_drift_unit(self, bary_layer, insert_after, batches, eps_h=1e-8):
        """D_rep(1). The probe hook computes G_bar(h) with the layer's true
        call arguments, accumulates ||G_bar(h)-h||_F^2 / (||h||_F^2 + eps),
        and passes h through UNCHANGED, so the probed forward equals the base
        model. For any gamma, D_rep(gamma) = gamma^2 * D_rep(1) exactly
        (identity of Section 3.6, from eq. 3.23)."""
        target = self.layers[insert_after + 1]
        acc = []

        def hook(module, args, kwargs):
            if len(args) > 0:
                h, rest = args[0], args[1:]
            else:
                h, rest = kwargs["hidden_states"], ()
            out = bary_layer(h, *rest, **{k: v for k, v in kwargs.items()
                                          if k != "hidden_states"})
            h_bar = out[0] if isinstance(out, (tuple, list)) else out
            num = (h_bar - h).float().pow(2).sum()
            den = h.float().pow(2).sum() + eps_h
            acc.append(float(num / den))
            return None  # pass-through: base model unchanged

        handle = target.register_forward_pre_hook(hook, with_kwargs=True)
        try:
            for b in batches:
                self.model(input_ids=b.to(self.device))
        finally:
            handle.remove()
        return float(np.mean(acc))

    # ---- latency (for C_eff, eq. 3.33) -------------------------------------
    @torch.no_grad()
    def measure_latency(self, bary_layer, insert_after, batches, n_rep=5):
        def timed(ctx):
            ts = []
            with ctx:
                self.logits_and_loss(batches[0])          # warm-up
                for r in range(n_rep):
                    b = batches[r % len(batches)]
                    if self.device == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    self.logits_and_loss(b)
                    if self.device == "cuda":
                        torch.cuda.synchronize()
                    ts.append(time.perf_counter() - t0)
            return float(np.median(ts))
        t_base = timed(contextlib.nullcontext())
        t_ext = timed(self.inserted(bary_layer, insert_after, 1.0))
        return t_base, t_ext
