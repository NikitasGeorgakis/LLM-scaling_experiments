"""Optimal-Transport Barycentric Layer Insertion (OT-BLI) experiments.

Companion code for:
  N. Georgakis & V. Panaretos, "Scaling without re-training" (EPFL, 2026).
"""
__version__ = "0.1.0"


def load_model(name: str, device: str = "cuda", revision: str = None,
               dtype=None):
    """Load any supported CausalLM in float32 (Section 6.1) together with its
    own tokenizer: GPT-NeoX/Pythia, GPT-2, and Llama-family gated models
    (Llama / Mistral / TinyLlama / Qwen2-style). `revision` selects an
    intermediate training checkpoint where the hub provides one (e.g. Pythia
    'step512' ... 'step143000' for the Section-6.5 trajectory)."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    kw = {"torch_dtype": dtype if dtype is not None else torch.float32}
    if revision is not None:
        kw["revision"] = revision
    model = AutoModelForCausalLM.from_pretrained(name, **kw).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(name, **({"revision": revision} if revision else {}))
    return model, tok


def load_pythia(*args, **kwargs):
    """Backward-compatible alias of load_model."""
    return load_model(*args, **kwargs)
