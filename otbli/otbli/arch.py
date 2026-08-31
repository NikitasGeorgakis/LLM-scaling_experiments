"""Model-tree navigation for the supported architectures.

The atomization (otbli/atomize.py) is layer-local; this module locates the
layer stack inside the various HF model wrappers so that insertion, screening
and diagnostics are architecture-agnostic.
"""

_LAYER_PATHS = ("gpt_neox.layers",   # GPTNeoXForCausalLM (Pythia)
                "model.layers",      # Llama / Mistral / TinyLlama / Qwen2-style
                "transformer.h")     # GPT2LMHeadModel


def get_layers(model):
    """Return the ModuleList of transformer blocks of a supported model."""
    for path in _LAYER_PATHS:
        obj, ok = model, True
        for part in path.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok:
            return obj
    raise ValueError(f"Unsupported model tree {type(model).__name__}; "
                     f"expected one of {_LAYER_PATHS} (see ARCHITECTURES.md)")
