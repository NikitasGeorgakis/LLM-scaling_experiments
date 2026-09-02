"""
Validation-data pools for the two-stage protocol (paper eq. 3.48).

The token stream is split in HALF: selection contexts come only from the first
half, confirmation contexts only from the second half (disjoint text), so the
confirmation test is honestly out-of-sample for the selected candidate.

DATASETS
  pile      (default) NeelNanda/pile-10k -- a standard 10k-document random
            sample of the Pile, i.e. Pythia's own pretraining distribution.
            This is the right choice for headline numbers.
  wikitext  WikiText-103 validation. Reliable but a DISTRIBUTION SHIFT
            relative to Pythia pretraining: paired comparisons remain valid,
            absolute losses will be higher and effect sizes may differ.
Documents are shuffled with a fixed seed before tokenization so both halves
mix domains.
"""
import torch


def load_token_stream(tokenizer, dataset="pile", max_tokens=2_000_000,
                      cache_dir=None):
    import random
    from datasets import load_dataset

    # trust_remote_code=True is defensive: some canonical HF datasets (wikitext
    # historically among them) ship a loading script rather than plain data
    # files, and recent `datasets` versions require explicit trust for that.
    # Falls back cleanly on older `datasets` versions that lack the kwarg.
    # This mirrors pythia_prefetch.py exactly, so a successful prefetch run
    # guarantees this call will not fail for the same reason.
    def _load(*a, **kw):
        try:
            return load_dataset(*a, **kw, trust_remote_code=True)
        except TypeError:
            return load_dataset(*a, **kw)

    if dataset == "pile":
        ds = _load("NeelNanda/pile-10k", split="train", cache_dir=cache_dir)
        texts = [t for t in ds["text"] if t.strip()]
    elif dataset == "wikitext":
        ds = _load("wikitext", "wikitext-103-raw-v1", split="validation",
                   cache_dir=cache_dir)
        texts = [t for t in ds["text"] if t.strip()]
    else:
        raise ValueError(f"unknown dataset {dataset!r}")

    random.Random(123).shuffle(texts)

    # we tokenize whole documents into a flat stream, so per-call length limits
    # are irrelevant; raising the limit silences a benign length warning
    try:
        tokenizer.model_max_length = 10 ** 9
    except Exception:
        pass

    ids = []
    for t in texts:
        ids.extend(tokenizer(t).input_ids)
        if len(ids) >= max_tokens:
            break
    ids = ids[:max_tokens]
    print(f"[data] {dataset}: {len(ids):,} tokens from {len(texts)} docs (shuffled, seed 123)")
    return torch.tensor(ids, dtype=torch.long)


def make_pools(stream, block_size=1024, batch=8,
               sel_batches=25, conf_batches=40,
               sel_seed=777, conf_seed=999):
    """(selection_batches, confirmation_batches): lists of LongTensors of
    shape (batch, block_size), drawn from disjoint halves of the stream."""
    half = len(stream) // 2
    sel_stream, conf_stream = stream[:half], stream[half:]

    def pool(s, n_batches, seed):
        g = torch.Generator().manual_seed(seed)
        hi = len(s) - block_size - 1
        if hi <= 0:
            raise ValueError("token stream too short for this block_size")
        out = []
        for _ in range(n_batches):
            ix = torch.randint(hi, (batch,), generator=g)
            out.append(torch.stack([s[i:i + block_size] for i in ix]))
        return out

    return pool(sel_stream, sel_batches, sel_seed), pool(conf_stream, conf_batches, conf_seed)
