"""Held-out Pile pools (Section 6.1).

Text is tokenized with each model's own tokenizer, document-shuffled with a
fixed seed, concatenated with EOS separators, and packed into blocks of 1,024
tokens. The stream is split into a SELECTION pool of 25 paired batches of
8 x 1,024 tokens (204,800 tokens) and a DISJOINT CONFIRMATION pool of 40
batches (327,680 tokens). All candidates are evaluated on identical batches
(paired design), so the pools are materialized once and cached to disk.
"""
import json
import os
import torch


def _doc_stream(dataset: str, seed: int):
    """Yield raw documents. `dataset` may be a local .jsonl/.txt path (one doc
    per line, {'text': ...} or plain text) or a HuggingFace dataset name."""
    if os.path.exists(dataset):
        def gen():
            with open(dataset, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line).get("text", "")
                    except json.JSONDecodeError:
                        yield line
        return gen()
    from datasets import load_dataset
    ds = load_dataset(dataset, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)   # document shuffle, fixed seed
    return (ex["text"] for ex in ds)


@torch.no_grad()
def build_pools(tokenizer, cfg):
    """Return (selection, confirmation) tensors of shape
    [n_batches, batch_size, block_len]."""
    need_blocks = (cfg.n_sel_batches + cfg.n_conf_batches) * cfg.batch_size
    eos = tokenizer.eos_token_id
    buf, blocks = [], []
    for text in _doc_stream(cfg.dataset, cfg.seed):
        if not text:
            continue
        buf.extend(tokenizer(text, add_special_tokens=False).input_ids)
        buf.append(eos)
        while len(buf) >= cfg.block_len and len(blocks) < need_blocks:
            blocks.append(torch.tensor(buf[:cfg.block_len], dtype=torch.long))
            buf = buf[cfg.block_len:]
        if len(blocks) >= need_blocks:
            break
    if len(blocks) < need_blocks:
        raise RuntimeError(
            f"dataset stream exhausted after {len(blocks)}/{need_blocks} blocks")
    T = torch.stack(blocks)
    B = cfg.batch_size
    S, C = cfg.n_sel_batches, cfg.n_conf_batches
    sel = T[:S * B].view(S, B, cfg.block_len).contiguous()
    conf = T[S * B:(S + C) * B].view(C, B, cfg.block_len).contiguous()
    return sel, conf


def load_or_build_pools(tokenizer, cfg, cache_path: str):
    """Materialize pools once per (tokenizer, seed) and reuse them for every
    candidate and for the (single) confirmation run."""
    if cache_path and os.path.exists(cache_path):
        blob = torch.load(cache_path, map_location="cpu")
        return blob["selection"], blob["confirmation"]
    sel, conf = build_pools(tokenizer, cfg)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        torch.save({"selection": sel, "confirmation": conf,
                    "dataset": cfg.dataset, "seed": cfg.seed}, cache_path)
    return sel, conf
