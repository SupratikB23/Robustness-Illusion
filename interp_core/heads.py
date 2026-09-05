from __future__ import annotations

import numpy as np


def topk_heads_by_norm(attn_weights, k: int) -> tuple[list[int], list[float]]:
    import torch
    if isinstance(attn_weights, np.ndarray):
        attn_weights = torch.from_numpy(attn_weights)
    scores = attn_weights.float().reshape(attn_weights.shape[0], -1).norm(dim=-1)
    vals, idx = torch.topk(scores, min(k, scores.numel()))
    return idx.cpu().tolist(), vals.cpu().tolist()


def topk_neurons_by_activation(acts, k: int) -> tuple[list[int], list[float]]:
    import torch
    flat = acts.detach().float().reshape(-1, acts.shape[-1]).abs().max(dim=0).values
    vals, idx = torch.topk(flat, min(k, flat.numel()))
    return idx.cpu().tolist(), vals.cpu().tolist()
