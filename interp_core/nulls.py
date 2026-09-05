from __future__ import annotations

import hashlib
import numpy as np


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def stable_seed(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "big") % (2 ** 31)


def random_subset_from_pool(pool: list[int], k: int, seed: int) -> set[int]:
    r = rng(seed)
    pool = list(dict.fromkeys(pool))
    if len(pool) <= k:
        return set(pool)
    return set(r.choice(pool, size=k, replace=False).tolist())


def shuffled_partner(n: int, i: int, seed: int) -> int:
    if n < 2:
        raise ValueError("Need at least 2 images for shuffled pairing")
    r = rng(seed + i)
    j = int(r.integers(0, n - 1))
    return j + 1 if j >= i else j


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def strict_subset(indices: list[int], values: list[float], factor: float = 2.0) -> set[int]:
    if not indices:
        return set()
    thresh = min(values) * factor
    return {idx for idx, v in zip(indices, values) if v > thresh}
