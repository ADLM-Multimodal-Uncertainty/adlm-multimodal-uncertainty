"""Independent seeds on a shared split.

Members retain ordinary run artifacts. Aggregation of their probabilities lives in
`src.evaluate` so this module does not import the evaluation code.
"""

from __future__ import annotations

import numpy as np


class DeepEnsemble:
    def __init__(self, seeds: list[int]):
        self.seeds = list(seeds)
        if len(self.seeds) < 2 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("deep ensembles require at least two distinct seeds")
        if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in self.seeds):
            raise ValueError("seeds must be integers in [0, 2**32)")


def uncertainty(member_probs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return predictive entropy, expected member entropy, and mutual information.

    Entropies use natural logarithms (nats) and ``0 log 0 = 0``. Inputs are cast
    to float64 first: clipping a float32 zero at the float64 minimum underflows
    back to zero, and ``0 * log(0)`` is NaN.
    """
    member_probs = np.asarray(member_probs, dtype=np.float64)

    def entropy(p: np.ndarray) -> np.ndarray:
        logged = np.zeros_like(p)
        np.log(p, out=logged, where=p > 0)
        return -(p * logged).sum(axis=-1)

    total = entropy(member_probs.mean(axis=0))
    aleatoric = entropy(member_probs).mean(axis=0)
    return total, aleatoric, np.maximum(total - aleatoric, 0)
