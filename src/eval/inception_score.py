"""Inception Score (IS).

The Inception Score measures both the confidence of per-image class
predictions and the diversity across generated images:

    IS = exp( E_x [ KL( p(y|x) || p(y) ) ] )

where ``p(y|x)`` is the softmax output of Inception-v3 for image ``x`` and
``p(y)`` is the marginal over the sample set. Higher is better. The score is
averaged over ``splits`` disjoint chunks to also report its standard deviation.
"""

import numpy as np


def inception_score_from_probs(probs, splits=10):
    """Compute ``(mean, std)`` Inception Score from softmax probabilities.

    Args:
        probs: ``(N, num_classes)`` array of per-image class probabilities.
        splits: Number of chunks used to estimate the score's variance.
    """
    probs = np.asarray(probs, dtype=np.float64)
    n = probs.shape[0]
    if n < splits:
        splits = max(1, n)

    scores = []
    for k in range(splits):
        part = probs[k * n // splits:(k + 1) * n // splits]
        py = part.mean(axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-16) - np.log(py + 1e-16))
        kl = kl.sum(axis=1).mean()
        scores.append(np.exp(kl))

    return float(np.mean(scores)), float(np.std(scores))


def compute_inception_score(probs, splits=10):
    """Alias kept for API symmetry with :func:`compute_fid`."""
    return inception_score_from_probs(probs, splits=splits)
