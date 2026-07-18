"""Fréchet Inception Distance (FID).

FID fits a multivariate Gaussian to the Inception features of two image sets
(typically generated vs. real) and measures the Fréchet distance between them:

    FID = ||mu_r - mu_g||^2 + Tr(S_r + S_g - 2 (S_r S_g)^{1/2})

Lower is better; 0 means the feature distributions are identical.
"""

import numpy as np
from scipy import linalg


def gaussian_statistics(features):
    """Return ``(mu, sigma)`` for an ``(N, D)`` array of features."""
    features = np.asarray(features, dtype=np.float64)
    mu = features.mean(axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Fréchet distance between two multivariate Gaussians."""
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        # Product of covariances is singular; nudge the diagonal and retry.
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def fid_from_features(features1, features2):
    """Compute FID directly from two sets of Inception features."""
    mu1, sigma1 = gaussian_statistics(features1)
    mu2, sigma2 = gaussian_statistics(features2)
    return frechet_distance(mu1, sigma1, mu2, sigma2)


def compute_fid(stats1, stats2):
    """Compute FID from two precomputed ``(mu, sigma)`` tuples."""
    (mu1, sigma1), (mu2, sigma2) = stats1, stats2
    return frechet_distance(mu1, sigma1, mu2, sigma2)
