"""Space analyzer — profiles the geometry of the candidate embedding space.

Computes variance, anisotropy, density classification, and score spread
to help the router understand the structure of the search space.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from refract.types import SpaceProfile


def _classify_density(variance: float) -> Literal["sparse", "medium", "dense"]:
    """Classify space density from embedding variance.

    Args:
        variance: Mean variance across embedding dimensions.

    Returns:
        Density category: "sparse", "medium", or "dense".
    """
    if variance < 0.05:
        return "dense"
    elif variance > 0.20:
        return "sparse"
    else:
        return "medium"


def _compute_anisotropy(candidates: np.ndarray) -> float:
    """Compute anisotropy ratio from candidate embeddings.

    Anisotropy is the ratio of the largest to smallest eigenvalue of
    the covariance matrix. Higher values mean the embedding space is
    more "cone-shaped" and cosine similarity is less discriminative.

    For large candidate pools (>5000), subsamples 2000 vectors for speed.

    Args:
        candidates: Candidate matrix of shape ``(n, dim)``.

    Returns:
        Anisotropy ratio (≥ 1.0). Returns 1.0 for degenerate cases.
    """
    n, _dim = candidates.shape

    if n < 2:
        return 1.0

    # Subsample for large corpora
    vecs = candidates
    if n > 5000:
        rng = np.random.default_rng(42)
        indices = rng.choice(n, size=2000, replace=False)
        vecs = vecs[indices]

    # For high-dimensional data where n < dim, compute on smaller matrix
    if vecs.shape[0] < vecs.shape[1]:
        # Use Gram matrix trick: eigenvalues of X@X.T / (n-1) are same as X.T@X / (n-1)
        centered = vecs - vecs.mean(axis=0)
        gram = centered @ centered.T / (len(centered) - 1)
        eigenvalues = np.linalg.eigvalsh(gram)
    else:
        cov = np.cov(vecs.T)
        if cov.ndim == 0:
            return 1.0
        eigenvalues = np.linalg.eigvalsh(cov)

    # Filter out near-zero eigenvalues
    pos_eigenvalues = eigenvalues[eigenvalues > 1e-10]
    if len(pos_eigenvalues) < 2:
        return 1.0

    ratio = float(pos_eigenvalues[-1] / pos_eigenvalues[0])
    return max(ratio, 1.0)


def analyze_space(
    candidates: np.ndarray,
    cosine_scores: np.ndarray | None = None,
) -> SpaceProfile:
    """Analyze the candidate embedding space.

    Computes geometric properties of the candidate pool that inform
    the router's metric weight decisions.

    Args:
        candidates: Candidate matrix of shape ``(n, dim)``.
        cosine_scores: Pre-computed cosine scores for score_spread.
            If None, score_spread is set to 0.0.

    Returns:
        A SpaceProfile with all geometry features.
    """
    n_candidates, embedding_dim = candidates.shape

    # Compute per-dimension variance and average
    variance = float(np.var(candidates, axis=0).mean())

    # Compute anisotropy
    anisotropy = _compute_anisotropy(candidates)

    # Classify density
    density = _classify_density(variance)

    # Score spread (if cosine scores available)
    if cosine_scores is not None and len(cosine_scores) > 0:
        score_spread = float(np.std(cosine_scores))
    else:
        score_spread = 0.0

    return SpaceProfile(
        n_candidates=n_candidates,
        embedding_dim=embedding_dim,
        variance=variance,
        anisotropy=anisotropy,
        density=density,
        score_spread=score_spread,
    )
