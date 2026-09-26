"""Confidence-region geometry for Gaussian estimates (pure NumPy, no plotting).

For a 2D Gaussian, the region {e : e^T P^-1 e <= k^2} holds probability p when
k^2 is the chi-square(2 DOF) quantile, which has the closed form
k = sqrt(-2 ln(1 - p)). Note that k = 1 (a "1-sigma ellipse") covers only ~39%
of the probability in 2D, so ellipses are parameterized by confidence level.
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np


def chi2_2dof_scale(confidence: float) -> float:
    """Mahalanobis radius k whose 2D ellipse contains `confidence` probability."""
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    return float(np.sqrt(-2.0 * np.log(1.0 - confidence)))


def normal_1d_scale(confidence: float) -> float:
    """Two-sided scale k with P(|x| <= k sigma) = `confidence` for a 1D Gaussian."""
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    return NormalDist().inv_cdf(0.5 * (1.0 + confidence))


def covariance_ellipse(covariance: np.ndarray, confidence: float = 0.95) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (width, height, angle_deg) of the `confidence` ellipse of 2x2 covariance(s).

    `covariance` has shape (..., 2, 2); outputs have shape (...). `width`/`height`
    are full axis lengths (as `matplotlib.patches.Ellipse` expects), with `width`
    along the major axis, and `angle_deg` is the major axis angle from +x.
    NaN covariances give NaN outputs.
    """
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape[-2:] != (2, 2):
        raise ValueError(f"expected (..., 2, 2) covariance, got shape {covariance.shape}")

    k = chi2_2dof_scale(confidence)
    valid = np.isfinite(covariance).all(axis=(-2, -1))
    safe = np.where(valid[..., None, None], covariance, np.eye(2))
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (safe + np.swapaxes(safe, -1, -2)))
    eigenvalues = np.clip(eigenvalues, 0.0, None)

    # eigh sorts ascending, so the last column is the major axis.
    major = eigenvectors[..., :, 1]
    width = 2.0 * k * np.sqrt(eigenvalues[..., 1])
    height = 2.0 * k * np.sqrt(eigenvalues[..., 0])
    angle_deg = np.rad2deg(np.arctan2(major[..., 1], major[..., 0]))

    nan = np.where(valid, 0.0, np.nan)
    return width + nan, height + nan, angle_deg + nan


def mahalanobis_squared(error: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """e^T P^-1 e for errors (..., n) and covariances (..., n, n); NaN inputs give NaN."""
    error = np.asarray(error, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    valid = np.isfinite(error).all(axis=-1) & np.isfinite(covariance).all(axis=(-2, -1))
    n = error.shape[-1]
    safe_error = np.where(valid[..., None], error, 0.0)
    safe_covariance = np.where(valid[..., None, None], covariance, np.eye(n))
    solved = np.linalg.solve(safe_covariance, safe_error[..., None])[..., 0]
    result = np.einsum("...i,...i->...", safe_error, solved)
    return np.where(valid, result, np.nan)
