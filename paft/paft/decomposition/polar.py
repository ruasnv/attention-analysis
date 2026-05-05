"""
Polar decomposition for PAFT.

For W_V_h  [n_embd, d_head]  — tall matrix (n_embd > d_head):
    Right polar:  W_V_h = Q_V_h @ S_V_h
        Q_V_h : [n_embd, d_head]  semi-unitary  (Q^T Q = I_{d_head})
        S_V_h : [d_head, d_head]  symmetric PSD

For W_O_h  [d_head, n_embd]  — wide matrix (d_head < n_embd):
    Left polar:   W_O_h = S_O_h @ Q_O_h
        S_O_h : [d_head, d_head]  symmetric PSD
        Q_O_h : [d_head, n_embd]  semi-unitary rows  (Q Q^T = I_{d_head})

In both cases S lives in [d_head, d_head] — the small per-head space.
Q is frozen; S is the trainable component in pure/hybrid PAFT.
"""

import torch
from typing import Tuple


def safe_svd(
    W: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    SVD with a fallback perturbation for numerical stability.

    Returns U, sigma, Vh such that W = U @ diag(sigma) @ Vh.
    Uses full_matrices=False (economy / thin SVD).
    """
    try:
        U, sigma, Vh = torch.linalg.svd(W, full_matrices=False)
    except torch.linalg.LinAlgError:
        # Rare on well-conditioned pretrained weights, but guard anyway.
        W_nudged = W + 1e-6 * torch.randn_like(W)
        U, sigma, Vh = torch.linalg.svd(W_nudged, full_matrices=False)
    return U, sigma, Vh


def polar_decompose(
    W: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Right polar decomposition for a tall (or square) matrix.

        W [m, n]  ->  Q [m, n],  S [n, n]    where m >= n
        W = Q @ S
        Q^T Q = I_n   (orthonormal columns)
        S symmetric PSD, eigenvalues = singular values of W

    Used for W_V_h  [n_embd, d_head].

    Derivation via SVD  W = U Sigma Vh:
        Q = U @ Vh           absorbs both unitary factors
        S = Vh.T @ diag(sigma) @ Vh    symmetric PSD
    Check:  Q @ S = U Vh Vh.T diag(sigma) Vh = U diag(sigma) Vh = W
    """
    assert W.ndim == 2, f"Expected 2-D tensor, got {W.shape}"
    m, n = W.shape
    if m < n:
        raise ValueError(
            f"polar_decompose (right) requires m >= n, got {W.shape}. "
            "Use polar_decompose_left for wide matrices."
        )
    U, sigma, Vh = safe_svd(W)              # U[m,n]  sigma[n]  Vh[n,n]
    Q = U @ Vh                              # [m, n]
    S = Vh.T @ torch.diag(sigma) @ Vh      # [n, n]
    return Q, S


def polar_decompose_left(
    W: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Left polar decomposition for a wide (or square) matrix.

        W [m, n]  ->  S [m, m],  Q [m, n]    where m <= n
        W = S @ Q
        Q @ Q^T = I_m   (orthonormal rows)
        S symmetric PSD, eigenvalues = singular values of W

    Used for W_O_h  [d_head, n_embd].

    Derivation via SVD  W = U Sigma Vh:
        Q = U @ Vh           [m, n] with orthonormal rows
        S = U @ diag(sigma) @ U.T    symmetric PSD
    Check:  S @ Q = U diag(sigma) U.T U Vh = U diag(sigma) Vh = W
    """
    assert W.ndim == 2, f"Expected 2-D tensor, got {W.shape}"
    m, n = W.shape
    if m > n:
        raise ValueError(
            f"polar_decompose_left requires m <= n, got {W.shape}. "
            "Use polar_decompose for tall matrices."
        )
    U, sigma, Vh = safe_svd(W)              # U[m,m]  sigma[m]  Vh[m,n]
    Q = U @ Vh                              # [m, n]
    S = U @ torch.diag(sigma) @ U.T        # [m, m]
    return S, Q


def reconstruct(Q: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """Reconstruct W = Q @ S  (right polar — used for W_V)."""
    return Q @ S


def reconstruct_left(S: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """Reconstruct W = S @ Q  (left polar — used for W_O)."""
    return S @ Q