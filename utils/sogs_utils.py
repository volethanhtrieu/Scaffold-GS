"""Second-order anchor feature processing used by the optional SOGS path.

The paper treats the ``D`` channels of the ``N x D`` anchor-feature matrix as
random variables and the anchors as observations.  The global correlation
matrix is decomposed once per feature evaluation; each selected eigenvector is
then paired with every anchor feature and passed through a small learned
two-layer MLP.  The resulting ``M`` branches are concatenated with the
original feature, giving ``D * (1 + M)`` render features.

This module intentionally has no CUDA assumptions.  All temporary tensors
inherit the input feature device and statistics are promoted to at least
float32 for stable covariance/eigendecomposition in mixed precision.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import torch
from torch import nn


class SecondOrderStatistics(NamedTuple):
    """Statistics and principal directions for an ``N x D`` feature matrix."""

    mean: torch.Tensor
    covariance: torch.Tensor
    correlation: torch.Tensor
    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor


def validate_second_order_dimensions(
    feat_dim: int, num_eigenvectors: int, *, enabled: bool = True
) -> None:
    """Validate dimensions shared by the model and standalone tests."""

    if not isinstance(feat_dim, int) or isinstance(feat_dim, bool) or feat_dim <= 0:
        raise ValueError("feat_dim must be a positive integer")
    if (
        not isinstance(num_eigenvectors, int)
        or isinstance(num_eigenvectors, bool)
        or num_eigenvectors < 0
    ):
        raise ValueError("num_eigenvectors must be a non-negative integer")
    if enabled and num_eigenvectors < 1:
        raise ValueError(
            "num_eigenvectors must be at least 1 when second-order features are enabled"
        )
    if num_eigenvectors > feat_dim:
        raise ValueError(
            "num_eigenvectors must not exceed feat_dim "
            f"({num_eigenvectors} > {feat_dim})"
        )


def _statistics_dtype(features: torch.Tensor) -> torch.dtype:
    """Return a dtype supported reliably by covariance/eigh operations."""

    if features.dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return features.dtype


def compute_second_order_statistics(
    anchor_features: torch.Tensor,
    num_eigenvectors: int,
    *,
    eps: float = 1e-6,
) -> SecondOrderStatistics:
    """Compute correlation statistics and the top principal co-variations.

    Parameters
    ----------
    anchor_features:
        A floating-point tensor with shape ``[N, D]``.  ``N`` is the number of
        anchors and ``D`` is the base feature dimension.
    num_eigenvectors:
        Number ``M`` of eigenvectors to return.  ``0`` is accepted by this
        standalone function and returns an empty ``[D, 0]`` selection.
    eps:
        Positive floor used when normalizing zero-variance channels.

    Returns
    -------
    SecondOrderStatistics
        ``mean`` has shape ``[D]``, covariance/correlation have shape
        ``[D, D]``, eigenvalues have shape ``[M]``, and eigenvectors have shape
        ``[D, M]``.

    Raises
    ------
    ValueError
        If the input is not a finite floating-point matrix or the dimensions
        are invalid.
    """

    if not isinstance(anchor_features, torch.Tensor):
        raise TypeError("anchor_features must be a torch.Tensor")
    if anchor_features.ndim != 2:
        raise ValueError(
            "anchor_features must have shape [N, D] "
            f"(received {tuple(anchor_features.shape)})"
        )
    if not anchor_features.is_floating_point():
        raise ValueError("anchor_features must use a floating-point dtype")
    if not torch.isfinite(anchor_features).all().item():
        raise ValueError("anchor_features contains NaN or infinite values")
    if not isinstance(num_eigenvectors, int) or isinstance(
        num_eigenvectors, bool
    ):
        raise ValueError("num_eigenvectors must be an integer")
    if num_eigenvectors < 0 or num_eigenvectors > anchor_features.shape[1]:
        raise ValueError(
            "num_eigenvectors must satisfy 0 <= num_eigenvectors <= feat_dim"
        )
    if not isinstance(eps, (float, int)) or eps <= 0:
        raise ValueError("eps must be positive")

    n_anchors, feat_dim = anchor_features.shape
    stats_dtype = _statistics_dtype(anchor_features)
    # PAPER: center the scene-wide anchor matrix before computing covariance.
    features = anchor_features.to(dtype=stats_dtype)
    if n_anchors:
        mean = features.mean(dim=0)
        centered = features - mean
        # PAPER: Eq. (4) uses the unbiased 1/(N-1) covariance estimator.
        denominator = float(max(n_anchors - 1, 1))
        covariance = centered.transpose(0, 1).matmul(centered) / denominator
    else:
        # INFERENCE: an empty scene is represented by zero covariance and an
        # identity correlation matrix so shape-only callers remain well-defined.
        mean = torch.zeros(
            (feat_dim,), dtype=stats_dtype, device=anchor_features.device
        )
        covariance = torch.zeros(
            (feat_dim, feat_dim),
            dtype=stats_dtype,
            device=anchor_features.device,
        )

    covariance = torch.nan_to_num(covariance, nan=0.0, posinf=0.0, neginf=0.0)
    covariance = (covariance + covariance.transpose(0, 1)) * 0.5

    variances = torch.diagonal(covariance, dim1=0, dim2=1).clamp_min(0.0)
    standard_deviation = variances.sqrt()
    denominator = standard_deviation[:, None] * standard_deviation[None, :]
    valid = (standard_deviation[:, None] > eps) & (
        standard_deviation[None, :] > eps
    )
    correlation = torch.where(
        valid,
        covariance / denominator.clamp_min(float(eps) ** 2),
        torch.zeros_like(covariance),
    )
    # INFERENCE: zero-variance channels have no measurable pairwise
    # correlation; setting their diagonal to one keeps the matrix a valid
    # correlation matrix and makes the single-anchor case numerically safe.
    identity = torch.eye(
        feat_dim, dtype=stats_dtype, device=anchor_features.device
    )
    correlation = correlation * (1.0 - identity) + identity
    correlation = torch.nan_to_num(
        correlation, nan=0.0, posinf=0.0, neginf=0.0
    )
    correlation = (correlation + correlation.transpose(0, 1)) * 0.5

    if num_eigenvectors == 0:
        return SecondOrderStatistics(
            mean=mean,
            covariance=covariance,
            correlation=correlation,
            eigenvalues=torch.empty(
                (0,), dtype=stats_dtype, device=anchor_features.device
            ),
            eigenvectors=torch.empty(
                (feat_dim, 0), dtype=stats_dtype, device=anchor_features.device
            ),
        )

    # INFERENCE: a tiny deterministic diagonal tie-breaker avoids undefined
    # eigenvector gradients for a constant/single-anchor feature matrix while
    # leaving the reported correlation matrix unchanged.
    tie_breaker = torch.arange(
        feat_dim, dtype=stats_dtype, device=anchor_features.device
    )
    tie_breaker = torch.diag(tie_breaker) * (float(eps) * 10.0)
    eig_matrix = correlation + tie_breaker
    eig_matrix = (eig_matrix + eig_matrix.transpose(0, 1)) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(eig_matrix)
    order = torch.argsort(eigenvalues, descending=True)
    order = order[:num_eigenvectors]
    # Keep the reported eigenvalues tied to the unperturbed correlation
    # matrix.  The infinitesimal tie-breaker is only an INFERENCE-time
    # numerical aid for selecting a stable basis in degenerate eigenspaces.
    reference_eigenvalues = torch.linalg.eigvalsh(correlation)
    reference_eigenvalues = torch.sort(
        reference_eigenvalues, descending=True
    ).values
    selected_values = reference_eigenvalues[:num_eigenvectors].clamp_min(0.0)
    selected_vectors = eigenvectors.index_select(1, order)
    selected_vectors = torch.nan_to_num(
        selected_vectors, nan=0.0, posinf=0.0, neginf=0.0
    )

    return SecondOrderStatistics(
        mean=mean,
        covariance=covariance,
        correlation=correlation,
        eigenvalues=selected_values,
        eigenvectors=selected_vectors,
    )


class SecondOrderFeatureAugmentor(nn.Module):
    """Learned feature augmentation for SOGS.

    ``forward`` accepts ``[N, D]`` base features and returns
    ``[N, D * (1 + M)]``.  The ``M`` branch MLPs are trainable and are exposed
    as normal module parameters so callers can add them to an optimizer and
    checkpoint state.
    """

    def __init__(
        self,
        feat_dim: int,
        num_eigenvectors: int = 2,
        *,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        validate_second_order_dimensions(
            feat_dim, num_eigenvectors, enabled=num_eigenvectors > 0
        )
        self.feat_dim = feat_dim
        self.num_eigenvectors = num_eigenvectors
        hidden = feat_dim if hidden_dim is None else int(hidden_dim)
        if hidden <= 0:
            raise ValueError("hidden_dim must be positive")
        # PAPER: each Fi is a two-layer MLP with ReLU activation.  The paper
        # does not prescribe a hidden width; using D preserves local MLP scale.
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * feat_dim, hidden),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden, feat_dim),
                )
                for _ in range(num_eigenvectors)
            ]
        )

    @property
    def augmented_dim(self) -> int:
        """The final feature width ``D * (1 + M)``."""

        return self.feat_dim * (1 + self.num_eigenvectors)

    def forward(self, anchor_features: torch.Tensor) -> torch.Tensor:
        if not isinstance(anchor_features, torch.Tensor):
            raise TypeError("anchor_features must be a torch.Tensor")
        if anchor_features.ndim != 2:
            raise ValueError(
                "anchor_features must have shape [N, D] "
                f"(received {tuple(anchor_features.shape)})"
            )
        if anchor_features.shape[1] != self.feat_dim:
            raise ValueError(
                f"expected base feature dimension {self.feat_dim}, "
                f"received {anchor_features.shape[1]}"
            )
        if not anchor_features.is_floating_point():
            raise ValueError("anchor_features must use a floating-point dtype")
        if not torch.isfinite(anchor_features).all().item():
            raise ValueError("anchor_features contains NaN or infinite values")
        if self.num_eigenvectors == 0:
            return anchor_features

        statistics = compute_second_order_statistics(
            anchor_features, self.num_eigenvectors
        )
        n_anchors = anchor_features.shape[0]
        branches = []
        for index, branch in enumerate(self.branches):
            eigenvector = statistics.eigenvectors[:, index].to(
                device=anchor_features.device, dtype=anchor_features.dtype
            )
            eigenvector = eigenvector.unsqueeze(0).expand(n_anchors, -1)
            # PAPER: combine the global principal direction Pi and local fa.
            branch_input = torch.cat([eigenvector, anchor_features], dim=-1)
            # INFERENCE: promote branch input to the parameter dtype for
            # float16/bfloat16 callers, then return each branch in the source
            # feature dtype.
            parameter = next(branch.parameters(), None)
            if parameter is not None:
                branch_input = branch_input.to(dtype=parameter.dtype)
            branch_output = branch(branch_input).to(dtype=anchor_features.dtype)
            branches.append(branch_output)

        augmented_features = torch.cat([anchor_features] + branches, dim=-1)
        expected_dim = self.feat_dim * (1 + self.num_eigenvectors)
        assert augmented_features.shape[-1] == expected_dim
        if not torch.isfinite(augmented_features).all().item():
            raise ValueError(
                "second-order feature augmentation produced NaN or infinite values"
            )
        return augmented_features


# COMPATIBILITY: descriptive alias for integrations that call the learned
# feature module a "second-order anchor".
SecondOrderAnchor = SecondOrderFeatureAugmentor


def augment_anchor_features(
    anchor_features: torch.Tensor,
    num_eigenvectors: int = 2,
    *,
    augmentor: Optional[SecondOrderFeatureAugmentor] = None,
) -> torch.Tensor:
    """Functional convenience wrapper used by tests and small integrations."""

    if augmentor is None:
        augmentor = SecondOrderFeatureAugmentor(
            anchor_features.shape[-1], num_eigenvectors
        ).to(device=anchor_features.device)
    return augmentor(anchor_features)


__all__ = [
    "SecondOrderFeatureAugmentor",
    "SecondOrderAnchor",
    "SecondOrderStatistics",
    "augment_anchor_features",
    "compute_second_order_statistics",
    "validate_second_order_dimensions",
]
