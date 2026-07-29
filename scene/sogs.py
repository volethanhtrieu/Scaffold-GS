"""Compatibility re-export for callers that expect SOGS under ``scene``."""

# COMPATIBILITY: the implementation lives in utils.sogs_utils so CPU-only
# feature tests can import it without initializing the full Scene package.
from utils.sogs_utils import (  # noqa: F401
    SecondOrderFeatureAugmentor,
    SecondOrderAnchor,
    SecondOrderStatistics,
    augment_anchor_features,
    compute_second_order_statistics,
    validate_second_order_dimensions,
)

__all__ = [
    "SecondOrderFeatureAugmentor",
    "SecondOrderAnchor",
    "SecondOrderStatistics",
    "augment_anchor_features",
    "compute_second_order_statistics",
    "validate_second_order_dimensions",
]
