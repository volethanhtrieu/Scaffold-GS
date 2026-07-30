"""Small, dependency-free helpers for bounded training runs."""

from __future__ import annotations

import math
import time
from typing import Callable, Iterable, Optional


def validate_max_runtime_minutes(value: float = 0.0) -> float:
    """Return a finite, non-negative wall-clock budget in minutes.

    A value of zero disables the runtime limit.
    """

    if isinstance(value, bool):
        raise ValueError("max_runtime_minutes must be a non-negative number")
    try:
        minutes = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "max_runtime_minutes must be a non-negative number"
        ) from error
    if not math.isfinite(minutes) or minutes < 0:
        raise ValueError("max_runtime_minutes must be finite and non-negative")
    return minutes


def validate_checkpoint_interval(value: int = 0) -> int:
    """Return a non-negative checkpoint interval; zero disables it."""

    if isinstance(value, bool):
        raise ValueError("checkpoint_interval must be a non-negative integer")
    try:
        interval = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "checkpoint_interval must be a non-negative integer"
        ) from error
    if interval != value or interval < 0:
        raise ValueError("checkpoint_interval must be a non-negative integer")
    return interval


def should_checkpoint_iteration(
    iteration: int,
    explicit_iterations: Iterable[int],
    interval: int,
) -> bool:
    """Return whether an iteration is explicitly or periodically checkpointed."""

    return iteration in explicit_iterations or (
        interval > 0 and iteration % interval == 0
    )


class TrainingBudget:
    """Track one process-wide monotonic training deadline.

    The deadline is created before scene initialization so data/model setup is
    included in the requested wall-clock budget. A disabled budget never
    expires.
    """

    def __init__(
        self,
        max_runtime_minutes: float = 0.0,
        *,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self.max_runtime_minutes = validate_max_runtime_minutes(
            max_runtime_minutes
        )
        self._clock = time.monotonic if clock is None else clock
        self._deadline = (
            self._clock() + self.max_runtime_minutes * 60.0
            if self.max_runtime_minutes > 0
            else None
        )

    @property
    def enabled(self) -> bool:
        return self._deadline is not None

    def expired(self) -> bool:
        return self._deadline is not None and self._clock() >= self._deadline

    def remaining_seconds(self) -> Optional[float]:
        if self._deadline is None:
            return None
        return max(0.0, self._deadline - self._clock())


__all__ = [
    "TrainingBudget",
    "should_checkpoint_iteration",
    "validate_checkpoint_interval",
    "validate_max_runtime_minutes",
]
