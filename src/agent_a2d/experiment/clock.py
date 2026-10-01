"""AgentA2D deterministic clock.

Generates reproducible timestamps based on an initial seed.
"""

from datetime import datetime, timedelta, timezone


class DeterministicClock:
    """A clock that ticks deterministically for reproducibility."""

    def __init__(self, seed: int) -> None:
        # Base timestamp anchored to the seed for reproducibility
        self._current_time = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seed)
        
    def now(self) -> str:
        """Return the current ISO timestamp and tick the clock forward slightly."""
        ts = self._current_time.isoformat()
        # Advance by exactly 1 millisecond per call
        self._current_time += timedelta(milliseconds=1)
        return ts
