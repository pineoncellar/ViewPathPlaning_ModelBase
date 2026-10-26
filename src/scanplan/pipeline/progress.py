"""Optional progress reporting for one planning run.

The planner is a synchronous pipeline, so a caller that wants to show progress needs
somewhere for it to arrive.  The contract is deliberately tiny - a hook that receives
`(stage, fraction, detail)` - so the CLI can ignore it and the browser interface can
forward it to a polling endpoint without either of them knowing the other's vocabulary.

Two properties matter for a progress bar that people trust:

* **monotonic** - a stage never moves backwards, even when a later stage restarts part
  of the work (a refinement round re-evaluates candidates);
* **throttled** - a hot inner loop calls the reporter thousands of times, and each call
  would otherwise become a publish.  Only a real advance, or a stage change, is sent.
"""

from __future__ import annotations

import time
from collections.abc import Callable

ProgressHook = Callable[[str, float, str], None]
"""`hook(stage, fraction, detail)` with `fraction` in 0..1 over the whole run."""

#: Minimum seconds between two publishes that are not a stage change or completion.
#: The page polls every 400 ms, so anything faster would only add noise.
_MIN_INTERVAL_S = 0.35
#: Minimum advance (0..1 of the whole run) that is worth publishing.
_MIN_STEP = 0.01


class ProgressReporter:
    """Maps a stage's local 0..1 onto a band of the whole run and forwards it.

    A reporter with no hook is a no-op, which is what keeps every existing caller
    (`plan(...)` from the CLI, tests, the scripts) unchanged.
    """

    def __init__(self, hook: ProgressHook | None = None, *,
                 low: float = 0.0, high: float = 1.0, stage: str = "") -> None:
        self.hook = hook
        self.low = float(low)
        self.high = float(high)
        self.stage = stage
        self._last_time = float("-inf")
        self._last_value = float("-inf")
        self._last_stage = ""

    @property
    def enabled(self) -> bool:
        return self.hook is not None

    def band(self, stage: str, low: float, high: float) -> ProgressReporter:
        """A reporter for one stage, covering `low..high` of this reporter's band."""
        if self.hook is None:
            return self
        span = self.high - self.low
        return ProgressReporter(
            self.hook,
            low=self.low + span * low,
            high=self.low + span * high,
            stage=stage,
        )

    def __call__(self, fraction: float, detail: str = "") -> None:
        if self.hook is None:
            return
        fraction = min(max(float(fraction), 0.0), 1.0)
        value = self.low + (self.high - self.low) * fraction
        changed_stage = self.stage != self._last_stage
        if not changed_stage and value <= self._last_value:
            # A stage only moves forwards, and repeating the same value is not news.
            return
        now = time.monotonic()
        finished = fraction >= 1.0 or value >= 1.0
        if (not finished and not changed_stage
                and now - self._last_time < _MIN_INTERVAL_S
                and value - self._last_value < _MIN_STEP):
            return
        self._last_time = now
        self._last_value = value
        self._last_stage = self.stage
        self.hook(self.stage, value, detail)

    def tick(self, unit: str = "") -> Callable[[int, int], None] | None:
        """A `(done, total)` callable for an inner loop, or None when nothing listens."""
        if self.hook is None:
            return None

        def report(done: int, total: int) -> None:
            total = max(int(total), 1)
            self(done / total, f"{done}/{total} {unit}".strip())

        return report

    def done(self, detail: str = "") -> None:
        self(1.0, detail)
