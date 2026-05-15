"""Simple resource-lane scheduler for resident stage services."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Callable

from .models import CancelToken


LaneLogger = Callable[[str], None]


@dataclass(slots=True)
class _LaneState:
    capacity: int
    in_use: int = 0


class ResourceScheduler:
    """Serialize risky model lanes while letting services stay resident."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._lanes: dict[str, _LaneState] = {
            "gpu_model_lane": _LaneState(capacity=1),
            "cpu_image_lane": _LaneState(capacity=2),
            "network_lane": _LaneState(capacity=2),
        }

    def acquire(
        self,
        lane_name: str,
        *,
        cancel_token: CancelToken | None = None,
        logger: LaneLogger | None = None,
    ) -> bool:
        normalized = str(lane_name or "").strip().lower()
        if not normalized:
            return True

        with self._condition:
            lane = self._lanes.setdefault(normalized, _LaneState(capacity=1))
            waiting_logged = False
            while lane.in_use >= lane.capacity:
                if cancel_token is not None and cancel_token.is_cancel_requested():
                    return False
                if logger is not None and not waiting_logged:
                    logger(f"Waiting for resource lane: {normalized}")
                    waiting_logged = True
                self._condition.wait(timeout=0.10)
            lane.in_use += 1
            if logger is not None:
                logger(f"Acquired resource lane: {normalized}")
            return True

    def release(self, lane_name: str, *, logger: LaneLogger | None = None) -> None:
        normalized = str(lane_name or "").strip().lower()
        if not normalized:
            return
        with self._condition:
            lane = self._lanes.get(normalized)
            if lane is None:
                return
            if lane.in_use > 0:
                lane.in_use -= 1
            if logger is not None:
                logger(f"Released resource lane: {normalized}")
            self._condition.notify_all()


__all__ = ["ResourceScheduler"]
