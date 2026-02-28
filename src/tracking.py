"""Temporal filtering for volleyball detections.

This module keeps a lightweight ball track so detections stay consistent across
camera angles and noisy frames.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import hypot

from .detector import BallDetection


@dataclass
class TrackingConfig:
    """Settings for temporal ball filtering."""

    max_jump_ratio: float = 0.14
    confidence_weight: float = 0.7
    distance_weight: float = 0.3
    recovery_confidence: float = 0.45
    max_lost_frames: int = 12
    history_size: int = 6


class BallTrackFilter:
    """Selects detections that fit a physically plausible ball trajectory."""

    def __init__(self, frame_width: int, frame_height: int, config: TrackingConfig | None = None):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.config = config or TrackingConfig()
        self.frame_diagonal = hypot(frame_width, frame_height)
        self.history: deque[tuple[float, float]] = deque(maxlen=self.config.history_size)
        self.last_detection: BallDetection | None = None
        self.lost_frames = 0

    def _predict_next(self) -> tuple[float, float] | None:
        if len(self.history) < 2:
            return self.history[-1] if self.history else None

        (x1, y1), (x2, y2) = self.history[-2], self.history[-1]
        return (x2 + (x2 - x1), y2 + (y2 - y1))

    def _score(self, detection: BallDetection, predicted: tuple[float, float] | None) -> float:
        if predicted is None:
            return detection.confidence

        dx = detection.x - predicted[0]
        dy = detection.y - predicted[1]
        distance = hypot(dx, dy)
        distance_norm = min(distance / (self.frame_diagonal * self.config.max_jump_ratio), 1.0)
        return (
            self.config.confidence_weight * detection.confidence
            + self.config.distance_weight * (1.0 - distance_norm)
        )

    def filter(self, detections: list[BallDetection]) -> list[BallDetection]:
        """Return either one temporally-consistent detection or an empty list."""
        if not detections:
            self.lost_frames += 1
            if self.lost_frames > self.config.max_lost_frames:
                self.history.clear()
                self.last_detection = None
            return []

        predicted = self._predict_next()
        ranked = sorted(detections, key=lambda det: self._score(det, predicted), reverse=True)
        best = ranked[0]

        if predicted is not None:
            max_jump = self.frame_diagonal * self.config.max_jump_ratio
            jump = hypot(best.x - predicted[0], best.y - predicted[1])
            needs_recovery = self.lost_frames > 0

            if jump > max_jump and best.confidence < self.config.recovery_confidence:
                # Likely false positive. Keep rally alive with timeout logic rather than hard reset.
                self.lost_frames += 1
                return []

            if needs_recovery and best.confidence < self.config.recovery_confidence:
                self.lost_frames += 1
                return []

        self.history.append((best.x, best.y))
        self.last_detection = best
        self.lost_frames = 0
        return [best]
