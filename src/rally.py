"""
Rally Detection Module

Identifies volleyball rallies from ball detection data.
A rally is a continuous period where the ball is in play.
"""

from dataclasses import dataclass, field
from typing import Optional

from .detector import BallDetection


@dataclass
class Rally:
    """Represents a detected rally segment."""

    start_frame: int
    end_frame: int
    fps: float
    detections: list[BallDetection] = field(default_factory=list)

    @property
    def start_time(self) -> float:
        """Start time in seconds."""
        return self.start_frame / self.fps

    @property
    def end_time(self) -> float:
        """End time in seconds."""
        return self.end_frame / self.fps

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.end_time - self.start_time

    @property
    def frame_count(self) -> int:
        """Number of frames in the rally."""
        return self.end_frame - self.start_frame + 1

    @property
    def detection_rate(self) -> float:
        """Percentage of frames with ball detected."""
        if self.frame_count == 0:
            return 0.0
        return len(self.detections) / self.frame_count * 100

    def format_time(self, seconds: float) -> str:
        """Format seconds as MM:SS."""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"

    @property
    def time_range(self) -> str:
        """Formatted time range string."""
        return (
            f"{self.format_time(self.start_time)} - {self.format_time(self.end_time)}"
        )

    def __str__(self) -> str:
        return (
            f"Rally({self.time_range}, "
            f"duration={self.duration:.1f}s, "
            f"detections={len(self.detections)})"
        )


class RallyDetector:
    """
    Detects rallies from a sequence of ball detections.

    A rally starts when the ball is detected and ends when
    the ball has been missing for longer than the timeout.
    """

    def __init__(
        self,
        fps: float,
        ball_missing_timeout: float = 2.0,
        min_rally_duration: float = 2.0,
        pre_rally_buffer: float = 1.0,
        post_rally_buffer: float = 2.0,
        min_detections: int = 5,
    ):
        """
        Initialize the rally detector.

        Args:
            fps: Video frames per second
            ball_missing_timeout: Seconds without ball before rally ends
            min_rally_duration: Minimum rally duration to keep (filters noise)
            pre_rally_buffer: Seconds to include before rally starts
            post_rally_buffer: Seconds to include after rally ends
            min_detections: Minimum ball detections to consider valid rally
        """
        self.fps = fps
        self.ball_missing_timeout = ball_missing_timeout
        self.min_rally_duration = min_rally_duration
        self.pre_rally_buffer = pre_rally_buffer
        self.post_rally_buffer = post_rally_buffer
        self.min_detections = min_detections

        # Convert time thresholds to frames
        self.timeout_frames = int(ball_missing_timeout * fps)
        self.min_duration_frames = int(min_rally_duration * fps)
        self.pre_buffer_frames = int(pre_rally_buffer * fps)
        self.post_buffer_frames = int(post_rally_buffer * fps)

        # State for incremental processing
        self._reset_state()

    def _reset_state(self):
        """Reset internal state."""
        self.current_rally: Optional[Rally] = None
        self.rallies: list[Rally] = []
        self.last_detection_frame: Optional[int] = None
        self.frames_since_detection: int = 0

    def process_frame(
        self,
        frame_number: int,
        detections: list[BallDetection],
        total_frames: int,
    ) -> Optional[Rally]:
        """
        Process a single frame's detections.

        Args:
            frame_number: Current frame number
            detections: Ball detections for this frame
            total_frames: Total frames in video (for buffer clamping)

        Returns:
            Completed Rally if one just ended, None otherwise
        """
        completed_rally = None
        ball_detected = len(detections) > 0

        if ball_detected:
            self.last_detection_frame = frame_number
            self.frames_since_detection = 0

            if self.current_rally is None:
                # Start new rally
                self.current_rally = Rally(
                    start_frame=frame_number,
                    end_frame=frame_number,
                    fps=self.fps,
                )

            # Update current rally
            self.current_rally.end_frame = frame_number
            self.current_rally.detections.extend(detections)

        else:
            self.frames_since_detection += 1

            if self.current_rally is not None:
                # Check if rally should end
                if self.frames_since_detection >= self.timeout_frames:
                    completed_rally = self._finalize_rally(total_frames)

        return completed_rally

    def _finalize_rally(self, total_frames: int) -> Optional[Rally]:
        """Finalize current rally and check if it's valid."""
        if self.current_rally is None:
            return None

        rally = self.current_rally
        self.current_rally = None

        # Check minimum duration
        if rally.duration < self.min_rally_duration:
            return None

        # Check minimum detections
        if len(rally.detections) < self.min_detections:
            return None

        # Apply buffers (clamp to valid range)
        rally.start_frame = max(0, rally.start_frame - self.pre_buffer_frames)
        rally.end_frame = min(
            total_frames - 1, rally.end_frame + self.post_buffer_frames
        )

        self.rallies.append(rally)
        return rally

    def finish(self, total_frames: int) -> Optional[Rally]:
        """
        Call when video processing is complete.
        Finalizes any in-progress rally.

        Returns:
            Final Rally if one was in progress, None otherwise
        """
        if self.current_rally is not None:
            return self._finalize_rally(total_frames)
        return None

    def detect_rallies(
        self,
        detections_by_frame: dict[int, list[BallDetection]],
        total_frames: int,
    ) -> list[Rally]:
        """
        Detect all rallies from a complete detection map.

        Args:
            detections_by_frame: Dict mapping frame numbers to detections
            total_frames: Total number of frames in video

        Returns:
            List of detected Rally objects
        """
        self._reset_state()

        for frame_num in range(total_frames):
            detections = detections_by_frame.get(frame_num, [])
            self.process_frame(frame_num, detections, total_frames)

        # Finalize any remaining rally
        self.finish(total_frames)

        return self.rallies

    def merge_close_rallies(
        self,
        rallies: list[Rally],
        gap_threshold: float = 3.0,
    ) -> list[Rally]:
        """
        Merge rallies that are close together.

        Sometimes a rally might be split due to brief detection gaps.
        This merges rallies that are within gap_threshold seconds.

        Args:
            rallies: List of detected rallies
            gap_threshold: Maximum gap (seconds) to merge

        Returns:
            Merged list of rallies
        """
        if len(rallies) <= 1:
            return rallies

        gap_frames = int(gap_threshold * self.fps)
        merged = []
        current = rallies[0]

        for next_rally in rallies[1:]:
            gap = next_rally.start_frame - current.end_frame

            if gap <= gap_frames:
                # Merge rallies
                current.end_frame = next_rally.end_frame
                current.detections.extend(next_rally.detections)
            else:
                merged.append(current)
                current = next_rally

        merged.append(current)
        return merged

    def get_summary(self, rallies: list[Rally]) -> str:
        """Generate a summary of detected rallies."""
        if not rallies:
            return "No rallies detected."

        total_duration = sum(r.duration for r in rallies)
        avg_duration = total_duration / len(rallies)

        lines = [
            f"Detected {len(rallies)} rallies:",
            f"  Total highlight time: {total_duration:.1f}s",
            f"  Average rally duration: {avg_duration:.1f}s",
            "",
            "Rally breakdown:",
        ]

        for i, rally in enumerate(rallies, 1):
            lines.append(
                f"  {i}. {rally.time_range} "
                f"({rally.duration:.1f}s, {len(rally.detections)} detections)"
            )

        return "\n".join(lines)
