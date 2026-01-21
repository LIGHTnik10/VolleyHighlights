"""
Progress Reporter Module

Provides a clean single-line progress output for video processing.
"""

import sys
import time
from typing import Optional


class ProgressReporter:
    """
    Reports processing progress with a single-line update.

    Designed to work well in both foreground and background modes.
    """

    def __init__(
        self,
        total: int,
        description: str = "Processing",
        update_interval: int = 30,
    ):
        """
        Initialize the progress reporter.

        Args:
            total: Total number of items to process
            description: Description of the task
            update_interval: Update every N items
        """
        self.total = total
        self.description = description
        self.update_interval = update_interval
        self.current = 0
        self.start_time = time.time()
        self.last_update = 0

    def update(self, n: int = 1):
        """Update progress by n items."""
        self.current += n

        # Update display at regular intervals or at completion
        if (
            self.current - self.last_update >= self.update_interval
            or self.current >= self.total
        ):
            self._display()
            self.last_update = self.current

    def _display(self):
        """Display current progress."""
        elapsed = time.time() - self.start_time

        # Calculate stats
        percent = (self.current / self.total * 100) if self.total > 0 else 0
        fps = self.current / elapsed if elapsed > 0 else 0

        # Estimate remaining time
        if self.current > 0 and fps > 0:
            remaining_frames = self.total - self.current
            eta_seconds = remaining_frames / fps
            eta_str = self._format_time(eta_seconds)
        else:
            eta_str = "calculating..."

        # Build progress message
        msg = (
            f"{self.description}: {self.current}/{self.total} frames "
            f"({percent:.1f}%) | {fps:.1f} fps | ETA: {eta_str}"
        )

        # Use carriage return for single-line update
        print(f"\r{msg}", end="", flush=True)

        # Add newline when complete
        if self.current >= self.total:
            print()  # Final newline

    def close(self):
        """Finalize progress display."""
        if self.current < self.total:
            self.current = self.total
            self._display()

        elapsed = time.time() - self.start_time
        print(f"\nCompleted in {self._format_time(elapsed)}")

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as human-readable time."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.close()
