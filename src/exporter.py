"""
Video Export Module

Extracts rally segments and combines them into a highlight video.
Uses ffmpeg for memory-efficient processing of long videos.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from .rally import Rally


class HighlightExporter:
    """
    Exports detected rallies as a highlight video.
    Uses ffmpeg directly for memory-efficient processing.
    """

    def __init__(
        self,
        output_path: str = "highlights.mp4",
        codec: str = "libx264",
        audio: bool = True,
        crf: int = 23,
    ):
        """
        Initialize the exporter.

        Args:
            output_path: Path for output video file
            codec: Video codec to use
            audio: Whether to include audio
            crf: Constant Rate Factor for quality (lower = better, 18-28 typical)
        """
        self.output_path = Path(output_path)
        self.codec = codec
        self.audio = audio
        self.crf = crf

    def _get_ffmpeg_path(self) -> str:
        """Get ffmpeg executable path."""
        # Try to find ffmpeg from imageio_ffmpeg (installed with moviepy)
        try:
            import imageio_ffmpeg

            return imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            return "ffmpeg"  # Assume it's in PATH

    def export(
        self,
        video_path: str,
        rallies: list[Rally],
        progress_callback: Optional[callable] = None,
    ) -> Path:
        """
        Export rallies as a highlight video using ffmpeg concat demuxer.

        This approach is memory-efficient as it doesn't load all clips into memory.

        Args:
            video_path: Path to source video
            rallies: List of Rally objects to export
            progress_callback: Optional callback for progress updates

        Returns:
            Path to exported video
        """
        if not rallies:
            raise ValueError("No rallies to export")

        ffmpeg = self._get_ffmpeg_path()
        video_path = Path(video_path)

        print(f"\nExporting {len(rallies)} rallies to {self.output_path}...")
        print(f"Using ffmpeg: {ffmpeg}")

        # Create temp directory for intermediate clips
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            clip_paths = []

            # Extract each rally as a separate clip
            for i, rally in enumerate(rallies):
                clip_path = temp_path / f"clip_{i:04d}.mp4"
                clip_paths.append(clip_path)

                print(f"  [{i + 1}/{len(rallies)}] Extracting {rally.time_range}")

                # Use ffmpeg to extract clip (fast, no re-encoding)
                cmd = [
                    ffmpeg,
                    "-y",  # Overwrite
                    "-ss",
                    str(rally.start_time),  # Start time
                    "-i",
                    str(video_path),  # Input
                    "-t",
                    str(rally.duration),  # Duration
                    "-c",
                    "copy",  # Copy streams (no re-encode, very fast)
                    "-avoid_negative_ts",
                    "make_zero",
                    str(clip_path),
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                )

                if result.returncode != 0:
                    print(f"    Warning: ffmpeg returned {result.returncode}")
                    # Try with re-encoding if copy fails
                    cmd = [
                        ffmpeg,
                        "-y",
                        "-ss",
                        str(rally.start_time),
                        "-i",
                        str(video_path),
                        "-t",
                        str(rally.duration),
                        "-c:v",
                        self.codec,
                        "-crf",
                        str(self.crf),
                        "-c:a",
                        "aac" if self.audio else "none",
                        str(clip_path),
                    ]
                    subprocess.run(cmd, capture_output=True)

            # Create concat file list
            concat_file = temp_path / "concat.txt"
            with open(concat_file, "w") as f:
                for clip_path in clip_paths:
                    # Escape single quotes and use proper format
                    escaped_path = str(clip_path).replace("'", "'\\''")
                    f.write(f"file '{escaped_path}'\n")

            # Concatenate all clips
            print(f"\nConcatenating {len(clip_paths)} clips...")

            concat_cmd = [
                ffmpeg,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-c",
                "copy",  # Try copy first
                str(self.output_path),
            ]

            result = subprocess.run(
                concat_cmd,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                print("  Copy concat failed, re-encoding...")
                # Fall back to re-encoding if copy fails
                concat_cmd = [
                    ffmpeg,
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(concat_file),
                    "-c:v",
                    self.codec,
                    "-crf",
                    str(self.crf),
                    "-c:a",
                    "aac" if self.audio else "none",
                    "-movflags",
                    "+faststart",
                    str(self.output_path),
                ]

                # Run with progress output
                process = subprocess.Popen(
                    concat_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

                for line in process.stdout:
                    if "frame=" in line or "time=" in line:
                        print(f"\r  {line.strip()}", end="", flush=True)

                process.wait()
                print()  # New line after progress

        # Verify output
        if self.output_path.exists():
            size_mb = self.output_path.stat().st_size / (1024 * 1024)
            total_duration = sum(r.duration for r in rallies)
            print(f"\nHighlight video saved: {self.output_path}")
            print(f"Duration: {total_duration:.1f}s")
            print(f"File size: {size_mb:.1f} MB")
        else:
            raise RuntimeError(f"Failed to create output video: {self.output_path}")

        return self.output_path

    def export_individual(
        self,
        video_path: str,
        rallies: list[Rally],
        output_dir: str = "rallies",
    ) -> list[Path]:
        """
        Export each rally as a separate video file.

        Args:
            video_path: Path to source video
            rallies: List of Rally objects to export
            output_dir: Directory for output files

        Returns:
            List of paths to exported videos
        """
        if not rallies:
            raise ValueError("No rallies to export")

        ffmpeg = self._get_ffmpeg_path()
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        print(f"\nExporting {len(rallies)} rallies to {output_dir}/...")

        exported = []
        for i, rally in enumerate(rallies):
            output_path = output_dir / f"rally_{i + 1:03d}.mp4"
            print(
                f"  [{i + 1}/{len(rallies)}] {rally.time_range} -> {output_path.name}"
            )

            cmd = [
                ffmpeg,
                "-y",
                "-ss",
                str(rally.start_time),
                "-i",
                str(video_path),
                "-t",
                str(rally.duration),
                "-c",
                "copy",
                str(output_path),
            ]

            subprocess.run(cmd, capture_output=True)
            exported.append(output_path)

        print(f"\nExported {len(exported)} rally videos to {output_dir}/")
        return exported
