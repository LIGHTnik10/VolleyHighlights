#!/usr/bin/env python3
"""
Process full video by detecting rallies and exporting highlights.
Designed to handle long videos with periodic checkpointing.
"""

import gc
import json
import sys
from pathlib import Path

import torch
import yaml
from src.detector import BallDetector, VideoProcessor
from src.exporter import HighlightExporter
from src.rally import Rally, RallyDetector
from tqdm import tqdm


def process_video(video_path: str, output_path: str = "highlights.mp4"):
    """Process full video and export highlights."""

    print(f"\n{'=' * 60}")
    print("VOLLEYBALL HIGHLIGHT GENERATOR")
    print(f"{'=' * 60}\n")

    video = VideoProcessor(video_path)
    info = video.info

    print(f"Video: {info['path']}")
    print(f"Duration: {info['duration_formatted']} ({info['frame_count']} frames)")
    print(f"FPS: {info['fps']:.2f}")

    # Load config
    config_path = Path(__file__).parent / "config" / "settings.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize detector
    det_config = config.get("ball_detection", {})
    detector = BallDetector(
        model_path=det_config.get("model", "yolov8n.pt"),
        confidence_threshold=det_config.get("confidence_threshold", 0.3),
        target_classes=det_config.get("target_classes", [32]),
        device=det_config.get("device", "cuda"),
        imgsz=det_config.get("imgsz", 640),
    )

    # Initialize rally detector
    rally_config = config.get("rally_detection", {})
    rally_detector = RallyDetector(
        fps=video.fps,
        ball_missing_timeout=rally_config.get("ball_missing_timeout", 2.0),
        min_rally_duration=rally_config.get("min_rally_duration", 2.0),
        pre_rally_buffer=rally_config.get("pre_rally_buffer", 1.0),
        post_rally_buffer=rally_config.get("post_rally_buffer", 2.0),
    )

    print("\nProcessing video for rally detection...\n")

    error_count = 0
    checkpoint_interval = 10000  # Save progress every 10k frames

    for frame_num, frame in tqdm(video, total=video.frame_count, desc="Detecting"):
        try:
            detections = detector.detect(frame, frame_num)
            rally_detector.process_frame(frame_num, detections, video.frame_count)
        except Exception as e:
            error_count += 1
            if error_count <= 5:
                print(f"\nWarning: Error at frame {frame_num}: {e}")
            continue

        # Periodic memory cleanup
        if frame_num > 0 and frame_num % 5000 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Progress checkpoint
        if frame_num > 0 and frame_num % checkpoint_interval == 0:
            current_rallies = len(rally_detector.rallies)
            print(
                f"\n  Checkpoint: {frame_num} frames, {current_rallies} rallies found so far"
            )

    if error_count > 0:
        print(f"\nTotal frames with errors: {error_count}")

    # Finalize
    rally_detector.finish(video.frame_count)
    rallies = rally_detector.rallies
    rallies = rally_detector.merge_close_rallies(rallies, gap_threshold=3.0)

    # Print results
    print(f"\n{'=' * 60}")
    print("RALLY DETECTION RESULTS")
    print(f"{'=' * 60}")
    print(rally_detector.get_summary(rallies))

    if not rallies:
        print("No rallies detected!")
        return

    # Export highlights
    print(f"\n{'=' * 60}")
    print("EXPORTING HIGHLIGHTS")
    print(f"{'=' * 60}")

    exporter = HighlightExporter(
        output_path=output_path,
        codec="libx264",
        audio=True,
    )

    exporter.export(video_path, rallies)

    print(f"\n{'=' * 60}")
    print("COMPLETE!")
    print(f"{'=' * 60}")
    print(f"Input: {video_path}")
    print(f"Output: {output_path}")
    print(f"Rallies: {len(rallies)}")
    total_duration = sum(r.duration for r in rallies)
    print(f"Highlight duration: {total_duration:.1f}s")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_full.py <video_path> [output_path]")
        sys.exit(1)

    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "highlights.mp4"

    process_video(video_path, output_path)
