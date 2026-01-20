#!/usr/bin/env python3
"""
VolleyHighlights - Main Entry Point

Automatic volleyball rally detection and highlight generation.

Usage:
    python main.py <video_path> [options]

Examples:
    # Test detection on a video (saves sample frames)
    python main.py test path/to/video.mp4

    # Process full video and generate highlights (Phase 2+)
    python main.py process path/to/video.mp4 -o highlights.mp4
"""

import argparse
import sys
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm

from src.detector import BallDetector, VideoProcessor


def load_config(config_path: str = "config/settings.yaml") -> dict:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"Config file not found: {config_path}, using defaults")
        return {}

    with open(config_file) as f:
        return yaml.safe_load(f)


def test_detection(video_path: str, config: dict, num_samples: int = 10):
    """
    Test ball detection on sample frames from a video.

    Saves annotated frames to debug_output/ directory.
    """
    print(f"\n{'='*60}")
    print("BALL DETECTION TEST")
    print(f"{'='*60}\n")

    # Load video
    print(f"Loading video: {video_path}")
    video = VideoProcessor(video_path)

    # Print video info
    info = video.info
    print(f"  Resolution: {info['width']}x{info['height']}")
    print(f"  FPS: {info['fps']:.2f}")
    print(f"  Duration: {info['duration_formatted']}")
    print(f"  Total frames: {info['frame_count']}")

    # Initialize detector
    det_config = config.get("ball_detection", {})
    detector = BallDetector(
        model_path=det_config.get("model", "yolov8n.pt"),
        confidence_threshold=det_config.get("confidence_threshold", 0.3),
        target_classes=det_config.get("target_classes", [32]),
        device=det_config.get("device", "cuda"),
        imgsz=det_config.get("imgsz", 640),
    )

    # Create output directory
    output_dir = Path("debug_output")
    output_dir.mkdir(exist_ok=True)

    # Sample frames evenly across the video
    sample_interval = video.frame_count // (num_samples + 1)
    sample_frames = [sample_interval * (i + 1) for i in range(num_samples)]

    print(f"\nTesting detection on {num_samples} sample frames...")
    print(f"Saving annotated frames to: {output_dir}/\n")

    detections_found = 0

    for i, frame_num in enumerate(sample_frames):
        frame = video.get_frame(frame_num)
        if frame is None:
            continue

        # Detect ball
        detections = detector.detect(frame, frame_num)

        # Draw detections
        annotated = detector.draw_detections(frame, detections)

        # Add frame info
        timestamp = frame_num / video.fps
        cv2.putText(
            annotated,
            f"Frame: {frame_num} | Time: {timestamp:.2f}s | Detections: {len(detections)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Save frame
        output_path = output_dir / f"sample_{i:02d}_frame_{frame_num}.jpg"
        cv2.imwrite(str(output_path), annotated)

        if detections:
            detections_found += 1
            best = detections[0]
            print(f"  Frame {frame_num}: Ball detected at ({best.x:.0f}, {best.y:.0f}) "
                  f"conf={best.confidence:.2f}")
        else:
            print(f"  Frame {frame_num}: No ball detected")

    print(f"\n{'='*60}")
    print(f"RESULTS: Ball detected in {detections_found}/{num_samples} sample frames")
    print(f"Annotated frames saved to: {output_dir}/")
    print(f"{'='*60}\n")

    return detections_found > 0


def analyze_video(video_path: str, config: dict):
    """
    Analyze entire video and print detection statistics.

    This is a preliminary step before full processing.
    """
    print(f"\n{'='*60}")
    print("FULL VIDEO ANALYSIS")
    print(f"{'='*60}\n")

    # Load video
    video = VideoProcessor(video_path)
    info = video.info

    print(f"Video: {info['path']}")
    print(f"Duration: {info['duration_formatted']} ({info['frame_count']} frames)")

    # Initialize detector
    det_config = config.get("ball_detection", {})
    detector = BallDetector(
        model_path=det_config.get("model", "yolov8n.pt"),
        confidence_threshold=det_config.get("confidence_threshold", 0.3),
        target_classes=det_config.get("target_classes", [32]),
        device=det_config.get("device", "cuda"),
        imgsz=det_config.get("imgsz", 640),
    )

    # Process all frames
    print("\nAnalyzing video (this may take a while)...\n")

    frames_with_ball = 0
    total_detections = 0
    detection_log = []

    for frame_num, frame in tqdm(video, total=video.frame_count, desc="Processing"):
        detections = detector.detect(frame, frame_num)

        if detections:
            frames_with_ball += 1
            total_detections += len(detections)
            detection_log.append({
                "frame": frame_num,
                "time": frame_num / video.fps,
                "detections": len(detections),
                "best_conf": detections[0].confidence,
                "best_pos": (detections[0].x, detections[0].y),
            })

    # Calculate statistics
    detection_rate = frames_with_ball / video.frame_count * 100

    print(f"\n{'='*60}")
    print("ANALYSIS RESULTS")
    print(f"{'='*60}")
    print(f"Total frames analyzed: {video.frame_count}")
    print(f"Frames with ball detected: {frames_with_ball} ({detection_rate:.1f}%)")
    print(f"Total detections: {total_detections}")

    if detection_log:
        # Find segments with continuous detections (potential rallies)
        print(f"\nFirst detection: Frame {detection_log[0]['frame']} "
              f"(t={detection_log[0]['time']:.2f}s)")
        print(f"Last detection: Frame {detection_log[-1]['frame']} "
              f"(t={detection_log[-1]['time']:.2f}s)")

    print(f"{'='*60}\n")

    return detection_log


def main():
    parser = argparse.ArgumentParser(
        description="VolleyHighlights - Automatic volleyball highlight generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py test video.mp4           Test detection on sample frames
  python main.py analyze video.mp4        Analyze full video (detection stats)
  python main.py process video.mp4        Generate highlights (coming in Phase 2)
        """,
    )

    parser.add_argument(
        "command",
        choices=["test", "analyze", "process"],
        help="Command to run",
    )
    parser.add_argument(
        "video",
        help="Path to input video file",
    )
    parser.add_argument(
        "-c", "--config",
        default="config/settings.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "-o", "--output",
        default="highlights.mp4",
        help="Output file path (for process command)",
    )
    parser.add_argument(
        "-n", "--num-samples",
        type=int,
        default=10,
        help="Number of sample frames to test (for test command)",
    )

    args = parser.parse_args()

    # Check video exists
    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    # Load config
    config = load_config(args.config)

    # Execute command
    if args.command == "test":
        success = test_detection(args.video, config, args.num_samples)
        sys.exit(0 if success else 1)

    elif args.command == "analyze":
        analyze_video(args.video, config)

    elif args.command == "process":
        print("Full video processing will be available in Phase 2!")
        print("For now, use 'test' or 'analyze' commands.")
        sys.exit(0)


if __name__ == "__main__":
    main()
