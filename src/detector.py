"""
Ball Detection Module

Uses YOLOv8 to detect volleyballs in video frames.
Supports both pre-trained COCO model (sports ball) and custom trained models.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class BallDetection:
    """Represents a detected ball in a frame."""

    x: float  # Center X coordinate
    y: float  # Center Y coordinate
    width: float  # Bounding box width
    height: float  # Bounding box height
    confidence: float  # Detection confidence (0-1)
    frame_number: int  # Frame this detection came from

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Return bounding box as (x1, y1, x2, y2)."""
        x1 = self.x - self.width / 2
        y1 = self.y - self.height / 2
        x2 = self.x + self.width / 2
        y2 = self.y + self.height / 2
        return (x1, y1, x2, y2)

    @property
    def center(self) -> tuple[float, float]:
        """Return center point as (x, y)."""
        return (self.x, self.y)


class BallDetector:
    """
    Detects volleyballs in video frames using YOLOv8.

    The detector can use either:
    - Pre-trained COCO model (detects "sports ball" class)
    - Custom trained model for volleyball-specific detection
    """

    # COCO class ID for sports ball
    SPORTS_BALL_CLASS = 32

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.3,
        target_classes: Optional[list[int]] = None,
        device: str = "cuda",
        imgsz: int = 640,
    ):
        """
        Initialize the ball detector.

        Args:
            model_path: Path to YOLO model weights or model name
            confidence_threshold: Minimum confidence for detections
            target_classes: List of class IDs to detect (None = all)
            device: Device to run inference on ('cuda' or 'cpu')
            imgsz: Input image size for inference
        """
        self.confidence_threshold = confidence_threshold
        self.target_classes = target_classes or [self.SPORTS_BALL_CLASS]
        self.device = device
        self.imgsz = imgsz

        # Load YOLO model
        print(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)

        # Move to device
        if device == "cuda":
            try:
                self.model.to("cuda")
                print("Using GPU for inference")
            except Exception as e:
                print(f"GPU not available, falling back to CPU: {e}")
                self.device = "cpu"
        else:
            print("Using CPU for inference")

    def detect(
        self, frame: np.ndarray, frame_number: int = 0
    ) -> list[BallDetection]:
        """
        Detect balls in a single frame.

        Args:
            frame: BGR image as numpy array
            frame_number: Frame number for tracking purposes

        Returns:
            List of BallDetection objects
        """
        # Run inference
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            classes=self.target_classes,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False,
        )

        detections = []

        for result in results:
            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                # Get bounding box coordinates (xyxy format)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                # Calculate center and dimensions
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1

                # Get confidence
                confidence = float(box.conf[0].cpu().numpy())

                detection = BallDetection(
                    x=center_x,
                    y=center_y,
                    width=width,
                    height=height,
                    confidence=confidence,
                    frame_number=frame_number,
                )
                detections.append(detection)

        # Sort by confidence (highest first)
        detections.sort(key=lambda d: d.confidence, reverse=True)

        return detections

    def detect_best(
        self, frame: np.ndarray, frame_number: int = 0
    ) -> Optional[BallDetection]:
        """
        Detect the most likely ball in a frame.

        Args:
            frame: BGR image as numpy array
            frame_number: Frame number for tracking purposes

        Returns:
            Best BallDetection or None if no ball detected
        """
        detections = self.detect(frame, frame_number)
        return detections[0] if detections else None

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: list[BallDetection],
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw detection boxes on a frame.

        Args:
            frame: BGR image as numpy array
            detections: List of detections to draw
            color: BGR color for boxes
            thickness: Line thickness

        Returns:
            Frame with drawn detections
        """
        frame_copy = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox

            # Draw bounding box
            cv2.rectangle(
                frame_copy,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                thickness,
            )

            # Draw center point
            cv2.circle(
                frame_copy,
                (int(det.x), int(det.y)),
                5,
                (0, 0, 255),
                -1,
            )

            # Draw confidence label
            label = f"Ball: {det.confidence:.2f}"
            cv2.putText(
                frame_copy,
                label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        return frame_copy


class VideoProcessor:
    """
    Processes video files frame by frame for ball detection.
    """

    def __init__(self, video_path: str):
        """
        Initialize the video processor.

        Args:
            video_path: Path to the input video file
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Open video
        self.cap = cv2.VideoCapture(str(self.video_path))

        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0

    def __del__(self):
        """Release video capture on cleanup."""
        if hasattr(self, "cap") and self.cap is not None:
            self.cap.release()

    def __iter__(self):
        """Iterate over video frames."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return self

    def __next__(self) -> tuple[int, np.ndarray]:
        """Get next frame."""
        frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = self.cap.read()

        if not ret:
            raise StopIteration

        return frame_number, frame

    def get_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """Get a specific frame by number."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        return frame if ret else None

    def get_frame_at_time(self, seconds: float) -> Optional[np.ndarray]:
        """Get frame at a specific timestamp."""
        frame_number = int(seconds * self.fps)
        return self.get_frame(frame_number)

    @property
    def info(self) -> dict:
        """Get video information."""
        return {
            "path": str(self.video_path),
            "fps": self.fps,
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "duration_seconds": self.duration,
            "duration_formatted": self._format_duration(self.duration),
        }

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration as HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
