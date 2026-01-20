# VolleyHighlights

Automatic volleyball rally detection and highlight generation using computer vision.

## Features

- Detects volleyballs in video using YOLOv8
- Identifies rally start/end based on ball visibility
- Extracts highlight clips automatically
- GPU-accelerated processing

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (recommended)
- ~4GB VRAM (for YOLOv8-nano)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/VolleyHighlights.git
cd VolleyHighlights
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) Install PyTorch with CUDA support if not already installed:
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Usage

### Test Detection
Test ball detection on sample frames from your video:
```bash
python main.py test path/to/your/video.mp4
```

This will:
- Extract 10 sample frames from the video
- Run ball detection on each frame
- Save annotated frames to `debug_output/`

### Analyze Video
Run full video analysis to see detection statistics:
```bash
python main.py analyze path/to/your/video.mp4
```

### Generate Highlights (Coming Soon)
```bash
python main.py process path/to/your/video.mp4 -o highlights.mp4
```

## Configuration

Edit `config/settings.yaml` to adjust:

- `ball_detection.confidence_threshold`: Detection sensitivity (0.0-1.0)
- `rally_detection.ball_missing_timeout`: Seconds without ball before rally ends
- `rally_detection.post_rally_buffer`: Seconds to keep after rally ends

## Project Structure

```
VolleyHighlights/
├── src/
│   ├── detector.py      # Ball detection with YOLO
│   ├── tracker.py       # Ball tracking (Phase 2)
│   ├── rally_detector.py # Rally state machine (Phase 2)
│   └── exporter.py      # Video export (Phase 4)
├── config/
│   └── settings.yaml    # Configuration
├── models/              # YOLO weights (auto-downloaded)
├── debug_output/        # Test output frames
├── main.py              # Entry point
└── requirements.txt
```

## Development Phases

- [x] Phase 1: Ball detection with YOLOv8
- [ ] Phase 2: Ball tracking and rally state machine
- [ ] Phase 3: Ground touch detection
- [ ] Phase 4: Video segmentation and export
- [ ] Phase 5: Refinement and edge cases

## Troubleshooting

### "No ball detected" in most frames
- Try lowering `confidence_threshold` in settings.yaml (e.g., 0.2)
- The COCO-trained model may not recognize beach volleyballs well
- Consider fine-tuning on volleyball-specific data

### CUDA out of memory
- Reduce `imgsz` in settings.yaml (e.g., 480)
- Use `device: cpu` (slower but works)

### Slow processing
- Ensure GPU is being used (check output for "Using GPU")
- Reduce `imgsz` for faster processing

## License

See LICENSE file.
