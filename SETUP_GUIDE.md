# VolleyHighlights - Complete Setup Guide

This guide will walk you through setting up the VolleyHighlights project on any device, from cloning the repository to processing your first video.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [GPU Setup (Recommended)](#gpu-setup-recommended)
4. [Running the Project](#running-the-project)
5. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software
- **Git**: For cloning the repository
  - Download: https://git-scm.com/downloads

- **Python 3.11 or 3.12**: For GPU support with PyTorch
  - **Important**: Python 3.14+ doesn't have CUDA wheels yet
  - Download Python 3.11: https://www.python.org/downloads/release/python-3110/
  - Download Python 3.12: https://www.python.org/downloads/release/python-3120/

### Recommended Hardware
- **NVIDIA GPU** with CUDA support (e.g., RTX 3060 Ti, RTX 4090, etc.)
- **Minimum 4GB VRAM** (for YOLOv8-nano)
- **8GB+ RAM**
- **Storage**: ~2GB for dependencies + space for videos

### Check Your GPU (Windows)
```bash
nvidia-smi
```
This will show your GPU model and CUDA version.

---

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/VolleyHighlights.git
cd VolleyHighlights
```

### Step 2: Verify Python Version
```bash
python --version
```
or
```bash
py -3.11 --version
```

**Important**: Ensure you're using Python 3.11 or 3.12 for GPU support.

### Step 3: Create Virtual Environment

#### On Windows (Python 3.11):
```bash
py -3.11 -m venv venv_cuda
venv_cuda\Scripts\activate
```

#### On Linux/Mac:
```bash
python3.11 -m venv venv_cuda
source venv_cuda/bin/activate
```

You should see `(venv_cuda)` in your terminal prompt.

---

## GPU Setup (Recommended)

### Step 1: Install PyTorch with CUDA Support

First, check your CUDA version from `nvidia-smi` output.

#### For CUDA 12.1 (Most common):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### For CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### For CUDA 12.4+:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Step 2: Install Project Dependencies
```bash
pip install ultralytics opencv-python supervision moviepy pyyaml tqdm
```

Alternatively, if you have a `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Step 3: Verify GPU Installation
```python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

Expected output:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 3060 Ti
```

---

## Running the Project

### Test Ball Detection
Test on sample frames to verify setup:
```bash
python main.py test "path/to/your/video.mp4"
```

Output will be saved to `debug_output/` directory.

### Analyze Full Video
Get detection statistics without generating output:
```bash
python main.py analyze "path/to/your/video.mp4"
```

### Generate Highlights
Process the full video and generate highlight clips:
```bash
python main.py process "path/to/input/video.mp4" -o "path/to/output/highlights.mp4"
```

**Example**:
```bash
python main.py process "D:\Videos\volleyball_game.mp4" -o "D:\Videos\highlights.mp4"
```

### Performance Tips
- **GPU Processing**: ~100-110 frames/second (RTX 3060 Ti)
- **CPU Processing**: ~30-40 frames/second (slower)
- For a 1-hour video (~119k frames): Expect 15-20 minutes on GPU

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'cv2'"
**Solution**: Install OpenCV
```bash
pip install opencv-python
```

### Issue: "GPU not available, falling back to CPU"
**Cause**: PyTorch CPU version installed or wrong Python version

**Solution**:
1. Check Python version: `python --version` (must be 3.11 or 3.12)
2. Reinstall PyTorch with CUDA:
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
3. Verify: `python -c "import torch; print(torch.cuda.is_available())"`

### Issue: "CUDA out of memory"
**Solution**: Reduce image size in `config/settings.yaml`:
```yaml
ball_detection:
  imgsz: 480  # Reduce from 640
```

### Issue: Python 3.14 installed but no CUDA support
**Solution**: Create a new venv with Python 3.11:
```bash
py -3.11 -m venv venv_cuda
venv_cuda\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Issue: "No ball detected in most frames"
**Solutions**:
1. Lower confidence threshold in `config/settings.yaml`:
   ```yaml
   ball_detection:
     confidence_threshold: 0.2  # Reduce from 0.5
   ```
2. Ensure good lighting and clear ball visibility in video
3. Consider fine-tuning YOLO on volleyball-specific dataset

### Issue: Very slow processing
**Solutions**:
1. Verify GPU is being used (look for "Using GPU for inference" in output)
2. Check GPU usage: `nvidia-smi` (should show python.exe using GPU)
3. Reduce `imgsz` in settings.yaml
4. Close other GPU-intensive applications

---

## Configuration

Edit `config/settings.yaml` to customize behavior:

```yaml
ball_detection:
  model: yolov8n.pt           # Model size (n/s/m/l/x)
  confidence_threshold: 0.5    # Detection sensitivity (0.0-1.0)
  imgsz: 640                   # Input image size
  device: null                 # null=auto, 'cuda', 'cpu'

rally_detection:
  ball_missing_timeout: 3.0    # Seconds without ball = rally end
  min_rally_duration: 2.0      # Minimum rally length
  pre_rally_buffer: 1.0        # Seconds before rally start
  post_rally_buffer: 2.0       # Seconds after rally end

video_export:
  codec: mp4v                  # Video codec
  fps: null                    # Output FPS (null=same as input)
```

---

## Quick Reference Commands

### View Background Tasks
```bash
/tasks
```

### Activate Virtual Environment
```bash
# Windows
venv_cuda\Scripts\activate

# Linux/Mac
source venv_cuda/bin/activate
```

### Check GPU Status
```bash
nvidia-smi
```

### Update Dependencies
```bash
pip install --upgrade ultralytics opencv-python supervision moviepy
```

---

## Project Structure

```
VolleyHighlights/
├── src/
│   ├── detector.py          # Ball detection with YOLOv8
│   ├── tracker.py           # Ball tracking
│   ├── rally_detector.py    # Rally state machine
│   └── exporter.py          # Video export
├── config/
│   └── settings.yaml        # Configuration file
├── models/                  # YOLO weights (auto-downloaded)
├── debug_output/            # Test output frames
├── venv_cuda/               # Virtual environment (GPU)
├── main.py                  # Entry point
├── requirements.txt         # Dependencies
├── README.md                # Project overview
└── SETUP_GUIDE.md          # This file
```

---

## Additional Resources

- **YOLOv8 Documentation**: https://docs.ultralytics.com/
- **PyTorch Installation**: https://pytorch.org/get-started/locally/
- **CUDA Toolkit**: https://developer.nvidia.com/cuda-downloads
- **OpenCV Documentation**: https://docs.opencv.org/

---

## Support

For issues or questions:
1. Check this guide's [Troubleshooting](#troubleshooting) section
2. Review `README.md` for feature documentation
3. Open an issue on GitHub with:
   - Your Python version: `python --version`
   - CUDA version: `nvidia-smi`
   - Full error message
   - Steps to reproduce

---

## Performance Benchmarks

| Hardware | Processing Speed | 1-hour Video |
|----------|------------------|--------------|
| RTX 4090 | ~150-200 fps | ~10 minutes |
| RTX 3060 Ti | ~100-110 fps | ~18 minutes |
| CPU (i7) | ~30-40 fps | ~60 minutes |

*Benchmarks based on 1920x1080 @ 30fps input*

---

Last Updated: 2026-01-21
