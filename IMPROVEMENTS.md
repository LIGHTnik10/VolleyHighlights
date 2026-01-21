# VolleyHighlights Improvements

## Changes Made - 2026-01-21

### 1. Fixed Terminal Progress Output ✅

**Problem**: Progress was creating hundreds of newlines, making it impossible to track processing status.

**Solution**: Created custom `src/progress.py` module
- Single-line progress updates using carriage return
- Shows: `Processing: X/Y frames (Z.Z%) | A.A fps | ETA: Bm Cs`
- Updates every 30 frames (reduced output spam)
- Works better in both foreground and background modes

**Files Changed**:
- `src/progress.py` (new)
- `main.py` (replaced tqdm with ProgressReporter)

---

### 2. Fixed Video Concatenation Issues ✅

**Problem**: Output video had lagging and repeated frames at clip transitions.

**Solution**: Changed from stream copy to proper re-encoding
- Individual clips are now re-encoded during extraction
- Concatenation uses constant frame rate (`-vsync cfr`)
- Added `preset fast` for good balance of speed and quality
- Audio bitrate set to 192k for consistent quality
- Added `-movflags +faststart` for better web playback

**Files Changed**:
- `src/exporter.py`

**Technical Details**:
- Changed from `-c copy` to `-c:v libx264 -crf 23`
- Added `-vsync cfr` to prevent duplicate/dropped frames
- Consistent encoding parameters across all clips

---

### 3. Improved Rally Detection Settings ✅

**Problem**: Only 15 minutes of highlights from 66-minute video felt too short.

**Solution**: Adjusted detection parameters to capture more content

**Changes to `config/settings.yaml`**:
- **Confidence threshold**: 0.3 → 0.2 (detect more balls)
- **Ball missing timeout**: 4.0s → 5.0s (allow brief gaps in detection)
- **Minimum rally duration**: 2.0s → 1.5s (capture shorter plays)
- **Post-rally buffer**: 3.0s → 4.0s (capture more reactions)
- **Pre-rally buffer**: 1.0s → 2.0s (capture serve setup)

**Fixed `main.py`**:
- Removed hardcoded values that were overriding config
- Now properly uses `config/settings.yaml` settings

---

## Testing the Improvements

### Quick Test
Run the same command again to see the improvements:
```bash
cd "D:\AI\My Custom Project (My Ideas)\VolleyHighlights\VolleyHighlights"
venv_cuda\Scripts\activate
python main.py process "D:\Temp Uploads\18 01 2026.MOV" -o "D:\Temp Uploads\highlights_v2.mp4"
```

### Expected Results

**Progress Output**:
```
Detecting rallies: 45000/119099 frames (37.8%) | 105.3 fps | ETA: 11m 45s
```
Instead of hundreds of newlines, you'll see a single updating line.

**Video Quality**:
- Smooth transitions between clips
- No repeated or dropped frames
- Consistent playback throughout

**More Content**:
- Likely 80-100+ rallies detected (vs previous 77)
- 20-25 minutes of highlights (vs previous 15)
- Better coverage of the full game

---

## Tuning Detection Parameters

If you want even more (or fewer) highlights, edit `config/settings.yaml`:

### To Get MORE Highlights:
```yaml
ball_detection:
  confidence_threshold: 0.15  # Lower = more detections

rally_detection:
  ball_missing_timeout: 6.0    # Longer gaps allowed
  min_rally_duration: 1.0      # Include very short plays
  post_rally_buffer: 5.0       # More celebration footage
```

### To Get FEWER Highlights (Higher Quality Only):
```yaml
ball_detection:
  confidence_threshold: 0.35   # Higher = only confident detections

rally_detection:
  ball_missing_timeout: 3.0    # Stricter rally definition
  min_rally_duration: 3.0      # Only longer rallies
  post_rally_buffer: 2.0       # Less buffer footage
```

---

## Performance Notes

**Video Re-encoding**:
- Extraction + concatenation now takes longer (~2-3x)
- But produces much better quality output
- Still faster than real-time playback

**Recommended Workflow**:
1. First run: Test with current settings
2. Review output video
3. Adjust `config/settings.yaml` if needed
4. Re-run with tweaked settings

---

## File Summary

**New Files**:
- `src/progress.py` - Custom progress reporter

**Modified Files**:
- `main.py` - Uses new progress reporter, reads config properly
- `src/exporter.py` - Re-encodes clips for smooth playback
- `config/settings.yaml` - Better default parameters
- `IMPROVEMENTS.md` - This file

**No Changes**:
- `src/detector.py` - Ball detection logic unchanged
- `src/rally.py` - Rally detection logic unchanged
- `requirements.txt` - Dependencies unchanged
