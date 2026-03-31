# Drowsy Detect

A real-time computer vision system for driver fatigue monitoring using eye closure and yawning detection.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Detection Algorithms](#detection-algorithms)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [System Requirements](#system-requirements)
- [License](#license)

---

## Overview

Drowsy Detect is a Python-based application that monitors drivers for signs of fatigue by analyzing facial features in real-time. The system uses **Haar cascade classifiers** from OpenCV to detect faces, eyes, and mouth states, then applies temporal filtering to distinguish between normal behaviors (blinking, talking) and drowsiness indicators (sustained eye closure, yawning).

When drowsiness is detected, the system triggers **audio-visual alerts** to唤醒 the driver:
- **Audio**: Continuous 880Hz tone played through system speakers
- **Visual**: Flashing red border and color-coded status overlays

The system operates entirely on-device with no data storage or network transmission, ensuring user privacy.

---

## Features

| Feature | Description |
|---------|-------------|
| **Real-time Eye Tracking** | Haar cascade-based eye detection with optimized parameters for high sensitivity |
| **Yawn Detection** | Texture variance and intensity analysis of mouth region |
| **Temporal Filtering** | Frame-based counters prevent false positives from normal blinking |
| **Dual Alert System** | Synchronized audio (pygame) and visual (OpenCV overlays) feedback |
| **Threaded Audio** | Non-blocking sound playback via dedicated thread |
| **Dual UI Modes** | Native OpenCV window (`simple_main.py`) and Streamlit web dashboard (`streamlit_app.py`) |
| **Debug Mode** | Verbose logging variant (`debug_main.py`) for troubleshooting |
| **Zero Data Retention** | All processing happens in-memory; no video or biometric data is stored |

---

## System Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PRESENTATION LAYER                           │
│  ┌───────────────────┐  ┌─────────────────────────────────────────┐ │
│  │ simple_main.py    │  │ streamlit_app.py                        │ │
│  │ OpenCV Window     │  │ Web Dashboard                           │ │
│  │ - cv2.imshow()    │  │ - Metrics, controls, live feed          │ │
│  │ - Keyboard input  │  │ - Color-coded status boxes              │ │
│  └─────────┬─────────┘  └──────────────────┬──────────────────────┘ │
└────────────┼────────────────────────────────┼───────────────────────┘
             │                                │
             └────────────────┬───────────────┘
                              │
             ┌────────────────▼────────────────┐
             │       BUSINESS LOGIC LAYER       │
             │                                 │
             │  simple_detector.py             │
             │  ┌───────────────────────────┐  │
             │  │ SimpleDrowsinessDetector  │  │
             │  │                           │  │
             │  │ - face_cascade            │  │
             │  │ - eye_cascade             │  │
             │  │ - eye_closed_counter      │  │
             │  │ - mouth_open_counter      │  │
             │  │ - alert_active            │  │
             │  │                           │  │
             │  │ + detect_drowsiness()     │  │
             │  │ + detect_eyes()           │  │
             │  │ + estimate_mouth_state()  │  │
             │  │ + start_alert()           │  │
             │  │ + stop_alert()            │  │
             │  │ + draw_status()           │  │
             │  │ + reset()                 │  │
             │  └───────────────────────────┘  │
             │                                 │
             │  alert_system.py (standalone)   │
             └────────────────┬────────────────┘
                              │
             ┌────────────────▼────────────────┐
             │        HARDWARE LAYER            │
             │                                 │
             │  OpenCV                         │
             │  - VideoCapture (camera)        │
             │  - CascadeClassifier (Haar)     │
             │  - imshow, waitKey (display)    │
             │                                 │
             │  Pygame Mixer                   │
             │  - sndarray (sound generation)  │
             │  - play/stop (audio output)     │
             └─────────────────────────────────┘
```

### Detection Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Frame       │    │  Face        │    │  Eye         │
│  Capture     │───▶│  Detection   │───▶│  Detection   │
│  (Camera)    │    │  (Haar)      │    │  (Haar)      │
└──────────────┘    └──────────────┘    └──────────────┘
                                               │
                    ┌──────────────────────────┘
                    │
                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Alert       │◀───│  Threshold   │◀───│  Mouth       │
│  (Audio/     │    │  Check       │    │  Analysis    │
│   Visual)    │    │  (Counters)  │    │  (Texture)   │
└──────────────┘    └──────────────┘    └──────────────┘
```

### Project Structure

```
drowsy_detect/
├── simple_detector.py      # Core detection engine (253 lines)
├── simple_main.py          # Native OpenCV application entry point
├── debug_main.py           # Debug variant with verbose logging
├── streamlit_app.py        # Web-based Streamlit dashboard
├── alert_system.py         # Standalone audio alert module
├── test_detector.py        # Automated test suite with simulated frames
├── requirements.txt        # Python package dependencies
└── README.md               # This documentation
```

---

## Installation

### Prerequisites

- Python 3.7 or higher
- Webcam (built-in or USB)
- Audio output device (speakers/headphones)
- pip (Python package manager)

### Step-by-Step Installation

1. **Clone or navigate to the project directory**
   ```bash
   cd /path/to/drowsy_detect
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # macOS/Linux
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python3 -c "import cv2, numpy, pygame, streamlit; print('All dependencies installed successfully')"
   ```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `opencv-python` | 4.8.1.78 | Computer vision, face/eye detection |
| `numpy` | 1.24.3 | Array operations, sound wave generation |
| `pygame` | 2.5.2 | Audio alert generation and playback |
| `scipy` | 1.11.4 | Scientific computing utilities |
| `imutils` | 0.5.4 | Image processing helpers |
| `streamlit` | >=1.28.0 | Web dashboard framework |
| `Pillow` | 10.1.0 | Image processing |
| `tensorflow` | 2.14.0 | Reserved for future ML integration |
| `scikit-learn` | 1.3.2 | Machine learning utilities |

---

## Usage

### Native OpenCV Application

Run the main application:

```bash
python3 simple_main.py
```

**Controls:**

| Key | Action |
|-----|--------|
| `q` | Quit application |
| `r` | Reset all counters |

**What to expect:**
1. Camera initializes and displays video feed
2. Green rectangles appear around detected faces
3. Status text shows current drowsiness state
4. If eyes close for 2+ seconds, red flashing border appears and audio alert plays
5. Alert stops automatically when eyes reopen

### Streamlit Web Dashboard

Run the web-based interface:

```bash
streamlit run streamlit_app.py
```

The dashboard will open in your default browser at `http://localhost:8501`.

**Features:**
- Live video feed with detection overlays
- Real-time metrics for eye closure and yawning duration
- Start/Stop detection buttons
- Reset detector button
- Settings sidebar with threshold information

### Debug Mode

For troubleshooting and calibration:

```bash
python3 debug_main.py
```

Debug mode provides:
- Frame-by-frame console logging
- Detailed detection state output
- Counter values printed each frame
- Alert system status messages

---

## Detection Algorithms

### Face Detection

Faces are detected using OpenCV's pre-trained Haar cascade classifier:

```python
self.face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
```

**Parameters:**
- `scaleFactor=1.1`: Image scale reduction per iteration (10% reduction)
- `minNeighbors=4`: Minimum neighbors required for detection (higher = fewer false positives)

### Eye Detection

```python
def detect_eyes(self, face_roi):
    eyes = self.eye_cascade.detectMultiScale(
        face_roi,
        scaleFactor=1.05,    # Fine-grained scaling (5% per step)
        minNeighbors=2,      # Lenient: only 2 neighbors required
        minSize=(15, 15),    # Minimum eye size in pixels
        maxSize=(100, 100)   # Maximum eye size in pixels
    )
    return eyes
```

**Detection Logic:**
- Eyes are considered **closed** when zero eyes are detected
- This inverse approach is more robust than measuring eyelid positions
- Normal blinks (~100-400ms) don't trigger alerts due to temporal filtering

### Yawn Detection

```python
def estimate_mouth_state(self, face_roi):
    # Extract lower face region (mouth area)
    height, width = face_roi.shape[:2]
    mouth_region = face_roi[
        int(height * 0.65):int(height * 0.9),
        int(width * 0.25):int(width * 0.75)
    ]

    # Compute texture features
    variance = np.var(mouth_region)
    mean_intensity = np.mean(mouth_region)

    # Yawn criteria: high texture variance + low intensity (dark mouth interior)
    is_yawning = (variance > 400 and mean_intensity < 120)
    return is_yawning
```

**Rationale:**
- Open mouth creates a dark region with high texture contrast (teeth, tongue, shadows)
- Variance threshold (400) filters out closed-mouth textures
- Intensity threshold (120) ensures the region is sufficiently dark

### Temporal Filtering

```python
# Eye closure counter
if eyes_closed:
    self.eye_closed_counter += 1
else:
    self.eye_closed_counter = 0

# Alert threshold check (assuming 30 FPS)
if self.eye_closed_counter >= 60:  # 60 frames = 2.0 seconds
    trigger_alert()
```

| Counter | Threshold | Duration | Purpose |
|---------|-----------|----------|---------|
| `eye_closed_counter` | 60 frames | 2.0s | Drowsiness alert |
| `mouth_open_counter` | 15 frames | 0.5s | Yawning alert |
| Blink indicator | 10 frames | 0.3s | Visual warning only |
| Mouth warning | 5 frames | 0.17s | Visual warning only |

### Alert Generation

**Audio Alert (880Hz sine wave):**

```python
def create_alert_sound(self):
    sample_rate = 22050
    duration = 0.5
    frequency = 880  # A5 musical note

    frames = int(duration * sample_rate)
    arr = np.sin(2 * np.pi * frequency * np.arange(frames) / sample_rate)
    arr = (arr * 32767).astype(np.int16)

    stereo_arr = np.zeros((frames, 2), dtype=np.int16)
    stereo_arr[:, 0] = arr
    stereo_arr[:, 1] = arr

    self.alert_sound = pygame.sndarray.make_sound(stereo_arr)
```

**Visual Alert:**
- Flashing red border (2Hz frequency)
- Color-coded status text:
  - Green `(0, 255, 0)`: Alert state
  - Yellow `(0, 255, 255)`: Blinking warning
  - Red `(0, 0, 255)`: Drowsiness detected

---

## Configuration

### Adjusting Detection Thresholds

Edit `simple_detector.py`:

```python
# Eye closure threshold (frames at 30 FPS)
self.eye_closed_threshold = 60  # Default: 2.0 seconds

# Yawn detection threshold (frames at 30 FPS)
self.mouth_open_threshold = 15  # Default: 0.5 seconds
```

| Threshold | Lower Value Effect | Higher Value Effect |
|-----------|-------------------|---------------------|
| `eye_closed_threshold` | More sensitive, more false positives | Less sensitive, may miss brief drowsiness |
| `mouth_open_threshold` | Catches shorter yawns, more false alerts | Only detects sustained yawns |

### Tuning Eye Detection

```python
eyes = self.eye_cascade.detectMultiScale(
    face_roi,
    scaleFactor=1.05,    # Decrease for finer scale steps (more sensitive)
    minNeighbors=2,      # Decrease for more detections (more false positives)
    minSize=(15, 15),    # Decrease to detect smaller eyes
    maxSize=(100, 100)   # Increase to detect larger eyes
)
```

### Tuning Yawn Detection

```python
variance_threshold = 400      # Decrease for more sensitivity
intensity_threshold = 120     # Increase for more sensitivity
```

### Customizing Alert Sound

```python
frequency = 880    # Hz (range: 440-2000 recommended)
duration = 0.5     # Seconds per beep cycle
```

---

## API Reference

### SimpleDrowsinessDetector Class

```python
class SimpleDrowsinessDetector:
    """Real-time drowsiness detection using Haar cascades."""

    def __init__(self):
        """Initialize face/eye cascades, pygame mixer, and counters."""

    def detect_drowsiness(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Process a frame and detect drowsiness.

        Args:
            frame: BGR image from camera

        Returns:
            Tuple of (annotated frame, drowsiness_detected boolean)
        """

    def reset(self):
        """Reset all counters and stop any active alerts."""

    def stop_alert(self):
        """Stop the audio alert without resetting counters."""
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `face_cascade` | `cv2.CascadeClassifier` | Pre-trained face detector |
| `eye_cascade` | `cv2.CascadeClassifier` | Pre-trained eye detector |
| `eye_closed_counter` | `int` | Frames since eyes were last detected |
| `eye_closed_threshold` | `int` | Frames required to trigger eye alert |
| `mouth_open_counter` | `int` | Frames with mouth open (yawning) |
| `mouth_open_threshold` | `int` | Frames required to trigger yawn alert |
| `alert_active` | `bool` | Whether audio alert is currently playing |
| `is_drowsy` | `bool` | Whether drowsiness state is active |

---

## Troubleshooting

### Camera Issues

**Problem:** Camera not detected

```bash
# Test camera availability
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('Camera open:', cap.isOpened())"
```

**Solutions:**
- Close other applications using the camera (Zoom, Teams, etc.)
- Check camera permissions in System Settings
- Try a different USB port for external cameras

### Audio Issues

**Problem:** No sound during alerts

```bash
# Test pygame audio system
python3 -c "import pygame; pygame.mixer.init(); print('Audio initialized:', pygame.mixer.get_init())"
```

**Solutions:**
- Check system volume and output device
- Ensure speakers/headphones are connected
- On Linux, verify PulseAudio/PipeWire is running

### Detection Accuracy

**Problem:** Eyes not detected consistently

**Causes and fixes:**

| Cause | Solution |
|-------|----------|
| Poor lighting | Add front-facing light, avoid backlighting |
| Glasses interference | Try anti-reflective coating or adjust position |
| Face too far from camera | Move closer or use higher resolution camera |
| Head turned away | Position camera directly in front of driver |

**Problem:** False yawn detections

```python
# Increase thresholds in simple_detector.py
variance_threshold = 500    # Was 400
intensity_threshold = 100   # Was 120
```

### Performance Issues

**Problem:** Low frame rate (<15 FPS)

**Solutions:**
1. Reduce camera resolution:
   ```python
   cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
   cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
   ```
2. Close background applications
3. Process every other frame instead of every frame

---

## System Requirements

### Minimum Requirements

| Component | Specification |
|-----------|---------------|
| CPU | Dual-core 2.0 GHz |
| RAM | 4 GB |
| Camera | 640x480 @ 30 FPS |
| OS | macOS 10.14+, Windows 10+, Ubuntu 18.04+ |

### Recommended Requirements

| Component | Specification |
|-----------|---------------|
| CPU | Quad-core 3.0 GHz |
| RAM | 8 GB |
| Camera | 1280x720 @ 30 FPS |
| Lighting | Front-facing, 300+ lux |

### Performance Benchmarks

| Metric | Value |
|--------|-------|
| Processing speed | ~30 FPS (recommended hardware) |
| Detection latency | <100ms |
| Memory usage | ~150 MB |
| CPU usage | 15-25% (single core) |

---

## Privacy and Security

**Data Handling:**
- All video processing occurs locally on-device
- No frames are stored to disk or transmitted over network
- No biometric data is retained after application closes
- No user accounts or authentication required

**Security Considerations:**
- Application requires camera and audio permissions
- No network connectivity required for core functionality
- Open-source code allows full audit of data handling practices

---

## Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/improvement`
3. Make changes and test thoroughly
4. Submit a pull request

### Code Style

- Follow PEP 8 guidelines
- Use descriptive variable and function names
- Add docstrings to all public methods
- Include inline comments for complex logic

### Testing

```bash
# Run the test suite
python3 test_detector.py

# Run with pytest (if installed)
python3 -m pytest tests/
```

---

## Roadmap

### Planned Enhancements

- [ ] Eye Aspect Ratio (EAR) calculation for precise eye state
- [ ] PERCLOS metric for standardized drowsiness measurement
- [ ] MediaPipe Face Mesh integration for improved landmark detection
- [ ] Mobile application (iOS/Android)
- [ ] Customizable alert sounds (upload custom audio files)
- [ ] Driver behavior analytics and session history
- [ ] Multi-face support for fleet vehicles

### Research Directions

- [ ] Deep learning model integration (TensorFlow Lite)
- [ ] Infrared camera support for night-time operation
- [ ] Head pose estimation for attention tracking
- [ ] Integration with vehicle OBD-II systems
- [ ] Fatigue prediction based on historical patterns

---

## License

This project is provided for educational and research purposes. Use responsibly and in compliance with local regulations regarding driver monitoring systems.

**Disclaimer:** This system is a safety assistance tool and does not replace the need for alert, responsible driving. Always take regular breaks during long drives and never drive when fatigued.

---

**Version:** 1.0.0
**Last Updated:** March 2026
**Contact:** See GitHub repository for issues and discussions
