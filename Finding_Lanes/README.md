<!--Please do not remove this part-->
![Star Badge](https://img.shields.io/static/v1?label=%F0%9F%8C%9F&message=If%20Useful&style=style=flat&color=BC4E99)
![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)

# ?? Finding Lanes

> Real-time and static road lane boundary detection using OpenCV and NumPy.

<p align="center">
  <img src="capture.png" width="80%" alt="Finding Lanes Demo">
</p>

## ??? Description

**Finding Lanes** is a modular Python computer vision project that detects and highlights road lane boundaries from static images, video files, or live camera feeds. 

The pipeline processes each frame through several standard computer vision stages:
1. **Grayscale Conversion & Gaussian Blur**: Reduces image noise and gradient variance.
2. **Canny Edge Detection**: Identifies sharp brightness transitions indicating potential lane boundaries.
3. **Dynamic Region of Interest (ROI)**: Dynamically computes a triangular polygon mask proportional to frame dimensions to eliminate irrelevant road surroundings and sky.
4. **Hough Transform Line Detection (`cv2.HoughLinesP`)**: Identifies line segments from edge pixels.
5. **Slope-Intercept Regression & Extrapolation**: Separates left (negative slope) and right (positive slope) line segments, filters out horizontal noise markings, and extrapolates continuous lane boundary lines.
6. **Alpha Blending (`cv2.addWeighted`)**: Overlays highlighted lane boundaries onto the original frame.

---

## ? Features

- **Dynamic ROI**: Adapts automatically to arbitrary image/video resolutions and aspect ratios.
- **Robust Line Filtering**: Ignores horizontal markings and guards against zero-division, `NaN`, or out-of-bound coordinates.
- **Multi-Source Support**: Seamlessly processes static images (`picture.jpg`), video files (`video.mp4`), or live camera streams (`--camera 0`).
- **Headless & Batch Support**: `--no-show` flag and `--output` options for saving results in automated/CI pipelines.
- **Pipeline Visualizer (`sub.py`)**: 4-panel subplot showing Original, Canny Edges, ROI Masked, and Lane Detection stages side-by-side.
- **Comprehensive Unit Tests (`test_lanes.py`)**: 100% test coverage over all core pipeline functions and edge cases.

---

## ?? Requirements & Installation

Ensure you have Python 3.8+ installed. Install the required dependencies:

```sh
pip install -r requirements.txt
```

Or install dependencies manually:

```sh
pip install opencv-python numpy matplotlib
```

---

## ?? How to Run

Navigate to the project folder:

```sh
cd Finding_Lanes
```

### 1. Run Lane Detection on Video (Default)
```sh
python lanes.py
```
> Press **`q`** on the video window to quit.

### 2. Run Lane Detection on a Static Image
```sh
python lanes.py --image picture.jpg
```

### 3. Run on Custom Video and Save Output
```sh
python lanes.py --video path/to/video.mp4 --output output.mp4
```

### 4. Run on Live Webcam
```sh
python lanes.py --camera 0
```

### 5. Multi-Stage Pipeline Visualizer (`sub.py`)
To inspect intermediate computer vision stages side-by-side using Matplotlib:
```sh
python sub.py
```
Or with custom image and save output:
```sh
python sub.py --image picture.jpg --output pipeline_stages.png
```

---

## ?? Running Unit Tests

Run the test suite using Python's built-in `unittest` runner:

```sh
python -m unittest test_lanes.py -v
```

---

## ?? Project Structure

```
Finding_Lanes/
??? README.md          # Project documentation and guide
??? lanes.py           # Core lane detection engine and CLI
??? sub.py             # 4-stage pipeline visualization utility
??? test_lanes.py      # Unit test suite
??? picture.jpg        # Sample input road image
??? video.mp4          # Sample input road driving video
??? capture.png        # Demo output screenshot
```

---

## ?? Demo

<p align="center">
  <img src="capture.png" width="100%" alt="Finding Lanes Demo Output">
</p>

---

## ?? Author

- Original Script: **zmdlw** ([@zmdlw](https://github.com/zmdlw))
- Enhancements & Tests: **Prayas Dey** ([@Prayas340](https://github.com/Prayas340))
