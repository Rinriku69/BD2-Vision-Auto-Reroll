# Infinite Gacha Reroll Automation (YOLO + OpenCV)

Automated rerolling system for gacha-based games using computer vision.
This project leverages YOLO object detection and OpenCV template matching to identify desired characters and fully automate the reroll loop.

---

## Overview

This tool continuously performs gacha pulls and evaluates results using trained detection models.
It simulates user input and interacts with the UI in real time, allowing for hands-free rerolling until target characters are found.

Core pipeline:

1. Capture screen
2. Detect UI elements (e.g. buttons)
3. Execute actions (click, skip)
4. Detect characters using YOLO
5. Verify targets using template matching
6. Repeat until success condition is met

---

## Features

* Real-time object detection using YOLO (Ultralytics)
* UI automation with PyAutoGUI and DirectInput
* Multi-scale template matching for character verification
* Hotkey control for starting/stopping the script
* Infinite reroll loop with stop condition
* Sound alert when targets are found

---

## Tech Stack

* Python 3.x
* OpenCV
* Ultralytics YOLO
* NumPy
* PyAutoGUI / PyDirectInput
* Keyboard
* WinSound (Windows only)

---

## Project Structure

```
.
├── model/
│   └── train/
│       └── runs/
│           └── detect/
│               ├── char/
│               └── ui/
├── public/
│   ├── characters/
│   └── sound/
├── main.py
└── README.md
```

---

## Installation

Clone the repository:

```
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Usage

Run the script:

```
python main.py
```

Hotkeys:

* Toggle script: `F9`
* Exit script: `ESC`

---

## Configuration

Modify parameters in `main.py`:

* Model paths:

```
CHAR_MODEL_PATH = 'model/.../char/.../best.pt'
UI_MODEL_PATH = 'model/.../ui/.../best.pt'
```

* Target characters:

```
["blade1"]
```

* Timing settings:

```
DELAY_AFTER_CLICK
DELAY_AFTER_PULL_SEQUENCE
DELAY_BEFORE_RETRY
```

---

## How It Works

### 1. Character Detection

Uses YOLO model to locate 5-star characters from screenshots.

### 2. UI Detection

Detects buttons such as "Redraw" and "Skip" to control flow.

### 3. Template Matching

Each detected character is verified using grayscale template matching across multiple scales.

### 4. Automation Loop

Continuously performs:

* Click redraw
* Skip animations
* Scan results
* Compare against targets

Stops when all required targets are detected.

---

## Requirements

* Windows OS (required for input simulation and sound playback)
* GPU recommended (CUDA support for YOLO)
* Minimum 16GB RAM suggested for stable performance

---

## Limitations

* Designed for specific screen resolution and UI layout
* Requires trained YOLO models
* Sensitive to UI changes or visual differences
* May consume high CPU/GPU resources during long runs

---

## Disclaimer

This project is intended for educational purposes only.
Automating gameplay may violate the terms of service of certain games. Use at your own risk.

---

## Future Improvements

* Memory optimization for long-running sessions
* Dynamic resolution scaling
* Model confidence tuning UI
* Logging and statistics dashboard

---

## Author

Developed as a computer vision automation experiment combining real-time detection and input control.

---
