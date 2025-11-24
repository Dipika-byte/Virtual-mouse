## Overview

This project implements a virtual mouse using hand gestures and MediaPipe. The system allows the user to control the mouse, click, drag, and take screenshots using predefined hand gestures.

# Virtual Mouse Using Hand Gestures

## Features
- Move cursor using index finger
- Left-click when index & middle fingers are close
- Right-click when index & middle fingers are far apart

## How to Run
1. Install requirements: `pip install -r requirements.txt`
2. Run: `python virtual_mouse.py`
3. Show hand gestures to control mouse

## Optional
- Collect your own gesture data using `data_collection.py`
- Train gesture recognition using `train_gesture_model.py`

# 🖱️ Virtual Mouse Using Hand Gestures

A Python-based virtual mouse that allows you to control your cursor using hand gestures via webcam. It uses MediaPipe for hand tracking and PyAutoGUI for mouse control.

## ✨ Features

- 🖐️ Show 5 fingers → Start system  
- ☝️ Index finger up → Move cursor  
- 👌 Thumb + Index touching → Left click  
- ✌️ Index + Middle touching → Right click  
- 👍 thumb + Ring finger touching → Drag  
- ✊ Fist (0 fingers) → Take Screenshot  
- 👍 Thumb only → Exit  
- 🛑 Press `Esc` or `q` → Emergency Exit 


## Folder Structure

```
VirtualMouseProject/
│
├── virtual_mouse.py        # Main script to run the system
├── data_collection.py      # Optional - for collecting gesture data
├── train_gesture_model.py  # Optional - for training a custom ML model
├── gesture_model.h5        # Optional - ML model for advanced gesture recognition
├── requirements.txt        # List of dependencies
├── README.md               # Project documentation
│
├── utils/
│   ├── hand_tracking.py    # Handles hand detection and tracking
│   └── gesture_utils.py    # Utility functions for gesture recognition
│
└── media/
    ├── sample_gesture.png  # Example gesture images
    └── demo_video.mp4      # Demo video of the system
```

---

# Details 

import cv2
import pyautogui
import numpy as np
from utils.hand\_tracking import HandTracker
from utils.gesture\_utils import (
fingers\_up,
get\_finger\_distance,
is\_thumb\_index\_touching,
is\_index\_middle\_touching,
is\_thumb\_ring\_touching
)

# Parameters

frame\_width, frame\_height = 640, 480
screen\_width, screen\_height = pyautogui.size()
dragging = False
screenshot\_counter = 0
system\_started = False

# Initialize HandTracker

hand\_tracker = HandTracker(max\_hands=1, detection\_confidence=0.5, tracking\_confidence=0.5)

# Start Video Capture

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP\_PROP\_FRAME\_WIDTH, frame\_width)
cap.set(cv2.CAP\_PROP\_FRAME\_HEIGHT, frame\_height)

while cap.isOpened():
ret, frame = cap.read()
if not ret:
print("Failed to grab frame")
break

```
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
result = hand_tracker.process_frame(frame_rgb)

# Debug: Display frame size
print("Frame Size:", frame.shape)

if result.multi_hand_landmarks:
    for hand_landmarks in result.multi_hand_landmarks:
        print("Hand Detected")

        # Debug: Show all landmarks coordinates
        for id, lm in enumerate(hand_landmarks.landmark):
            print(f"ID: {id}, X: {lm.x}, Y: {lm.y}")

        finger_status = fingers_up(hand_landmarks)
        print("Finger Status:", finger_status)

        if finger_status == [1, 1, 1, 1, 1]:
            system_started = True
            cv2.putText(frame, "🟢 System Started", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            continue

        if not system_started:
            cv2.putText(frame, "✋ Show 5 fingers to start", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            continue

        ix, iy = int(hand_landmarks.landmark[8].x * frame_width), int(hand_landmarks.landmark[8].y * frame_height)
        pyautogui.moveTo(screen_width / frame_width * ix, screen_height / frame_height * iy)

        if is_thumb_index_touching(hand_landmarks, frame_width, frame_height):
            pyautogui.click()
            cv2.putText(frame, "Left Click", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        elif is_index_middle_touching(hand_landmarks, frame_width, frame_height):
            pyautogui.rightClick()
            cv2.putText(frame, "Right Click", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        elif is_thumb_ring_touching(hand_landmarks, frame_width, frame_height):
            if not dragging:
                pyautogui.mouseDown()
                dragging = True
            cv2.putText(frame, "Dragging...", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            if dragging:
                pyautogui.mouseUp()
                dragging = False

        if sum(finger_status) == 0:
            cv2.imwrite(f"screenshot_{screenshot_counter}.png", frame)
            screenshot_counter += 1
            cv2.putText(frame, "📸 Screenshot", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 255), 2)

        if finger_status == [1, 0, 0, 0, 0]:
            cv2.putText(frame, "👋 Exiting...", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            break

cv2.imshow("Virtual Mouse", frame)

# Exit Conditions
if cv2.waitKey(1) & 0xFF == ord('q'):
    break
```

cap.release()
cv2.destroyAllWindows()
hand\_tracker.close()





## Installation

1. Clone the repository:

   ```bash
   git clone <repository_url>
   cd VirtualMouseProject
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv env
   .\env\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Adjust system settings for optimal performance (Optional):

   * Reduce frame size to 640x480
   * Limit FPS to 15 for better CPU performance

---

## How to Run

1. Navigate to the project directory:

   ```bash
   cd VirtualMouseProject
   ```

2. Run the virtual mouse system:

   ```bash
   python virtual_mouse.py
   ```

3. Press 'q' at any time to exit the program.

---

## Troubleshooting

* If the camera feed is not showing, ensure that your webcam is properly connected.
* If the system is slow, consider reducing the frame size or FPS in `virtual_mouse.py`.
* Ensure all dependencies are installed correctly.

---

## Future Enhancements

* Implement advanced gestures using custom ML models.
* Add multi-hand support.
* Enable voice commands for accessibility.

