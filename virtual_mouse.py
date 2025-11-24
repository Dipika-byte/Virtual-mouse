import cv2
import mediapipe as mp
import pyautogui
import time
from utils.gesture_utils import (
    fingers_up,
    is_thumb_index_touching,
    is_index_middle_touching,
    is_thumb_ring_touching,
    is_fist,
    is_thumb_up
)
from utils.hand_tracking import HandTracker

# Constants
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
FPS = 15
PROCESS_INTERVAL = 2
screen_width, screen_height = pyautogui.size()

# Initialize Hand Tracker
tracker = HandTracker(max_hands=1, detection_confidence=0.7, tracking_confidence=0.7)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)

frame_counter = 0
screenshot_counter = 0
system_started = False
dragging = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip for mirror effect
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_counter += 1

    if frame_counter % PROCESS_INTERVAL != 0:
        cv2.imshow("Virtual Mouse", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Process frame
    result = tracker.process_frame(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            finger_status = fingers_up(hand_landmarks)

            if finger_status == [1, 1, 1, 1, 1] and not system_started:
                system_started = True
                cv2.putText(frame, "System Started", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                continue

            if not system_started:
                cv2.putText(frame, "Show 5 Fingers to Start", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Virtual Mouse", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            ix, iy = int(hand_landmarks.landmark[8].x * FRAME_WIDTH), int(hand_landmarks.landmark[8].y * FRAME_HEIGHT)
            pyautogui.moveTo(screen_width / FRAME_WIDTH * ix, screen_height / FRAME_HEIGHT * iy)

            # Left Click
            if is_thumb_index_touching(hand_landmarks, FRAME_WIDTH, FRAME_HEIGHT):
                pyautogui.click()
                cv2.putText(frame, "Left Click", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            # Right Click
            elif is_index_middle_touching(hand_landmarks, FRAME_WIDTH, FRAME_HEIGHT):
                pyautogui.rightClick()
                cv2.putText(frame, "Right Click", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Drag
            elif is_thumb_ring_touching(hand_landmarks, FRAME_WIDTH, FRAME_HEIGHT):
                if not dragging:
                    pyautogui.mouseDown()
                    dragging = True
                cv2.putText(frame, "Dragging...", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False

            # Screenshot
            if is_fist(hand_landmarks, FRAME_WIDTH, FRAME_HEIGHT):
                cv2.imwrite(f"screenshot_{screenshot_counter}.png", frame)
                screenshot_counter += 1
                cv2.putText(frame, "Screenshot", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 255), 2)

            # Exit
            if is_thumb_up(hand_landmarks):
                cv2.putText(frame, "Exiting...", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                break

    cv2.imshow("Virtual Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()