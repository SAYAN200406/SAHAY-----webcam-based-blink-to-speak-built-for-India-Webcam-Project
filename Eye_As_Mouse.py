"""
SAHAY - Head/Eye Controlled Assistive Interface
------------------------------------------------
Flask bridge for the existing MediaPipe + OpenCV + PyAutoGUI tracker.

Run:
    pip install flask opencv-python mediapipe pyautogui numpy
    python Eye_As_Mouse.py

Then open:
    http://127.0.0.1:5000

The browser UI is served by Flask. Clicking the START button should POST
to /start from Index.html. The actual tracking runs as a separate Python
process so OpenCV's webcam/window handling does not block Flask.

Controls:
    - Calibration: look at each red dot and press SPACE
    - Tracking: blink the LEFT eye to click
    - Tracking window: press 'q' to quit
"""

import os
import sys
import subprocess
import urllib.request
import time
import json

from flask import Flask, render_template, jsonify

import cv2
import numpy as np
import mediapipe as mp
import pyautogui
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# ============================================================
# PATHS / CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "face_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

SMOOTHING = 5
BLINK_THRESHOLD = 0.21
CLICK_COOLDOWN = 5
CAM_INDEX = 0
CALIB_SAMPLES = 20
CALIB_MARGIN = 0.08


# ============================================================
# FLASK SERVER
# ============================================================

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)

tracking_process = None


@app.route("/")
def home():
    return render_template("Index.html")


@app.route("/start", methods=["POST"])
def start_tracking():
    global tracking_process

    if tracking_process is not None and tracking_process.poll() is None:
        return jsonify({
            "status": "already_running",
            "message": "Eye tracking is already running."
        })

    tracking_process = subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), "--track"],
        cwd=BASE_DIR,
    )

    return jsonify({
        "status": "started",
        "message": "Eye tracking started."
    })


@app.route("/stop", methods=["POST"])
def stop_tracking():
    global tracking_process

    if tracking_process is not None and tracking_process.poll() is None:
        tracking_process.terminate()
        tracking_process.wait(timeout=3)
        tracking_process = None
        return jsonify({
            "status": "stopped",
            "message": "Eye tracking stopped."
        })

    return jsonify({
        "status": "not_running",
        "message": "Eye tracking is not running."
    })


@app.route("/status")
def status():
    global tracking_process

    running = (
        tracking_process is not None
        and tracking_process.poll() is None
    )

    return jsonify({
        "running": running
    })
@app.route("/fps")
def get_fps():
    try:
        with open(FPS_FILE, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {"fps": 0}
    return jsonify(data)

# ============================================================
# EYE / HEAD TRACKING
# ============================================================

def eye_aspect_ratio(landmarks, top_idx, bottom_idx, left_idx, right_idx):
    top = landmarks[top_idx]
    bottom = landmarks[bottom_idx]
    left = landmarks[left_idx]
    right = landmarks[right_idx]

    vertical = (
        (top.x - bottom.x) ** 2 +
        (top.y - bottom.y) ** 2
    ) ** 0.5

    horizontal = (
        (left.x - right.x) ** 2 +
        (left.y - right.y) ** 2
    ) ** 0.5

    return vertical / horizontal if horizontal else 1.0


def get_gaze_raw(landmarks):
    """
    Uses the nose tip as the main head-position tracking point.
    This is intentionally kept from the original SAHAY implementation.
    """
    NOSE_TIP = 1
    nose = landmarks[NOSE_TIP]
    return nose.x, nose.y


def draw_eye_cage(frame, landmarks):
    """Draw the head tracking cage and left-eye blink reference box."""
    h, w, _ = frame.shape

    def to_px(landmark):
        return int(landmark.x * w), int(landmark.y * h)

    # Main head tracking point
    NOSE_TIP = 1
    nose_px = to_px(landmarks[NOSE_TIP])

    cage_size = 60

    cv2.rectangle(
        frame,
        (nose_px[0] - cage_size, nose_px[1] - cage_size),
        (nose_px[0] + cage_size, nose_px[1] + cage_size),
        (0, 255, 255),
        2,
    )

    cv2.line(
        frame,
        (nose_px[0] - cage_size, nose_px[1]),
        (nose_px[0] + cage_size, nose_px[1]),
        (0, 150, 150),
        1,
    )

    cv2.line(
        frame,
        (nose_px[0], nose_px[1] - cage_size),
        (nose_px[0], nose_px[1] + cage_size),
        (0, 150, 150),
        1,
    )

    cv2.circle(frame, nose_px, 6, (0, 0, 255), 2)
    cv2.circle(frame, nose_px, 1, (0, 0, 255), -1)

    cv2.putText(
        frame,
        "TRACKING POINT",
        (nose_px[0] - 70, nose_px[1] - cage_size - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
    )

    # Left eye box used for blink detection
    LEFT_EYE_LEFT = 33
    LEFT_EYE_RIGHT = 133
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145

    left_pt = to_px(landmarks[LEFT_EYE_LEFT])
    right_pt = to_px(landmarks[LEFT_EYE_RIGHT])
    top_pt = to_px(landmarks[LEFT_EYE_TOP])
    bottom_pt = to_px(landmarks[LEFT_EYE_BOTTOM])

    box_left = min(left_pt[0], right_pt[0]) - 8
    box_right = max(left_pt[0], right_pt[0]) + 8
    box_top = min(top_pt[1], bottom_pt[1]) - 8
    box_bottom = max(top_pt[1], bottom_pt[1]) + 8

    cv2.rectangle(
        frame,
        (box_left, box_top),
        (box_right, box_bottom),
        (255, 0, 255),
        1,
    )


def run_eye_tracking():
    """Run the original calibration + tracking workflow."""

    # ---------- DOWNLOAD MODEL IF MISSING ----------
    if not os.path.exists(MODEL_PATH):
        print("Downloading face landmark model (one-time, ~5MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")

    # ---------- SETUP ----------
    screen_w, screen_h = pyautogui.size()
    pyautogui.FAILSAFE = False

    base_options = mp_python.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    cam = cv2.VideoCapture(CAM_INDEX)

    if not cam.isOpened():
        print("ERROR: Could not open webcam.")
        landmarker.close()
        return

    # Face landmark indices
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145
    LEFT_EYE_LEFT = 33
    LEFT_EYE_RIGHT = 133
    NOSE_TIP = 1

    frame_timestamp_ms = 0

    def detect_landmarks(frame_bgr):
        nonlocal frame_timestamp_ms

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        frame_timestamp_ms += 33

        result = landmarker.detect_for_video(
            mp_image,
            frame_timestamp_ms,
        )

        if result.face_landmarks:
            return result.face_landmarks[0]

        return None

    # ---------- CALIBRATION ----------
    CALIB_POINTS = [
        ("TOP-LEFT", CALIB_MARGIN, CALIB_MARGIN),
        ("TOP-RIGHT", 1 - CALIB_MARGIN, CALIB_MARGIN),
        ("BOTTOM-LEFT", CALIB_MARGIN, 1 - CALIB_MARGIN),
        ("BOTTOM-RIGHT", 1 - CALIB_MARGIN, 1 - CALIB_MARGIN),
    ]

    calib_window = "SAHAY Calibration - SPACE to capture"
    cv2.namedWindow(calib_window, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(
        calib_window,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN,
    )

    collected = {}

    for label, fx, fy in CALIB_POINTS:
        px = int(fx * screen_w)
        py = int(fy * screen_h)

        while True:
            canvas = np.zeros(
                (screen_h, screen_w, 3),
                dtype=np.uint8,
            )

            cv2.circle(
                canvas,
                (px, py),
                25,
                (0, 0, 255),
                -1,
            )

            cv2.putText(
                canvas,
                f"Look at {label} - press SPACE",
                (int(screen_w * 0.30), int(screen_h * 0.05)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            cv2.imshow(calib_window, canvas)

            key = cv2.waitKey(1) & 0xFF

            if key == 32:
                break

            if key == ord("q"):
                cam.release()
                landmarker.close()
                cv2.destroyAllWindows()
                return

            cam.read()

        samples_x = []
        samples_y = []

        while len(samples_x) < CALIB_SAMPLES:
            success, frame = cam.read()

            if not success:
                continue

            frame = cv2.flip(frame, 1)
            landmarks = detect_landmarks(frame)

            canvas = np.zeros(
                (screen_h, screen_w, 3),
                dtype=np.uint8,
            )

            cv2.circle(
                canvas,
                (px, py),
                25,
                (0, 255, 0),
                -1,
            )

            cv2.putText(
                canvas,
                "Hold steady, capturing...",
                (int(screen_w * 0.30), int(screen_h * 0.05)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            cv2.imshow(calib_window, canvas)

            if landmarks:
                draw_eye_cage(frame, landmarks)

            cv2.imshow(
                "Webcam (SAHAY tracking preview)",
                frame,
            )

            cv2.waitKey(1)

            if landmarks:
                rx, ry = get_gaze_raw(landmarks)
                samples_x.append(rx)
                samples_y.append(ry)

        collected[label] = (
            sum(samples_x) / len(samples_x),
            sum(samples_y) / len(samples_y),
        )

    cv2.destroyWindow(calib_window)

    # ---------- BUILD CALIBRATION BOUNDS ----------
    left_x = (
        collected["TOP-LEFT"][0] +
        collected["BOTTOM-LEFT"][0]
    ) / 2

    right_x = (
        collected["TOP-RIGHT"][0] +
        collected["BOTTOM-RIGHT"][0]
    ) / 2

    top_y = (
        collected["TOP-LEFT"][1] +
        collected["TOP-RIGHT"][1]
    ) / 2

    bottom_y = (
        collected["BOTTOM-LEFT"][1] +
        collected["BOTTOM-RIGHT"][1]
    ) / 2

    print(
        f"Calibration done. x range: "
        f"{left_x:.3f} -> {right_x:.3f}, "
        f"y range: {top_y:.3f} -> {bottom_y:.3f}"
    )

    def map_to_screen(rel_x, rel_y):
        norm_x = (
            (rel_x - left_x) /
            (right_x - left_x + 1e-6)
        )

        norm_y = (
            (rel_y - top_y) /
            (bottom_y - top_y + 1e-6)
        )

        tx = int(norm_x * screen_w)
        ty = int(norm_y * screen_h)

        tx = max(0, min(screen_w - 1, tx))
        ty = max(0, min(screen_h - 1, ty))

        return tx, ty

    # ---------- MAIN TRACKING LOOP ----------
    prev_x = screen_w // 2
    prev_y = screen_h // 2
    click_cooldown_counter = 0

    while True:
        success, frame = cam.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)
        landmarks = detect_landmarks(frame)

        if landmarks:
            draw_eye_cage(frame, landmarks)

            rel_x, rel_y = get_gaze_raw(landmarks)

            target_x, target_y = map_to_screen(
                rel_x,
                rel_y,
            )

            cur_x = (
                prev_x +
                (target_x - prev_x) // SMOOTHING
            )

            cur_y = (
                prev_y +
                (target_y - prev_y) // SMOOTHING
            )

            pyautogui.moveTo(cur_x, cur_y)

            prev_x = cur_x
            prev_y = cur_y

            left_ear = eye_aspect_ratio(
                landmarks,
                LEFT_EYE_TOP,
                LEFT_EYE_BOTTOM,
                LEFT_EYE_LEFT,
                LEFT_EYE_RIGHT,
            )

            if click_cooldown_counter > 0:
                click_cooldown_counter -= 1

            elif left_ear < BLINK_THRESHOLD:
                pyautogui.click()
                click_cooldown_counter = CLICK_COOLDOWN

                cv2.putText(
                    frame,
                    "CLICK",
                    (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

            cv2.putText(
                frame,
                f"EAR: {left_ear:.2f}",
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        cv2.imshow(
            "SAHAY Eye Mouse - press q to quit",
            frame,
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    landmarker.close()
    cv2.destroyAllWindows()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    if "--track" in sys.argv:
        run_eye_tracking()
    else:
        # debug=False is intentional: Flask's reloader would create
        # another process and can cause duplicate tracking processes.
        app.run(
            host="127.0.0.1",
            port=5000,
            debug=False,
        )
