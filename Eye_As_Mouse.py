"""
SAHAY - Head/Eye Controlled Assistive Interface
------------------------------------------------
Flask bridge for the existing MediaPipe + OpenCV + PyAutoGUI tracker.

Run:
    pip install flask opencv-python mediapipe pyautogui numpy twilio psutil
    python Eye_As_Mouse.py

Then open:
    http://127.0.0.1:5000
"""

import os
import sys
import subprocess
import urllib.request
import time
import json
import threading

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from flask import Flask, render_template, jsonify

import cv2
import numpy as np
import mediapipe as mp
import pyautogui
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Optional Twilio Import
try:
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False

# ============================================================
# TWILIO & EMERGENCY CREDENTIALS
# ============================================================
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Explicitly load .env from the SAHAY project folder
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
CAREGIVER_PHONE_NUMBER = os.getenv("CAREGIVER_PHONE_NUMBER")

def send_caregiver_alert(reason="Manual Emergency Button Pressed"):
    """
    Sends an SMS alert via Twilio.
    """

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    alert_msg = (
        f"🚨 SAHAY EMERGENCY ALERT [{timestamp}]: "
        f"{reason}. Please check on the patient immediately!"
    )

    print(f"\n[EMERGENCY TRIGGERED]: {alert_msg}\n")

    # Check credentials
    if not all([
        TWILIO_AVAILABLE,
        TWILIO_ACCOUNT_SID,
        TWILIO_AUTH_TOKEN,
        TWILIO_PHONE_NUMBER,
        CAREGIVER_PHONE_NUMBER
    ]):
        print("[Twilio ERROR] Missing Twilio configuration.")
        return False, "Twilio configuration is missing."

    try:
        # Create Twilio client
        client = Client(
            TWILIO_ACCOUNT_SID,
            TWILIO_AUTH_TOKEN
        )

        # Send SMS
        message = client.messages.create(
            body=alert_msg,
            from_=TWILIO_PHONE_NUMBER,
            to=CAREGIVER_PHONE_NUMBER
        )

        print(f"[Twilio SUCCESS] SMS Sent! SID: {message.sid}")

        return True, "Caregiver alerted via SMS."

    except Exception as e:
        print(f"[Twilio ERROR] Failed to send SMS: {e}")

        return False, f"Failed to send SMS: {str(e)}"

# ============================================================
# PATHS / CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "face_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

FPS_FILE = os.path.join(BASE_DIR, "fps_status.json")

SMOOTHING = 5
BLINK_THRESHOLD = 0.21
CLICK_COOLDOWN = 5
CAM_INDEX = 0
CALIB_SAMPLES = 20
CALIB_MARGIN = 0.08

# Inactivity / Emergency Config
NO_RESPONSE_TIMEOUT = 30  # Seconds without face detection before alert


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
        ["py", os.path.abspath(__file__), "--track"],
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
        if PSUTIL_AVAILABLE:
            try:
                parent = psutil.Process(tracking_process.pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
            except Exception:
                tracking_process.terminate()
        else:
            tracking_process.terminate()

        tracking_process.wait(timeout=3)
        tracking_process = None

        try:
            with open(FPS_FILE, "w") as f:
                json.dump({"fps": 0}, f)
        except Exception:
            pass

        return jsonify({
            "status": "stopped",
            "message": "Eye tracking stopped."
        })

    return jsonify({
        "status": "not_running",
        "message": "Eye tracking is not running."
    })


@app.route("/emergency", methods=["POST"])
def trigger_emergency():
    """Manual Emergency route triggered by frontend button."""
    # Runs in a separate thread so UI does not wait for HTTP SMS network response
    sms_thread = threading.Thread(
        target=send_caregiver_alert, 
        kwargs={"reason": "Patient pressed the emergency button"}
    )
    sms_thread.start()

    return jsonify({
        "status": "success",
        "message": "🚨 Emergency alert sent to caregiver!"
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
    if not os.path.exists(FPS_FILE):
        return jsonify({"fps": 0})
    try:
        with open(FPS_FILE, "r") as f:
            content = f.read().strip()
            data = json.loads(content) if content else {"fps": 0}
    except Exception:
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
    NOSE_TIP = 1
    nose = landmarks[NOSE_TIP]
    return nose.x, nose.y


def draw_eye_cage(frame, landmarks):
    h, w, _ = frame.shape

    def to_px(landmark):
        return int(landmark.x * w), int(landmark.y * h)

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
    """Run calibration + tracking workflow with continuous safety checks."""

    if not os.path.exists(MODEL_PATH):
        print("Downloading face landmark model (one-time, ~5MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")

    screen_w, screen_h = pyautogui.size()
    pyautogui.FAILSAFE = False

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
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

    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145
    LEFT_EYE_LEFT = 33
    LEFT_EYE_RIGHT = 133

    frame_timestamp_ms = 0

    def detect_landmarks(frame_bgr):
        nonlocal frame_timestamp_ms

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        frame_timestamp_ms += 33
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

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
            canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)

            cv2.circle(canvas, (px, py), 25, (0, 0, 255), -1)
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

            canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            cv2.circle(canvas, (px, py), 25, (0, 255, 0), -1)
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

            cv2.imshow("Webcam (SAHAY tracking preview)", frame)
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

    # ---------- BOUNDS & MAPPING ----------
    left_x = (collected["TOP-LEFT"][0] + collected["BOTTOM-LEFT"][0]) / 2
    right_x = (collected["TOP-RIGHT"][0] + collected["BOTTOM-RIGHT"][0]) / 2
    top_y = (collected["TOP-LEFT"][1] + collected["TOP-RIGHT"][1]) / 2
    bottom_y = (collected["BOTTOM-LEFT"][1] + collected["BOTTOM-RIGHT"][1]) / 2

    def map_to_screen(rel_x, rel_y):
        norm_x = (rel_x - left_x) / (right_x - left_x + 1e-6)
        norm_y = (rel_y - top_y) / (bottom_y - top_y + 1e-6)

        tx = int(norm_x * screen_w)
        ty = int(norm_y * screen_h)

        return max(0, min(screen_w - 1, tx)), max(0, min(screen_h - 1, ty))

    # ---------- TRACKING LOOP WITH NO-RESPONSE SAFETY ----------
    prev_x, prev_y = screen_w // 2, screen_h // 2
    click_cooldown_counter = 0

    fps_last_time = time.time()
    fps_frame_count = 0

    # Inactivity Tracking
    last_face_detected_time = time.time()
    no_response_alert_sent = False

    while True:
        success, frame = cam.read()
        if not success:
            break

        fps_frame_count += 1
        fps_now = time.time()
        fps_elapsed = fps_now - fps_last_time

        if fps_elapsed >= 1.0:
            current_fps = round(fps_frame_count / fps_elapsed)
            fps_frame_count = 0
            fps_last_time = fps_now
            try:
                with open(FPS_FILE, "w") as f:
                    json.dump({"fps": current_fps}, f)
            except Exception:
                pass

        frame = cv2.flip(frame, 1)
        landmarks = detect_landmarks(frame)

        if landmarks:
            # Reset inactivity timer whenever face is seen
            last_face_detected_time = time.time()
            no_response_alert_sent = False

            draw_eye_cage(frame, landmarks)
            rel_x, rel_y = get_gaze_raw(landmarks)
            target_x, target_y = map_to_screen(rel_x, rel_y)

            cur_x = prev_x + (target_x - prev_x) // SMOOTHING
            cur_y = prev_y + (target_y - prev_y) // SMOOTHING

            pyautogui.moveTo(cur_x, cur_y)
            prev_x, prev_y = cur_x, cur_y

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
                    frame, "CLICK", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
                )

            cv2.putText(
                frame, f"EAR: {left_ear:.2f}", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

        else:
            # Handle No Face Detected / Patient Inactivity
            inactive_seconds = time.time() - last_face_detected_time

            if inactive_seconds > 5:
                cv2.putText(
                    frame,
                    f"NO USER DETECTED ({int(inactive_seconds)}s)",
                    (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            # Trigger automatic caregiver alert after 30 continuous seconds of no detection
            if inactive_seconds >= NO_RESPONSE_TIMEOUT and not no_response_alert_sent:
                send_caregiver_alert(
                    reason=f"No user detected for {NO_RESPONSE_TIMEOUT} seconds"
                )
                no_response_alert_sent = True

        cv2.imshow("SAHAY Eye Mouse - press q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    landmarker.close()
    cv2.destroyAllWindows()

    try:
        with open(FPS_FILE, "w") as f:
            json.dump({"fps": 0}, f)
    except Exception:
        pass


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    if "--track" in sys.argv:
        run_eye_tracking()
    else:
        app.run(
            host="127.0.0.1",
            port=5000,
            debug=False,
        )