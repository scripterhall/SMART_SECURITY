
import argparse
import logging
import math
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

from gpiozero import DistanceSensor
from ultralytics import YOLO

from config.settings import (
    TRIG_PIN,
    ECHO_PIN,
    MAX_DISTANCE_METERS,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CONFIDENCE_THRESHOLD,
    IMAGE_SIZE,
    MODEL_PATH,
    PERSON_CLASS_ID,
    WARNING_DISTANCE_CM,
    CRITICAL_DISTANCE_CM,
    WINDOW_NAME
)

RASPBERRY_PI_DIR = Path(__file__).resolve().parents[1]
if str(RASPBERRY_PI_DIR) not in sys.path:
    sys.path.insert(0, str(RASPBERRY_PI_DIR))

from camera.day_night_controller import (
    CameraConfig,
    DayNightCameraController
)


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger("SmartSecurity")


# =========================================================
# DISTANCE SENSOR
# =========================================================

class DistanceReader:

    def __init__(self, sensor):
        self.sensor = sensor
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._read_loop,
            name="ultrasonic-reader",
            daemon=True
        )
        self._distance = None
        self._updated_at = 0.0

    def start(self):
        self._thread.start()

    def _read_loop(self):
        while not self._stop_event.is_set():
            try:
                distance = float(self.sensor.distance) * 100
                if not math.isfinite(distance):
                    raise ValueError("Ultrasonic sensor returned a non-finite value")

                with self._lock:
                    self._distance = distance
                    self._updated_at = time.monotonic()

            except Exception:
                logger.exception("Ultrasonic distance read failed")
                with self._lock:
                    self._distance = None
                    self._updated_at = 0.0
                self._stop_event.wait(0.1)
                continue

            self._stop_event.wait(0.05)

    def get_distance(self):
        with self._lock:
            if time.monotonic() - self._updated_at > 0.5:
                return None
            return self._distance

    def stop(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


class DetectionWorker:

    def __init__(self, camera, model):
        self.camera = camera
        self.model = model
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._detect_loop,
            name="person-detection",
            daemon=True
        )
        self._detections = []
        self._frame_id = -1
        self._fps = 0.0
        self._last_run = time.monotonic()

    def start(self):
        self._thread.start()

    def _detect_loop(self):
        while not self._stop_event.is_set():
            frame, frame_id = self.camera.get_latest_frame()
            if frame is None or frame_id == self._frame_id:
                self._stop_event.wait(0.005)
                continue

            self._frame_id = frame_id
            input_height = IMAGE_SIZE * frame.shape[0] // frame.shape[1]
            inference_frame = cv2.resize(
                frame,
                (IMAGE_SIZE, input_height),
                interpolation=cv2.INTER_AREA
            )

            try:
                results = self.model.track(
                    inference_frame,
                    persist=True,
                    classes=[PERSON_CLASS_ID],
                    conf=CONFIDENCE_THRESHOLD,
                    imgsz=IMAGE_SIZE,
                    device="cpu",
                    verbose=False
                )

                result = results[0]
                detections = []
                if result.boxes is not None:
                    boxes = result.boxes
                    coordinates = boxes.xyxy.tolist()
                    track_ids = (
                        boxes.id.tolist()
                        if boxes.id is not None
                        else [None] * len(coordinates)
                    )
                    scale_x = frame.shape[1] / IMAGE_SIZE
                    scale_y = frame.shape[0] / input_height

                    for coordinates_for_box, track_id in zip(
                        coordinates,
                        track_ids
                    ):
                        x1, y1, x2, y2 = coordinates_for_box
                        detections.append((
                            round(x1 * scale_x),
                            round(y1 * scale_y),
                            round(x2 * scale_x),
                            round(y2 * scale_y),
                            int(track_id) if track_id is not None else None
                        ))

                now = time.monotonic()
                elapsed = now - self._last_run
                self._last_run = now
                inference_fps = 1.0 / elapsed if elapsed > 0 else 0.0

                with self._lock:
                    self._detections = detections
                    self._fps = inference_fps

            except Exception:
                logger.exception("Person detection failed")
                self._stop_event.wait(0.05)

    def get_detections(self):
        with self._lock:
            return self._detections.copy(), self._fps

    def stop(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("Detection thread did not stop within 5 seconds")


def get_distance_status(distance):

    if distance is None:
        return "SENSOR ERROR", (0, 165, 255)

    if distance <= CRITICAL_DISTANCE_CM:
        return "CRITICAL", (0, 0, 255)

    if distance <= WARNING_DISTANCE_CM:
        return "WARNING", (0, 165, 255)

    return "NORMAL", (0, 255, 0)


# =========================================================
# DRAWING
# =========================================================

def draw_persons(frame, detections, color, displayed_at):

    person_count = len(detections)
    scale = max(0.7, min(1.1, frame.shape[1] / 1280))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75 * scale
    thickness = max(2, round(2 * scale))
    padding = max(5, round(8 * scale))

    for x1, y1, x2, y2, person_id in detections:

        id_label = (
            f"ID: {person_id}"
            if person_id is not None
            else "ID: en attente"
        )
        date_label = displayed_at
        id_size, id_baseline = cv2.getTextSize(
            id_label,
            font,
            font_scale,
            thickness
        )
        date_size, date_baseline = cv2.getTextSize(
            date_label,
            font,
            font_scale,
            thickness
        )
        label_width = max(id_size[0], date_size[0]) + padding * 2
        label_height = (
            id_size[1]
            + date_size[1]
            + id_baseline
            + date_baseline
            + padding * 3
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            max(2, round(3 * scale))
        )

        label_x = min(
            max(0, x1),
            max(0, frame.shape[1] - label_width)
        )
        label_y = y2 + 1
        if label_y + label_height > frame.shape[0]:
            label_y = max(0, y1 - label_height)

        cv2.rectangle(
            frame,
            (label_x, label_y),
            (
                min(frame.shape[1] - 1, label_x + label_width),
                min(frame.shape[0] - 1, label_y + label_height)
            ),
            color,
            cv2.FILLED
        )
        cv2.putText(
            frame,
            id_label,
            (label_x + padding, label_y + padding + id_size[1]),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            date_label,
            (
                label_x + padding,
                label_y + padding * 2 + id_size[1] + date_size[1]
            ),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

        person_count += 1

    return person_count


# =========================================================
# MAIN APPLICATION
# =========================================================

def main(headless=False):

    camera = None
    sensor = None
    distance_reader = None
    detection_worker = None

    try:

        # -------------------------------------------------
        # 1. INITIALIZE CAMERA CONTROLLER
        # -------------------------------------------------

        logger.info("Initializing camera controller...")

        camera = DayNightCameraController(
            CameraConfig(
                width=CAMERA_WIDTH,
                height=CAMERA_HEIGHT,
                frame_duration_min=33333,
                frame_duration_max=40000
            )
        )
        camera.start()

        # -------------------------------------------------
        # 2. INITIALIZE HC-SR04
        # -------------------------------------------------

        logger.info("Initializing HC-SR04...")

        sensor = DistanceSensor(
            echo=ECHO_PIN,
            trigger=TRIG_PIN,
            max_distance=MAX_DISTANCE_METERS,
            queue_len=3,
            partial=True
        )
        distance_reader = DistanceReader(sensor)
        distance_reader.start()

        # -------------------------------------------------
        # 3. LOAD YOLO MODEL
        # -------------------------------------------------

        logger.info("Loading YOLO model...")

        model_path = str(MODEL_PATH)
        if not MODEL_PATH.is_file():
            model_path = MODEL_PATH.name
            logger.warning(
                "Local model not found at %s; asking Ultralytics to load %s",
                MODEL_PATH,
                model_path
            )

        model = YOLO(model_path)

        logger.info("YOLO model loaded successfully")
        detection_worker = DetectionWorker(camera, model)
        detection_worker.start()

        # -------------------------------------------------
        # 4. START APPLICATION
        # -------------------------------------------------

        logger.info("Smart Security System started")

        previous_time = time.perf_counter()
        fps = 0.0

        while True:

            # ---------------------------------------------
            # A. GET CAMERA FRAME
            # ---------------------------------------------

            frame, _ = camera.get_latest_frame()

            if frame is None:
                time.sleep(0.01)
                continue

            # ---------------------------------------------
            # B. GET CAMERA STATUS
            # ---------------------------------------------

            camera_status = camera.get_status()
            displayed_at = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

            detections, detection_fps = detection_worker.get_detections()

            # Use the latest sensor reading after inference, so a slow
            # detection pass does not make the displayed distance stale.
            distance = distance_reader.get_distance()
            status, color = get_distance_status(distance)

            # ---------------------------------------------
            # E. DRAW DETECTED PERSONS
            # ---------------------------------------------

            person_count = draw_persons(
                frame,
                detections,
                color,
                displayed_at
            )

            # ---------------------------------------------
            # F. CALCULATE FPS
            # ---------------------------------------------

            current_time = time.perf_counter()

            elapsed = current_time - previous_time

            if elapsed > 0:

                instantaneous_fps = 1.0 / elapsed

                fps = (
                    0.9 * fps
                    + 0.1 * instantaneous_fps
                )

            previous_time = current_time

            # ---------------------------------------------
            # G. SYSTEM INFORMATION
            # ---------------------------------------------

            distance_text = (
                f"{distance:.1f} cm"
                if distance is not None
                else "Unavailable"
            )

            cv2.putText(
                frame,
                f"Distance: {distance_text}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color,
                3,
                cv2.LINE_AA
            )

            cv2.putText(
                frame,
                f"Status: {status}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color,
                3,
                cv2.LINE_AA
            )

            cv2.putText(
                frame,
                f"Persons: {person_count}",
                (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                3,
                cv2.LINE_AA
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                3,
                cv2.LINE_AA
            )

            cv2.putText(
                frame,
                f"Detection FPS: {detection_fps:.1f}",
                (20, 270),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                3,
                cv2.LINE_AA
            )

            cv2.putText(
                frame,
                f"Camera: {camera_status['mode'].upper()}",
                (20, 225),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                3,
                cv2.LINE_AA
            )

            # ---------------------------------------------
            # H. DISPLAY
            # ---------------------------------------------

            if not headless:

                cv2.imshow(
                    WINDOW_NAME,
                    frame
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:

        logger.info("Stopping application...")

    except Exception:
        logger.exception("Application error")
        raise

    finally:

        # ---------------------------------------------
        # CLEANUP
        # ---------------------------------------------

        if detection_worker is not None:
            detection_worker.stop()

        if camera is not None:
            try:
                camera.stop()
            except Exception:
                logger.exception("Failed to stop camera cleanly")

        if sensor is not None:
            if distance_reader is not None:
                distance_reader.stop()
            try:
                sensor.close()
            except Exception:
                logger.exception("Failed to close distance sensor cleanly")

        if not headless:
            cv2.destroyAllWindows()

        logger.info("System stopped successfully")


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without graphical window"
    )

    args = parser.parse_args()

    main(headless=args.headless)
