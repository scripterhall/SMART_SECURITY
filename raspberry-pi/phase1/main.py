
import argparse
import logging
import time

import cv2

from gpiozero import DistanceSensor
from ultralytics import YOLO

from camera.day_night_controller import DayNightCameraController

from config.settings import (
    TRIG_PIN,
    ECHO_PIN,
    MAX_DISTANCE_METERS,
    CONFIDENCE_THRESHOLD,
    IMAGE_SIZE,
    MODEL_PATH,
    PERSON_CLASS_ID,
    WARNING_DISTANCE_CM,
    CRITICAL_DISTANCE_CM,
    WINDOW_NAME
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

def get_distance(sensor):

    try:
        return sensor.distance * 100

    except Exception:
        return None


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

def draw_persons(frame, result, color):

    person_count = 0

    if result.boxes is None:
        return person_count

    boxes = result.boxes

    for index, box in enumerate(boxes):

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        person_id = None

        if boxes.id is not None:
            person_id = int(
                boxes.id[index].item()
            )

        label = (
            f"ID-{person_id}"
            if person_id is not None
            else "PERSON"
        )

        # Bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        # Label
        cv2.putText(
            frame,
            label,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        person_count += 1

    return person_count


# =========================================================
# MAIN APPLICATION
# =========================================================

def main(headless=False):

    camera = None
    sensor = None

    try:

        # -------------------------------------------------
        # 1. INITIALIZE CAMERA CONTROLLER
        # -------------------------------------------------

        logger.info("Initializing camera controller...")

        camera = DayNightCameraController()
        camera.start()

        # -------------------------------------------------
        # 2. INITIALIZE HC-SR04
        # -------------------------------------------------

        logger.info("Initializing HC-SR04...")

        sensor = DistanceSensor(
            echo=ECHO_PIN,
            trigger=TRIG_PIN,
            max_distance=MAX_DISTANCE_METERS,
            queue_len=5
        )

        # -------------------------------------------------
        # 3. LOAD YOLO MODEL
        # -------------------------------------------------

        logger.info("Loading YOLO model...")

        model = YOLO(str(MODEL_PATH))

        logger.info("YOLO model loaded successfully")

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

            frame = camera.get_frame()

            if frame is None:
                time.sleep(0.01)
                continue

            # ---------------------------------------------
            # B. GET CAMERA STATUS
            # ---------------------------------------------

            camera_status = camera.get_status()

            # ---------------------------------------------
            # C. MEASURE DISTANCE
            # ---------------------------------------------

            distance = get_distance(sensor)

            status, color = get_distance_status(distance)

            # ---------------------------------------------
            # D. HUMAN DETECTION + TRACKING
            # ---------------------------------------------

            results = model.track(
                frame,
                persist=True,
                classes=[PERSON_CLASS_ID],
                conf=CONFIDENCE_THRESHOLD,
                imgsz=IMAGE_SIZE,
                device="cpu",
                verbose=False
            )

            result = results[0]

            # ---------------------------------------------
            # E. DRAW DETECTED PERSONS
            # ---------------------------------------------

            person_count = draw_persons(
                frame,
                result,
                color
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
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

            cv2.putText(
                frame,
                f"Status: {status}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

            cv2.putText(
                frame,
                f"Persons: {person_count}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Camera: {camera_status['mode'].upper()}",
                (20, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
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

    finally:

        # ---------------------------------------------
        # CLEANUP
        # ---------------------------------------------

        if camera is not None:
            camera.stop()

        if sensor is not None:
            sensor.close()

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
