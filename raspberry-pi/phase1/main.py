import argparse
import time
import cv2

from picamera2 import Picamera2
from gpiozero import DistanceSensor
from ultralytics import YOLO

from config.settings import (
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_FPS,
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


def main(headless=False):

    print("Initializing camera...")

    camera = Picamera2()

    camera_config = camera.create_preview_configuration(
        main={
            "size": (CAMERA_WIDTH, CAMERA_HEIGHT),
            "format": "RGB888"
        },
        controls={"FrameRate": CAMERA_FPS}
    )

    camera.configure(camera_config)

    print("Initializing HC-SR04...")

    sensor = DistanceSensor(
        echo=ECHO_PIN,
        trigger=TRIG_PIN,
        max_distance=MAX_DISTANCE_METERS,
        queue_len=5
    )

    print("Loading AI model...")

    model = YOLO(str(MODEL_PATH))

    camera.start()

    time.sleep(2)

    print("Smart Security System started")

    previous_time = time.perf_counter()

    fps = 0.0

    try:

        while True:

            # 1. Capture image
            frame = camera.capture_array()

            # 2. Measure distance
            distance = get_distance(sensor)

            status, color = get_distance_status(distance)

            # 3. Person detection and tracking
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

            # 4. Draw detected persons
            if result.boxes is not None:

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

                    # Color indicative de la proximité globale.
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2
                    )

            # 5. FPS calculation
            current_time = time.perf_counter()

            elapsed = current_time - previous_time

            if elapsed > 0:
                instantaneous_fps = 1.0 / elapsed

                fps = (
                    0.9 * fps
                    + 0.1 * instantaneous_fps
                )

            previous_time = current_time

            # 6. Display system information
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
                0.7,
                color,
                2
            )

            cv2.putText(
                frame,
                f"Status: {status}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # 7. Display
            if not headless:

                cv2.imshow(WINDOW_NAME, frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:

        print("Stopping application...")

    finally:

        camera.stop()
        sensor.close()

        if not headless:
            cv2.destroyAllWindows()

        print("System stopped successfully")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without graphical window"
    )

    args = parser.parse_args()

    main(headless=args.headless)