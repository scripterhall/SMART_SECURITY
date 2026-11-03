
import logging
import time

import cv2

from camera.day_night_controller import (
    DayNightCameraController
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

camera = DayNightCameraController()

try:
    camera.start()

    print("Caméra active. Appuie sur Q pour quitter.")

    while True:
        frame = camera.get_frame()

        if frame is not None:
            status = camera.get_status()

            display = frame.copy()

            text = (
                f"Mode: {status['mode']} | "
                f"FPS: {status['fps']} | "
                f"Gain: {status['analogue_gain']} | "
                f"Exposure: {status['exposure_us']} us"
            )

            cv2.putText(
                display,
                text,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                255 if len(display.shape) == 2 else (0, 255, 0),
                2
            )

            cv2.imshow("Smart Security Camera", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        time.sleep(0.01)

finally:
    camera.stop()
    cv2.destroyAllWindows()
