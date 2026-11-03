
import time
import logging
from pathlib import Path

import cv2
from picamera2 import Picamera2


# ==========================================================
# CONFIGURATION
# ==========================================================

TUNING_FILE = (
    "/usr/share/libcamera/ipa/rpi/pisp/ov5647_noir.json"
)

OUTPUT_DIR = Path("camera/output")

WIDTH = 1296
HEIGHT = 972

# Fréquence fixe : environ 15 FPS
FRAME_DURATION_US = 66667

# Seuils initiaux de luminosité (0 à 255)
NIGHT_THRESHOLD = 45
DAY_THRESHOLD = 60

# Nombre de secondes avant de valider un changement de mode
TRANSITION_DELAY = 2.0

# Sauvegarde des images de contrôle
SAVE_INTERVAL = 5.0


# ==========================================================
# LOGGING
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("DayNightController")


# ==========================================================
# CONTRÔLEUR JOUR / NUIT
# ==========================================================

class DayNightController:

    def __init__(self):
        self.mode = None
        self.candidate_mode = None
        self.candidate_since = None

        self.smoothed_brightness = None

        self.clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

    def calculate_brightness(self, gray):
        """
        Mesure la luminosité de l'image originale.
        On utilise une moyenne exponentiellement lissée.
        """

        brightness = float(cv2.mean(gray)[0])

        if self.smoothed_brightness is None:
            self.smoothed_brightness = brightness
        else:
            alpha = 0.10
            self.smoothed_brightness = (
                alpha * brightness
                + (1 - alpha) * self.smoothed_brightness
            )

        return self.smoothed_brightness

    def detect_mode(self, brightness):
        """
        Détection avec hystérésis :
        - passage en nuit sous 45
        - retour en jour au-dessus de 60
        """

        if self.mode is None:
            desired_mode = (
                "night" if brightness < NIGHT_THRESHOLD else "day"
            )

        elif self.mode == "day":
            if brightness < NIGHT_THRESHOLD:
                desired_mode = "night"
            else:
                desired_mode = "day"

        else:
            if brightness > DAY_THRESHOLD:
                desired_mode = "day"
            else:
                desired_mode = "night"

        if desired_mode == self.mode:
            self.candidate_mode = None
            self.candidate_since = None
            return self.mode

        if desired_mode != self.candidate_mode:
            self.candidate_mode = desired_mode
            self.candidate_since = time.monotonic()
            return self.mode

        elapsed = time.monotonic() - self.candidate_since

        if elapsed >= TRANSITION_DELAY:
            self.mode = desired_mode
            self.candidate_mode = None
            self.candidate_since = None

            logger.info(
                "Mode changed: %s | Brightness: %.2f",
                self.mode,
                brightness
            )

        return self.mode

    def process(self, frame):
        """
        Produit l'image finale selon le mode actif.
        """

        # Picamera2 RGB888 utilise ici l'ordre BGR des octets.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        brightness = self.calculate_brightness(gray)
        mode = self.detect_mode(brightness)

        if mode == "night":
            processed = self.clahe.apply(gray)
        else:
            processed = frame

        return processed, mode, brightness


# ==========================================================
# APPLICATION PRINCIPALE
# ==========================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    camera = Picamera2(tuning=TUNING_FILE)
    controller = DayNightController()

    config = camera.create_video_configuration(
        main={
            "size": (WIDTH, HEIGHT),
            "format": "RGB888"
        },
        controls={
            "AeEnable": True,
            "AwbEnable": True,
            "FrameDurationLimits": (
                FRAME_DURATION_US,
                FRAME_DURATION_US
            )
        }
    )

    camera.configure(config)

    logger.info("Camera configured.")
    logger.info("Resolution: %sx%s", WIDTH, HEIGHT)
    logger.info("Target FPS: approximately 15")
    logger.info("NoIR tuning profile loaded.")

    camera.start()

    try:
        logger.info("Waiting for automatic exposure...")
        time.sleep(3)

        last_save = 0
        last_log = 0

        while True:

            frame = camera.capture_array("main")

            processed, mode, brightness = controller.process(frame)

            now = time.monotonic()

            # Sauvegarder périodiquement les résultats
            if now - last_save >= SAVE_INTERVAL:

                output_file = OUTPUT_DIR / "latest.jpg"

                cv2.imwrite(
                    str(output_file),
                    processed,
                    [cv2.IMWRITE_JPEG_QUALITY, 90]
                )

                last_save = now

            # Afficher les métadonnées toutes les 5 secondes
            if now - last_log >= 5:

                metadata = camera.capture_metadata()

                exposure = metadata.get("ExposureTime")
                gain = metadata.get("AnalogueGain")
                frame_duration = metadata.get("FrameDuration")

                logger.info(
                    "Mode=%s | Brightness=%.2f | "
                    "Exposure=%s us | Gain=%.2f | "
                    "FrameDuration=%s us",
                    mode,
                    brightness,
                    exposure,
                    gain or 0,
                    frame_duration
                )

                last_log = now

    except KeyboardInterrupt:
        logger.info("Stopping camera controller...")

    finally:
        camera.stop()
        camera.close()
        logger.info("Camera stopped safely.")


if __name__ == "__main__":
    main()
