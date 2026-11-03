
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, time as datetime_time
from typing import Optional

import cv2
import numpy as np
from picamera2 import Picamera2
from libcamera import controls


@dataclass
class CameraConfig:
    width: int = 2592
    height: int = 1944

    frame_duration_min: int = 40000
    frame_duration_max: int = 100000

    brightness_filter_start: datetime_time = datetime_time(19, 30)
    brightness_boost: int = 8

    tuning_file: str = (
        "/usr/share/libcamera/ipa/rpi/pisp/ov5647_noir.json"
    )


class DayNightCameraController:

    def __init__(self, config: Optional[CameraConfig] = None):
        self.config = config or CameraConfig()

        self.camera = None

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._capture_thread = None

        self._latest_frame = None

        self._status = {
            "running": False,
            "mode": "day",
            "brightness_filter": False,
            "brightness": 0.0,
            "fps": 0.0,
        }

        self._fps_count = 0
        self._fps_start = time.monotonic()

        self.logger = logging.getLogger("CameraController")

    def _configure_camera(self):
        video_config = self.camera.create_video_configuration(
            main={
                "size": (self.config.width, self.config.height),
                "format": "RGB888"
            },
            controls={
                "AeEnable": True,
                "AeMeteringMode": controls.AeMeteringModeEnum.CentreWeighted,
                "AwbEnable": True,
                "FrameDurationLimits": (
                    self.config.frame_duration_min,
                    self.config.frame_duration_max
                )
            },
            buffer_count=4
        )

        self.camera.configure(video_config)

    # =====================================================
    # 3. INITIALISATION CAMERA
    # =====================================================

    def start(self):

        if self.camera is not None:
            return

        self.logger.info(
            "Initialisation de la caméra OV5647 NoIR"
        )

        self.camera = Picamera2(
            tuning=self.config.tuning_file
        )

        self._configure_camera()

        self.camera.start()

        # Attendre la stabilisation initiale.
        time.sleep(1.0)

        self._stop_event.clear()

        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="camera-capture",
            daemon=True
        )

        self._capture_thread.start()

        with self._lock:
            self._status["running"] = True

        self.logger.info(
            "Camera started successfully | Profile=DAY | Resolution=%dx%d",
            self.config.width,
            self.config.height
        )

    # =====================================================
    # 4. MESURE DE LUMINOSITE (INFORMATION UNIQUEMENT)
    # =====================================================

    def _measure_brightness(self, frame: np.ndarray) -> float:

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2GRAY
        )

        return float(np.percentile(gray, 50))

    # =====================================================
    # 5. TRAITEMENT IMAGE
    # =====================================================

    def _process_frame(
        self,
        rgb_frame: np.ndarray,
        brightness_filter: bool = False
    ):
        processed = cv2.cvtColor(
            rgb_frame,
            cv2.COLOR_RGB2BGR
        )

        if brightness_filter:
            processed = cv2.convertScaleAbs(
                processed,
                alpha=1.0,
                beta=self.config.brightness_boost
            )

        return processed

    # =====================================================
    # 6. BOUCLE PRINCIPALE
    # =====================================================

    def _capture_loop(self):

        while not self._stop_event.is_set():

            try:

                # Capture image.
                rgb_frame = self.camera.capture_array()

                brightness = self._measure_brightness(
                    rgb_frame
                )

                brightness_filter = (
                    datetime.now().time()
                    >= self.config.brightness_filter_start
                )

                processed = self._process_frame(
                    rgb_frame,
                    brightness_filter=brightness_filter
                )

                # Calcul FPS réel.
                self._fps_count += 1

                now = time.monotonic()
                elapsed = now - self._fps_start

                if elapsed >= 1.0:

                    fps = self._fps_count / elapsed

                    self._fps_count = 0
                    self._fps_start = now

                else:

                    fps = self._status["fps"]

                # Mise à jour des données partagées.
                with self._lock:

                    self._latest_frame = processed

                    self._status.update({

                        "mode": "day",

                        "brightness_filter": brightness_filter,

                        "brightness": round(
                            brightness,
                            2
                        ),

                        "fps": round(fps, 2),

                    })

            except Exception:

                self.logger.exception(
                    "Erreur pendant la capture caméra"
                )

                time.sleep(0.1)

    # =====================================================
    # 7. RECUPERATION IMAGE
    # =====================================================

    def get_frame(self) -> Optional[np.ndarray]:

        with self._lock:

            if self._latest_frame is None:
                return None

            return self._latest_frame.copy()

    # =====================================================
    # 8. RECUPERATION ETAT CAMERA
    # =====================================================

    def get_status(self) -> dict:

        with self._lock:
            return self._status.copy()

    # =====================================================
    # 9. ARRET CAMERA
    # =====================================================

    def stop(self):

        if self.camera is None:
            return

        self.logger.info("Arrêt de la caméra")

        self._stop_event.set()

        if self._capture_thread is not None:

            self._capture_thread.join(timeout=5)

        self.camera.stop()
        self.camera.close()

        self.camera = None

        with self._lock:

            self._latest_frame = None
            self._status["running"] = False
            self._status["mode"] = "stopped"
            self._status["brightness_filter"] = False

        self.logger.info("Caméra arrêtée")
