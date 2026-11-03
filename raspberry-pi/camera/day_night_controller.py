
import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from picamera2 import Picamera2
from libcamera import controls


@dataclass
class CameraConfig:
    width: int = 1296
    height: int = 972

    # 15 FPS maximum : durée d'une image d'environ 66,7 ms
    frame_duration_min: int = 40000
    frame_duration_max: int = 66667

    # Seuils de luminosité avec hystérésis
    night_threshold: float = 40.0
    day_threshold: float = 44.0

    # Exposition : microsecondes
    day_exposure_us: int = 10000
    night_exposure_us: int = 25000

    # Gain analogique
    day_gain: float = 2.0
    night_gain_min: float = 4.0
    night_gain_max: float = 16.0

    # Mise à jour des réglages
    control_interval: float = 1.0

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
            "mode": "initializing",
            "brightness": 0.0,
            "exposure_us": 0,
            "analogue_gain": 0.0,
            "fps": 0.0,
        }

        self._mode = "day"
        self._brightness_ema = None
        self._last_control_time = 0.0

        self._current_exposure = self.config.day_exposure_us
        self._current_gain = self.config.day_gain

        self._fps_count = 0
        self._fps_start = time.monotonic()

        self.logger = logging.getLogger("CameraController")

    def start(self):
        if self.camera is not None:
            return

        self.logger.info("Initialisation de la caméra OV5647 NoIR")

        self.camera = Picamera2(
            tuning=self.config.tuning_file
        )

        video_config = self.camera.create_video_configuration(
            main={
                "size": (
                    self.config.width,
                    self.config.height
                ),
                "format": "RGB888"
            },
            controls={
                "FrameDurationLimits": (
                    self.config.frame_duration_min,
                    self.config.frame_duration_max
                )
            },
            buffer_count=4
        )

        self.camera.configure(video_config)

        self.camera.start()

        # Désactivation explicite de l'AEC/AGC :
        # l'exposition et le gain sont contrôlés par notre logiciel.
        self.camera.set_controls({
            "AeEnable": False,
            "ExposureTimeMode": controls.ExposureTimeModeEnum.Manual,
            "AnalogueGainMode": controls.AnalogueGainModeEnum.Manual,
            "ExposureTime": self._current_exposure,
            "AnalogueGain": self._current_gain,
            "AwbEnable": True,
        })

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

        self.logger.info("Caméra démarrée")

    def _measure_brightness(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2GRAY
        )

        # On évite que quelques zones très lumineuses
        # faussent toute la mesure.
        return float(np.percentile(gray, 50))

    def _update_day_night_mode(self, brightness: float):
        if self._brightness_ema is None:
            self._brightness_ema = brightness
        else:
            alpha = 0.20
            self._brightness_ema = (
                alpha * brightness
                + (1 - alpha) * self._brightness_ema
            )

        if (
            self._mode == "day"
            and self._brightness_ema < self.config.night_threshold
        ):
            self._mode = "night"
            self.logger.info("Passage en mode NUIT")

        elif (
            self._mode == "night"
            and self._brightness_ema > self.config.day_threshold
        ):
            self._mode = "day"
            self.logger.info("Passage en mode JOUR")

    def _adjust_exposure(self, brightness: float):
        now = time.monotonic()

        if now - self._last_control_time < self.config.control_interval:
            return

        self._last_control_time = now

        if self._mode == "day":
            target_exposure = self.config.day_exposure_us
            target_gain = self.config.day_gain

        else:
            # Priorité à la netteté :
            # exposition plafonnée à 25 ms.
            target_exposure = self.config.night_exposure_us

            # Compensation de luminosité par le gain.
            # Plus la scène est sombre, plus le gain augmente.
            if self._brightness_ema is None:
                target_gain = self.config.night_gain_min
            else:
                darkness = max(
                    0.0,
                    min(
                        1.0,
                        (60.0 - self._brightness_ema) / 60.0
                    )
                )

                target_gain = (
                    self.config.night_gain_min
                    + darkness * (
                        self.config.night_gain_max
                        - self.config.night_gain_min
                    )
                )

        # Limitation des changements pour éviter
        # des variations brutales d'exposition.
        self._current_exposure = int(
            0.7 * self._current_exposure
            + 0.3 * target_exposure
        )

        self._current_gain = (
            0.7 * self._current_gain
            + 0.3 * target_gain
        )

        self.camera.set_controls({
            "ExposureTime": self._current_exposure,
            "AnalogueGain": self._current_gain,
        })

    def _capture_loop(self):
        while not self._stop_event.is_set():
            try:
                rgb_frame = self.camera.capture_array()

                brightness = self._measure_brightness(
                    rgb_frame
                )

                self._update_day_night_mode(brightness)
                if int(time.monotonic()) % 2 == 0:
                    self.logger.info(
                        "Luminosité brute=%.2f | Filtrée=%.2f | Mode=%s",
                        brightness,
                        self._brightness_ema,
                        self._mode
                    )

                self._adjust_exposure(brightness)

                if self._mode == "night":
                    gray = cv2.cvtColor(
                        rgb_frame,
                        cv2.COLOR_RGB2GRAY
                    )

                    clahe = cv2.createCLAHE(
                        clipLimit=2.0,
                        tileGridSize=(8, 8)
                    )

                    processed = clahe.apply(gray)

                else:
                    processed = cv2.cvtColor(
                        rgb_frame,
                        cv2.COLOR_RGB2BGR
                    )

                self._fps_count += 1
                now = time.monotonic()
                elapsed = now - self._fps_start

                if elapsed >= 1.0:
                    fps = self._fps_count / elapsed
                    self._fps_count = 0
                    self._fps_start = now
                else:
                    fps = self._status["fps"]

                with self._lock:
                    self._latest_frame = processed

                    self._status.update({
                        "mode": self._mode,
                        "brightness": round(
                            self._brightness_ema or 0.0,
                            2
                        ),
                        "exposure_us": self._current_exposure,
                        "analogue_gain": round(
                            self._current_gain,
                            2
                        ),
                        "fps": round(fps, 2),
                    })

            except Exception:
                self.logger.exception(
                    "Erreur pendant la capture caméra"
                )
                time.sleep(0.1)

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_frame is None:
                return None

            return self._latest_frame.copy()

    def get_status(self) -> dict:
        with self._lock:
            return self._status.copy()

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

        self.logger.info("Caméra arrêtée")
