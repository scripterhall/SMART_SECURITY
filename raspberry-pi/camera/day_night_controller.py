
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
    night_width: int = 1296
    night_height: int = 972

    day_width: int = 2592
    day_height: int = 1944

    # Limites générales de cadence :
    # jusqu'à 100 ms par image.
    frame_duration_min: int = 40000
    frame_duration_max: int = 100000

    # PROFIL JOUR : exposition et gain automatiques.
    day_start = datetime_time(6, 0)
    day_frame_duration_us: int = 100000

    # PROFIL NUIT
    night_start = datetime_time(19, 1)
    night_exposure_us: int = 70000
    night_gain: float = 120.0
    night_frame_duration_us: int = 75000

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

        self._mode = None

        self._current_exposure = 0
        self._current_gain = 0.0
        self._configured_size = None

        self._fps_count = 0
        self._fps_start = time.monotonic()

        # Création unique du CLAHE.
        self.clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        self.logger = logging.getLogger("CameraController")

    # =====================================================
    # 1. DETERMINATION DU MODE SELON L'HEURE
    # =====================================================

    def _get_scheduled_mode(self) -> str:

        current_time = datetime.now().time()

        if (
            self.config.day_start
            <= current_time
            < self.config.night_start
        ):
            return "day"

        return "night"

    # =====================================================
    # 2. APPLICATION DU PROFIL CAMERA
    # =====================================================

    def _apply_mode(self, mode: str):

        if mode == "day":
            width = self.config.day_width
            height = self.config.day_height
            frame_duration = self.config.day_frame_duration_us
            exposure = None
            gain = None
            mode_controls = {
                "AeEnable": True,
                "AeMeteringMode": controls.AeMeteringModeEnum.CentreWeighted,
                "AwbEnable": True,
            }
        else:
            width = self.config.night_width
            height = self.config.night_height
            exposure = self.config.night_exposure_us
            gain = self.config.night_gain
            frame_duration = self.config.night_frame_duration_us
            mode_controls = {
                "AeEnable": False,
                "ExposureTime": exposure,
                "AnalogueGain": gain,
                "AeMeteringMode": controls.AeMeteringModeEnum.CentreWeighted,
                "AwbEnable": True,
            }

        size = (width, height)
        if size != self._configured_size:
            self.camera.stop()
            self._configure_camera(mode)
            self.camera.start()

        self.camera.set_controls({
            **mode_controls,
            "FrameDurationLimits": (
                frame_duration,
                frame_duration
            ),
        })

        self._current_exposure = exposure or 0
        self._current_gain = gain or 0.0
        self._mode = mode

        self.logger.info(
            "MODE %s ACTIVE | Exposure=%s | Gain=%s | FrameDuration=%d us",
            mode.upper(),
            f"{exposure} us" if exposure is not None else "auto",
            gain if gain is not None else "auto",
            frame_duration
        )

    def _configure_camera(self, mode: str):
        if mode == "day":
            width = self.config.day_width
            height = self.config.day_height
        else:
            width = self.config.night_width
            height = self.config.night_height

        video_config = self.camera.create_video_configuration(
            main={
                "size": (width, height),
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
        self._configured_size = (width, height)

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

        initial_mode = self._get_scheduled_mode()
        self._configure_camera(initial_mode)

        self.camera.start()

        # Attendre la stabilisation initiale.
        time.sleep(1.0)

        # Appliquer directement le mode correspondant
        # à l'heure actuelle.
        self._apply_mode(initial_mode)

        time.sleep(0.5)

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
            "Camera started successfully | Mode=%s",
            initial_mode.upper()
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

    def _process_frame(self, rgb_frame: np.ndarray):

        if self._mode == "night":

            gray = cv2.cvtColor(
                rgb_frame,
                cv2.COLOR_RGB2GRAY
            )

            return self.clahe.apply(gray)

        return cv2.cvtColor(
            rgb_frame,
            cv2.COLOR_RGB2BGR
        )

    # =====================================================
    # 6. BOUCLE PRINCIPALE
    # =====================================================

    def _capture_loop(self):

        while not self._stop_event.is_set():

            try:

                # Vérification de l'heure à chaque image.
                scheduled_mode = self._get_scheduled_mode()

                # Application uniquement lors d'un changement.
                if scheduled_mode != self._mode:

                    self.logger.info(
                        "Changement horaire détecté"
                    )

                    self._apply_mode(scheduled_mode)

                # Capture image.
                rgb_frame = self.camera.capture_array()

                brightness = self._measure_brightness(
                    rgb_frame
                )

                # Traitement selon le mode actif.
                processed = self._process_frame(
                    rgb_frame
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

                        "mode": self._mode,

                        "brightness": round(
                            brightness,
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

        self.logger.info("Caméra arrêtée")
