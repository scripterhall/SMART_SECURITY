import time
from pathlib import Path

from picamera2 import Picamera2

TUNING_FILE = (
    "/usr/share/libcamera/ipa/rpi/pisp/ov5647_noir.json"
)

OUTPUT_FILE = "/home/pi/camera-picamera2-test.jpg"

def main():
    camera = Picamera2(tuning=TUNING_FILE)

    try:
        config = camera.create_video_configuration(
            main={
                "size": (1296, 972),
                "format": "RGB888"
            },
            controls={
                # Intervalle de 40 à 66,7 ms :
                # environ 15 à 25 FPS.
                "FrameDurationLimits": (40000, 66667),

                # Exposition et gain automatiques.
                "AeEnable": True
            }
        )

        camera.configure(config)

        print("Camera configured successfully.")
        print("Available controls:")

        for name in (
            "ExposureTime",
            "AnalogueGain",
            "FrameDurationLimits"
        ):
            print(name, camera.camera_controls.get(name))

        camera.start()

        print("Camera started. Waiting for auto exposure...")

        # Laisser l'AEC/AGC se stabiliser.
        time.sleep(5)

        metadata = camera.capture_metadata()

        print("\nCamera metadata:")
        for name in (
            "ExposureTime",
            "AnalogueGain",
            "DigitalGain",
            "FrameDuration",
            "ColourGains"
        ):
            print(f"{name}: {metadata.get(name)}")

        camera.capture_file(OUTPUT_FILE)

        print(f"\nImage saved: {OUTPUT_FILE}")

    finally:
        camera.stop()
        camera.close()

if __name__ == "__main__":
    main()