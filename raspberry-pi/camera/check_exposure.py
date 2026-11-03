
import time
from picamera2 import Picamera2

TUNING_FILE = (
    "/usr/share/libcamera/ipa/rpi/pisp/ov5647_noir.json"
)

camera = Picamera2(tuning=TUNING_FILE)

try:
    print("\n=== CAMERA CONTROLS ===")

    for name in (
        "AeEnable",
        "AwbEnable",
        "ExposureTime",
        "AnalogueGain",
        "DigitalGain",
        "FrameDurationLimits",
        "AeConstraintMode",
        "AeExposureMode",
        "ExposureValue"
    ):
        print(name, camera.camera_controls.get(name))

    config = camera.create_video_configuration(
        main={
            "size": (1296, 972),
            "format": "RGB888"
        },
        controls={
            "AeEnable": True,
            "AwbEnable": True,
            "FrameDurationLimits": (40000, 66667)
        }
    )

    camera.configure(config)
    camera.start()

    print("\n=== EXPOSURE OBSERVATION ===")

    for i in range(12):
        time.sleep(2)

        metadata = camera.capture_metadata()

        print(
            f"Sample {i + 1}: "
            f"Exposure={metadata.get('ExposureTime')} us | "
            f"Gain={metadata.get('AnalogueGain')} | "
            f"DigitalGain={metadata.get('DigitalGain')} | "
            f"FrameDuration={metadata.get('FrameDuration')}"
        )

finally:
    camera.stop()
    camera.close()
