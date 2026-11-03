
from picamera2 import Picamera2
import libcamera
from libcamera import controls

TUNING_FILE = (
    "/usr/share/libcamera/ipa/rpi/pisp/ov5647_noir.json"
)

camera = Picamera2(tuning=TUNING_FILE)

try:
    config = camera.create_video_configuration(
        main={
            "size": (1296, 972),
            "format": "RGB888"
        },
        controls={
            "FrameDurationLimits": (33333, 66667)
        }
    )

    camera.configure(config)

    print("\n=== CONTROLES AEC/AGC DISPONIBLES ===")

    for name in (
        "AeEnable",
        "AeState",
        "ExposureTimeMode",
        "AnalogueGainMode",
        "ExposureTime",
        "AnalogueGain",
        "FrameDurationLimits"
    ):
        print(f"{name}: {camera.camera_controls.get(name)}")

    print("\n=== ENUMERATIONS LIBCAMERA ===")

    for name in dir(controls):
        if any(key in name for key in (
            "ExposureTimeMode",
            "AnalogueGainMode",
            "AeState"
        )):
            obj = getattr(controls, name)
            print(f"\n{name}: {obj}")

            try:
                print("Valeurs:", list(obj))
            except TypeError:
                pass

finally:
    camera.close()
