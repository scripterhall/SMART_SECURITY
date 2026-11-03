import time
from picamera2 import Picamera2
from libcamera import controls

camera = Picamera2(
    tuning="/usr/share/libcamera/ipa/rpi/pisp/ov5647_noir.json"
)

try:
    config = camera.create_video_configuration(
        main={"size": (1296, 972), "format": "RGB888"},
        controls={"FrameDurationLimits": (40000, 66667)}
    )

    camera.configure(config)
    camera.start()

    camera.set_controls({
        "AeEnable": False,
        "ExposureTimeMode": controls.ExposureTimeModeEnum.Manual,
        "AnalogueGainMode": controls.AnalogueGainModeEnum.Manual,
    })

    settings = [
        ("TEST 1", 10000, 2.0),
        ("TEST 2", 60000, 16.0),
    ]

    for label, exposure, gain in settings:
        print(f"\n--- {label} ---")

        camera.set_controls({
            "ExposureTime": exposure,
            "AnalogueGain": gain,
        })

        time.sleep(3)

        metadata = camera.capture_metadata()

        print("Exposure demandé :", exposure)
        print("Gain demandé :", gain)
        print("Exposure réel :", metadata.get("ExposureTime"))
        print("Gain réel :", metadata.get("AnalogueGain"))

        camera.capture_file(f"camera/output/{label.lower()}.jpg")

finally:
    camera.stop()
    camera.close()