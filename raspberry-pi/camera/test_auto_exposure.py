import time
from picamera2 import Picamera2
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
            "FrameDurationLimits": (40000, 66667),
        }
    )

    camera.configure(config)

    print("\n=== ENUMERATIONS ===")

    exposure_auto = controls.ExposureTimeModeEnum.Auto
    gain_auto = controls.AnalogueGainModeEnum.Auto

    print("ExposureTimeMode.Auto =", exposure_auto)
    print("AnalogueGainMode.Auto =", gain_auto)

    camera.start()

    # Activation explicite de l'exposition et du gain automatiques
    camera.set_controls({
        "AeEnable": True,
        "ExposureTimeMode": exposure_auto,
        "AnalogueGainMode": gain_auto,
    })

    print("\n=== TEST AEC / AGC ===")

    for i in range(15):
        time.sleep(2)

        metadata = camera.capture_metadata()

        exposure = metadata.get("ExposureTime")
        gain = metadata.get("AnalogueGain")
        digital_gain = metadata.get("DigitalGain")
        frame_duration = metadata.get("FrameDuration")
        ae_state = metadata.get("AeState")

        print(
            f"[{i + 1:02d}] "
            f"Exposure={exposure} us | "
            f"Gain={gain} | "
            f"DigitalGain={digital_gain} | "
            f"FrameDuration={frame_duration} | "
            f"AeState={ae_state}"
        )

finally:
    camera.stop()
    camera.close()