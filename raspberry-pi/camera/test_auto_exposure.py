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
        "AeEnable": True,
        "ExposureTimeMode": controls.ExposureTimeModeEnum.Auto,
        "AnalogueGainMode": controls.AnalogueGainModeEnum.Auto,
    })

    def measure(label):
        print(f"\n--- {label} ---")
        for _ in range(5):
            time.sleep(2)
            m = camera.capture_metadata()
            print(
                f"Exposure={m.get('ExposureTime')} us | "
                f"Gain={m.get('AnalogueGain')} | "
                f"AeState={m.get('AeState')}"
            )

    measure("LUMIERE NORMALE")

    input("\nCouvre complètement l'objectif puis appuie sur Entrée...")
    measure("OBJECTIF COUVERT")

    input("\nDécouvre l'objectif puis appuie sur Entrée...")
    measure("LUMIERE RESTAUREE")

finally:
    camera.stop()
    camera.close()