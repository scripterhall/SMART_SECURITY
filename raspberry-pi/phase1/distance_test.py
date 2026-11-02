import time

from gpiozero import DistanceSensor

TRIG_PIN = 23
ECHO_PIN = 24

sensor = DistanceSensor(
    echo=ECHO_PIN,
    trigger=TRIG_PIN,
    max_distance=4.0,
    queue_len=5
)

try:

    print("HC-SR04 started")

    while True:

        distance_m = sensor.distance

        distance_cm = distance_m * 100

        print(
            f"Distance: {distance_cm:.1f} cm"
        )

        time.sleep(0.2)

except KeyboardInterrupt:

    print("\nSensor stopped")

finally:

    sensor.close()