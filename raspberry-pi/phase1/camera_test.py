from picamera2 import Picamera2
import cv2
import time


def main():

    camera = Picamera2()

    config = camera.create_preview_configuration(
        main={
            "size": (2592, 1944),
            "format": "RGB888"
        }
    )

    camera.configure(config)

    camera.start()

    time.sleep(2)

    print("Camera started successfully")

    try:

        while True:

            frame = camera.capture_array()

            # Afficher l'image
            cv2.imshow("IMX219 Camera Test", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:

        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()