
import cv2

def main():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open the webcam.")
        print("Check camera permissions or whether another app is using it.")
        return

    print("Webcam opened successfully.")
    print("Press Q to quit.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("ERROR: Could not read a frame.")
                break

            cv2.putText(
                frame,
                "Webcam Test - Press Q to Quit",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Driver Safety System - Webcam Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Webcam released successfully.")

if __name__ == "__main__":
    main()