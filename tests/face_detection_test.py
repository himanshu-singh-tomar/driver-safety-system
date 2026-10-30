
import cv2
import mediapipe as mp
from pathlib import Path

# Project root: D:\driver-safety-system
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "face_landmarker.task"

def main():
    if not MODEL_PATH.is_file():
        print(f"ERROR: Model file not found: {MODEL_PATH}")
        return

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return

    try:
        with FaceLandmarker.create_from_options(options) as landmarker:
            print("Face Landmarker started.")
            print("Look at the camera. Press Q to quit.")

            start_time = cv2.getTickCount()
            fps = cv2.getTickFrequency()

            while True:
                success, frame = camera.read()

                if not success:
                    print("ERROR: Could not read webcam frame.")
                    break

                # Mirror the image for a natural webcam preview.
                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]

                # MediaPipe expects RGB input.
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )

                # VIDEO mode requires a monotonically increasing timestamp.
                elapsed_ms = int(
                    (cv2.getTickCount() - start_time) * 1000 / fps
                )

                result = landmarker.detect_for_video(mp_image, elapsed_ms)

                if result.face_landmarks:
                    for face_landmarks in result.face_landmarks:
                        for landmark in face_landmarks:
                            x = int(landmark.x * width)
                            y = int(landmark.y * height)

                            if 0 <= x < width and 0 <= y < height:
                                cv2.circle(
                                    frame, (x, y), 1, (0, 255, 0), -1
                                )

                    status = "FACE DETECTED"
                    color = (0, 255, 0)
                else:
                    status = "NO FACE DETECTED"
                    color = (0, 0, 255)

                cv2.putText(
                    frame,
                    status,
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                )

                cv2.putText(
                    frame,
                    "Press Q to quit",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow("Face Landmark Test", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except Exception as exc:
        print(f"Face landmark test failed: {exc}")

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Webcam released.")

if __name__ == "__main__":
    main()