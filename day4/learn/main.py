import cv2


MODEL = "face_detection_yunet_2026may.onnx"


def main():
    # Create the YuNet face detector
    detector = cv2.FaceDetectorYN.create(
        MODEL,
        "",
        (320, 320),
        0.9,
        0.3,
        5000,
    )

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not access the webcam.")
        return

    while True:
        # Read frame
        ret, frame = cap.read()

        if not ret:
            print("Could not read frame.")
            break

        # Get frame dimensions
        height, width = frame.shape[:2]

        # Tell detector the current image size
        detector.setInputSize((width, height))

        # Detect faces
        _, faces = detector.detect(frame)

        # Draw detected faces
        if faces is not None:
            for face in faces:
                x, y, w, h = face[:4].astype(int)

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2,
                )

                # Face confidence
                confidence = face[-1]

                cv2.putText(
                    frame,
                    f"Face: {confidence:.2f}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        # Show result
        cv2.imshow("OpenCV 5 Face Detection", frame)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
