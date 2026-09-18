import json
import sys
import statistics

import cv2


TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920
SAMPLE_COUNT = 18


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def detect_focus(video_path, start, end):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("Could not open source video for smart framing.")

    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    if source_width <= 0 or source_height <= 0:
        cap.release()
        raise RuntimeError("Could not determine source video dimensions.")

    duration = max(0.1, float(end) - float(start))

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    profile_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_profileface.xml"
    )

    upper_body_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_upperbody.xml"
    )

    centers = []
    weights = []

    for index in range(SAMPLE_COUNT):
        fraction = (
            index / (SAMPLE_COUNT - 1)
            if SAMPLE_COUNT > 1
            else 0.5
        )

        timestamp = float(start) + duration * fraction

        cap.set(
            cv2.CAP_PROP_POS_MSEC,
            timestamp * 1000
        )

        ok, frame = cap.read()

        if not ok or frame is None:
            continue

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.equalizeHist(gray)

        detections = []

        # Normal frontal faces.
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )

        for x, y, w, h in faces:
            detections.append(
                (x, y, w, h, "face")
            )

        # Profile faces help when someone is looking sideways.
        profiles = profile_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(40, 40)
        )

        for x, y, w, h in profiles:
            detections.append(
                (x, y, w, h, "profile")
            )

        # If no face is visible, try upper-body detection.
        if not detections:
            bodies = upper_body_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(60, 60)
            )

            for x, y, w, h in bodies:
                detections.append(
                    (x, y, w, h, "body")
                )

        if not detections:
            continue

        # Prefer the largest detected subject.
        detections.sort(
            key=lambda item: item[2] * item[3],
            reverse=True
        )

        x, y, w, h, detection_type = detections[0]

        center_x = x + (w / 2)

        area = max(1.0, float(w * h))

        # Faces are more reliable than body detections.
        if detection_type in ("face", "profile"):
            weight = area * 2.0
        else:
            weight = area

        centers.append(
            center_x / source_width
        )

        weights.append(weight)

    cap.release()

    if centers:
        # Median is deliberately used instead of a simple average.
        # This prevents one bad detection from pulling the crop
        # too far toward the wrong side.
        focus_x = statistics.median(centers)
        detected = True
    else:
        focus_x = 0.5
        detected = False

    focus_x = clamp(focus_x, 0.0, 1.0)

    # Scale the original video enough to completely cover
    # a 1080x1920 vertical frame.
    scale = max(
        TARGET_WIDTH / source_width,
        TARGET_HEIGHT / source_height
    )

    scaled_width = max(
        TARGET_WIDTH,
        int(round(source_width * scale))
    )

    scaled_height = max(
        TARGET_HEIGHT,
        int(round(source_height * scale))
    )

    max_crop_x = max(
        0,
        scaled_width - TARGET_WIDTH
    )

    max_crop_y = max(
        0,
        scaled_height - TARGET_HEIGHT
    )

    # Put the detected subject near the horizontal center
    # of the vertical frame.
    crop_x = int(
        round(
            focus_x * scaled_width -
            TARGET_WIDTH / 2
        )
    )

    crop_x = clamp(
        crop_x,
        0,
        max_crop_x
    )

    crop_y = int(
        max_crop_y / 2
    )

    return {
        "sourceWidth": source_width,
        "sourceHeight": source_height,
        "scaleWidth": scaled_width,
        "scaleHeight": scaled_height,
        "cropX": crop_x,
        "cropY": crop_y,
        "focusX": round(focus_x, 4),
        "facesDetected": detected
    }


def main():
    if len(sys.argv) != 4:
        print(
            "Usage: python smart_crop.py "
            "<video> <start> <end>",
            file=sys.stderr
        )
        sys.exit(1)

    video_path = sys.argv[1]
    start = float(sys.argv[2])
    end = float(sys.argv[3])

    result = detect_focus(
        video_path,
        start,
        end
    )

    print(json.dumps(result))


if __name__ == "__main__":
    main()
