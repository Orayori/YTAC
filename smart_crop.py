import json
import math
import statistics
import sys

import cv2


TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920

# More samples give us a better picture of who remains on screen.
SAMPLE_COUNT = 30

# Minimum face size as a fraction of frame width.
# Tiny background faces are ignored.
MIN_FACE_WIDTH_RATIO = 0.045

# Maximum number of tracked people.
MAX_TRACKS = 8

# A face can move this fraction of the frame width and still
# be considered the same person between samples.
TRACK_DISTANCE_RATIO = 0.22

# How strongly face size contributes to dominance.
SIZE_WEIGHT = 1.7

# How strongly persistence contributes to dominance.
PERSISTENCE_WEIGHT = 2.8

# Small bonus for a face that is reasonably close to the
# horizontal center. This prevents extreme edge detections
# from winning purely because of noise.
CENTER_BONUS_WEIGHT = 0.20

# Extra horizontal safety margin around the detected face.
FACE_MARGIN_RATIO = 0.10


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def safe_float(value, default=0.0):
    try:
        result = float(value)

        if math.isfinite(result):
            return result

    except Exception:
        pass

    return default


def detection_center_x(detection):
    x, y, w, h = detection[:4]

    return x + (w / 2.0)


def detection_center_y(detection):
    x, y, w, h = detection[:4]

    return y + (h / 2.0)


def detection_area(detection):
    x, y, w, h = detection[:4]

    return max(
        1,
        int(w)
    ) * max(
        1,
        int(h)
    )


def iou(a, b):
    """
    Intersection-over-Union between two face rectangles.
    """

    ax, ay, aw, ah = a[:4]
    bx, by, bw, bh = b[:4]

    ax2 = ax + aw
    ay2 = ay + ah

    bx2 = bx + bw
    by2 = by + bh

    intersection_x1 = max(ax, bx)
    intersection_y1 = max(ay, by)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(
        0,
        intersection_x2 - intersection_x1
    )

    intersection_height = max(
        0,
        intersection_y2 - intersection_y1
    )

    intersection_area = (
        intersection_width *
        intersection_height
    )

    if intersection_area <= 0:
        return 0.0

    area_a = max(1, aw * ah)
    area_b = max(1, bw * bh)

    union = (
        area_a +
        area_b -
        intersection_area
    )

    if union <= 0:
        return 0.0

    return intersection_area / union


def face_distance(a, b, frame_width, frame_height):
    """
    Normalized center distance between two detections.
    """

    ax = detection_center_x(a)
    ay = detection_center_y(a)

    bx = detection_center_x(b)
    by = detection_center_y(b)

    dx = (ax - bx) / max(
        1.0,
        float(frame_width)
    )

    dy = (ay - by) / max(
        1.0,
        float(frame_height)
    )

    return math.sqrt(
        dx * dx +
        dy * dy
    )


def build_face_cascades():
    """
    Load OpenCV Haar cascades.

    We use frontal + profile detection because people in short-form
    videos are not always looking directly at the camera.
    """

    frontal_path = (
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    profile_path = (
        cv2.data.haarcascades +
        "haarcascade_profileface.xml"
    )

    frontal = cv2.CascadeClassifier(
        frontal_path
    )

    profile = cv2.CascadeClassifier(
        profile_path
    )

    if frontal.empty():
        raise RuntimeError(
            "Could not load OpenCV frontal face cascade."
        )

    if profile.empty():
        raise RuntimeError(
            "Could not load OpenCV profile face cascade."
        )

    return frontal, profile


def detect_faces(
    frame,
    frontal,
    profile,
    source_width,
    source_height
):
    """
    Detect faces and remove obviously tiny detections.
    """

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.equalizeHist(
        gray
    )

    min_face_width = max(
        35,
        int(
            source_width *
            MIN_FACE_WIDTH_RATIO
        )
    )

    detections = []

    frontal_faces = frontal.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=5,
        minSize=(
            min_face_width,
            min_face_width
        )
    )

    for x, y, w, h in frontal_faces:
        detections.append(
            (
                int(x),
                int(y),
                int(w),
                int(h),
                "frontal"
            )
        )

    profile_faces = profile.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=4,
        minSize=(
            min_face_width,
            min_face_width
        )
    )

    for x, y, w, h in profile_faces:
        detections.append(
            (
                int(x),
                int(y),
                int(w),
                int(h),
                "profile"
            )
        )

    # Remove duplicate detections.
    cleaned = []

    for detection in detections:
        duplicate = False

        for existing in cleaned:
            overlap = iou(
                detection,
                existing
            )

            distance = face_distance(
                detection,
                existing,
                source_width,
                source_height
            )

            if (
                overlap >= 0.35 or
                distance <= 0.035
            ):
                duplicate = True

                # Keep the larger detection.
                if (
                    detection_area(detection) >
                    detection_area(existing)
                ):
                    cleaned.remove(
                        existing
                    )
                    cleaned.append(
                        detection
                    )

                break

        if not duplicate:
            cleaned.append(
                detection
            )

    return cleaned


def create_track(detection, frame_index):
    return {
        "detections": [
            detection
        ],
        "firstFrame": frame_index,
        "lastFrame": frame_index,
        "framesSeen": 1,
        "totalArea": detection_area(
            detection
        ),
        "centerXs": [
            detection_center_x(
                detection
            )
        ],
        "centerYs": [
            detection_center_y(
                detection
            )
        ]
    }


def track_detection(
    tracks,
    detection,
    frame_index,
    source_width,
    source_height
):
    """
    Assign a detection to an existing person track.

    We combine position and box overlap so that people moving around
    the frame can still be followed without constantly switching
    identities.
    """

    best_track = None
    best_score = float("inf")

    for track in tracks:
        if (
            frame_index -
            track["lastFrame"]
        ) > 3:
            continue

        previous = track[
            "detections"
        ][-1]

        distance = face_distance(
            detection,
            previous,
            source_width,
            source_height
        )

        overlap = iou(
            detection,
            previous
        )

        # Position is the main identity signal.
        score = distance

        # Strong overlap means very likely same person.
        if overlap >= 0.15:
            score *= 0.35

        if score < best_score:
            best_score = score
            best_track = track

    if (
        best_track is None or
        best_score >
        TRACK_DISTANCE_RATIO
    ):
        return False

    best_track[
        "detections"
    ].append(
        detection
    )

    best_track[
        "lastFrame"
    ] = frame_index

    best_track[
        "framesSeen"
    ] += 1

    best_track[
        "totalArea"
    ] += detection_area(
        detection
    )

    best_track[
        "centerXs"
    ].append(
        detection_center_x(
            detection
        )
    )

    best_track[
        "centerYs"
    ].append(
        detection_center_y(
            detection
        )
    )

    return True


def track_faces(
    samples,
    source_width,
    source_height
):
    """
    Build persistent face tracks from all sampled frames.
    """

    tracks = []

    for frame_index, detections in enumerate(
        samples
    ):
        # Largest detections first.
        detections = sorted(
            detections,
            key=detection_area,
            reverse=True
        )

        assigned_tracks = set()

        for detection in detections:
            best_track = None
            best_score = float("inf")

            for track_index, track in enumerate(
                tracks
            ):
                if track_index in assigned_tracks:
                    continue

                if (
                    frame_index -
                    track["lastFrame"]
                ) > 3:
                    continue

                previous = track[
                    "detections"
                ][-1]

                distance = face_distance(
                    detection,
                    previous,
                    source_width,
                    source_height
                )

                overlap = iou(
                    detection,
                    previous
                )

                score = distance

                if overlap >= 0.15:
                    score *= 0.35

                if score < best_score:
                    best_score = score
                    best_track = (
                        track_index,
                        track
                    )

            if (
                best_track is not None and
                best_score <=
                TRACK_DISTANCE_RATIO
            ):
                track_index, track = (
                    best_track
                )

                track[
                    "detections"
                ].append(
                    detection
                )

                track[
                    "lastFrame"
                ] = frame_index

                track[
                    "framesSeen"
                ] += 1

                track[
                    "totalArea"
                ] += detection_area(
                    detection
                )

                track[
                    "centerXs"
                ].append(
                    detection_center_x(
                        detection
                    )
                )

                track[
                    "centerYs"
                ].append(
                    detection_center_y(
                        detection
                    )
                )

                assigned_tracks.add(
                    track_index
                )

            elif len(tracks) < MAX_TRACKS:
                tracks.append(
                    create_track(
                        detection,
                        frame_index
                    )
                )

    return tracks


def score_track(
    track,
    total_frames,
    source_width,
    source_height
):
    """
    Calculate how likely this track is to be the dominant person.

    Persistence matters more than simply being large in one frame.
    """

    persistence = (
        track["framesSeen"] /
        max(
            1,
            total_frames
        )
    )

    average_area = (
        track["totalArea"] /
        max(
            1,
            track["framesSeen"]
        )
    )

    frame_area = (
        source_width *
        source_height
    )

    area_ratio = (
        average_area /
        max(
            1,
            frame_area
        )
    )

    # Convert area into a useful bounded score.
    size_score = clamp(
        math.sqrt(
            max(
                0.0,
                area_ratio
            ) * 100.0
        ),
        0.0,
        1.0
    )

    median_x = statistics.median(
        track["centerXs"]
    )

    center_distance = abs(
        median_x -
        source_width / 2.0
    ) / max(
        1.0,
        source_width / 2.0
    )

    center_bonus = 1.0 - clamp(
        center_distance,
        0.0,
        1.0
    )

    score = (
        persistence *
        PERSISTENCE_WEIGHT
    ) + (
        size_score *
        SIZE_WEIGHT
    ) + (
        center_bonus *
        CENTER_BONUS_WEIGHT
    )

    return score


def select_dominant_track(
    tracks,
    total_frames,
    source_width,
    source_height
):
    if not tracks:
        return None

    scored = []

    for track in tracks:
        score = score_track(
            track,
            total_frames,
            source_width,
            source_height
        )

        scored.append(
            (
                score,
                track
            )
        )

    scored.sort(
        key=lambda item:
            item[0],
        reverse=True
    )

    return scored[0][1]


def smooth_positions(
    positions
):
    """
    Remove occasional detection jumps.

    Median filtering is intentionally simple and robust.
    """

    if not positions:
        return []

    if len(positions) <= 2:
        return positions

    result = []

    for index in range(
        len(positions)
    ):
        left = max(
            0,
            index - 2
        )

        right = min(
            len(positions),
            index + 3
        )

        window = positions[
            left:right
        ]

        result.append(
            statistics.median(
                window
            )
        )

    return result


def calculate_focus(
    track,
    source_width,
    source_height
):
    """
    Calculate the horizontal focus point.

    We use the detected face positions but weight larger,
    clearer detections more heavily.
    """

    detections = track[
        "detections"
    ]

    if not detections:
        return 0.5

    weighted_positions = []

    for detection in detections:
        x = detection_center_x(
            detection
        )

        area = detection_area(
            detection
        )

        weight = math.sqrt(
            max(
                1,
                area
            )
        )

        weighted_positions.append(
            (
                x,
                weight
            )
        )

    total_weight = sum(
        weight
        for _, weight
        in weighted_positions
    )

    if total_weight <= 0:
        focus_x = (
            statistics.median(
                track["centerXs"]
            ) /
            source_width
        )
    else:
        focus_pixels = sum(
            x * weight
            for x, weight
            in weighted_positions
        ) / total_weight

        focus_x = (
            focus_pixels /
            source_width
        )

    return clamp(
        focus_x,
        0.0,
        1.0
    )


def calculate_crop(
    source_width,
    source_height,
    focus_x
):
    """
    Calculate a 1080x1920 crop after scaling the source
    while preserving its aspect ratio.
    """

    scale = max(
        TARGET_WIDTH /
        source_width,

        TARGET_HEIGHT /
        source_height
    )

    scaled_width = max(
        TARGET_WIDTH,
        int(
            round(
                source_width *
                scale
            )
        )
    )

    scaled_height = max(
        TARGET_HEIGHT,
        int(
            round(
                source_height *
                scale
            )
        )
    )

    max_crop_x = max(
        0,
        scaled_width -
        TARGET_WIDTH
    )

    max_crop_y = max(
        0,
        scaled_height -
        TARGET_HEIGHT
    )

    # Convert normalized focus to scaled pixels.
    focus_pixel = (
        focus_x *
        scaled_width
    )

    # Center the crop around the speaker.
    crop_x = int(
        round(
            focus_pixel -
            TARGET_WIDTH / 2.0
        )
    )

    crop_x = clamp(
        crop_x,
        0,
        max_crop_x
    )

    # We don't currently perform vertical tracking.
    # Keep the crop vertically centered.
    crop_y = int(
        max_crop_y / 2
    )

    return (
        scaled_width,
        scaled_height,
        crop_x,
        crop_y
    )


def detect_focus(
    video_path,
    start,
    end
):
    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open source video "
            "for smart framing."
        )

    source_width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    source_height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    if (
        source_width <= 0 or
        source_height <= 0
    ):
        cap.release()

        raise RuntimeError(
            "Could not determine source "
            "video dimensions."
        )

    duration = max(
        0.1,
        safe_float(end) -
        safe_float(start)
    )

    frontal, profile = (
        build_face_cascades()
    )

    samples = []

    for index in range(
        SAMPLE_COUNT
    ):
        if SAMPLE_COUNT > 1:
            fraction = (
                index /
                (SAMPLE_COUNT - 1)
            )
        else:
            fraction = 0.5

        timestamp = (
            safe_float(start) +
            duration *
            fraction
        )

        cap.set(
            cv2.CAP_PROP_POS_MSEC,
            timestamp * 1000
        )

        ok, frame = cap.read()

        if (
            not ok or
            frame is None
        ):
            samples.append([])
            continue

        detections = detect_faces(
            frame,
            frontal,
            profile,
            source_width,
            source_height
        )

        samples.append(
            detections
        )

    cap.release()

    total_frames_with_samples = (
        len(samples)
    )

    tracks = track_faces(
        samples,
        source_width,
        source_height
    )

    # Ignore extremely weak tracks.
    reliable_tracks = []

    for track in tracks:
        if (
            track["framesSeen"] >=
            max(
                2,
                int(
                    SAMPLE_COUNT *
                    0.10
                )
            )
        ):
            reliable_tracks.append(
                track
            )

    dominant = select_dominant_track(
        reliable_tracks,
        total_frames_with_samples,
        source_width,
        source_height
    )

    if dominant is None:
        focus_x = 0.5
        faces_detected = False
        frames_tracked = 0
        track_count = 0
        dominant_score = 0.0
    else:
        focus_x = calculate_focus(
            dominant,
            source_width,
            source_height
        )

        faces_detected = True
        frames_tracked = (
            dominant["framesSeen"]
        )
        track_count = len(
            reliable_tracks
        )

        dominant_score = score_track(
            dominant,
            total_frames_with_samples,
            source_width,
            source_height
        )

    (
        scaled_width,
        scaled_height,
        crop_x,
        crop_y
    ) = calculate_crop(
        source_width,
        source_height,
        focus_x
    )

    return {
        "sourceWidth": source_width,
        "sourceHeight": source_height,

        "scaleWidth": scaled_width,
        "scaleHeight": scaled_height,

        "cropX": crop_x,
        "cropY": crop_y,

        "focusX": round(
            focus_x,
            4
        ),

        "facesDetected": faces_detected,

        "facesTracked": frames_tracked,

        "peopleDetected": track_count,

        "dominantScore": round(
            dominant_score,
            3
        )
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

    try:
        start = float(
            sys.argv[2]
        )

        end = float(
            sys.argv[3]
        )

    except ValueError:
        print(
            "Start and end must be numbers.",
            file=sys.stderr
        )

        sys.exit(1)

    try:
        result = detect_focus(
            video_path,
            start,
            end
        )

        print(
            json.dumps(
                result
            )
        )

    except Exception as error:
        print(
            str(error),
            file=sys.stderr
        )

        sys.exit(1)


if __name__ == "__main__":
    main()