import cv2
from pathlib import Path
from ultralytics import YOLO


# ============================================================
# ROADINTEL TRACKING + SEVERITY DETECTOR
# ============================================================

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "deployment"
    / "best.onnx"
)

model = YOLO(str(MODEL_PATH))


# ============================================================
# SEVERITY CALCULATION
# ============================================================

def calculate_severity(
    box_width,
    box_height,
    frame_width,
    frame_height
):
    """
    Estimate pothole severity based on the relative
    bounding-box area within the video frame.

    Returns:
        severity label
        severity score
    """

    if frame_width <= 0 or frame_height <= 0:
        return "Low", 0

    pothole_area = box_width * box_height

    frame_area = frame_width * frame_height

    area_ratio = pothole_area / frame_area

    # --------------------------------------------------------
    # Severity thresholds
    # --------------------------------------------------------

    if area_ratio < 0.02:

        severity = "Low"
        score = 1

    elif area_ratio < 0.06:

        severity = "Medium"
        score = 2

    else:

        severity = "High"
        score = 3

    return severity, score


# ============================================================
# TRACKING + SEVERITY
# ============================================================

def track_and_analyze(video_path, confidence=0.25):
    """
    Track potholes through a road video and estimate severity.

    Uses Ultralytics ByteTrack with streaming video inference.

    Returns:
        output_video_path
        total_detections
        unique_tracks
        low_count
        medium_count
        high_count
        total_frames
        frames_with_detections
    """

    video_path = str(video_path)

    # ========================================================
    # VIDEO INFORMATION
    # ========================================================

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(
            "Unable to open the uploaded video."
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    cap.release()

    # ========================================================
    # OUTPUT
    # ========================================================

    output_dir = (
        Path(__file__).resolve().parent.parent
        / "outputs"
        / "tracking"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir
        / "roadintel_tracking.mp4"
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        raise ValueError(
            "Unable to create the tracking output video."
        )

    # ========================================================
    # STATISTICS
    # ========================================================

    total_detections = 0

    frames_with_detections = 0

    # All track IDs observed during the video
    unique_track_ids = set()

    # Highest severity observed for each track
    severity_by_track = {}

    # Largest bounding-box area observed for each track
    max_area_by_track = {}

    frame_number = 0

    # ========================================================
    # STREAMING YOLO + BYTE TRACK
    # ========================================================
    #
    # Tracking remains sequential because ByteTrack needs
    # frame-to-frame continuity for persistent track IDs.
    #
    # The dynamic ONNX model is still used as the detector.
    # ========================================================

    results_stream = model.track(
        source=video_path,
        imgsz=640,
        conf=confidence,
        persist=True,
        tracker="bytetrack.yaml",
        stream=True,
        verbose=False
    )

    # ========================================================
    # PROCESS STREAMED RESULTS
    # ========================================================

    for result in results_stream:

        frame_number += 1

        # ----------------------------------------------------
        # GET ORIGINAL FRAME
        # ----------------------------------------------------

        frame = result.orig_img.copy()

        # ----------------------------------------------------
        # DETECTIONS
        # ----------------------------------------------------

        detection_count = 0

        if result.boxes is not None:

            detection_count = len(
                result.boxes
            )

        total_detections += detection_count

        if detection_count > 0:
            frames_with_detections += 1

        # ----------------------------------------------------
        # PROCESS DETECTIONS
        # ----------------------------------------------------

        if (
            result.boxes is not None
            and len(result.boxes) > 0
        ):

            boxes = result.boxes

            xyxy = (
                boxes.xyxy
                .cpu()
                .numpy()
            )

            # ------------------------------------------------
            # TRACKING IDS
            # ------------------------------------------------

            if boxes.id is not None:

                track_ids = (
                    boxes.id
                    .int()
                    .cpu()
                    .tolist()
                )

            else:

                track_ids = (
                    [None] * len(xyxy)
                )

            # ------------------------------------------------
            # PROCESS EVERY DETECTION
            # ------------------------------------------------

            for box, track_id in zip(
                xyxy,
                track_ids
            ):

                x1, y1, x2, y2 = box

                x1 = max(
                    0,
                    min(width - 1, x1)
                )

                y1 = max(
                    0,
                    min(height - 1, y1)
                )

                x2 = max(
                    0,
                    min(width - 1, x2)
                )

                y2 = max(
                    0,
                    min(height - 1, y2)
                )

                box_width = max(
                    0,
                    x2 - x1
                )

                box_height = max(
                    0,
                    y2 - y1
                )

                severity, severity_score = (
                    calculate_severity(
                        box_width,
                        box_height,
                        width,
                        height
                    )
                )

                # ============================================
                # TRACK INFORMATION
                # ============================================

                if track_id is not None:

                    track_id = int(
                        track_id
                    )

                    unique_track_ids.add(
                        track_id
                    )

                    # ----------------------------------------
                    # Store largest observed area
                    # ----------------------------------------

                    current_area = (
                        box_width
                        * box_height
                    )

                    previous_area = (
                        max_area_by_track.get(
                            track_id,
                            0
                        )
                    )

                    if current_area > previous_area:

                        max_area_by_track[
                            track_id
                        ] = current_area

                    # ----------------------------------------
                    # Store highest severity observed
                    # ----------------------------------------

                    previous_score = (
                        severity_by_track.get(
                            track_id,
                            0
                        )
                    )

                    if severity_score > previous_score:

                        severity_by_track[
                            track_id
                        ] = severity_score

                    label = (
                        f"Pothole #{track_id}"
                        f" | {severity}"
                    )

                else:

                    label = (
                        f"Pothole | {severity}"
                    )

                # ============================================
                # DRAW DETECTION BOX
                # ============================================

                cv2.rectangle(
                    frame,
                    (
                        int(x1),
                        int(y1)
                    ),
                    (
                        int(x2),
                        int(y2)
                    ),
                    (255, 165, 0),
                    2
                )

                # ============================================
                # LABEL BACKGROUND
                # ============================================

                text_x = int(x1)

                text_y = max(
                    25,
                    int(y1) - 10
                )

                (
                    text_width,
                    text_height
                ), baseline = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    2
                )

                cv2.rectangle(
                    frame,
                    (
                        text_x,
                        text_y
                        - text_height
                        - baseline
                    ),
                    (
                        text_x
                        + text_width,
                        text_y
                        + baseline
                    ),
                    (255, 165, 0),
                    -1
                )

                # ============================================
                # LABEL
                # ============================================

                cv2.putText(
                    frame,
                    label,
                    (
                        text_x,
                        text_y
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

        # ====================================================
        # VIDEO INFORMATION
        # ====================================================

        cv2.putText(
            frame,
            (
                f"Frame: "
                f"{frame_number}/{total_frames}"
            ),
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            (
                f"Tracked potholes: "
                f"{len(unique_track_ids)}"
            ),
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            (
                f"Detections: "
                f"{detection_count}"
            ),
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        # ====================================================
        # WRITE FRAME
        # ====================================================

        writer.write(frame)

    # ========================================================
    # CLEANUP
    # ========================================================

    writer.release()

    # ========================================================
    # FINAL SEVERITY COUNTS
    # ========================================================

    low_count = 0
    medium_count = 0
    high_count = 0

    for track_id in unique_track_ids:

        score = severity_by_track.get(
            track_id,
            1
        )

        if score == 1:

            low_count += 1

        elif score == 2:

            medium_count += 1

        elif score == 3:

            high_count += 1

    # ========================================================
    # FINAL RESULT
    # ========================================================

    return (
        str(output_path),
        total_detections,
        len(unique_track_ids),
        low_count,
        medium_count,
        high_count,
        total_frames,
        frames_with_detections
    )