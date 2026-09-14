import cv2
from pathlib import Path
from ultralytics import YOLO


# ============================================================
# ROADINTEL VIDEO DETECTOR
# ============================================================

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "deployment"
    / "best.onnx"
)

model = YOLO(str(MODEL_PATH))


# ============================================================
# VIDEO DETECTION
# ============================================================

def detect_video(video_path, confidence=0.25):
    """
    Run pothole detection on a road video.

    Frames are processed in batches using the dynamic-batch
    RoadIntel ONNX model.

    Returns:
        output_video_path
        total_detections
        total_frames
        frames_with_detections
    """

    video_path = str(video_path)

    # ========================================================
    # VIDEO INPUT
    # ========================================================

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Unable to open the uploaded video.")

    # --------------------------------------------------------
    # VIDEO INFORMATION
    # --------------------------------------------------------

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ========================================================
    # OUTPUT
    # ========================================================

    output_dir = (
        Path(__file__).resolve().parent.parent
        / "outputs"
        / "video"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "roadintel_detection.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise ValueError(
            "Unable to create the output video."
        )

    # ========================================================
    # DETECTION COUNTERS
    # ========================================================

    total_detections = 0
    frames_with_detections = 0

    # ========================================================
    # BATCH CONFIGURATION
    # ========================================================

    BATCH_SIZE = 8

    # ========================================================
    # PROCESS VIDEO IN BATCHES
    # ========================================================

    while True:

        frames = []

        # ----------------------------------------------------
        # Read up to BATCH_SIZE frames
        # ----------------------------------------------------

        for _ in range(BATCH_SIZE):

            ret, frame = cap.read()

            if not ret:
                break

            frames.append(frame)

        # ----------------------------------------------------
        # No more frames
        # ----------------------------------------------------

        if not frames:
            break

        # ====================================================
        # BATCH INFERENCE
        # ====================================================

        results = model.predict(
            source=frames,
            imgsz=640,
            conf=confidence,
            batch=len(frames),
            verbose=False
        )

        # ====================================================
        # PROCESS RESULTS
        # ====================================================

        for frame, result in zip(frames, results):

            detection_count = len(result.boxes)

            total_detections += detection_count

            if detection_count > 0:
                frames_with_detections += 1

            # ------------------------------------------------
            # Draw detections
            # ------------------------------------------------

            annotated_frame = result.plot()

            # ------------------------------------------------
            # Preserve original frame order
            # ------------------------------------------------

            writer.write(annotated_frame)

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()
    writer.release()

    # ========================================================
    # RETURN SAME INTERFACE AS BEFORE
    # ========================================================

    return (
        str(output_path),
        total_detections,
        total_frames,
        frames_with_detections
    )