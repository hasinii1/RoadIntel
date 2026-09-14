from ultralytics import YOLO
from pathlib import Path


# ============================================================
# ROADINTEL IMAGE DETECTOR
# ============================================================

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "deployment"
    / "best.onnx"
)

model = YOLO(str(MODEL_PATH))


# ============================================================
# IMAGE DETECTION
# ============================================================

def detect_potholes(image_path, confidence=0.25):
    """
    Run pothole detection on a single road image.

    Uses the trained RoadIntel YOLO11s ONNX model.

    Returns:
        Ultralytics detection results
    """

    results = model.predict(
        source=image_path,
        imgsz=640,
        conf=confidence,
        verbose=False
    )

    return results