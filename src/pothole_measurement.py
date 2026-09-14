import cv2
import numpy as np


def create_pothole_mask(image, box):
    """
    Create a refined pothole-region mask inside the YOLO bounding box.

    This is a geometry refinement step using OpenCV.
    It does not claim centimetre-level physical dimensions.
    """

    if image is None or box is None:
        return None

    height, width = image.shape[:2]

    x1, y1, x2, y2 = [int(v) for v in box]

    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Reduce small image noise.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Local contrast enhancement.
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(blurred)

    # Adaptive threshold helps isolate darker damaged regions.
    mask = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        5
    )

    # Remove tiny noise.
    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    # Find contours.
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        # Fallback to the complete detection box.
        full_mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        full_mask[y1:y2, x1:x2] = 255

        return full_mask

    # Select the largest contour because the YOLO box already
    # identifies the pothole region.
    largest = max(
        contours,
        key=cv2.contourArea
    )

    refined = np.zeros_like(mask)

    cv2.drawContours(
        refined,
        [largest],
        -1,
        255,
        thickness=-1
    )

    full_mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    full_mask[y1:y2, x1:x2] = refined

    return full_mask


def calculate_geometry(mask):
    """
    Calculate measurable image geometry.

    Units:
        width  -> pixels
        height -> pixels
        area   -> pixels²
        perimeter -> pixels
    """

    if mask is None:
        return None

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    contour = max(
        contours,
        key=cv2.contourArea
    )

    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))

    x, y, width, height = cv2.boundingRect(contour)

    return {
        "width_px": float(width),
        "height_px": float(height),
        "area_px2": area,
        "perimeter_px": perimeter,
        "x": int(x),
        "y": int(y),
    }


def analyze_pothole_measurements(image, detection_result):
    """
    Calculate measurements for every detected pothole.

    Returns a list of dictionaries.
    """

    if image is None or detection_result is None:
        return []

    if isinstance(image, str):
        image = cv2.imread(image)

    if image is None:
        return []

    measurements = []

    boxes = detection_result.boxes

    if boxes is None or len(boxes) == 0:
        return measurements

    image_height, image_width = image.shape[:2]

    image_area = float(
        max(1, image_width * image_height)
    )

    for index, box in enumerate(boxes.xyxy):

        coordinates = box.cpu().numpy()

        mask = create_pothole_mask(
            image,
            coordinates
        )

        geometry = calculate_geometry(mask)

        if geometry is None:
            continue

        area_ratio = (
            geometry["area_px2"] /
            image_area
        )

        width_ratio = (
            geometry["width_px"] /
            max(1, image_width)
        )

        height_ratio = (
            geometry["height_px"] /
            max(1, image_height)
        )

        measurements.append({
            "id": index + 1,
            "width_px": geometry["width_px"],
            "height_px": geometry["height_px"],
            "area_px2": geometry["area_px2"],
            "perimeter_px": geometry["perimeter_px"],
            "area_percent": area_ratio * 100.0,
            "width_percent": width_ratio * 100.0,
            "height_percent": height_ratio * 100.0,
            "mask": mask,
        })

    return measurements