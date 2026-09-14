import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline


MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"

_depth_pipeline = None


def get_depth_model():
    """
    Load Depth Anything V2 only once.

    The model produces monocular relative depth.
    It does NOT directly provide calibrated centimetre measurements.
    """
    global _depth_pipeline

    if _depth_pipeline is None:
        device = 0 if torch.cuda.is_available() else -1

        _depth_pipeline = pipeline(
            task="depth-estimation",
            model=MODEL_NAME,
            device=device
        )

    return _depth_pipeline


def estimate_depth(image):
    """
    Estimate a relative depth map for an image.

    Returns:
        depth_map: numpy array with same H x W dimensions as input.
    """

    if image is None:
        return None

    if isinstance(image, str):
        image = cv2.imread(image)

    if image is None:
        return None

    if not isinstance(image, np.ndarray):
        return None

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)

    pipe = get_depth_model()

    result = pipe(pil_image)

    depth_image = result["depth"]

    depth_array = np.asarray(depth_image).astype(np.float32)

    original_height, original_width = image.shape[:2]

    depth_array = cv2.resize(
        depth_array,
        (original_width, original_height),
        interpolation=cv2.INTER_CUBIC
    )

    # Normalize only for relative comparison.
    minimum = float(np.min(depth_array))
    maximum = float(np.max(depth_array))

    if maximum - minimum > 1e-8:
        depth_array = (
            depth_array - minimum
        ) / (
            maximum - minimum
        )

    return depth_array


def get_region_depth(depth_map, mask):
    """
    Calculate depth statistics inside a pothole region.

    Returns:
        mean_depth
        max_depth
        min_depth
        relative_depth_percent
    """

    if depth_map is None or mask is None:
        return None, None, None, None

    valid_values = depth_map[mask > 0]

    if valid_values.size == 0:
        return None, None, None, None

    mean_depth = float(np.mean(valid_values))
    max_depth = float(np.max(valid_values))
    min_depth = float(np.min(valid_values))

    relative_depth_percent = mean_depth * 100.0

    return (
        mean_depth,
        max_depth,
        min_depth,
        relative_depth_percent
    )