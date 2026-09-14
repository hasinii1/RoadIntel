import gradio as gr
import sys
from pathlib import Path
import cv2
import numpy as np
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import detect_potholes, MODEL_PATH
from src.video_detector import detect_video
from src.tracking_detector import track_and_analyze
from src.depth_estimator import estimate_depth, get_region_depth
from src.pothole_measurement import analyze_pothole_measurements
from ultralytics import YOLO


# ============================================================
# VIDEO PLAYBACK / COMPATIBILITY
# ============================================================

def make_browser_compatible_video(video_path, output_name):
    """Convert OpenCV-generated MP4/AVI to H.264 MP4 for browser playback."""
    if not video_path:
        return None

    source = Path(str(video_path))
    if not source.exists():
        return str(video_path)

    output_dir = PROJECT_ROOT / "outputs" / "video"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name

    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        command = [
            ffmpeg,
            "-y",
            "-i", str(source),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",
            str(output_path),
        ]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if output_path.exists() and output_path.stat().st_size > 0:
            return str(output_path)
    except Exception:
        pass

    # Fallback: return the original result if FFmpeg conversion is unavailable.
    return str(video_path)


# ============================================================
# IMAGE LOADING HELPER
# ============================================================

def load_image_for_analysis(image):
    """Convert a Gradio image/path into an OpenCV BGR numpy array."""
    if image is None:
        return None

    if isinstance(image, str):
        img = cv2.imread(image)
        return img

    if isinstance(image, np.ndarray):
        return image.copy()

    return None


# ============================================================
# IMAGE GEOMETRY + MONOCULAR DEPTH
# ============================================================

def analyze_image_measurements(image, results):
    """
    Use the project's dedicated measurement and depth modules.

    Geometry is measured from the OpenCV-refined pothole mask in pixels.
    Depth is the normalized relative signal produced by Depth Anything V2.
    No physical centimetre values are invented or reported.
    """

    if image is None or results is None:
        return "—", "—", "—", "—", "—", "Waiting for image detection."

    if isinstance(image, str):
        img = cv2.imread(image)
    elif isinstance(image, np.ndarray):
        img = image.copy()
    else:
        img = None

    if img is None:
        return "—", "—", "—", "—", "—", "Unable to analyze image dimensions."

    result = results[0]
    pothole_count = len(result.boxes)

    if pothole_count == 0:
        return (
            "0 px",
            "0 px",
            "0 px²",
            "—",
            "0 / 0 potholes received a relative depth estimate.",
            "No potholes detected, so no pothole geometry or depth estimate is available."
        )

    # Real image-space geometry from the dedicated pothole measurement module.
    measurements = analyze_pothole_measurements(img, result)

    if not measurements:
        geometry_width = "—"
        geometry_height = "—"
        geometry_area = "—"
    else:
        largest = max(measurements, key=lambda item: item.get("area_px2", 0.0))
        geometry_width = f"{largest['width_px']:.0f} px"
        geometry_height = f"{largest['height_px']:.0f} px"
        geometry_area = f"{largest['area_px2']:.0f} px²"

    # Real model output from Depth Anything V2.
    depth_map = estimate_depth(img)
    relative_depth_values = []

    if depth_map is not None:
        for measurement in measurements:
            mask = measurement.get("mask")
            if mask is None:
                continue

            _, _, _, relative_depth_percent = get_region_depth(
                depth_map,
                mask
            )

            if relative_depth_percent is not None:
                relative_depth_values.append(float(relative_depth_percent))

    if relative_depth_values:
        # Show the strongest relative depth signal among detected potholes.
        relative_depth = max(relative_depth_values)
        depth_text = f"{relative_depth:.1f} / 100"
    else:
        depth_text = "Unavailable"

    depth_coverage = (
        f"{len(relative_depth_values)} / {len(measurements)} potholes received a relative depth estimate."
        if measurements
        else f"0 / {pothole_count} potholes received a relative depth estimate."
    )

    status = (
        "Width, height, area and perimeter are measured from the OpenCV-refined "
        "pothole region in image pixels. Depth Anything V2 provides a normalized "
        "relative depth signal; it is not calibrated physical depth in cm."
    )

    return (
        geometry_width,
        geometry_height,
        geometry_area,
        depth_text,
        depth_coverage,
        status,
    )


# ============================================================
# IMAGE SEVERITY ANALYSIS
# ============================================================

def analyze_image_severity(image, results):
    if image is None or results is None:
        return "—", "—", "—", "—", "Waiting for image detection."

    result = results[0]
    pothole_count = len(result.boxes)

    if pothole_count == 0:
        return "0", "0", "0", "0", "No potholes detected. Severity analysis completed."

    img = load_image_for_analysis(image)
    if img is None:
        return str(pothole_count), "—", "—", "—", "Unable to analyze image."

    height, width = img.shape[:2]
    image_area = max(1, width * height)
    low_count = medium_count = high_count = 0

    for box in result.boxes.xyxy:
        x1, y1, x2, y2 = box.cpu().numpy()
        box_width = max(0, x2 - x1)
        box_height = max(0, y2 - y1)
        area_ratio = (box_width * box_height) / image_area

        if area_ratio < 0.02:
            low_count += 1
        elif area_ratio < 0.05:
            medium_count += 1
        else:
            high_count += 1

    if high_count > 0:
        overall = "HIGH"
    elif medium_count > 0:
        overall = "MEDIUM"
    else:
        overall = "LOW"

    return (
        str(pothole_count),
        str(low_count),
        str(medium_count),
        str(high_count),
        f"Image severity analysis completed. Overall severity: {overall}."
    )


# ============================================================
# IMAGE DETECTION
# ============================================================

def run_image_detection(image, confidence):
    if image is None:
        return (None, "Please upload a road image.", None, "—", "—", "—", "—", "Please upload a road image.", "—", "—", "—", "—", "—")

    try:
        results = detect_potholes(image, confidence)
        result = results[0]
        annotated_image = result.plot()
        pothole_count = len(result.boxes)

        # Save the processed/annotated image in the project's outputs folder.
        # The original uploaded image is not copied or stored.
        images_dir = PROJECT_ROOT / "outputs" / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        output_image_path = images_dir / "roadintel_detection.png"
        cv2.imwrite(str(output_image_path), annotated_image)

        if pothole_count == 0:
            message = "No potholes detected."
        elif pothole_count == 1:
            message = "1 pothole detected."
        else:
            message = f"{pothole_count} potholes detected."

        detected, low, medium, high, severity_status = analyze_image_severity(image, results)
        largest_width, largest_height, largest_area, relative_depth, depth_coverage, measurement_status = analyze_image_measurements(image, results)
        combined_status = f"{severity_status}\n\n{measurement_status}"
        return (
            annotated_image,
            message,
            annotated_image,
            detected,
            low,
            medium,
            high,
            combined_status,
            largest_width,
            largest_height,
            largest_area,
            relative_depth,
            depth_coverage
        )
    except Exception as e:
        return (None, f"Detection error: {str(e)}", None, "—", "—", "—", "—", f"Analysis failed: {str(e)}", "—", "—", "—", "—", "—")


# ============================================================
# VIDEO DETECTION
# ============================================================

def run_video_detection(video, confidence):
    if video is None:
        return None, "Please upload a road video."

    try:
        output_video, total_detections, total_frames, frames_with_detections = detect_video(video, confidence)
        playable_video = make_browser_compatible_video(output_video, "roadintel_detection_browser.mp4")
        message = (
            f"Detection completed successfully.\n"
            f"Total detections: {total_detections}\n"
            f"Frames analyzed: {total_frames}\n"
            f"Frames with potholes: {frames_with_detections}"
        )
        return playable_video, message
    except Exception as e:
        return None, f"Video processing error: {str(e)}"


# ============================================================
# TRACKING + VIDEO SEVERITY
# ============================================================

def fallback_video_tracking_and_severity(video_path, confidence):
    """Reliable software-only video tracking fallback.

    Some Ultralytics tracker configurations can return bounding boxes but no
    persistent IDs. In that case we run normal YOLO prediction and maintain
    lightweight IoU-based IDs ourselves. This guarantees that Page 3 receives
    real pothole counts and severity counts instead of zero values.
    """
    model = YOLO(str(MODEL_PATH))

    # track_id -> latest bounding box [x1, y1, x2, y2]
    active_tracks = {}
    # track_id -> maximum observed bounding-box area ratio
    max_area_by_id = {}
    # Frames since a track was last matched
    missed_by_id = {}

    next_track_id = 1
    total_detections = 0
    frames_with_detections = 0
    total_frames = 0
    frame_width = 0
    frame_height = 0

    cap = cv2.VideoCapture(str(video_path))
    if cap.isOpened():
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    def iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        intersection = iw * ih
        if intersection <= 0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0.0

    # Normal prediction is intentional here: it does not depend on the
    # tracker producing IDs, which is the problem we are recovering from.
    results_stream = model.predict(
        source=str(video_path),
        imgsz=640,
        conf=confidence,
        stream=True,
        verbose=False,
    )

    for result in results_stream:
        boxes = result.boxes
        count = len(boxes) if boxes is not None else 0
        total_detections += count
        if count > 0:
            frames_with_detections += 1

        # Age existing tracks. Tracks are allowed to disappear briefly so
        # temporary missed detections do not immediately create new IDs.
        for track_id in list(missed_by_id):
            missed_by_id[track_id] += 1
            if missed_by_id[track_id] > 8:
                active_tracks.pop(track_id, None)
                missed_by_id.pop(track_id, None)

        if count == 0:
            continue

        current_boxes = []
        for box in boxes.xyxy.cpu().numpy():
            current_boxes.append([float(v) for v in box])

        # Greedy highest-IoU matching between current detections and active tracks.
        candidates = []
        for det_idx, det_box in enumerate(current_boxes):
            for track_id, old_box in active_tracks.items():
                score = iou(det_box, old_box)
                if score >= 0.30:
                    candidates.append((score, det_idx, track_id))

        candidates.sort(reverse=True)
        matched_detections = set()
        matched_tracks = set()
        det_to_track = {}

        for score, det_idx, track_id in candidates:
            if det_idx in matched_detections or track_id in matched_tracks:
                continue
            matched_detections.add(det_idx)
            matched_tracks.add(track_id)
            det_to_track[det_idx] = track_id
            active_tracks[track_id] = current_boxes[det_idx]
            missed_by_id[track_id] = 0

        # Create new IDs for detections that could not be matched.
        for det_idx, det_box in enumerate(current_boxes):
            if det_idx in matched_detections:
                continue
            track_id = next_track_id
            next_track_id += 1
            active_tracks[track_id] = det_box
            missed_by_id[track_id] = 0
            det_to_track[det_idx] = track_id

        # Record the largest visible area for every track.
        denominator = float(frame_width * frame_height) if frame_width and frame_height else 1.0
        for det_idx, det_box in enumerate(current_boxes):
            matched_id = det_to_track.get(det_idx)
            if matched_id is None:
                continue
            x1, y1, x2, y2 = det_box
            ratio = max(0.0, (x2 - x1) * (y2 - y1)) / denominator
            max_area_by_id[matched_id] = max(max_area_by_id.get(matched_id, 0.0), ratio)

    unique_tracks = len(max_area_by_id)

    low = medium = high = 0
    for ratio in max_area_by_id.values():
        if ratio < 0.02:
            low += 1
        elif ratio < 0.05:
            medium += 1
        else:
            high += 1

    return (
        unique_tracks,
        low,
        medium,
        high,
        total_detections,
        total_frames,
        frames_with_detections,
    )


def run_tracking(video, confidence):
    if video is None:
        return None, "No video was processed on the Detection page.", "—", "—", "—", "—", "Process a road video on Detection first."

    try:
        (
            output_video,
            total_detections,
            unique_tracks,
            low_count,
            medium_count,
            high_count,
            total_frames,
            frames_with_detections
        ) = track_and_analyze(video, confidence)

        # Some tracker configurations can detect potholes but fail to return
        # persistent track IDs. In that case, run a reliable Ultralytics
        # fallback tracker so Page 3 and Page 4 still receive real severity data.
        if total_detections > 0 and unique_tracks == 0:
            (
                unique_tracks,
                low_count,
                medium_count,
                high_count,
                fallback_detections,
                fallback_frames,
                fallback_frames_with_detections
            ) = fallback_video_tracking_and_severity(video, confidence)
            if fallback_detections > 0:
                total_detections = fallback_detections
            if fallback_frames > 0:
                total_frames = fallback_frames
            frames_with_detections = fallback_frames_with_detections

        playable_video = make_browser_compatible_video(output_video, "roadintel_tracking_browser.mp4")
        summary = (
            f"Tracking and severity analysis completed successfully.\n"
            f"Total detections: {total_detections}\n"
            f"Unique potholes tracked: {unique_tracks}\n"
            f"Frames analyzed: {total_frames}\n"
            f"Frames with potholes: {frames_with_detections}\n"
            f"Severity — Low: {low_count} | Medium: {medium_count} | High: {high_count}"
        )
        return (
            playable_video,
            summary,
            str(unique_tracks),
            str(low_count),
            str(medium_count),
            str(high_count),
            "Video tracking and severity analysis completed successfully."
        )
    except Exception as e:
        return None, f"Tracking error: {str(e)}", "—", "—", "—", "—", f"Analysis failed: {str(e)}"


# ============================================================
# ROAD INTELLIGENCE
# ============================================================

def calculate_road_intelligence(potholes, low_count, medium_count, high_count):
    try:
        total = int(potholes)
        low = int(low_count)
        medium = int(medium_count)
        high = int(high_count)
    except (ValueError, TypeError):
        return "—", "—", "—", "—", "—", "WAITING", "Waiting for detection or tracking results."

    # If a tracker reports severity counts but its unique-ID counter is zero,
    # use the severity counts as the pothole total for intelligence scoring.
    if total <= 0 and (low + medium + high) > 0:
        total = low + medium + high

    if total <= 0:
        return "100 / 100", "0", "0", "0", "0", "EXCELLENT", "No potholes were detected. The analyzed road section appears to be in excellent condition."

    # Normalize the severity mix to a 0-100 road-condition score.
    # This prevents the number of potholes from making the score negative simply
    # because there are many tracked potholes. Each unique pothole contributes
    # according to its Page 3 severity: Low=1, Medium=3, High=5.
    #
    # Example: 26 unique potholes = 23 Low + 2 Medium + 1 High
    # weighted_damage = 23*1 + 2*3 + 1*5 = 34
    # maximum_damage = 26*5 = 130
    # score = 100 * (1 - 34/130) = 73.85 -> 74/100
    weighted_damage = (low * 1) + (medium * 3) + (high * 5)
    maximum_damage = total * 5
    severity_ratio = weighted_damage / maximum_damage if maximum_damage > 0 else 0.0
    score = round(max(0.0, min(100.0, 100.0 * (1.0 - severity_ratio))))

    # Overall road condition is based on the resulting score. A single high-
    # severity pothole is still reflected in the score and safety risk, but it
    # does not by itself force the entire road-condition score to POOR.
    if score < 20:
        condition = "CRITICAL"
        assessment = "The analyzed road section shows severe visual damage and should be prioritized for inspection and maintenance."
    elif score < 40:
        condition = "POOR"
        assessment = "The analyzed road section shows significant visual pothole damage and should be inspected for maintenance."
    elif score < 60:
        condition = "MODERATE"
        assessment = "The analyzed road section shows moderate visual damage and may require maintenance attention."
    elif score < 80:
        condition = "GOOD"
        assessment = "The analyzed road section has some visible pothole damage but remains comparatively less affected."
    else:
        condition = "EXCELLENT"
        assessment = "The analyzed road section has relatively low visual pothole impact."

    return f"{score} / 100", str(total), str(low), str(medium), str(high), condition, assessment


# ============================================================
# CAMERA-BASED ROAD SAFETY
# ============================================================

def _parse_analysis_value(value):
    """Return an integer analysis value, or None when the source is not ready."""
    if value is None:
        return None
    text = str(value).strip()
    if text in ("", "—", "-", "None", "Waiting for image detection.",
                "Waiting for video processing.", "Waiting for image analysis.",
                "Waiting for video analysis."):
        return None
    try:
        return int(float(text))
    except (ValueError, TypeError):
        return None


def mark_processed_if_valid(total, low, medium, high):
    """Set processed state only when Page 2/3 actually produced numeric results."""
    return all(
        _parse_analysis_value(value) is not None
        for value in (total, low, medium, high)
    )


def calculate_combined_safety(
    image_ready,
    image_total, image_low, image_medium, image_high,
    video_ready,
    video_total, video_low, video_medium, video_high
):
    """
    Camera-based safety screening.

    A source contributes only when its processed flag is True.
    Therefore:
      - image only -> image evidence only
      - video only -> video evidence only
      - both -> both sources combined
      - neither -> waiting
    """
    image_ready = bool(image_ready)
    video_ready = bool(video_ready)

    if not image_ready and not video_ready:
        return "—", "—", "Waiting for image or video analysis."

    it = _parse_analysis_value(image_total) if image_ready else 0
    il = _parse_analysis_value(image_low) if image_ready else 0
    im = _parse_analysis_value(image_medium) if image_ready else 0
    ih = _parse_analysis_value(image_high) if image_ready else 0

    vt = _parse_analysis_value(video_total) if video_ready else 0
    vl = _parse_analysis_value(video_low) if video_ready else 0
    vm = _parse_analysis_value(video_medium) if video_ready else 0
    vh = _parse_analysis_value(video_high) if video_ready else 0

    # A processed source with zero potholes is still valid evidence.
    it = 0 if it is None else it
    il = 0 if il is None else il
    im = 0 if im is None else im
    ih = 0 if ih is None else ih

    vt = 0 if vt is None else vt
    vl = 0 if vl is None else vl
    vm = 0 if vm is None else vm
    vh = 0 if vh is None else vh

    total = it + vt
    low = il + vl
    medium = im + vm
    high = ih + vh

    if total <= 0:
        source_text = "the analyzed image" if image_ready and not video_ready else (
            "the analyzed video" if video_ready and not image_ready else
            "the analyzed image and video"
        )
        return (
            "LOW",
            "LOW",
            f"No visible potholes were detected in {source_text}. "
            "Camera-based safety risk is therefore LOW."
        )

    # Visual safety weighting: severity contributes more than raw count.
    risk_points = low + (medium * 5) + (high * 12)

    if high >= 1 or risk_points >= 20:
        risk = "HIGH"
        assessment = (
            "High camera-based safety risk. The analyzed road media contains "
            "significant visible pothole damage and should be prioritized for "
            "inspection and maintenance."
        )
    elif medium >= 2 or risk_points >= 5:
        risk = "MEDIUM"
        assessment = (
            "Moderate camera-based safety risk. Visible road damage should be "
            "monitored and considered for maintenance."
        )
    else:
        risk = "LOW"
        assessment = (
            "Lower camera-based safety risk based on the detected pothole "
            "count and visual severity."
        )

    return risk, risk, assessment


# ============================================================
# REPORT GENERATION — PAGE 4
# ============================================================

def _safe_report_path():
    """Store reports in the existing outputs folder.
    Only create outputs/reports when it does not already exist.
    If outputs/reports is an existing file, use outputs directly.
    """
    outputs_dir = PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    reports_path = outputs_dir / "reports"

    if reports_path.exists() and reports_path.is_dir():
        return reports_path / "roadintel_report.txt"
    if reports_path.exists() and reports_path.is_file():
        return outputs_dir / "roadintel_report.txt"

    reports_path.mkdir(parents=True, exist_ok=True)
    return reports_path / "roadintel_report.txt"


def generate_report(
    image_score, image_total, image_low, image_medium, image_high,
    image_condition, image_assessment,
    image_largest_width, image_largest_height, image_largest_area, image_relative_depth,
    video_score, video_total, video_low, video_medium, video_high,
    video_condition, video_assessment,
    safety_risk, safety_assessment
):
    image_ready = image_score not in (None, "—", "") and image_total not in (None, "—", "")
    video_ready = video_score not in (None, "—", "") and video_total not in (None, "—", "")

    if not image_ready and not video_ready:
        return None, "Please run image detection and/or video processing first."

    try:
        report_path = _safe_report_path()
        sections = [
            "ROADINTEL — ROAD CONDITION & SAFETY REPORT",
            "===========================================",
            "",
        ]

        if image_ready:
            sections += [
                "IMAGE ANALYSIS",
                "--------------",
                f"Road Condition Score : {image_score}",
                f"Total Potholes       : {image_total}",
                f"Low Severity         : {image_low}",
                f"Medium Severity      : {image_medium}",
                f"High Severity        : {image_high}",
                f"Overall Condition    : {image_condition}",
                f"Assessment           : {image_assessment}",
                f"Largest Width        : {image_largest_width}",
                f"Largest Height       : {image_largest_height}",
                f"Largest Area         : {image_largest_area}",
                f"Relative Depth       : {image_relative_depth}",
                "Measurement note    : Width/height/area are image-pixel measurements; relative depth is not physical cm.",
                "",
            ]

        if video_ready:
            sections += [
                "VIDEO ANALYSIS",
                "---------------",
                f"Road Condition Score : {video_score}",
                f"Unique Potholes      : {video_total}",
                f"Low Severity         : {video_low}",
                f"Medium Severity      : {video_medium}",
                f"High Severity        : {video_high}",
                f"Overall Condition    : {video_condition}",
                f"Assessment           : {video_assessment}",
                "",
            ]

        sections += [
            "CAMERA-BASED ROAD SAFETY",
            "------------------------",
            f"Safety Risk          : {safety_risk}",
            f"Safety Assessment    : {safety_assessment}",
            "",
            "LIMITATIONS",
            "-----------",
            "RoadIntel uses camera-based computer vision to detect visible potholes, measure image-space geometry, and estimate a model-relative depth signal. Width, height and area are reported in pixels. The depth output is not calibrated physical depth in cm and the system does not directly measure pavement strength, structural integrity, vehicle speed, or certify actual road safety. These results support inspection and maintenance prioritization and are not engineering certification.",
            "",
            "Generated by RoadIntel • YOLO11s",
        ]

        report_text = "\n".join(sections)
        report_path.write_text(report_text, encoding="utf-8")
        preview = "ROADINTEL REPORT GENERATED SUCCESSFULLY\n\n" + report_text
        return str(report_path), preview
    except Exception as e:
        return None, f"Report generation failed: {str(e)}"


# ============================================================
# PAGE NAVIGATION
# ============================================================

def show_overview():

    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
    )


def show_detection():

    return (
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
    )


def show_tracking():

    return (
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
    )


def _source_is_ready(processed, total, low, medium, high):
    """A source is valid only when its own Page 2/3 outputs contain real numeric results."""
    if not bool(processed):
        return False
    values = (
        _parse_analysis_value(total),
        _parse_analysis_value(low),
        _parse_analysis_value(medium),
        _parse_analysis_value(high),
    )
    return all(value is not None for value in values)


def metric_card_html(label, value):
    """Render a guaranteed-visible Page 4 metric value."""
    return f"""
    <div class=\"metric-card-inner\">
        <div class=\"metric-card-label\">{label}</div>
        <div class=\"metric-card-value\">{value}</div>
    </div>
    """


def show_intelligence(image_ready, image_total, image_low, image_medium, image_high, video_ready, video_total, video_low, video_medium, video_high):
    # Page 4 is a pure consumer of Page 2/3 results.
    # It never creates fallback analysis data of its own.
    image_ready = _source_is_ready(image_ready, image_total, image_low, image_medium, image_high)
    video_ready = _source_is_ready(video_ready, video_total, video_low, video_medium, video_high)

    if image_ready:
        ir = calculate_road_intelligence(image_total, image_low, image_medium, image_high)
    else:
        ir = ("—", "—", "—", "—", "—", "WAITING", "Waiting for image analysis.")

    if video_ready:
        vr = calculate_road_intelligence(video_total, video_low, video_medium, video_high)
    else:
        vr = ("—", "—", "—", "—", "—", "WAITING", "Waiting for video analysis.")

    sr, sl, sa = calculate_combined_safety(
        image_ready,
        image_total, image_low, image_medium, image_high,
        video_ready,
        video_total, video_low, video_medium, video_high,
    )

    return (
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=image_ready),
        gr.update(visible=video_ready),
        metric_card_html("ROAD CONDITION SCORE", ir[0]),
        ir[1], ir[2], ir[3], ir[4], ir[0],
        ir[5],
        ir[6],
        metric_card_html("ROAD CONDITION SCORE", vr[0]),
        vr[1], vr[2], vr[3], vr[4], vr[0],
        vr[5],
        vr[6],
        metric_card_html("OVERALL SAFETY RISK", sr),
        metric_card_html("RISK LEVEL", sl),
        sa,
        sr,
        sl,
    )


# ============================================================
# LIGHT MAIN + DARK SIDEBAR THEME
# ============================================================

CUSTOM_CSS = """

/* ============================================================
   GLOBAL
   ============================================================ */

html,
body,
gradio-app {

    margin: 0 !important;
    padding: 0 !important;

    width: 100% !important;

    min-height: 100% !important;

    background: #f4f7fb !important;
}

body {

    overflow-x: hidden !important;
}

.gradio-container {

    width: 100vw !important;
    max-width: 100vw !important;

    min-height: 100vh !important;

    margin: 0 !important;
    padding: 0 !important;

    background: #f4f7fb !important;

    color: #172033 !important;
}

.gradio-container .contain {

    width: 100% !important;
    max-width: none !important;

    margin: 0 !important;
    padding: 0 !important;
}


/* ============================================================
   APP SHELL
   ============================================================ */

.app-shell {

    display: flex !important;

    flex-direction: row !important;

    flex-wrap: nowrap !important;

    align-items: stretch !important;

    width: 100vw !important;

    min-height: 100vh !important;

    margin-left: calc(50% - 50vw) !important;

    padding: 0 !important;

    gap: 0 !important;

    box-sizing: border-box !important;

    background: #f4f7fb !important;
}


/* ============================================================
   DARK SIDEBAR
   ============================================================ */

.sidebar {

    flex: 0 0 260px !important;

    width: 260px !important;
    min-width: 260px !important;
    max-width: 260px !important;

    min-height: 100vh !important;

    box-sizing: border-box !important;

    margin: 0 !important;

    padding: 30px 20px !important;

    background:
        linear-gradient(
            180deg,
            #0b172a 0%,
            #101f36 55%,
            #0a1425 100%
        ) !important;

    border-right:
        1px solid #1d3453 !important;

    border-radius: 0 !important;

    box-shadow:
        8px 0 35px rgba(2, 12, 27, 0.18) !important;

    color: #ffffff !important;

    z-index: 5 !important;
}


/* ============================================================
   SIDEBAR BRAND
   ============================================================ */

.sidebar-brand {

    display: flex;

    align-items: center;

    gap: 13px;

    padding: 2px 7px 25px 7px;

    margin-bottom: 27px;

    border-bottom:
        1px solid #243852;
}

.sidebar-icon {

    width: 46px;
    height: 46px;

    display: flex;

    align-items: center;
    justify-content: center;

    flex-shrink: 0;

    border-radius: 12px;

    background:
        linear-gradient(
            135deg,
            #2563eb,
            #0ea5e9
        );

    font-size: 23px;

    box-shadow:
        0 8px 25px rgba(14, 165, 233, 0.20);
}

.sidebar-title {

    color: #ffffff !important;

    font-size: 21px;

    font-weight: 800;

    letter-spacing: 1px;

    line-height: 1.1;
}

.sidebar-subtitle {

    margin-top: 5px;

    color: #91a4bd !important;

    font-size: 9px;

    font-weight: 500;

    letter-spacing: 0.8px;

    line-height: 1.4;
}


/* ============================================================
   MENU
   ============================================================ */

.nav-label {

    margin: 0 8px 13px 8px;

    color: #7186a2 !important;

    font-size: 10px;

    font-weight: 700;

    letter-spacing: 1.4px;

    text-transform: uppercase;
}

.nav-item {

    display: flex !important;

    align-items: center !important;

    gap: 13px !important;

    width: 100% !important;

    box-sizing: border-box !important;

    padding: 13px 12px !important;

    margin: 5px 0 !important;

    border-radius: 10px !important;

    color: #9aabc0 !important;

    font-size: 13px !important;

    font-weight: 500 !important;

    border-left: 3px solid transparent !important;

    background: transparent !important;
}

.nav-item:hover {

    background:
        rgba(59, 130, 246, 0.10) !important;

    color: #dbeafe !important;
}


/* ============================================================
   NAVIGATION BUTTONS
   ============================================================ */

.nav-button {

    width: 100% !important;

    min-height: 46px !important;

    margin: 5px 0 !important;

    padding: 13px 12px !important;

    border-radius: 10px !important;

    border: none !important;

    border-left: 3px solid transparent !important;

    background: transparent !important;

    color: #9aabc0 !important;

    box-shadow: none !important;

    font-size: 13px !important;

    font-weight: 500 !important;

    text-align: left !important;

    justify-content: flex-start !important;

    transition: all 0.2s ease !important;
}

.nav-button:hover {

    background:
        rgba(59, 130, 246, 0.12) !important;

    color: #ffffff !important;

    border-left:
        3px solid #3b82f6 !important;
}

.nav-button:focus {

    box-shadow: none !important;
}

.nav-button span {

    color: inherit !important;
}


/* ============================================================
   SIDEBAR STATUS
   ============================================================ */

.sidebar-status-box {

    margin-top: 75px;

    padding: 15px;

    border-radius: 12px;

    background:
        rgba(34, 197, 94, 0.07) !important;

    border:
        1px solid rgba(34, 197, 94, 0.18);
}

.sidebar-status {

    display: flex;

    align-items: center;

    gap: 8px;

    color: #4ade80 !important;

    font-size: 11px;

    font-weight: 700;
}

.sidebar-status-dot {

    width: 7px;
    height: 7px;

    border-radius: 50%;

    background: #22c55e;

    box-shadow:
        0 0 10px rgba(34, 197, 94, 0.45);
}

.sidebar-version {

    margin-top: 8px;

    color: #7186a2 !important;

    font-size: 10px;
}


/* ============================================================
   MAIN CONTENT — LIGHT
   ============================================================ */

.main-content {

    flex: 1 1 auto !important;

    width: auto !important;

    min-width: 0 !important;

    max-width: none !important;

    min-height: 100vh !important;

    box-sizing: border-box !important;

    margin: 0 !important;

    padding: 32px 42px 40px 42px !important;

    background:
        radial-gradient(
            circle at 80% 10%,
            rgba(37, 99, 235, 0.06),
            transparent 28%
        ),
        #f4f7fb !important;

    border: none !important;

    border-radius: 0 !important;

    color: #172033 !important;

    overflow: visible !important;
}


/* ============================================================
   PAGE HEADER
   ============================================================ */

.page-header {

    display: flex;

    align-items: center;

    justify-content: space-between;

    width: 100%;

    margin-bottom: 25px;
}

.page-heading {

    margin: 0 !important;
    padding: 0 !important;

    color: #172033 !important;

    font-size: 30px !important;

    font-weight: 800 !important;

    line-height: 1.2 !important;
}

.page-description {

    margin-top: 7px;

    color: #64748b !important;

    font-size: 13px;

    line-height: 1.5;
}

.system-status {

    display: flex;

    align-items: center;

    gap: 8px;

    padding: 9px 15px;

    border-radius: 999px;

    background:
        rgba(34, 197, 94, 0.07);

    border:
        1px solid rgba(34, 197, 94, 0.22);

    color: #15803d !important;

    font-size: 11px;

    font-weight: 700;
}

.status-dot {

    width: 7px;
    height: 7px;

    border-radius: 50%;

    background: #22c55e;

    box-shadow:
        0 0 10px rgba(34, 197, 94, 0.35);
}


/* ============================================================
   HERO
   ============================================================ */

.hero-card {

    position: relative;

    overflow: hidden;

    width: 100%;

    box-sizing: border-box;

    padding: 40px 45px;

    margin-bottom: 27px;

    border-radius: 20px;

    background:
        linear-gradient(
            135deg,
            #e8f1ff 0%,
            #eef6ff 55%,
            #f4f9ff 100%
        ) !important;

    border:
        1px solid #cfe0f5 !important;

    box-shadow:
        0 20px 50px rgba(30, 64, 175, 0.08);
}

.hero-glow {

    position: absolute;

    width: 420px;
    height: 420px;

    right: -180px;
    top: -230px;

    border-radius: 50%;

    background:
        rgba(37, 99, 235, 0.08);

    filter: blur(25px);
}

.hero-content {

    position: relative;

    z-index: 2;
}

.hero-badge {

    display: inline-block;

    padding: 7px 13px;

    margin-bottom: 16px;

    border-radius: 999px;

    background:
        rgba(59, 130, 246, 0.08);

    border:
        1px solid rgba(96, 165, 250, 0.20);

    color: #2563eb !important;

    font-size: 10px;

    font-weight: 700;

    letter-spacing: 1px;
}

.hero-title {

    margin: 0 !important;

    color: #172033 !important;

    font-size: 40px !important;

    font-weight: 800 !important;

    line-height: 1.15 !important;
}

.hero-title span {

    color: #2563eb !important;
}

.hero-description {

    max-width: 850px;

    margin: 17px 0 24px 0;

    color: #526174 !important;

    font-size: 14px;

    line-height: 1.75;
}


/* ============================================================
   FEATURE PILLS
   ============================================================ */

.hero-features {

    display: flex;

    flex-wrap: wrap;

    gap: 9px;
}

.feature-pill {

    padding: 8px 13px;

    border-radius: 8px;

    background:
        rgba(255, 255, 255, 0.75) !important;

    border:
        1px solid #d7e3f1;

    color: #526174 !important;

    font-size: 11px;

    font-weight: 500;
}


/* ============================================================
   SECTION TITLES
   ============================================================ */

.section-title {

    margin: 0 0 13px 2px !important;

    color: #334155 !important;

    font-size: 13px !important;

    font-weight: 750 !important;

    letter-spacing: 0.7px !important;
}


/* ============================================================
   METRIC ROW
   ============================================================ */

.metrics-row {

    display: flex !important;

    flex-direction: row !important;

    flex-wrap: nowrap !important;

    gap: 15px !important;

    width: 100% !important;

    margin-bottom: 27px !important;
}


/* ============================================================
   METRIC CARDS
   ============================================================ */

.metric-card {

    flex: 1 1 0 !important;

    min-width: 0 !important;

    min-height: 130px;

    box-sizing: border-box;

    padding: 20px;

    border-radius: 15px;

    background:
        linear-gradient(
            145deg,
            #ffffff,
            #f8fbff
        ) !important;

    border:
        1px solid #dbe4ef;

    box-shadow:
        0 12px 30px rgba(30, 64, 175, 0.06);
}

.metric-label {

    color: #718096 !important;

    font-size: 10px;

    font-weight: 700;

    letter-spacing: 1px;

    text-transform: uppercase;
}

.metric-value {

    margin-top: 9px;

    color: #172033 !important;

    font-size: 29px;

    font-weight: 800;

    line-height: 1.1;
}

.metric-description {

    margin-top: 7px;

    color: #8492a6 !important;

    font-size: 10px;

    line-height: 1.45;
}


/* ============================================================
   LOWER ROW
   ============================================================ */

.lower-row {

    display: flex !important;

    flex-direction: row !important;

    flex-wrap: nowrap !important;

    gap: 18px !important;

    width: 100% !important;
}

.lower-column {

    flex: 1 1 0 !important;

    min-width: 0 !important;
}


/* ============================================================
   INFO CARDS
   ============================================================ */

.info-card {

    width: 100%;

    min-height: 220px;

    box-sizing: border-box;

    padding: 22px;

    border-radius: 16px;

    background:
        linear-gradient(
            145deg,
            #ffffff,
            #f8fbff
        ) !important;

    border:
        1px solid #dbe4ef;

    box-shadow:
        0 12px 30px rgba(30, 64, 175, 0.06);
}


/* ============================================================
   SYSTEM ROW
   ============================================================ */

.system-row {

    display: flex;

    align-items: center;

    justify-content: space-between;

    padding: 13px 0;

    border-bottom:
        1px solid #e8edf3;
}

.system-row:last-child {

    border-bottom: none;
}

.system-name {

    color: #526174 !important;

    font-size: 12px;
}

.system-ready {

    color: #15803d !important;

    font-size: 11px;

    font-weight: 700;
}


/* ============================================================
   WORKFLOW
   ============================================================ */

.workflow-step {

    display: flex;

    align-items: center;

    gap: 12px;

    margin: 11px 0;
}

.workflow-number {

    width: 31px;
    height: 31px;

    flex-shrink: 0;

    display: flex;

    align-items: center;
    justify-content: center;

    border-radius: 8px;

    background:
        rgba(37, 99, 235, 0.08);

    border:
        1px solid rgba(59, 130, 246, 0.17);

    color: #2563eb !important;

    font-size: 10px;

    font-weight: 700;
}

.workflow-text {

    color: #526174 !important;

    font-size: 12px;
}


/* ============================================================
   DETECTION PAGE
   ============================================================ */

.detection-section-title {

    margin-top: 5px;

    color: #334155 !important;

    font-size: 15px;

    font-weight: 800;

    letter-spacing: 0.8px;
}

.detection-section-description {

    margin-top: 6px;

    margin-bottom: 15px;

    color: #718096 !important;

    font-size: 12px;

    line-height: 1.5;
}

.video-title {

    margin-top: 35px;
}

.detection-row {

    display: flex !important;

    flex-direction: row !important;

    flex-wrap: nowrap !important;

    gap: 18px !important;

    width: 100% !important;
}

.detection-card {

    flex: 1 1 0 !important;

    min-width: 0 !important;

    box-sizing: border-box !important;

    padding: 20px !important;

    border-radius: 16px !important;

    background:
        linear-gradient(
            145deg,
            #ffffff,
            #f8fbff
        ) !important;

    border:
        1px solid #dbe4ef !important;

    box-shadow:
        0 12px 30px rgba(30, 64, 175, 0.06) !important;
}

.detection-card label {

    color: #526174 !important;
}

.detection-card input {

    color: #172033 !important;
}


/* ============================================================
   DETECTION BUTTON
   ============================================================ */

.detection-button {

    margin-top: 8px !important;

    border-radius: 10px !important;

    border: none !important;

    background:
        linear-gradient(
            135deg,
            #2563eb,
            #0ea5e9
        ) !important;

    color: #ffffff !important;

    font-weight: 700 !important;

    box-shadow:
        0 8px 20px rgba(37, 99, 235, 0.18) !important;
}

.detection-button:hover {

    filter: brightness(1.05);
}


/* ============================================================
   INPUTS / TEXTBOXES
   ============================================================ */

.detection-card textarea,
.detection-card input,
.tracking-status textarea,
.tracking-status input {

    background: #f8fafc !important;

    border:
        1px solid #dbe4ef !important;

    color: #172033 !important;
}


/* ============================================================
   TRACKING METRICS
   ============================================================ */

.tracking-metric {

    flex: 1 1 0 !important;

    min-width: 0 !important;

    box-sizing: border-box !important;

    padding: 18px !important;

    border-radius: 15px !important;

    background:
        linear-gradient(
            145deg,
            #ffffff,
            #f8fbff
        ) !important;

    border:
        1px solid #dbe4ef !important;

    box-shadow:
        0 12px 30px rgba(30, 64, 175, 0.06) !important;
}

.tracking-metric label {

    color: #718096 !important;

    font-size: 10px !important;

    font-weight: 700 !important;

    letter-spacing: 1px !important;

    text-transform: uppercase !important;
}

.tracking-metric input,
.tracking-metric textarea {

    margin-top: 6px !important;

    background: transparent !important;

    border: none !important;

    color: #172033 !important;

    -webkit-text-fill-color: #172033 !important;

    opacity: 1 !important;

    font-size: 27px !important;

    font-weight: 800 !important;

    padding-left: 0 !important;
}

.tracking-metric input:disabled,
.tracking-metric textarea:disabled,
.tracking-metric input[readonly],
.tracking-metric textarea[readonly] {

    color: #172033 !important;

    -webkit-text-fill-color: #172033 !important;

    opacity: 1 !important;
}


.tracking-metric input,
.tracking-metric textarea {
    pointer-events: none !important;
}

.tracking-metric-html {
    width: 100% !important;
    min-height: 145px !important;
    box-sizing: border-box !important;
    padding: 18px !important;
    border-radius: 15px !important;
    background: linear-gradient(145deg, #ffffff, #f8fbff) !important;
    border: 1px solid #dbe4ef !important;
    box-shadow: 0 12px 30px rgba(30, 64, 175, 0.06) !important;
}

.tracking-metric-html .metric-card-inner {
    width: 100% !important;
    min-height: 105px !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
}

.tracking-metric-html .metric-card-label {
    color: #718096 !important;
    font-size: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    margin-bottom: 10px !important;
}

.tracking-metric-html .metric-card-value {
    color: #172033 !important;
    font-size: 32px !important;
    line-height: 1.2 !important;
    font-weight: 850 !important;
    word-break: break-word !important;
}

/* ============================================================
   TRACKING + INTELLIGENCE PAGES
   ============================================================ */

.tracking-page,
.intelligence-page {

    width: 100% !important;
}


/* ============================================================
   FOOTER
   ============================================================ */

.footer {

    width: 100%;

    margin-top: 28px;

    padding-top: 18px;

    border-top:
        1px solid #e1e8f0;

    text-align: center;

    color: #8794a6 !important;

    font-size: 10px;
}


/* ============================================================
   HIDE GRADIO FOOTER
   ============================================================ */

footer {

    display: none !important;
}


/* ============================================================
   PAGE 4 SOURCE SECTIONS
   ============================================================ */

.analysis-source-section {
    width: 100% !important;
    margin-bottom: 22px !important;
    padding: 18px !important;
    box-sizing: border-box !important;
    border: 1px solid #dbe4ef !important;
    border-radius: 18px !important;
    background: rgba(255,255,255,0.72) !important;
    box-shadow: 0 10px 28px rgba(30,64,175,0.05) !important;
}

.source-badge {
    display: inline-flex !important;
    align-items: center !important;
    padding: 7px 12px !important;
    margin: 0 0 14px 2px !important;
    border-radius: 999px !important;
    font-size: 10px !important;
    font-weight: 800 !important;
    letter-spacing: 1px !important;
}

.image-badge {
    color: #2563eb !important;
    background: #eff6ff !important;
    border: 1px solid #bfdbfe !important;
}

.video-badge {
    color: #0f766e !important;
    background: #ecfeff !important;
    border: 1px solid #a5f3fc !important;
}

.intelligence-score-card {
    min-height: 145px !important;
}

.intelligence-page .tracking-metric input,
.intelligence-page .tracking-metric textarea {
    font-size: 32px !important;
    font-weight: 850 !important;
    color: #172033 !important;
    -webkit-text-fill-color: #172033 !important;
    opacity: 1 !important;
}

.intelligence-page .tracking-status input,
.intelligence-page .tracking-status textarea {
    font-weight: 750 !important;
}

.intelligence-page .detection-section-description {
    max-width: 900px !important;
}

/* ============================================================
   MOBILE
   ============================================================ */

@media (max-width: 850px) {

    .app-shell {

        flex-direction: column !important;

        margin-left: 0 !important;
    }

    .sidebar {

        flex: none !important;

        width: 100% !important;

        min-width: 100% !important;

        max-width: 100% !important;

        min-height: auto !important;
    }

    .main-content {

        width: 100% !important;

        padding: 25px !important;
    }

    .metrics-row,
    .lower-row {

        flex-wrap: wrap !important;
    }

    .hero-title {

        font-size: 31px !important;
    }

    .detection-row {

        flex-wrap: wrap !important;
    }
}

"""


# ============================================================
# ROADINTEL APPLICATION
# ============================================================

with gr.Blocks(
    title="RoadIntel | AI Road Intelligence",
    css=CUSTOM_CSS,
    theme=gr.themes.Base(
        primary_hue="blue",
        neutral_hue="slate"
    )
) as demo:


    # ========================================================
    # ANALYSIS STATE
    # ========================================================
    # These flags distinguish "not processed" from a valid result of
    # zero potholes. They also let Page 4 show only relevant sections.
    image_processed_state = gr.State(False)
    video_processed_state = gr.State(False)

    # ========================================================
    # APPLICATION SHELL
    # ========================================================

    with gr.Row(
        elem_classes="app-shell",
        variant="panel"
    ):


        # ====================================================
        # SIDEBAR
        # ====================================================

        with gr.Column(
            elem_classes="sidebar",
            scale=0,
            min_width=260
        ):

            gr.HTML(
                """
                <div class="sidebar-brand">

                    <div class="sidebar-icon">
                        🚧
                    </div>

                    <div>

                        <div class="sidebar-title">
                            ROADINTEL
                        </div>

                        <div class="sidebar-subtitle">
                            ROAD INTELLIGENCE<br>
                            PLATFORM
                        </div>

                    </div>

                </div>
                """
            )


            gr.HTML(
                """
                <div class="nav-label">
                    MAIN MENU
                </div>
                """
            )


            overview_nav = gr.Button(
                "🏠  Overview",
                elem_classes="nav-button",
                variant="secondary"
            )


            detection_nav = gr.Button(
                "📷  Detection",
                elem_classes="nav-button",
                variant="secondary"
            )


            tracking_nav = gr.Button(
                "🚗  Tracking & Severity",
                elem_classes="nav-button",
                variant="secondary"
            )


            intelligence_nav = gr.Button(
                "🛣️  Road Intelligence",
                elem_classes="nav-button",
                variant="secondary"
            )



            # =================================================
            # SIDEBAR STATUS
            # =================================================

            gr.HTML(
                """
                <div class="sidebar-status-box">

                    <div class="sidebar-status">

                        <span class="sidebar-status-dot"></span>

                        AI SYSTEM ONLINE

                    </div>

                    <div class="sidebar-version">
                        YOLO11s • Depth Anything V2
                    </div>

                </div>
                """
            )


        # ====================================================
        # PAGE 1 — OVERVIEW
        # ====================================================

        with gr.Column(
            elem_classes="main-content",
            scale=1,
            min_width=0,
            visible=True
        ) as overview_page:


            gr.HTML(
                """
                <div class="page-header">

                    <div>

                        <h1 class="page-heading">
                            Overview
                        </h1>

                        <div class="page-description">
                            AI-powered road condition monitoring
                        </div>

                    </div>

                    <div class="system-status">

                        <span class="status-dot"></span>

                        SYSTEM ONLINE

                    </div>

                </div>
                """
            )


            # =================================================
            # HERO
            # =================================================

            gr.HTML(
                """
                <div class="hero-card">

                    <div class="hero-glow"></div>

                    <div class="hero-content">

                        <div class="hero-badge">
                            COMPUTER VISION • YOLO11s
                        </div>

                        <h1 class="hero-title">
                            Smarter Roads with
                            <span>AI Vision</span>
                        </h1>

                        <p class="hero-description">

                            RoadIntel is an AI-powered road intelligence
                            platform designed to detect potholes from
                            road images and videos using computer vision.
                            Detected road damage can then be analyzed,
                            tracked and transformed into infrastructure
                            insights.

                        </p>

                        <div class="hero-features">

                            <div class="feature-pill">
                                📷 Image Detection
                            </div>

                            <div class="feature-pill">
                                🎥 Video Analysis
                            </div>

                            <div class="feature-pill">
                                🚗 Object Tracking
                            </div>

                            <div class="feature-pill">
                                📊 Road Analytics
                            </div>

                            <div class="feature-pill">
                                🛡️ Road Safety Analysis
                            </div>

                        </div>

                    </div>

                </div>
                """
            )


            # =================================================
            # MODEL PERFORMANCE
            # =================================================

            gr.HTML(
                """
                <div class="section-title">
                    📊 MODEL PERFORMANCE
                </div>
                """
            )


            with gr.Row(
                elem_classes="metrics-row"
            ):

                gr.HTML(
                    """
                    <div class="metric-card">

                        <div class="metric-label">
                            PRECISION
                        </div>

                        <div class="metric-value">
                            72.7%
                        </div>

                        <div class="metric-description">
                            Accuracy among predicted potholes
                        </div>

                    </div>
                    """
                )


                gr.HTML(
                    """
                    <div class="metric-card">

                        <div class="metric-label">
                            RECALL
                        </div>

                        <div class="metric-value">
                            60.8%
                        </div>

                        <div class="metric-description">
                            Potholes successfully detected
                        </div>

                    </div>
                    """
                )


                gr.HTML(
                    """
                    <div class="metric-card">

                        <div class="metric-label">
                            mAP@50
                        </div>

                        <div class="metric-value">
                            66.8%
                        </div>

                        <div class="metric-description">
                            Validation detection performance
                        </div>

                    </div>
                    """
                )


                gr.HTML(
                    """
                    <div class="metric-card">

                        <div class="metric-label">
                            MODEL
                        </div>

                        <div class="metric-value">
                            YOLO11s
                        </div>

                        <div class="metric-description">
                            Trained RoadIntel detector
                        </div>

                    </div>
                    """
                )


            # =================================================
            # LOWER SECTION
            # =================================================

            with gr.Row(
                elem_classes="lower-row"
            ):

                with gr.Column(
                    elem_classes="lower-column",
                    scale=1
                ):

                    gr.HTML(
                        """
                        <div class="section-title">
                            🧠 SYSTEM STATUS
                        </div>

                        <div class="info-card">

                            <div class="system-row">

                                <div class="system-name">
                                    🧠 YOLO11s Model
                                </div>

                                <div class="system-ready">
                                    ✓ READY
                                </div>

                            </div>

                            <div class="system-row">

                                <div class="system-name">
                                    📦 Trained Weights
                                </div>

                                <div class="system-ready">
                                    ✓ LOADED
                                </div>

                            </div>

                            <div class="system-row">

                                <div class="system-name">
                                    🔍 Image Detection
                                </div>

                                <div class="system-ready">
                                    ✓ AVAILABLE
                                </div>

                            </div>

                            <div class="system-row">

                                <div class="system-name">
                                    ⚡ Inference Engine
                                </div>

                                <div class="system-ready">
                                    ✓ ACTIVE
                                </div>

                            </div>

                        </div>
                        """
                    )


                with gr.Column(
                    elem_classes="lower-column",
                    scale=1
                ):

                    gr.HTML(
                        """
                        <div class="section-title">
                            ⚙️ HOW ROADINTEL WORKS
                        </div>

                        <div class="info-card">

                            <div class="workflow-step">

                                <div class="workflow-number">
                                    01
                                </div>

                                <div class="workflow-text">
                                    Upload a road image or video
                                </div>

                            </div>

                            <div class="workflow-step">

                                <div class="workflow-number">
                                    02
                                </div>

                                <div class="workflow-text">
                                    YOLO analyzes the road scene
                                </div>

                            </div>

                            <div class="workflow-step">

                                <div class="workflow-number">
                                    03
                                </div>

                                <div class="workflow-text">
                                    Potholes are detected and localized
                                </div>

                            </div>

                            <div class="workflow-step">

                                <div class="workflow-number">
                                    04
                                </div>

                                <div class="workflow-text">
                                    Results become road intelligence
                                </div>

                            </div>

                        </div>
                        """
                    )


            gr.HTML(
                """
                <div class="footer">
                    ROADINTEL
                    &nbsp;•&nbsp;
                    AI-POWERED ROAD INTELLIGENCE
                    &nbsp;•&nbsp;
                    YOLO11s
                </div>
                """
            )


        # ====================================================
        # PAGE 2 — DETECTION
        # ====================================================

        with gr.Column(visible=False, elem_classes="main-content", scale=1, min_width=0) as detection_page:
            gr.HTML("""
            <div class="page-header"><div><h1 class="page-heading">Detection</h1><div class="page-description">Detect potholes from road images and road videos</div></div><div class="system-status"><span class="status-dot"></span> YOLO11s READY</div></div>
            """)

            gr.HTML("""
            <div class="detection-section-title">📷 IMAGE DETECTION</div>
            <div class="detection-section-description">Upload a road image to detect and localize potholes. The same result is automatically passed to Tracking & Severity.</div>
            """)
            with gr.Row(elem_classes="detection-row"):
                with gr.Column(elem_classes="detection-card"):
                    image_input = gr.Image(type="filepath", label="Road Image")
                    image_confidence = gr.Slider(0.10, 0.90, value=0.25, step=0.05, label="Detection Confidence")
                    image_detect_button = gr.Button("🔍 Detect Potholes", variant="primary", elem_classes="detection-button")
                with gr.Column(elem_classes="detection-card"):
                    image_output = gr.Image(label="Detection Result", interactive=False)
                    image_result = gr.Textbox(label="Detection Summary", lines=4, interactive=False)

            gr.HTML("""
            <div class="detection-section-title video-title">🎥 VIDEO DETECTION</div>
            <div class="detection-section-description">Upload road footage to detect potholes frame-by-frame. The processed detection video appears here, while tracking and severity are automatically calculated on Page 3.</div>
            """)
            with gr.Row(elem_classes="detection-row"):
                with gr.Column(elem_classes="detection-card"):
                    video_input = gr.Video(label="Road Video")
                    video_confidence = gr.Slider(0.10, 0.90, value=0.25, step=0.05, label="Detection Confidence")
                    video_detect_button = gr.Button("🎥 Process Video", variant="primary", elem_classes="detection-button")
                with gr.Column(elem_classes="detection-card"):
                    video_output = gr.Video(label="Detection Result", interactive=False, format="mp4", autoplay=False)
                    video_result = gr.Textbox(label="Video Detection Summary", lines=5, interactive=False)

            gr.HTML("""
            <div class="footer">ROADINTEL &nbsp;•&nbsp; AI-POWERED ROAD INTELLIGENCE &nbsp;•&nbsp; YOLO11s</div>
            """)


        # ====================================================
        # PAGE 3 — TRACKING & SEVERITY
        # ====================================================

        with gr.Column(visible=False, elem_classes="main-content tracking-page", scale=1, min_width=0) as tracking_page:
            gr.HTML("""
            <div class="page-header"><div><h1 class="page-heading">Tracking & Severity</h1><div class="page-description">Review severity for images and track road damage across video frames</div></div><div class="system-status"><span class="status-dot"></span> ANALYSIS READY</div></div>
            """)

            gr.HTML("""
            <div class="detection-section-title">📷 IMAGE SEVERITY ANALYSIS</div>
            <div class="detection-section-description">The image processed on Detection is automatically analyzed here. No second upload is required.</div>
            """)
            with gr.Row(elem_classes="detection-row"):
                with gr.Column(elem_classes="detection-card"):
                    image_severity_image = gr.Image(label="Image Detection Result", interactive=False)
                with gr.Column(elem_classes="detection-card"):
                    image_severity_status = gr.Textbox(label="Image Analysis", value="Waiting for image detection.", lines=5, interactive=False)
                    with gr.Row(elem_classes="metrics-row"):
                        image_pothole_count = gr.Textbox(label="Potholes", value="—", interactive=False)
                        image_low_output = gr.Textbox(label="Low", value="—", interactive=False)
                        image_medium_output = gr.Textbox(label="Medium", value="—", interactive=False)
                        image_high_output = gr.Textbox(label="High", value="—", interactive=False)
                    with gr.Row(elem_classes="metrics-row"):
                        image_largest_width = gr.Textbox(label="Largest Width", value="—", interactive=False)
                        image_largest_height = gr.Textbox(label="Largest Height", value="—", interactive=False)
                        image_largest_area = gr.Textbox(label="Largest Area", value="—", interactive=False)
                        image_relative_depth = gr.Textbox(label="Relative Depth", value="—", interactive=False)
                    image_depth_status = gr.Textbox(label="Measurement Notes", value="Waiting for image detection.", lines=3, interactive=False)

            gr.HTML("""
            <div class="detection-section-title video-title">🎯 VIDEO TRACKING & SEVERITY ANALYSIS</div>
            <div class="detection-section-description">The video processed on Detection is automatically tracked here to identify unique potholes and classify their visual severity. No second upload is required.</div>
            """)
            with gr.Row(elem_classes="detection-row"):
                with gr.Column(elem_classes="detection-card"):
                    tracking_video_output = gr.Video(label="Tracking Result", interactive=False, format="mp4", autoplay=False)
                with gr.Column(elem_classes="detection-card"):
                    tracking_summary = gr.Textbox(label="Tracking & Severity Summary", lines=6, interactive=False)
                    with gr.Row(elem_classes="metrics-row"):
                        video_unique_tracks = gr.Textbox(label="Unique Potholes", value="—", interactive=False)
                        video_low_output = gr.Textbox(label="Low", value="—", interactive=False)
                        video_medium_output = gr.Textbox(label="Medium", value="—", interactive=False)
                        video_high_output = gr.Textbox(label="High", value="—", interactive=False)
                    video_tracking_status = gr.Textbox(label="Video Analysis", value="Waiting for video processing.", lines=3, interactive=False)

            gr.HTML("""
            <div class="footer">ROADINTEL &nbsp;•&nbsp; AI-POWERED ROAD INTELLIGENCE &nbsp;•&nbsp; YOLO11s</div>
            """)


        # ====================================================
        # PAGE 4 — ROAD INTELLIGENCE
        # ====================================================

        with gr.Column(visible=False, elem_classes="main-content intelligence-page", scale=1, min_width=0) as intelligence_page:
            gr.HTML("""
            <div class="page-header"><div><h1 class="page-heading">Road Intelligence</h1><div class="page-description">Turn camera-based pothole analysis into road-condition, safety and inspection insights</div></div><div class="system-status"><span class="status-dot"></span> ANALYTICS READY</div></div>
            """)

            gr.HTML("""
            <div class="detection-section-title">🛣️ ROAD CONDITION INTELLIGENCE</div>
            <div class="detection-section-description">The condition score uses severity-weighted visual evidence. Image geometry and model-relative depth evidence are shown separately; they do not represent calibrated physical measurements.</div>
            """)

            with gr.Column(visible=False, elem_classes="analysis-source-section") as image_insight_section:
                gr.HTML('<div class="source-badge image-badge">📷 IMAGE ANALYSIS</div>')
                with gr.Row(elem_classes="detection-row"):
                    with gr.Column(elem_classes="detection-card intelligence-score-card"):
                        image_road_score_output = gr.HTML(
                            value=metric_card_html("ROAD CONDITION SCORE", "—"),
                            elem_classes="tracking-metric-html"
                        )
                    with gr.Column(elem_classes="detection-card"):
                        image_road_condition_output = gr.Textbox(label="Overall Condition", value="Waiting for image analysis.", interactive=False, elem_classes="tracking-status")
                    with gr.Column(elem_classes="detection-card"):
                        image_road_assessment_output = gr.Textbox(label="Road Assessment", value="Waiting for image analysis.", lines=4, interactive=False)

                image_road_total_hidden = gr.Textbox(value="—", visible=False)
                image_road_low_hidden = gr.Textbox(value="—", visible=False)
                image_road_medium_hidden = gr.Textbox(value="—", visible=False)
                image_road_high_hidden = gr.Textbox(value="—", visible=False)
                image_road_score_raw = gr.Textbox(value="—", visible=False)

            with gr.Column(visible=False, elem_classes="analysis-source-section") as video_insight_section:
                gr.HTML('<div class="source-badge video-badge">🎥 VIDEO ANALYSIS</div>')
                with gr.Row(elem_classes="detection-row"):
                    with gr.Column(elem_classes="detection-card intelligence-score-card"):
                        video_road_score_output = gr.HTML(
                            value=metric_card_html("ROAD CONDITION SCORE", "—"),
                            elem_classes="tracking-metric-html"
                        )
                    with gr.Column(elem_classes="detection-card"):
                        video_road_condition_output = gr.Textbox(label="Overall Condition", value="Waiting for video analysis.", interactive=False, elem_classes="tracking-status")
                    with gr.Column(elem_classes="detection-card"):
                        video_road_assessment_output = gr.Textbox(label="Road Assessment", value="Waiting for video analysis.", lines=4, interactive=False)

                video_road_total_hidden = gr.Textbox(value="—", visible=False)
                video_road_low_hidden = gr.Textbox(value="—", visible=False)
                video_road_medium_hidden = gr.Textbox(value="—", visible=False)
                video_road_high_hidden = gr.Textbox(value="—", visible=False)
                video_road_score_raw = gr.Textbox(value="—", visible=False)

            gr.HTML("""
            <div class="detection-section-title video-title">🛡️ CAMERA-BASED ROAD SAFETY</div>
            <div class="detection-section-description">The safety screen uses camera-visible pothole size and severity. It is a maintenance-prioritization indicator, not a certified engineering safety measurement.</div>
            """)

            with gr.Row(elem_classes="detection-row"):
                with gr.Column(elem_classes="detection-card intelligence-score-card"):
                    safety_risk_output = gr.HTML(
                        value=metric_card_html("OVERALL SAFETY RISK", "—"),
                        elem_classes="tracking-metric-html"
                    )
                with gr.Column(elem_classes="detection-card intelligence-score-card"):
                    safety_level_output = gr.HTML(
                        value=metric_card_html("RISK LEVEL", "—"),
                        elem_classes="tracking-metric-html"
                    )
                with gr.Column(elem_classes="detection-card"):
                    safety_assessment_output = gr.Textbox(label="Safety Assessment", value="Waiting for image or video analysis.", lines=5, interactive=False)

            safety_risk_raw = gr.Textbox(value="—", visible=False)
            safety_level_raw = gr.Textbox(value="—", visible=False)

            gr.HTML("""
            <div class="info-card" style="margin-top:20px;">
                <div class="system-name" style="line-height:1.7;">
                    <strong>🛡️ CAMERA ROAD-SAFETY FRAMEWORK</strong><br><br>
                    <b>Potholes</b> → size / visual severity → <b>surface-damage risk</b><br>
                    <b>Road scene</b> → camera-visible evidence → <b>safety screening</b><br><br>
                    <b>Geometry</b> → pixel width / height / area → <b>visible damage extent</b><br>
                    <b>Depth model</b> → relative depth contrast → <b>depth evidence</b><br><br>
                    Physical depth in centimetres is not reported without calibration or ground-truth scale.
                </div>
            </div>
            """)

            gr.HTML("""
            <div class="detection-section-title">📄 INSPECTION REPORT</div>
            <div class="detection-section-description">Export the latest image and/or video intelligence together with the camera-based safety assessment.</div>
            """)
            with gr.Row(elem_classes="detection-row"):
                with gr.Column(elem_classes="detection-card"):
                    generate_report_button = gr.Button("📄 Generate Road Report", variant="primary", elem_classes="detection-button")
                with gr.Column(elem_classes="detection-card"):
                    report_preview = gr.Textbox(label="Report Preview", value="No report generated yet.", lines=12, interactive=False)
                    report_file = gr.File(label="Download Report", interactive=False)

            gr.HTML("""
            <div class="footer">ROADINTEL &nbsp;•&nbsp; AI-POWERED ROAD INTELLIGENCE &nbsp;•&nbsp; YOLO11s</div>
            """)
    # ========================================================
    # PAGE 2 IMAGE → PAGE 3 IMAGE ANALYSIS
    # ========================================================

    # Selecting a new source invalidates the previous analysis until
    # the corresponding Analyze/Process button is pressed again.
    image_input.change(
        fn=lambda _: False,
        inputs=[image_input],
        outputs=[image_processed_state]
    )

    video_input.change(
        fn=lambda _: False,
        inputs=[video_input],
        outputs=[video_processed_state]
    )

    image_event = image_detect_button.click(
        fn=run_image_detection,
        inputs=[image_input, image_confidence],
        outputs=[image_output, image_result, image_severity_image, image_pothole_count, image_low_output, image_medium_output, image_high_output, image_severity_status, image_largest_width, image_largest_height, image_largest_area, image_relative_depth, image_depth_status]
    )

    image_event = image_event.then(
        fn=mark_processed_if_valid,
        inputs=[image_pothole_count, image_low_output, image_medium_output, image_high_output],
        outputs=[image_processed_state]
    )

    image_event = image_event.then(
        fn=calculate_road_intelligence,
        inputs=[image_pothole_count, image_low_output, image_medium_output, image_high_output],
        outputs=[image_road_score_raw, image_road_total_hidden, image_road_low_hidden, image_road_medium_hidden, image_road_high_hidden, image_road_condition_output, image_road_assessment_output]
    )

    # ========================================================
    # PAGE 2 VIDEO → PAGE 3 VIDEO TRACKING + SEVERITY
    # ========================================================

    video_event = video_detect_button.click(
        fn=run_video_detection,
        inputs=[video_input, video_confidence],
        outputs=[video_output, video_result]
    )

    # After detection, automatically run tracking and severity on the same uploaded video.
    video_tracking_event = video_event.then(
        fn=run_tracking,
        inputs=[video_input, video_confidence],
        outputs=[tracking_video_output, tracking_summary, video_unique_tracks, video_low_output, video_medium_output, video_high_output, video_tracking_status]
    )

    video_tracking_event = video_tracking_event.then(
        fn=mark_processed_if_valid,
        inputs=[video_unique_tracks, video_low_output, video_medium_output, video_high_output],
        outputs=[video_processed_state]
    )

    video_tracking_event = video_tracking_event.then(
        fn=calculate_road_intelligence,
        inputs=[video_unique_tracks, video_low_output, video_medium_output, video_high_output],
        outputs=[video_road_score_raw, video_road_total_hidden, video_road_low_hidden, video_road_medium_hidden, video_road_high_hidden, video_road_condition_output, video_road_assessment_output]
    )

    # ========================================================
    # PAGE 4 SAFETY ANALYSIS
    # ========================================================
    # Safety is calculated only after the corresponding source has:
    # 1) completed Page 2/3 processing,
    # 2) passed the numeric-result validation, and
    # 3) updated its processed state.
    #
    # Keeping this callback serial prevents Page 4 from reading the old
    # processed-state value and displaying a false LOW/blank result.

    image_event = image_event.then(
        fn=calculate_combined_safety,
        inputs=[
            image_processed_state,
            image_pothole_count, image_low_output, image_medium_output, image_high_output,
            video_processed_state,
            video_unique_tracks, video_low_output, video_medium_output, video_high_output
        ],
        outputs=[safety_risk_raw, safety_level_raw, safety_assessment_output]
    )

    video_tracking_event = video_tracking_event.then(
        fn=calculate_combined_safety,
        inputs=[
            image_processed_state,
            image_pothole_count, image_low_output, image_medium_output, image_high_output,
            video_processed_state,
            video_unique_tracks, video_low_output, video_medium_output, video_high_output
        ],
        outputs=[safety_risk_raw, safety_level_raw, safety_assessment_output]
    )

    # Convert the validated raw values into dedicated HTML metric cards.
    # HTML is used only for display; raw text values remain hidden for reports.
    image_event = image_event.then(
        fn=lambda score: metric_card_html("ROAD CONDITION SCORE", score),
        inputs=[image_road_score_raw],
        outputs=[image_road_score_output]
    )

    video_tracking_event = video_tracking_event.then(
        fn=lambda score: metric_card_html("ROAD CONDITION SCORE", score),
        inputs=[video_road_score_raw],
        outputs=[video_road_score_output]
    )

    image_event = image_event.then(
        fn=lambda risk, level: (metric_card_html("OVERALL SAFETY RISK", risk), metric_card_html("RISK LEVEL", level)),
        inputs=[safety_risk_raw, safety_level_raw],
        outputs=[safety_risk_output, safety_level_output]
    )

    video_tracking_event = video_tracking_event.then(
        fn=lambda risk, level: (metric_card_html("OVERALL SAFETY RISK", risk), metric_card_html("RISK LEVEL", level)),
        inputs=[safety_risk_raw, safety_level_raw],
        outputs=[safety_risk_output, safety_level_output]
    )

    # ========================================================
    # PAGE 4 REPORT
    # ========================================================

    generate_report_button.click(
        fn=generate_report,
        inputs=[
            image_road_score_raw, image_pothole_count, image_low_output, image_medium_output, image_high_output,
            image_road_condition_output, image_road_assessment_output,
            image_largest_width, image_largest_height, image_largest_area, image_relative_depth,
            video_road_score_raw, video_unique_tracks, video_low_output, video_medium_output, video_high_output,
            video_road_condition_output, video_road_assessment_output,
            safety_risk_raw, safety_assessment_output
        ],
        outputs=[report_file, report_preview]
    )

    # ========================================================
    # NAVIGATION
    # ========================================================

    overview_nav.click(
        fn=show_overview,
        inputs=None,
        outputs=[overview_page, detection_page, tracking_page, intelligence_page]
    )

    detection_nav.click(
        fn=show_detection,
        inputs=None,
        outputs=[overview_page, detection_page, tracking_page, intelligence_page]
    )

    tracking_nav.click(
        fn=show_tracking,
        inputs=None,
        outputs=[overview_page, detection_page, tracking_page, intelligence_page]
    )

    intelligence_nav.click(
        fn=show_intelligence,
        inputs=[
            image_processed_state,
            image_pothole_count, image_low_output, image_medium_output, image_high_output,
            video_processed_state,
            video_unique_tracks, video_low_output, video_medium_output, video_high_output,
        ],
        outputs=[
            overview_page,
            detection_page,
            tracking_page,
            intelligence_page,
            image_insight_section,
            video_insight_section,

            image_road_score_output,
            image_road_score_raw,
            image_road_total_hidden,
            image_road_low_hidden,
            image_road_medium_hidden,
            image_road_high_hidden,
            image_road_condition_output,
            image_road_assessment_output,

            video_road_score_output,
            video_road_score_raw,
            video_road_total_hidden,
            video_road_low_hidden,
            video_road_medium_hidden,
            video_road_high_hidden,
            video_road_condition_output,
            video_road_assessment_output,

            safety_risk_output,
            safety_level_output,
            safety_assessment_output,
            safety_risk_raw,
            safety_level_raw,
        ]
    )

# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":

    import os

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000))
    )