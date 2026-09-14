import streamlit as st
import sys
from pathlib import Path
import tempfile
import os
import subprocess
import cv2
import numpy as np

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import detect_potholes
from src.video_detector import detect_video
from src.tracking_detector import track_and_analyze
from src.pothole_measurement import analyze_pothole_measurements
from src.depth_estimator import estimate_depth, get_region_depth


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RoadIntel | AI Road Intelligence",
    page_icon="🚧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "page": "Overview",
    "image_path": None,
    "image_result": None,
    "image_processed": False,
    "image_metrics": None,
    "video_path": None,
    "video_result": None,
    "video_processed": False,
    "video_metrics": None,
    "tracking_result": None,
    "report_text": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

[data-testid="stAppViewContainer"] {
    background: #f5f8fc;
}

[data-testid="stSidebar"] {
    background: #10213f;
}

[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

.roadintel-brand {
    padding: 10px 4px 28px 4px;
}

.brand-row {
    display: flex;
    align-items: center;
    gap: 12px;
}

.brand-icon {
    width: 46px;
    height: 46px;
    border-radius: 13px;
    background: #ffffff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 25px;
}

.brand-title {
    font-size: 22px;
    font-weight: 850;
    letter-spacing: 1px;
    color: white !important;
}

.brand-subtitle {
    font-size: 9px;
    line-height: 1.45;
    letter-spacing: 1.5px;
    color: #a9b9d2 !important;
    margin-top: 3px;
}

.nav-label {
    font-size: 10px;
    letter-spacing: 1.5px;
    font-weight: 800;
    color: #91a4c3 !important;
    margin-bottom: 10px;
}

.sidebar-status {
    margin-top: 25px;
    padding: 16px;
    border: 1px solid rgba(255,255,255,.12);
    border-radius: 14px;
    background: rgba(255,255,255,.05);
}

.online {
    color: #74e5b0 !important;
    font-weight: 800;
    font-size: 11px;
    letter-spacing: .8px;
}

.version {
    color: #a9b9d2 !important;
    font-size: 10px;
    margin-top: 7px;
}

.page-heading {
    font-size: 38px;
    line-height: 1.1;
    font-weight: 850;
    color: #172033;
    margin: 0;
}

.page-description {
    color: #718096;
    margin-top: 7px;
    font-size: 14px;
}

.page-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 28px;
}

.system-status {
    border: 1px solid #dbe4ef;
    background: white;
    border-radius: 999px;
    padding: 9px 14px;
    color: #2563eb;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1px;
}

.hero-card {
    position: relative;
    overflow: hidden;
    border-radius: 22px;
    padding: 42px;
    margin-bottom: 34px;
    background: linear-gradient(135deg,#10213f,#1d4f91);
    box-shadow: 0 18px 45px rgba(30,64,175,.16);
}

.hero-title {
    color: white;
    font-size: 44px;
    line-height: 1.1;
    font-weight: 850;
    margin: 15px 0;
}

.hero-title span {
    color: #76b7ff;
}

.hero-description {
    max-width: 820px;
    color: #dce8f8;
    line-height: 1.7;
    font-size: 15px;
}

.hero-badge {
    display: inline-block;
    padding: 7px 12px;
    border-radius: 999px;
    background: rgba(255,255,255,.1);
    color: #b9d8ff;
    border: 1px solid rgba(255,255,255,.18);
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1px;
}

.feature-pill {
    display: inline-block;
    margin: 7px 7px 0 0;
    padding: 9px 13px;
    border-radius: 999px;
    background: rgba(255,255,255,.1);
    color: white;
    font-size: 11px;
}

.section-title {
    font-size: 12px;
    font-weight: 850;
    letter-spacing: 1.4px;
    color: #4a5568;
    margin: 22px 0 14px;
}

.metric-card {
    min-height: 145px;
    padding: 22px;
    border-radius: 16px;
    background: linear-gradient(145deg,#ffffff,#f8fbff);
    border: 1px solid #dbe4ef;
    box-shadow: 0 12px 30px rgba(30,64,175,.06);
}

.metric-label {
    color: #718096;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1px;
}

.metric-value {
    color: #172033;
    font-size: 31px;
    font-weight: 850;
    margin-top: 12px;
}

.metric-description {
    color: #8794a6;
    font-size: 10px;
    margin-top: 8px;
}

.info-card {
    padding: 22px;
    border-radius: 17px;
    background: white;
    border: 1px solid #dbe4ef;
    box-shadow: 0 10px 28px rgba(30,64,175,.05);
}

.system-row,
.workflow-step {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 13px 0;
    border-bottom: 1px solid #edf1f6;
}

.system-row:last-child,
.workflow-step:last-child {
    border-bottom: none;
}

.system-name {
    color: #344054;
    font-size: 13px;
}

.system-ready {
    color: #16a36c;
    font-size: 10px;
    font-weight: 800;
}

.workflow-number {
    width: 32px;
    height: 32px;
    border-radius: 9px;
    background: #eff6ff;
    color: #2563eb;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 10px;
    font-weight: 850;
}

.workflow-text {
    flex: 1;
    margin-left: 13px;
    color: #344054;
    font-size: 13px;
}

.section-card {
    padding: 23px;
    border-radius: 18px;
    background: white;
    border: 1px solid #dbe4ef;
    box-shadow: 0 10px 28px rgba(30,64,175,.05);
    margin-bottom: 22px;
}

.section-heading {
    font-size: 16px;
    font-weight: 850;
    color: #172033;
    margin-bottom: 5px;
}

.section-description {
    color: #718096;
    font-size: 12px;
    margin-bottom: 18px;
}

.big-score {
    padding: 25px;
    border-radius: 17px;
    background: linear-gradient(145deg,#ffffff,#f8fbff);
    border: 1px solid #dbe4ef;
    text-align: center;
}

.big-score-label {
    color: #718096;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1px;
}

.big-score-value {
    color: #172033;
    font-size: 43px;
    font-weight: 900;
    margin-top: 8px;
}

.footer {
    margin-top: 32px;
    padding: 18px 0;
    border-top: 1px solid #dbe4ef;
    text-align: center;
    color: #8794a6;
    font-size: 10px;
    letter-spacing: .5px;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def save_uploaded_file(uploaded_file, folder_name):
    folder = PROJECT_ROOT / "outputs" / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / uploaded_file.name
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(path)


def make_browser_compatible_video(video_path):
    """Convert an OpenCV-generated video to browser-friendly H.264 MP4."""
    if not video_path:
        return video_path

    source = Path(video_path)
    if not source.exists():
        return video_path

    if source.suffix.lower() != ".mp4":
        output = source.with_suffix(".mp4")
    else:
        output = source.with_name(source.stem + "_browser.mp4")

    try:
        if imageio_ffmpeg is None:
            return video_path

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        command = [
            ffmpeg_exe,
            "-y",
            "-i", str(source),
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",
            str(output),
        ]

        subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True,
        )

        if output.exists() and output.stat().st_size > 0:
            return str(output)

    except Exception:
        pass

    return video_path


def result_image(result):
    return result.plot()


def classify_severity(area_percent):
    ratio = area_percent / 100.0
    if ratio < 0.02:
        return "LOW"
    if ratio < 0.05:
        return "MEDIUM"
    return "HIGH"


def severity_counts_from_measurements(measurements):
    low = medium = high = 0
    for item in measurements:
        severity = classify_severity(item["area_percent"])
        if severity == "LOW":
            low += 1
        elif severity == "MEDIUM":
            medium += 1
        else:
            high += 1
    return low, medium, high


def road_score(total, low, medium, high):
    if total <= 0:
        return 100
    weighted_damage = (low * 1) + (medium * 3) + (high * 5)
    maximum_damage = total * 5
    ratio = weighted_damage / maximum_damage
    return round(max(0.0, min(100.0, 100.0 * (1.0 - ratio))))


def road_condition(score):
    if score < 20:
        return "CRITICAL"
    if score < 40:
        return "POOR"
    if score < 60:
        return "MODERATE"
    if score < 80:
        return "GOOD"
    return "EXCELLENT"


def safety_result(low, medium, high):
    points = low + medium * 5 + high * 12
    if high >= 1 or points >= 20:
        return "HIGH", "URGENT"
    if medium >= 2 or points >= 5:
        return "MEDIUM", "SOON"
    return "LOW", "ROUTINE"


def assessment_text(total, low, medium, high, score, risk, priority):
    if total == 0:
        return (
            "No potholes were detected in the analyzed input. "
            "The available visual evidence indicates a clean road surface."
        )

    condition = road_condition(score)
    return (
        f"{total} pothole(s) were identified. "
        f"Severity distribution: {low} low, {medium} medium, {high} high. "
        f"Overall road condition is {condition} with a score of {score}/100. "
        f"Camera-based safety risk is {risk}; maintenance priority is {priority}."
    )


def render_page_header(title, description, status):
    st.markdown(
        f"""
        <div class="page-header">
            <div>
                <div class="page-heading">{title}</div>
                <div class="page-description">{description}</div>
            </div>
            <div class="system-status">{status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer():
    st.markdown(
        '<div class="footer">ROADINTEL &nbsp;•&nbsp; AI-POWERED ROAD INTELLIGENCE &nbsp;•&nbsp; YOLO11s</div>',
        unsafe_allow_html=True,
    )


def run_image_pipeline(image_path, confidence):
    results = detect_potholes(image_path, confidence=confidence)
    result = results[0]
    image = cv2.imread(image_path)

    measurements = analyze_pothole_measurements(image, result)
    low, medium, high = severity_counts_from_measurements(measurements)
    total = len(measurements)

    largest = max(measurements, key=lambda x: x["area_px2"], default=None)

    relative_depth = None
    depth_note = "Relative depth unavailable."

    if largest is not None:
        try:
            depth_map = estimate_depth(image)
            mean_depth, _, _, depth_percent = get_region_depth(
                depth_map, largest["mask"]
            )
            if depth_percent is not None:
                relative_depth = depth_percent
                depth_note = (
                    "Depth Anything V2 provides model-relative depth. "
                    "It is not a calibrated physical depth measurement."
                )
        except Exception as exc:
            depth_note = f"Relative depth unavailable: {exc}"

    score = road_score(total, low, medium, high)
    risk, priority = safety_result(low, medium, high)

    metrics = {
        "total": total,
        "low": low,
        "medium": medium,
        "high": high,
        "score": score,
        "condition": road_condition(score),
        "risk": risk,
        "priority": priority,
        "largest": largest,
        "relative_depth": relative_depth,
        "depth_note": depth_note,
        "measurements": measurements,
        "assessment": assessment_text(
            total, low, medium, high, score, risk, priority
        ),
    }

    return result, result_image(result), metrics


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="roadintel-brand">
            <div class="brand-row">
                <div class="brand-icon">🚧</div>
                <div>
                    <div class="brand-title">ROADINTEL</div>
                    <div class="brand-subtitle">ROAD INTELLIGENCE<br>PLATFORM</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="nav-label">MAIN MENU</div>', unsafe_allow_html=True)

    pages = [
        "Overview",
        "Detection",
        "Tracking & Severity",
        "Road Intelligence",
    ]

    selected = st.radio(
        "Navigation",
        pages,
        index=pages.index(st.session_state.page),
        label_visibility="collapsed",
    )

    st.session_state.page = selected

    st.markdown(
        """
        <div class="sidebar-status">
            <div class="online">● AI SYSTEM ONLINE</div>
            <div class="version">YOLO11s • Depth Anything V2</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================

if st.session_state.page == "Overview":

    render_page_header(
        "Overview",
        "AI-powered road condition monitoring",
        "● SYSTEM ONLINE",
    )

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-badge">COMPUTER VISION • YOLO11s</div>
            <div class="hero-title">Smarter Roads with <span>AI Vision</span></div>
            <div class="hero-description">
                RoadIntel is an AI-powered road intelligence platform designed
                to detect potholes from road images and videos using computer
                vision. Detected road damage can then be analyzed, tracked
                and transformed into infrastructure insights.
            </div>
            <div style="margin-top:20px">
                <span class="feature-pill">📷 Image Detection</span>
                <span class="feature-pill">🎥 Video Analysis</span>
                <span class="feature-pill">🚗 Object Tracking</span>
                <span class="feature-pill">📊 Road Analytics</span>
                <span class="feature-pill">🛡️ Road Safety Analysis</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">📊 MODEL PERFORMANCE</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    cards = [
        ("PRECISION", "72.7%", "Accuracy among predicted potholes"),
        ("RECALL", "60.8%", "Potholes successfully detected"),
        ("mAP@50", "66.8%", "Validation detection performance"),
        ("MODEL", "YOLO11s", "Trained RoadIntel detector"),
    ]

    for col, (label, value, desc) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-description">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    left, right = st.columns(2)

    with left:
        st.markdown(
            '<div class="section-title">🧠 SYSTEM STATUS</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="info-card">
                <div class="system-row"><div class="system-name">🧠 YOLO11s Model</div><div class="system-ready">✓ READY</div></div>
                <div class="system-row"><div class="system-name">📦 Trained Weights</div><div class="system-ready">✓ LOADED</div></div>
                <div class="system-row"><div class="system-name">🔍 Image Detection</div><div class="system-ready">✓ AVAILABLE</div></div>
                <div class="system-row"><div class="system-name">⚡ Inference Engine</div><div class="system-ready">✓ ACTIVE</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            '<div class="section-title">⚙️ HOW ROADINTEL WORKS</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="info-card">
                <div class="workflow-step"><div class="workflow-number">01</div><div class="workflow-text">Upload a road image or video</div></div>
                <div class="workflow-step"><div class="workflow-number">02</div><div class="workflow-text">YOLO analyzes the road scene</div></div>
                <div class="workflow-step"><div class="workflow-number">03</div><div class="workflow-text">Potholes are detected and localized</div></div>
                <div class="workflow-step"><div class="workflow-number">04</div><div class="workflow-text">Results become road intelligence</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_footer()


# ============================================================
# PAGE 2 — DETECTION
# ============================================================

elif st.session_state.page == "Detection":

    render_page_header(
        "Detection",
        "Detect potholes from road images and road videos",
        "● YOLO11s READY",
    )

    st.markdown(
        """
        <div class="section-card">
            <div class="section-heading">📷 IMAGE DETECTION</div>
            <div class="section-description">
                Upload a road image to detect and localize potholes.
                The result is automatically retained for Tracking & Severity.
            </div>
        """,
        unsafe_allow_html=True,
    )

    image_col, output_col = st.columns(2)

    with image_col:
        uploaded_image = st.file_uploader(
            "Road Image",
            type=["jpg", "jpeg", "png", "webp"],
            key="road_image_upload",
        )

        image_confidence = st.slider(
            "Detection Confidence",
            0.10,
            0.90,
            0.25,
            0.05,
            key="image_confidence",
        )

        image_button = st.button(
            "🔍 Detect Potholes",
            type="primary",
            use_container_width=True,
            key="image_detect",
        )

    if uploaded_image is not None and image_button:
        with st.spinner("Analyzing road image..."):
            path = save_uploaded_file(uploaded_image, "images")
            try:
                result, annotated, metrics = run_image_pipeline(
                    path, image_confidence
                )
                st.session_state.image_path = path
                st.session_state.image_result = result
                st.session_state.image_processed = True
                st.session_state.image_metrics = metrics
            except Exception as exc:
                st.error(f"Image analysis failed: {exc}")

    with output_col:
        if st.session_state.image_processed and st.session_state.image_result is not None:
            st.image(
                result_image(st.session_state.image_result),
                caption="Detection Result",
                use_container_width=True,
            )
            m = st.session_state.image_metrics
            st.info(
                f"Detected {m['total']} pothole(s) • "
                f"Low: {m['low']} • Medium: {m['medium']} • High: {m['high']}"
            )

            output_path = (
                PROJECT_ROOT / "outputs" / "images" / "roadintel_detection.png"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(output_path),
                result_image(st.session_state.image_result),
            )

            with open(output_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Detection Result",
                    f,
                    file_name="roadintel_detection.png",
                    mime="image/png",
                    use_container_width=True,
                )
        else:
            st.info("Waiting for image detection.")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="section-card">
            <div class="section-heading">🎥 VIDEO DETECTION</div>
            <div class="section-description">
                Upload road footage to detect potholes frame-by-frame.
                The processed video is retained for Tracking & Severity.
            </div>
        """,
        unsafe_allow_html=True,
    )

    video_col, video_output_col = st.columns(2)

    with video_col:
        uploaded_video = st.file_uploader(
            "Road Video",
            type=["mp4", "avi", "mov", "mkv"],
            key="road_video_upload",
        )

        video_confidence = st.slider(
            "Detection Confidence",
            0.10,
            0.90,
            0.25,
            0.05,
            key="video_confidence",
        )

        video_button = st.button(
            "🎥 Process Video",
            type="primary",
            use_container_width=True,
            key="video_detect",
        )

    if uploaded_video is not None and video_button:
        with st.spinner("Processing road video..."):
            path = save_uploaded_file(uploaded_video, "video")
            try:
                (
                    output_video,
                    total_detections,
                    total_frames,
                    frames_with_detections,
                ) = detect_video(path, confidence=video_confidence)

                browser_video = make_browser_compatible_video(output_video)

                st.session_state.video_path = path
                st.session_state.video_result = browser_video
                st.session_state.video_processed = True
                st.session_state.video_metrics = {
                    "total_detections": total_detections,
                    "total_frames": total_frames,
                    "frames_with_detections": frames_with_detections,
                    "confidence": video_confidence,
                }
            except Exception as exc:
                st.error(f"Video processing failed: {exc}")

    with video_output_col:
        if st.session_state.video_processed:
            st.video(st.session_state.video_result)

            v = st.session_state.video_metrics
            st.info(
                f"Frames: {v['total_frames']} • "
                f"Frames with potholes: {v['frames_with_detections']} • "
                f"Total detections: {v['total_detections']}"
            )
        else:
            st.info("Waiting for video processing.")

    st.markdown("</div>", unsafe_allow_html=True)

    render_footer()


# ============================================================
# PAGE 3 — TRACKING & SEVERITY
# ============================================================

elif st.session_state.page == "Tracking & Severity":

    render_page_header(
        "Tracking & Severity",
        "Review severity for images and track road damage across video frames",
        "● ANALYSIS READY",
    )

    st.markdown(
        """
        <div class="section-card">
            <div class="section-heading">📷 IMAGE SEVERITY ANALYSIS</div>
            <div class="section-description">
                The image processed on Detection is automatically analyzed here.
                No second upload is required.
            </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.image_processed:
        st.warning(
            "Please process an image on the Detection page first."
        )
    else:
        m = st.session_state.image_metrics

        image_col, metrics_col = st.columns([1.1, 1])

        with image_col:
            st.image(
                result_image(st.session_state.image_result),
                caption="Image Detection Result",
                use_container_width=True,
            )

        with metrics_col:
            st.success(
                f"Image analysis complete — {m['total']} pothole(s) detected."
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Potholes", m["total"])
            c2.metric("Low", m["low"])
            c3.metric("Medium", m["medium"])
            c4.metric("High", m["high"])

            largest = m["largest"]

            if largest:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Largest Width", f"{largest['width_px']:.0f} px")
                c2.metric("Largest Height", f"{largest['height_px']:.0f} px")
                c3.metric("Largest Area", f"{largest['area_px2']:.0f} px²")
                depth_text = (
                    f"{m['relative_depth']:.1f}%"
                    if m["relative_depth"] is not None
                    else "—"
                )
                c4.metric("Relative Depth", depth_text)

                st.caption(m["depth_note"])
            else:
                st.info("No pothole geometry is available.")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="section-card">
            <div class="section-heading">🎯 VIDEO TRACKING & SEVERITY ANALYSIS</div>
            <div class="section-description">
                The video processed on Detection is automatically tracked here
                to identify unique potholes and classify their visual severity.
            </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.video_processed:
        st.warning(
            "Please process a video on the Detection page first."
        )
    else:
        if st.button(
            "🚗 Run Tracking & Severity Analysis",
            type="primary",
            use_container_width=True,
            key="run_tracking",
        ):
            with st.spinner("Tracking potholes across video frames..."):
                try:
                    values = track_and_analyze(
                        st.session_state.video_path,
                        st.session_state.video_metrics["confidence"],
                    )
                    values = list(values)
                    values[0] = make_browser_compatible_video(values[0])
                    st.session_state.tracking_result = tuple(values)
                except Exception as exc:
                    st.error(f"Tracking failed: {exc}")

        if st.session_state.tracking_result is not None:
            (
                tracking_video,
                total_detections,
                unique_tracks,
                low_count,
                medium_count,
                high_count,
                total_frames,
                frames_with_detections,
            ) = st.session_state.tracking_result

            st.video(tracking_video)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Unique Potholes", unique_tracks)
            c2.metric("Low", low_count)
            c3.metric("Medium", medium_count)
            c4.metric("High", high_count)

            st.info(
                f"Tracking complete • {total_detections} detections across "
                f"{total_frames} frames • {unique_tracks} unique pothole tracks."
            )
        else:
            st.info(
                "Click the button above to run video tracking and severity."
            )

    st.markdown("</div>", unsafe_allow_html=True)

    render_footer()


# ============================================================
# ============================================================
# PAGE 4 — ROAD INTELLIGENCE
# ============================================================

elif st.session_state.page == "Road Intelligence":

    render_page_header(
        "Road Intelligence",
        "Turn pothole detection, severity and tracking into road-condition and safety insights",
        "● ANALYTICS READY",
    )

    # ------------------------------------------------------------
    # COLLECT AVAILABLE ANALYSIS
    # ------------------------------------------------------------

    image_data = None
    video_data = None

    if st.session_state.image_processed and st.session_state.image_metrics:
        im = st.session_state.image_metrics
        image_data = {
            "source": "IMAGE",
            "total": im["total"],
            "low": im["low"],
            "medium": im["medium"],
            "high": im["high"],
            "score": im["score"],
            "condition": im["condition"],
            "risk": im["risk"],
            "priority": im["priority"],
            "assessment": im["assessment"],
        }

    if (
        st.session_state.video_processed
        and st.session_state.tracking_result is not None
    ):
        (
            tracking_video,
            total_detections,
            unique_tracks,
            low_count,
            medium_count,
            high_count,
            total_frames,
            frames_with_detections,
        ) = st.session_state.tracking_result

        video_score = road_score(
            unique_tracks, low_count, medium_count, high_count
        )
        video_condition = road_condition(video_score)
        video_risk, video_priority = safety_result(
            low_count, medium_count, high_count
        )
        video_assessment = assessment_text(
            unique_tracks,
            low_count,
            medium_count,
            high_count,
            video_score,
            video_risk,
            video_priority,
        )

        video_data = {
            "source": "VIDEO",
            "tracking_video": tracking_video,
            "total_detections": total_detections,
            "unique_tracks": unique_tracks,
            "low": low_count,
            "medium": medium_count,
            "high": high_count,
            "total_frames": total_frames,
            "frames_with_detections": frames_with_detections,
            "score": video_score,
            "condition": video_condition,
            "risk": video_risk,
            "priority": video_priority,
            "assessment": video_assessment,
        }

    has_image = image_data is not None
    has_video = video_data is not None

    # ------------------------------------------------------------
    # ROAD CONDITION INTELLIGENCE
    # ------------------------------------------------------------

    st.markdown(
        """
        <div class="section-card">
            <div class="section-heading">🛣️ ROAD CONDITION INTELLIGENCE</div>
            <div class="section-description">
                Road condition is calculated from severity-weighted pothole evidence.
                Scores are software-based visual indicators, not certified engineering ratings.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not has_image and not has_video:
        st.info(
            "Process an image or complete video tracking on the Detection and "
            "Tracking & Severity pages first."
        )
    else:

        # --------------------------------------------------------
        # IMAGE ROAD INTELLIGENCE
        # --------------------------------------------------------

        if has_image:
            st.markdown(
                """
                <div class="section-card">
                    <div class="section-heading">📷 IMAGE ROAD INTELLIGENCE</div>
                    <div class="section-description">
                        Road intelligence calculated from the processed road image.
                    </div>
                """,
                unsafe_allow_html=True,
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.markdown(
                    f"""
                    <div class="big-score">
                        <div class="big-score-label">ROAD CONDITION SCORE</div>
                        <div class="big-score-value">{image_data["score"]} / 100</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            c2.metric("Overall Condition", image_data["condition"])
            c3.metric("Potholes", image_data["total"])
            c4.metric("High Severity", image_data["high"])

            st.markdown(
                f"""
                <div class="info-card" style="margin-top:18px">
                    <div class="section-heading">📊 IMAGE SEVERITY DISTRIBUTION</div>
                    <div class="system-row">
                        <div class="system-name">Low Severity</div>
                        <div class="system-ready">{image_data["low"]}</div>
                    </div>
                    <div class="system-row">
                        <div class="system-name">Medium Severity</div>
                        <div class="system-ready">{image_data["medium"]}</div>
                    </div>
                    <div class="system-row">
                        <div class="system-name">High Severity</div>
                        <div class="system-ready">{image_data["high"]}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("</div>", unsafe_allow_html=True)

        # --------------------------------------------------------
        # VIDEO ROAD INTELLIGENCE
        # --------------------------------------------------------

        if has_video:
            st.markdown(
                """
                <div class="section-card">
                    <div class="section-heading">🎥 VIDEO ROAD INTELLIGENCE</div>
                    <div class="section-description">
                        Road intelligence calculated from unique pothole tracks and video severity analysis.
                    </div>
                """,
                unsafe_allow_html=True,
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.markdown(
                    f"""
                    <div class="big-score">
                        <div class="big-score-label">ROAD CONDITION SCORE</div>
                        <div class="big-score-value">{video_data["score"]} / 100</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            c2.metric("Overall Condition", video_data["condition"])
            c3.metric("Unique Potholes", video_data["unique_tracks"])
            c4.metric("High Severity", video_data["high"])

            st.markdown(
                f"""
                <div class="info-card" style="margin-top:18px">
                    <div class="section-heading">📊 VIDEO SEVERITY DISTRIBUTION</div>
                    <div class="system-row">
                        <div class="system-name">Low Severity</div>
                        <div class="system-ready">{video_data["low"]}</div>
                    </div>
                    <div class="system-row">
                        <div class="system-name">Medium Severity</div>
                        <div class="system-ready">{video_data["medium"]}</div>
                    </div>
                    <div class="system-row">
                        <div class="system-name">High Severity</div>
                        <div class="system-ready">{video_data["high"]}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("</div>", unsafe_allow_html=True)

        # --------------------------------------------------------
        # BOTH SOURCES
        # --------------------------------------------------------

        if has_image and has_video:
            st.markdown(
                """
                <div class="section-card">
                    <div class="section-heading">🔗 IMAGE + VIDEO INTELLIGENCE</div>
                    <div class="section-description">
                        Both processed sources are available. Their results are shown separately
                        so image evidence and video tracking evidence are not incorrectly counted
                        as the same physical potholes.
                    </div>
                """,
                unsafe_allow_html=True,
            )

            c1, c2 = st.columns(2)

            with c1:
                st.markdown(
                    f"""
                    <div class="info-card">
                        <div class="section-heading">📷 IMAGE EVIDENCE</div>
                        <div class="system-row">
                            <div class="system-name">Road Score</div>
                            <div class="system-ready">{image_data["score"]}/100</div>
                        </div>
                        <div class="system-row">
                            <div class="system-name">Condition</div>
                            <div class="system-ready">{image_data["condition"]}</div>
                        </div>
                        <div class="system-row">
                            <div class="system-name">Potholes</div>
                            <div class="system-ready">{image_data["total"]}</div>
                        </div>
                        <div class="system-row">
                            <div class="system-name">Safety Risk</div>
                            <div class="system-ready">{image_data["risk"]}</div>
                        </div>
                        <div class="system-row">
                            <div class="system-name">Maintenance Priority</div>
                            <div class="system-ready">{image_data["priority"]}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c2:
                st.markdown(
                    f"""
                    <div class="info-card">
                        <div class="section-heading">🎥 VIDEO EVIDENCE</div>
                        <div class="system-row">
                            <div class="system-name">Road Score</div>
                            <div class="system-ready">{video_data["score"]}/100</div>
                        </div>
                        <div class="system-row">
                            <div class="system-name">Condition</div>
                            <div class="system-ready">{video_data["condition"]}</div>
                        </div>
                        <div class="system-row">
                            <div class="system-name">Unique Potholes</div>
                            <div class="system-ready">{video_data["unique_tracks"]}</div>
                        </div>
                        <div class="system-row">
                            <div class="system-name">Safety Risk</div>
                            <div class="system-ready">{video_data["risk"]}</div>
                        </div>
                        <div class="system-row">
                            <div class="system-name">Maintenance Priority</div>
                            <div class="system-ready">{video_data["priority"]}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.caption(
                "Combined view preserves the independent image and video evidence; "
                "it does not claim that detections from both sources are unique physical potholes."
            )

            st.markdown("</div>", unsafe_allow_html=True)

        # --------------------------------------------------------
        # CAMERA-BASED SAFETY
        # --------------------------------------------------------

        st.markdown(
            """
            <div class="section-card">
                <div class="section-heading">🛡️ CAMERA-BASED ROAD SAFETY</div>
                <div class="section-description">
                    Safety risk is derived from visible pothole severity and extent.
                    It is intended for maintenance prioritization and is not a certified
                    structural or traffic-safety measurement.
                </div>
            """,
            unsafe_allow_html=True,
        )

        if has_image and has_video:
            c1, c2, c3 = st.columns(3)
            c1.metric("Image Safety Risk", image_data["risk"])
            c2.metric("Video Safety Risk", video_data["risk"])
            combined_priority = (
                "URGENT"
                if "URGENT" in (image_data["priority"], video_data["priority"])
                else "SOON"
                if "SOON" in (image_data["priority"], video_data["priority"])
                else "ROUTINE"
            )
            c3.metric("Overall Priority", combined_priority)

            st.markdown(
                f"""
                <div class="info-card" style="margin-top:18px">
                    <div class="system-row">
                        <div class="system-name">Image safety assessment</div>
                        <div class="system-ready">{image_data["risk"]}</div>
                    </div>
                    <div class="system-row">
                        <div class="system-name">Video safety assessment</div>
                        <div class="system-ready">{video_data["risk"]}</div>
                    </div>
                    <div class="system-row">
                        <div class="system-name">Overall maintenance priority</div>
                        <div class="system-ready">{combined_priority}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif has_image:
            c1, c2, c3 = st.columns(3)
            c1.metric("Safety Risk", image_data["risk"])
            c2.metric("Risk Level", image_data["risk"])
            c3.metric("Maintenance Priority", image_data["priority"])

            st.markdown(
                f"""
                <div class="info-card" style="margin-top:18px">
                    <div class="system-row">
                        <div class="system-name">Safety assessment</div>
                        <div class="system-ready">{image_data["risk"]}</div>
                    </div>
                    <div class="system-row">
                        <div class="system-name">Recommended priority</div>
                        <div class="system-ready">{image_data["priority"]}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Safety Risk", video_data["risk"])
            c2.metric("Risk Level", video_data["risk"])
            c3.metric("Maintenance Priority", video_data["priority"])

            st.markdown(
                f"""
                <div class="info-card" style="margin-top:18px">
                    <div class="system-row">
                        <div class="system-name">Safety assessment</div>
                        <div class="system-ready">{video_data["risk"]}</div>
                    </div>
                    <div class="system-row">
                        <div class="system-name">Recommended priority</div>
                        <div class="system-ready">{video_data["priority"]}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # --------------------------------------------------------
        # VIDEO ANALYSIS SUMMARY
        # --------------------------------------------------------

        if has_video:
            st.markdown(
                """
                <div class="section-card">
                    <div class="section-heading">🎥 VIDEO ANALYSIS SUMMARY</div>
                    <div class="section-description">
                        Tracking and severity information retained from Page 3.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Unique Potholes", video_data["unique_tracks"])
            c2.metric("Total Detections", video_data["total_detections"])
            c3.metric("Total Frames", video_data["total_frames"])
            c4.metric(
                "Frames With Potholes",
                video_data["frames_with_detections"],
            )

        # --------------------------------------------------------
        # REPORT GENERATION
        # --------------------------------------------------------

        st.markdown(
            """
            <div class="section-card">
                <div class="section-heading">📄 ROADINTEL REPORT</div>
                <div class="section-description">
                    Generate, preview, save and download the current road intelligence report.
                </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "📄 Generate RoadIntel Report",
            type="primary",
            use_container_width=True,
            key="generate_report",
        ):
            report_parts = [
                "ROADINTEL — AI-POWERED ROAD INTELLIGENCE",
                "=" * 60,
                "",
            ]

            if has_image:
                report_parts.extend([
                    "IMAGE ANALYSIS",
                    "--------------",
                    f"Road Condition Score : {image_data['score']}/100",
                    f"Overall Condition    : {image_data['condition']}",
                    "",
                    "POTHOLE ANALYSIS",
                    "----------------",
                    f"Potholes             : {image_data['total']}",
                    f"Low Severity         : {image_data['low']}",
                    f"Medium Severity      : {image_data['medium']}",
                    f"High Severity        : {image_data['high']}",
                    "",
                    "CAMERA-BASED SAFETY",
                    "-------------------",
                    f"Safety Risk          : {image_data['risk']}",
                    f"Risk Level           : {image_data['risk']}",
                    f"Maintenance Priority : {image_data['priority']}",
                    "",
                    "ROAD ASSESSMENT",
                    "---------------",
                    image_data["assessment"],
                    "",
                ])

            if has_video:
                report_parts.extend([
                    "VIDEO ANALYSIS",
                    "--------------",
                    f"Road Condition Score : {video_data['score']}/100",
                    f"Overall Condition    : {video_data['condition']}",
                    "",
                    "POTHOLE ANALYSIS",
                    "----------------",
                    f"Unique Potholes      : {video_data['unique_tracks']}",
                    f"Total Detections     : {video_data['total_detections']}",
                    f"Low Severity         : {video_data['low']}",
                    f"Medium Severity      : {video_data['medium']}",
                    f"High Severity        : {video_data['high']}",
                    f"Total Frames         : {video_data['total_frames']}",
                    f"Frames With Potholes : {video_data['frames_with_detections']}",
                    "",
                    "CAMERA-BASED SAFETY",
                    "-------------------",
                    f"Safety Risk          : {video_data['risk']}",
                    f"Risk Level           : {video_data['risk']}",
                    f"Maintenance Priority : {video_data['priority']}",
                    "",
                    "ROAD ASSESSMENT",
                    "---------------",
                    video_data["assessment"],
                    "",
                ])

            if has_image and has_video:
                combined_priority = (
                    "URGENT"
                    if "URGENT" in (image_data["priority"], video_data["priority"])
                    else "SOON"
                    if "SOON" in (image_data["priority"], video_data["priority"])
                    else "ROUTINE"
                )
                report_parts.extend([
                    "IMAGE + VIDEO EVIDENCE",
                    "----------------------",
                    "Both sources were analyzed independently.",
                    "Their detections are not assumed to represent unique physical potholes across sources.",
                    f"Image Road Score     : {image_data['score']}/100",
                    f"Video Road Score     : {video_data['score']}/100",
                    f"Image Safety Risk    : {image_data['risk']}",
                    f"Video Safety Risk    : {video_data['risk']}",
                    f"Overall Priority     : {combined_priority}",
                    "",
                ])

            report_parts.extend([
                "MEASUREMENT NOTE",
                "----------------",
                "Pothole geometry from RGB imagery is represented in image-space pixels.",
                "Monocular depth is model-relative and is not a calibrated physical depth",
                "measurement. Physical centimetre measurements require camera calibration",
                "and/or ground-truth scale.",
                "",
                "MODEL",
                "-----",
                "YOLO11s",
                "Depth Anything V2",
                "",
                "=" * 60,
                "Generated by RoadIntel",
            ])

            report_text = "\n".join(report_parts)

            report_dir = PROJECT_ROOT / "outputs" / "reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / "roadintel_report.txt"
            report_path.write_text(report_text, encoding="utf-8")
            st.session_state.report_text = report_text

            st.success(
                f"Report saved successfully to: "
                f"{report_path.relative_to(PROJECT_ROOT)}"
            )

        if st.session_state.report_text:
            st.text_area(
                "Report Preview",
                st.session_state.report_text,
                height=400,
                disabled=True,
            )

            report_path = (
                PROJECT_ROOT / "outputs" / "reports" / "roadintel_report.txt"
            )
            if report_path.exists():
                with open(report_path, "rb") as report_file:
                    st.download_button(
                        "⬇️ Download RoadIntel Report",
                        report_file,
                        file_name="roadintel_report.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )
        else:
            st.info(
                "Click 'Generate RoadIntel Report' to create and save the report."
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # --------------------------------------------------------
        # FRAMEWORK
        # --------------------------------------------------------

        st.markdown(
            """
            <div class="section-card">
                <div class="section-heading">🧠 ROAD INTELLIGENCE FRAMEWORK</div>
                <div class="section-description">
                    Software-only intelligence derived from ordinary RGB camera inputs.
                </div>
                <div class="workflow-step">
                    <div class="workflow-number">01</div>
                    <div class="workflow-text"><b>Surface Condition</b> — pothole count, visual size and severity</div>
                </div>
                <div class="workflow-step">
                    <div class="workflow-number">02</div>
                    <div class="workflow-text"><b>Structural Risk</b> — large, repeated or extensive visible damage</div>
                </div>
                <div class="workflow-step">
                    <div class="workflow-number">03</div>
                    <div class="workflow-text"><b>Safety & Priority</b> — severity-based risk and maintenance priority</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_footer()
