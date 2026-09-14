# 🚧 RoadIntel — AI-Powered Road Condition Intelligence System

> An AI-powered computer vision platform for detecting, tracking, analyzing, and assessing potholes from road images and videos.

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![YOLO](https://img.shields.io/badge/YOLO11s-Ultralytics-orange)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)](https://opencv.org/)

---

## 🌐 Live Demo

**Try RoadIntel online:**

https://roadintel.streamlit.app/

---

# 📌 Overview

RoadIntel is an **AI-powered road condition intelligence system** designed to analyze road imagery and identify visible pothole damage.

Instead of limiting the system to simple pothole detection, RoadIntel processes the detected road damage and converts it into meaningful infrastructure insights through:

- Pothole detection
- Image and video analysis
- Pothole tracking
- Severity classification
- Image-space geometric measurements
- Monocular relative depth estimation
- Road condition scoring
- Safety risk assessment
- Maintenance prioritization
- Automated road intelligence reports

The system is designed as a **software-only computer vision solution**, requiring only road images or videos as input.

---

# 🎯 Problem Statement

Poor road conditions and potholes can negatively affect:

- Road safety
- Vehicle movement
- Driving comfort
- Infrastructure quality
- Maintenance planning

Traditional road inspection can be manual, time-consuming, and difficult to scale.

RoadIntel aims to provide an AI-assisted approach for analyzing road imagery and converting visible road damage into structured road condition information.

---

# 💡 Proposed Solution

RoadIntel uses a computer vision pipeline that takes a road image or video as input and processes it through multiple stages.

```text
                Road Image / Video
                        │
                        ▼
              ┌──────────────────┐
              │ Pothole Detection│
              │     YOLO11s      │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Tracking &       │
              │ Severity Analysis│
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Pothole Geometry │
              │  Measurements    │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Relative Depth   │
              │  Estimation      │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Road Intelligence│
              └────────┬─────────┘
                       │
              ┌────────┴─────────┐
              ▼                  ▼
       Safety Assessment    Maintenance
                            Priority
              │                  │
              └────────┬─────────┘
                       ▼
              ┌──────────────────┐
              │ RoadIntel Report │
              └──────────────────┘
