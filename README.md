# moilapp-plugin-opencv

This repository provides an **OpenCV-based plugin for MoilApp** that is designed to **experimentally validate camera calibration results**, particularly for **fisheye cameras**.  
The plugin enables visualization, feature detection, and geometric evaluation of calibrated cameras, including **rectilinear projection**, **checkerboard detection**, and **stereo 3D triangulation**.

---

## 📌 Overview

The main purpose of this plugin is to serve as an **experimental and validation tool** for OpenCV camera calibration results.  
It allows users to verify intrinsic parameters by observing rectified images, detect calibration patterns, and evaluate geometric consistency through stereo triangulation.

Key capabilities include:
- Rectilinear image generation from fisheye cameras
- Checkerboard (chessboard) detection on rectified images
- Stereo camera support
- 3D point triangulation inside the plugin environment

---

## ✨ Features

- **Fisheye to Rectilinear Projection**
  - Visualize rectilinear images generated from fisheye inputs
  - Validate calibration quality through visual inspection

- **Checkerboard Detection**
  - Detect chessboard patterns on rectified views
  - Support for configurable checkerboard size and square dimensions

- **Stereo Camera Support**
  - Work with two calibrated cameras simultaneously
  - Independent intrinsic parameter loading per camera

- **3D Triangulation**
  - Compute 3D points from corresponding checkerboard detections
  - Useful for depth estimation and geometric validation experiments

- **Experimental Validation Tool**
  - Designed for research, calibration evaluation, and prototyping
  - No modification of calibration parameters required

---

## 🧩 Plugin Role in MoilApp

This plugin is **not a calibration tool itself**.  
Instead, it focuses on **using and validating calibration results** produced by OpenCV.

Typical workflow:
1. Calibrate camera(s) using OpenCV (fisheye model)
2. Load intrinsic parameters into this plugin
3. Generate rectilinear views
4. Detect checkerboard features
5. Perform stereo triangulation for 3D evaluation

---

## ⚙️ Requirements

- Python 3.9+
- OpenCV (with fisheye module)
- NumPy
- MoilApp framework




