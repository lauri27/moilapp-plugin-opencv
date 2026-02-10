# OpenCV-only fisheye rectilinear views (5 directions) + chessboard detection + reorder combobox
# RESULT button overlays detected points from each rectified direction back onto ORIGINAL fisheye image.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Any

import json
import math

import cv2
import numpy as np
import os
from datetime import datetime
import pandas as pd
import json
from pathlib import Path


# PyQt6 / PyQt5 compatibility
try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QImage, QPixmap
    from PyQt6.QtWidgets import QWidget, QFileDialog, QMessageBox
    QT6 = True
except Exception:  # pragma: no cover
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QImage, QPixmap
    from PyQt5.QtWidgets import QWidget, QFileDialog, QMessageBox
    QT6 = False

from src.plugin_interface import PluginInterface
from .ui_main import Ui_Form
from .opencv_fisheye_triangulation import FisheyeCamera, triangulate_midpoint_from_pixels, project_world_points_to_fisheye_pixels
from .overlay_compare_point_opencv import show_3d_point_2cam_ori_visualization,compare_reprojection_with_original_opencv



DIRECTIONS = ("west", "east", "north", "south", "center")
SIDES = ("left", "right")


def _deg2rad(d: float) -> float:
    return float(d) * math.pi / 180.0


def _rot_x(rx: float) -> np.ndarray:
    c, s = math.cos(rx), math.sin(rx)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]], dtype=np.float64)


def _rot_y(ry: float) -> np.ndarray:
    c, s = math.cos(ry), math.sin(ry)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], dtype=np.float64)


def _ensure_gray(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim == 2:
        return img_bgr
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def _qimage_from_bgr(img_bgr: np.ndarray) -> QImage:
    if img_bgr is None:
        return QImage()
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    bytes_per_line = 3 * w
    fmt = QImage.Format.Format_RGB888 if QT6 else QImage.Format_RGB888
    qimg = QImage(rgb.data, w, h, bytes_per_line, fmt)
    return qimg.copy()


def _set_label_pixmap(label, img_bgr: Optional[np.ndarray]) -> None:
    if label is None:
        return
    if img_bgr is None:
        label.clear()
        return
    qimg = _qimage_from_bgr(img_bgr)
    pix = QPixmap.fromImage(qimg)
    if QT6:
        pix = pix.scaled(label.width(), label.height(),
                         Qt.AspectRatioMode.KeepAspectRatio,
                         Qt.TransformationMode.SmoothTransformation)
    else:
        pix = pix.scaled(label.width(), label.height(),
                         Qt.KeepAspectRatio,
                         Qt.SmoothTransformation)
    label.setPixmap(pix)


def _try_parse_intrinsics(data: Any) -> Tuple[np.ndarray, np.ndarray]:
    if not isinstance(data, dict):
        raise ValueError("Invalid JSON: root is not an object")

    candidates = [data]
    for k in ("intrinsic", "intrinsics", "fisheye", "camera", "calibration"):
        v = data.get(k)
        if isinstance(v, dict):
            candidates.append(v)

    def get_arr(obj: dict, keys) -> Optional[np.ndarray]:
        for kk in keys:
            if kk in obj:
                return np.array(obj[kk], dtype=np.float64)
        return None

    K = None
    D = None
    for obj in candidates:
        K_try = get_arr(obj, ("K", "camera_matrix", "cameraMatrix", "k"))
        D_try = get_arr(obj, ("D", "dist_coeffs", "distCoeffs", "distortion", "d"))
        if K_try is not None and D_try is not None:
            K, D = K_try, D_try
            break

    if K is None or D is None:
        raise ValueError("Cannot find intrinsics. Need K+D or camera_matrix+dist_coeffs in JSON.")

    K = K.reshape(3, 3).astype(np.float64)
    D = D.reshape(-1).astype(np.float64)
    if D.size >= 4:
        D = D[:4].copy()
    else:
        D = np.pad(D, (0, 4 - D.size), mode="constant")

    return K, D


def _build_virtual_K(out_w: int, out_h: int, zoom: float) -> np.ndarray:
    z = float(zoom)
    z = max(0.05, min(20.0, z))
    f0 = 0.5 * min(out_w, out_h)
    f = f0 * z
    cx = (out_w - 1) / 2.0
    cy = (out_h - 1) / 2.0
    return np.array([[f, 0, cx],
                     [0, f, cy],
                     [0, 0, 1]], dtype=np.float64)


def rectify_fisheye_view_with_map(
    img_bgr: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    zoom: float,
    out_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    out_w, out_h = int(out_size[0]), int(out_size[1])
    if out_w <= 0 or out_h <= 0:
        raise ValueError("Invalid output size")

    yaw = _deg2rad(float(yaw_deg))
    pitch = _deg2rad(float(pitch_deg))

    R = _rot_y(yaw) @ _rot_x(pitch)
    P = _build_virtual_K(out_w, out_h, zoom)

    map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
        K, D, R, P, (out_w, out_h), cv2.CV_32FC1
    )
    rect = cv2.remap(
        img_bgr, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )
    return rect, map_x, map_y


def _draw_corners_with_index(img_bgr: np.ndarray, corners: Optional[np.ndarray], color=(0, 255, 0)) -> np.ndarray:
    vis = img_bgr.copy()
    if corners is None or len(corners) == 0:
        return vis
    for i, pt in enumerate(corners.reshape(-1, 2)):
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(vis, (x, y), 3, color, -1)
        cv2.putText(vis, str(i), (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return vis


def _corners_to_grid(corners: np.ndarray, rows: int, cols: int) -> np.ndarray:
    pts = corners.reshape(-1, 2)
    return pts.reshape((rows, cols, 2)).astype(np.float32)


def _grid_to_corners(grid: np.ndarray) -> np.ndarray:
    flat = grid.reshape((-1, 2)).astype(np.float32)
    return flat.reshape((-1, 1, 2))


def _apply_reorder(grid: np.ndarray, mode: str) -> np.ndarray:
    if mode == "default":
        return grid
    if mode == "flip_horizontal":
        return grid[:, ::-1, :]
    if mode == "flip_vertical":
        return grid[::-1, :, :]
    if mode == "rotate_90":
        return np.transpose(grid, (1, 0, 2))[::-1, :, :]
    if mode == "rotate_180":
        return grid[::-1, ::-1, :]
    if mode == "rotate_270":
        return np.transpose(grid, (1, 0, 2))[:, ::-1, :]
    return grid


def _adaptive_reorder(grid: np.ndarray, img_shape: Tuple[int, int, int] | Tuple[int, int]) -> np.ndarray:
    h, w = img_shape[:2]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

    modes = ["default", "flip_horizontal", "flip_vertical", "rotate_90", "rotate_180", "rotate_270"]
    best = grid
    best_d = float("inf")
    for m in modes:
        g = _apply_reorder(grid, m)
        p0 = g.reshape(-1, 2)[0]
        d = float((p0[0] - cx) ** 2 + (p0[1] - cy) ** 2)
        if d < best_d:
            best_d = d
            best = g
    return best


def _parse_rows_cols_text(text: str) -> Optional[Tuple[int, int]]:
    t = str(text).strip().lower().replace(" ", "")
    if "x" not in t:
        return None
    parts = t.split("x")
    if len(parts) != 2:
        return None
    try:
        r = int(parts[0]); c = int(parts[1])
    except Exception:
        return None
    if r <= 1 or c <= 1:
        return None
    return (r, c)


def _bilinear_sample(map_x: np.ndarray, map_y: np.ndarray, x: float, y: float) -> Tuple[float, float]:
    h, w = map_x.shape[:2]
    if x < 0 or y < 0 or x > (w - 1) or y > (h - 1):
        return float("nan"), float("nan")

    x0 = int(math.floor(x)); y0 = int(math.floor(y))
    x1 = min(x0 + 1, w - 1); y1 = min(y0 + 1, h - 1)
    dx = x - x0; dy = y - y0

    u00 = float(map_x[y0, x0]); v00 = float(map_y[y0, x0])
    u10 = float(map_x[y0, x1]); v10 = float(map_y[y0, x1])
    u01 = float(map_x[y1, x0]); v01 = float(map_y[y1, x0])
    u11 = float(map_x[y1, x1]); v11 = float(map_y[y1, x1])

    u0 = u00 * (1 - dx) + u10 * dx
    v0 = v00 * (1 - dx) + v10 * dx
    u1 = u01 * (1 - dx) + u11 * dx
    v1 = v01 * (1 - dx) + v11 * dx

    u = u0 * (1 - dy) + u1 * dy
    v = v0 * (1 - dy) + v1 * dy
    return u, v


@dataclass
class SideState:
    image_path: str = ""
    image_bgr: Optional[np.ndarray] = None

    param_path: str = ""
    K: Optional[np.ndarray] = None
    D: Optional[np.ndarray] = None

    views: Dict[str, np.ndarray] = field(default_factory=dict)
    map_x: Dict[str, np.ndarray] = field(default_factory=dict)
    map_y: Dict[str, np.ndarray] = field(default_factory=dict)

    corners: Dict[str, np.ndarray] = field(default_factory=dict)
    corners_raw: Dict[str, np.ndarray] = field(default_factory=dict)
    pattern_rc: Dict[str, Tuple[int, int]] = field(default_factory=dict)


class Controller(QWidget):
    def __init__(self, model=None, parent=None):
        super().__init__(parent)
        self.model = model  # not used
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.state: Dict[str, SideState] = {s: SideState() for s in SIDES}

        self.image_btn_map = {"left": getattr(self.ui, "btn_left_image", None),
                              "right": getattr(self.ui, "btn_right_image", None)}
        self.param_btn_map = {"left": getattr(self.ui, "btn_left_parameter", None),
                              "right": getattr(self.ui, "btn_right_parameter", None)}
        self.image_path_edit_map = {"left": getattr(self.ui, "lineedit_left_image_path", None),
                                    "right": getattr(self.ui, "lineedit_right_image_path", None)}
        self.param_path_edit_map = {"left": getattr(self.ui, "lineedit_left_parameter_path", None),
                                    "right": getattr(self.ui, "lineedit_right_parameter_path", None)}
        self.original_label_map = {"left": getattr(self.ui, "label_original_left", None),
                                   "right": getattr(self.ui, "label_original_right", None)}

        # Start triangulation button
        self.btn_start_calculated = getattr(self.ui, "pushButton_start_calculated", None)

        self.cam_pos_spin = {
            "left": (
                getattr(self.ui, "spinBox_x_left", None),
                getattr(self.ui, "spinBox_y_left", None),
                getattr(self.ui, "spinBox_z_left", None),
            ),
            "right": (
                getattr(self.ui, "spinBox_x_right", None),
                getattr(self.ui, "spinBox_y_right", None),
                getattr(self.ui, "spinBox_z_right", None),
            ),
        }

        self.view_label_map = {
            ("left", "west"): getattr(self.ui, "label_west_left", None),
            ("left", "east"): getattr(self.ui, "label_east_left", None),
            ("left", "north"): getattr(self.ui, "label_north_left", None),
            ("left", "south"): getattr(self.ui, "label_south_left", None),
            ("left", "center"): getattr(self.ui, "label_center_left", None),

            ("right", "west"): getattr(self.ui, "label_west_right", None),
            ("right", "east"): getattr(self.ui, "label_east_right", None),
            ("right", "north"): getattr(self.ui, "label_north_right", None),
            ("right", "south"): getattr(self.ui, "label_south_right", None),
            ("right", "center"): getattr(self.ui, "label_center_right", None),
        }

        self.spin_alpha = {(s, d): getattr(self.ui, f"doubleSpinBox_alpha_{s}_{d}", None) for s in SIDES for d in DIRECTIONS}
        self.spin_beta = {(s, d): getattr(self.ui, f"doubleSpinBox_beta_{s}_{d}", None) for s in SIDES for d in DIRECTIONS}
        self.spin_zoom = {(s, d): getattr(self.ui, f"doubleSpinBox_zoom_{s}_{d}", None) for s in SIDES for d in DIRECTIONS}

        # single lineEdit: lineEdit_point_{cam}_{direction} with "rowsxcols"
        self.edit_point = {(s, d): getattr(self.ui, f"lineEdit_point_{s}_{d}", None) for s in SIDES for d in DIRECTIONS}

        self.detect_btn_map = {(s, d): getattr(self.ui, f"pushButton_Detect_{s}_{d}", None) for s in SIDES for d in DIRECTIONS}

        self.detect_result_btn = {"left": getattr(self.ui, "pushButton_Detect_left_result", None),
                                  "right": getattr(self.ui, "pushButton_Detect_right_result", None)}
        self.result_label = {"left": getattr(self.ui, "label_left_result", None),
                             "right": getattr(self.ui, "label_right_result", None)}

        self.reorder_combo_map = {
            ('left', 'west'): getattr(self.ui, "comboBox_reorder_left_west", None),
            ('left', 'east'): getattr(self.ui, "comboBox_reorder_left_east", None),
            ('left', 'north'): getattr(self.ui, "comboBox_reorder_left_north", None),
            ('left', 'south'): getattr(self.ui, "comboBox_reorder_left_south", None),
            ('left', 'center'): getattr(self.ui, "comboBox_reorder_left_center", None),

            ('right', 'west'): getattr(self.ui, "comboBox_reorder_right_west", None),
            ('right', 'east'): getattr(self.ui, "comboBox_reorder_right_east", None),
            ('right', 'north'): getattr(self.ui, "comboBox_reorder_right_north", None),
            ('right', 'south'): getattr(self.ui, "comboBox_reorder_right_south", None),
            ('right', 'center'): getattr(self.ui, "comboBox_reorder_right_center", None),
        }
        combo_items = ["default", "flip_horizontal", "flip_vertical", "rotate_90", "rotate_180", "rotate_270", "adaptive"]
        for combo in self.reorder_combo_map.values():
            if combo is None:
                continue
            combo.clear()
            combo.addItems(combo_items)

        self._connect_signals()

    def _connect_signals(self) -> None:
        for side, btn in self.image_btn_map.items():
            if btn is not None:
                btn.clicked.connect(lambda _=False, s=side: self.open_image(s))
        for side, btn in self.param_btn_map.items():
            if btn is not None:
                btn.clicked.connect(lambda _=False, s=side: self.open_params(s))

        for s in SIDES:
            for d in DIRECTIONS:
                a = self.spin_alpha[(s, d)]
                b = self.spin_beta[(s, d)]
                z = self.spin_zoom[(s, d)]
                if a is not None:
                    a.valueChanged.connect(lambda _=0.0, ss=s, dd=d: self.update_view(ss, dd))
                if b is not None:
                    b.valueChanged.connect(lambda _=0.0, ss=s, dd=d: self.update_view(ss, dd))
                if z is not None:
                    z.valueChanged.connect(lambda _=0.0, ss=s, dd=d: self.update_view(ss, dd))

        for (s, d), btn in self.detect_btn_map.items():
            if btn is not None:
                btn.clicked.connect(lambda _=False, ss=s, dd=d: self.detect_chessboard(ss, dd))

        for (s, d), combo in self.reorder_combo_map.items():
            if combo is not None:
                combo.currentTextChanged.connect(lambda _txt="", ss=s, dd=d: self._on_reorder_changed(ss, dd))

        # RESULT: overlay remapped points on original
        for s, btn in self.detect_result_btn.items():
            if btn is not None:
                btn.clicked.connect(lambda _=False, ss=s: self.show_result_overlay_on_original(ss))

        for side in ("left", "right"):
            sx, sy, sz = self.cam_pos_spin.get(side, (None, None, None))
            for w in (sx, sy, sz):
                if w is None:
                    continue
                w.valueChanged.connect(lambda _=0, s=side: print(
                    f"[CAM] {s} position changed -> {self._get_camera_center_from_ui(s).tolist()}"))

        # Start triangulation
        if self.btn_start_calculated is not None:
            self.btn_start_calculated.clicked.connect(self.handle_start_calculated)

    def open_image(self, side: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, f"Open {side} image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if not path:
            return
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            QMessageBox.warning(self, "Open image", f"Cannot read image:\n{path}")
            return

        st = self.state[side]
        st.image_path = path
        st.image_bgr = img
        st.views.clear(); st.map_x.clear(); st.map_y.clear()
        st.corners.clear(); st.corners_raw.clear(); st.pattern_rc.clear()

        if self.image_path_edit_map.get(side) is not None:
            self.image_path_edit_map[side].setText(path)
        _set_label_pixmap(self.original_label_map.get(side), img)

        for d in DIRECTIONS:
            self.update_view(side, d)

    def open_params(self, side: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, f"Open {side} calibration JSON", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            K, D = _try_parse_intrinsics(data)
        except Exception as e:
            QMessageBox.warning(self, "Open params", f"Failed to load calibration params:\n{e}")
            return

        st = self.state[side]
        st.param_path = path
        st.K = K; st.D = D
        st.views.clear(); st.map_x.clear(); st.map_y.clear()
        st.corners.clear(); st.corners_raw.clear(); st.pattern_rc.clear()

        if self.param_path_edit_map.get(side) is not None:
            self.param_path_edit_map[side].setText(path)

        for d in DIRECTIONS:
            self.update_view(side, d)

    def _get_out_size_from_label(self, label) -> Tuple[int, int]:
        w = max(80, int(label.width())) if label is not None else 640
        h = max(80, int(label.height())) if label is not None else 480
        return (w, h)

    def update_view(self, side: str, direction: str) -> None:
        st = self.state[side]
        label = self.view_label_map.get((side, direction))

        if st.image_bgr is None:
            _set_label_pixmap(label, None)
            return

        if st.K is None or st.D is None:
            _set_label_pixmap(label, st.image_bgr)
            return

        yaw = float(self.spin_alpha[(side, direction)].value()) if self.spin_alpha[(side, direction)] is not None else 0.0
        pitch = float(self.spin_beta[(side, direction)].value()) if self.spin_beta[(side, direction)] is not None else 0.0
        zoom = float(self.spin_zoom[(side, direction)].value()) if self.spin_zoom[(side, direction)] is not None else 1.0

        out_w, out_h = self._get_out_size_from_label(label)

        try:
            rect, mx, my = rectify_fisheye_view_with_map(
                st.image_bgr, st.K, st.D,
                yaw_deg=yaw, pitch_deg=pitch, zoom=zoom,
                out_size=(out_w, out_h),
            )
        except Exception as e:
            QMessageBox.warning(self, "Undistort/Rectify", f"Failed to rectify {side}-{direction}:\n{e}")
            return

        st.views[direction] = rect
        st.map_x[direction] = mx
        st.map_y[direction] = my

        if direction in st.corners:
            rect_show = _draw_corners_with_index(rect, st.corners[direction])
            _set_label_pixmap(label, rect_show)
        else:
            _set_label_pixmap(label, rect)

    def _read_pattern_rc(self, side: str, direction: str) -> Optional[Tuple[int, int]]:
        w = self.edit_point.get((side, direction))
        if w is None:
            return None
        return _parse_rows_cols_text(w.text())


    def detect_chessboard(self, side: str, direction: str) -> None:
        st = self.state[side]
        if direction not in st.views:
            QMessageBox.information(self, "Detect", f"No rectified image for {side}-{direction}.")
            return

        img = st.views[direction]
        rc = self._read_pattern_rc(side, direction)
        if rc is None:
            QMessageBox.warning(self, "Detect", f"Invalid pattern for {side}-{direction}. Use 'rowsxcols', e.g. 8x11.")
            return

        rows, cols = rc
        gray = _ensure_gray(img)

        found = False
        corners = None

        if hasattr(cv2, "findChessboardCornersSB"):
            try:
                flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
                found, corners = cv2.findChessboardCornersSB(gray, (cols, rows), flags)
            except Exception:
                found = False

        if not found:
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
            found, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
            if found and corners is not None:
                term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
                cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)

        if not found or corners is None:
            QMessageBox.information(self, "Detect", f"Chessboard NOT found: {side}-{direction}")
            self.update_view(side, direction)
            return

        st.pattern_rc[direction] = (rows, cols)
        st.corners_raw[direction] = corners.copy()


        corners2 = self._apply_reorder_for(side, direction, img, corners, rows, cols)
        st.corners[direction] = corners2
        print(f"[DETECT] OK {side}-{direction}: corners={len(corners2)} rowsxcols={rows}x{cols}")

        label = self.view_label_map.get((side, direction))
        vis = _draw_corners_with_index(img, corners2)
        _set_label_pixmap(label, vis)

    def _get_camera_center_from_ui(self, side: str) -> np.ndarray:
        """
        Read camera center (C_w) from UI spinboxes.
        Returns np.array([x,y,z], float64)
        """
        sx, sy, sz = self.cam_pos_spin.get(side, (None, None, None))

        def _val(w):
            if w is None:
                return 0.0
            # spinBox biasanya integer; tetap cast float untuk konsistensi
            return float(w.value())

        C = np.array([_val(sx), _val(sy), _val(sz)], dtype=np.float64)

        # DEBUG print (opsional, sangat membantu)
        print(f"[CAM] {side} center from UI: C_w={C.tolist()}")

        return C

    def _apply_reorder_for(self, side: str, direction: str, img_bgr: np.ndarray, corners: np.ndarray, rows: int, cols: int) -> np.ndarray:
        combo = self.reorder_combo_map.get((side, direction))
        mode = combo.currentText().strip() if combo is not None else "default"
        grid = _corners_to_grid(corners, rows, cols)
        if mode == "adaptive":
            grid2 = _adaptive_reorder(grid, img_bgr.shape)
        else:
            grid2 = _apply_reorder(grid, mode)
        return _grid_to_corners(grid2)

    def _on_reorder_changed(self, side: str, direction: str) -> None:
        st = self.state[side]
        if direction not in st.corners_raw or direction not in st.views:
            return
        rows_cols = st.pattern_rc.get(direction)
        if not rows_cols:
            return

        rows, cols = rows_cols
        img = st.views[direction]
        corners_raw = st.corners_raw[direction]
        corners2 = self._apply_reorder_for(side, direction, img, corners_raw, rows, cols)
        st.corners[direction] = corners2

        label = self.view_label_map.get((side, direction))
        vis = _draw_corners_with_index(img, corners2)
        _set_label_pixmap(label, vis)

    def _remap_rect_points_to_fisheye(self, side: str, direction: str, corners_rect: np.ndarray) -> np.ndarray:
        st = self.state[side]
        mx = st.map_x.get(direction)
        my = st.map_y.get(direction)
        if mx is None or my is None:
            raise RuntimeError(f"Missing rect->fisheye map for {side}-{direction}.")
        pts = corners_rect.reshape(-1, 2)
        uv = np.zeros((pts.shape[0], 2), dtype=np.float32)
        for i, (x, y) in enumerate(pts):
            u, v = _bilinear_sample(mx, my, float(x), float(y))
            uv[i, 0] = u
            uv[i, 1] = v
        return uv

    def show_result_overlay_on_original(self, side: str) -> None:
        st = self.state[side]
        if st.image_bgr is None:
            QMessageBox.information(self, "Result", f"No original image loaded for {side}.")
            return

        base = st.image_bgr.copy()

        for d in DIRECTIONS:
            if d not in st.corners:
                continue
            try:
                uv = self._remap_rect_points_to_fisheye(side, d, st.corners[d])
            except Exception as e:
                QMessageBox.warning(self, "Result", f"Failed to remap points for {side}-{d}:\n{e}")
                continue

            for i, (u, v) in enumerate(uv):
                if not np.isfinite(u) or not np.isfinite(v):
                    continue
                x = int(round(float(u)))
                y = int(round(float(v)))
                if x < 0 or y < 0 or x >= base.shape[1] or y >= base.shape[0]:
                    continue
                cv2.circle(base, (x, y), 5, (0, 0, 255), -1)
                if i % 10 == 0:
                    cv2.putText(base, d[0].upper(), (x + 2, y - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)

        # --- EXPORT EXCEL ---
        excel_path = self._export_points_to_excel(side)
        if excel_path:
            print(f"[EXPORT] Saved Excel for {side}: {excel_path}")
        else:
            print(f"[EXPORT] No points to export for {side} (maybe detection not done).")
        _set_label_pixmap(self.result_label.get(side), base)

    def _collect_matched_uv_pairs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Collect matched fisheye pixel pairs (uv_L, uv_R) from detected corners.
        Strategy (simple & deterministic):
          - per direction: pair by index (0..min(NL,NR)-1)
        Returns:
          uvL: (N,2), uvR: (N,2), meta: (N,) object array with (direction, idx)
        """
        stL = self.state["left"]
        stR = self.state["right"]

        uvL_all = []
        uvR_all = []
        meta = []

        for d in DIRECTIONS:
            if d not in stL.corners or d not in stR.corners:
                continue

            uvL = self._remap_rect_points_to_fisheye("left", d, stL.corners[d])
            uvR = self._remap_rect_points_to_fisheye("right", d, stR.corners[d])

            n = int(min(len(uvL), len(uvR)))
            if n <= 0:
                continue

            uvL_all.append(uvL[:n])
            uvR_all.append(uvR[:n])
            meta.extend([(d, i) for i in range(n)])

            print(f"[MATCH] {d}: NL={len(uvL)} NR={len(uvR)} -> use {n}")

        if not uvL_all:
            return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 2), dtype=np.float64), np.array([], dtype=object)

        uvL_all = np.vstack(uvL_all).astype(np.float64)
        uvR_all = np.vstack(uvR_all).astype(np.float64)
        meta = np.array(meta, dtype=object)

        print(f"[MATCH] total pairs = {len(uvL_all)}")
        return uvL_all, uvR_all, meta

    def handle_start_calculated(self) -> None:
        """
        Start stereo triangulation (OpenCV-only) when pushButton_start_calculated is clicked.
        Requirements:
          - Both sides loaded images + intrinsics
          - Both sides have detected corners (at least one direction)
          - Camera positions provided from UI spinBoxes
        Outputs:
          - triangulation Excel in output/ folder
        """
        # Basic checks
        if self.state["left"].image_bgr is None or self.state["right"].image_bgr is None:
            QMessageBox.warning(self, "Triangulation", "Please load BOTH left and right images first.")
            return
        if self.state["left"].K is None or self.state["right"].K is None:
            QMessageBox.warning(self, "Triangulation", "Please load BOTH left and right calibration JSON (K,D) first.")
            return

        uvL, uvR, meta = self._collect_matched_uv_pairs()
        if len(uvL) == 0:
            QMessageBox.warning(self, "Triangulation", "No matched points.\nRun detection on BOTH sides first.")
            return

        # Camera centers (baseline) from UI
        C_L = self._get_camera_center_from_ui("left")
        C_R = self._get_camera_center_from_ui("right")
        baseline = float(np.linalg.norm(C_R - C_L))
        print(f"[CAM] baseline = {baseline}")

        # Build camera models (rotation identity for now)
        camL = FisheyeCamera(
            K=self.state["left"].K,
            D=self.state["left"].D,
            R_wc=np.eye(3, dtype=np.float64),
            C_w=C_L,
        )
        camR = FisheyeCamera(
            K=self.state["right"].K,
            D=self.state["right"].D,
            R_wc=np.eye(3, dtype=np.float64),
            C_w=C_R,
        )

        # Triangulate
        out = triangulate_midpoint_from_pixels(camL, camR, uvL, uvR)
        P_mid = out["P_mid"]
        P_1 = out["P1"]
        P_2 = out["P2"]
        dist12 = out["dist12"]
        rays1 = out["rays1_w"]
        rays2 = out["rays2_w"]
        print(f"[TRI] done: N={len(P_mid)} dist_mean={float(np.mean(dist12))} dist_max={float(np.max(dist12))}")

        # Export to Excel
        out_dir = self._get_output_dir()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(out_dir, f"triangulation_{ts}.xlsx")

        rows = []
        json_rows = []
        for i in range(len(P_mid)):
            d, idx = meta[i]  # meta dari matching: (direction, index)

            # angle between rays (deg)
            cosang = float(np.clip(np.dot(rays1[i], rays2[i]), -1.0, 1.0))
            ray_angle_deg = float(np.degrees(np.arccos(cosang)))

            d_mid_to_center = float(np.linalg.norm(P_mid[i] - baseline))

            rec = {
                "direction": str(d),
                "point_id": int(idx),

                "x": float(P_mid[i, 0]),
                "y": float(P_mid[i, 1]),
                "z": float(P_mid[i, 2]),

                "point_p": [float(P_1[i, 0]), float(P_1[i, 1]), float(P_1[i, 2])],  # cam L
                "point_q": [float(P_2[i, 0]), float(P_2[i, 1]), float(P_2[i, 2])],  # cam R

                "ray_gap": float(dist12[i]),
                "ray_angle_deg": float(ray_angle_deg),
                "d_mid_to_center": float(d_mid_to_center),

                # optional placeholder
                "confidence": float(1.0 / (1.0 + dist12[i])) if np.isfinite(dist12[i]) else None,
            }

            rows.append({
                "direction": rec["direction"],
                "point_id": rec["point_id"],
                "x": rec["x"], "y": rec["y"], "z": rec["z"],
                "ray_gap": rec["ray_gap"],
                "ray_angle_deg": rec["ray_angle_deg"],
                "d_mid_to_center": rec["d_mid_to_center"],
            })
            json_rows.append(rec)

        df_3d = pd.DataFrame(rows)

        self._overlay_reprojection_opencv("left", camL, df_3d, out_dir, ts)
        self._overlay_reprojection_opencv("right", camR, df_3d, out_dir, ts)

        # OpenCV-only reprojection compare (CSV)
        compare_reprojection_with_original_opencv(
            df_3d=df_3d[["direction", "point_id", "x", "y", "z"]].copy(),
            cam_L=C_L,
            cam_R=C_R,
            K_L=self.state["left"].K,
            D_L=self.state["left"].D,
            K_R=self.state["right"].K,
            D_R=self.state["right"].D,
            gt_csv_L=None,  # optional: kalau Anda punya file GT csv
            gt_csv_R=None,  # optional
            output_dir=self._get_output_dir(),
        )

        # ---------------- Excel triangulation ----------------
        out_dir = self._get_output_dir()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        xlsx_path = os.path.join(out_dir, f"triangulation_{ts}.xlsx")
        df_3d.to_excel(xlsx_path, index=False, sheet_name="triangulation")
        print(f"[TRI][XLSX] saved: {xlsx_path}")

        baseline_center = (C_L + C_R)

        # ---------------- JSON triangulation (full, include p/q) ----------------
        payload = {
            "meta": {
                "created_at": ts,
                "cam_left": {"C_w": C_L.tolist()},
                "cam_right": {"C_w": C_R.tolist()},
                "baseline_center": baseline_center.tolist(),
            },
            "points": json_rows,
        }
        json_path = self._save_triangulation_json(payload, f"triangulation_{ts}.json")

        # ---------------- Overlay/visualization (3D html) ----------------
        # show_3d_point_2cam_ori_visualization butuh df yang punya x,y,z,direction,point_id,point_p,point_q,ray_gap,ray_angle_deg,confidence,d_mid_to_center
        pattern_size = {}

        # Kita pakai salah satu sisi saja (left), karena grid sama
        for d in DIRECTIONS:
            rc = self._read_pattern_rc("left", d)
            if rc is None:
                continue

            rows, cols = rc  # dari "rowsxcols"
            if rows > 0 and cols > 0:
                # INGAT: overlay pakai (cols, rows)
                pattern_size[d] = (cols, rows)

        if not pattern_size:
            pattern_size = None  # fallback kalau user belum isi apa pun

        print("[GRID] pattern_size from UI:", pattern_size)

        df_vis = pd.DataFrame(json_rows)
        path_basic, path_full, angle_map, mean_maps, thickness_map = show_3d_point_2cam_ori_visualization(
            df_vis,
            cam_L=C_L,
            cam_R=C_R,
            pattern_size=pattern_size  # atau isi jika Anda sudah punya mapping cols,rows per direction
        )
        print("[TRI][HTML] basic:", path_basic)
        print("[TRI][HTML] full :", path_full)



    def _get_output_dir(self) -> str:
        """
        Output folder: satu level dengan file controller ini: .../output
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        out_dir = os.path.join(base_dir, "output")
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _save_triangulation_json(self, payload: dict, filename: str) -> str:
        out_dir = self._get_output_dir()
        path = Path(out_dir) / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[TRI][JSON] saved: {str(path)}")
        return str(path)

    def _collect_all_points_for_excel(self, side: str) -> "pd.DataFrame":
        """
        Kumpulkan:
        - points rectified (x_rect,y_rect)
        - points mapped-back to fisheye original (x_fisheye,y_fisheye)
        - metadata: direction, index, rows/cols, yaw/pitch/zoom
        """
        st = self.state[side]
        rows_list = []

        for d in DIRECTIONS:
            if d not in st.corners:
                continue

            corners_rect = st.corners[d]  # (N,1,2) in rect coords
            pts_rect = corners_rect.reshape(-1, 2)

            # # ambil yaw/pitch/zoom yang dipakai
            # yaw = float(self.spin_alpha[(side, d)].value()) if self.spin_alpha[(side, d)] is not None else 0.0
            # pitch = float(self.spin_beta[(side, d)].value()) if self.spin_beta[(side, d)] is not None else 0.0
            # zoom = float(self.spin_zoom[(side, d)].value()) if self.spin_zoom[(side, d)] is not None else 1.0

            # rows/cols pattern (kalau ada)
            # rc = st.pattern_rc.get(d, (None, None))
            # rows_c, cols_c = rc

            # remap balik ke fisheye
            uv = self._remap_rect_points_to_fisheye(side, d, corners_rect)  # (N,2)
            cam = FisheyeCamera(
                K=st.K,
                D=st.D,
                R_wc=np.eye(3),
                C_w=np.zeros(3),
            )
            ab = cam.pixel_to_alpha_beta(uv)
            for i, ((xr, yr), (uf, vf)) in enumerate(zip(pts_rect, uv)):
                rows_list.append({
                    "side": side,
                    "direction": d,
                    "idx": i,
                    "x_rect": float(xr),
                    "y_rect": float(yr),
                    "x_fisheye": float(uf),
                    "y_fisheye": float(vf),
                    "alpha_deg": float(ab[i, 0]),
                    "beta_deg": float(ab[i, 1]),
                })

        return pd.DataFrame(rows_list)

    def _export_points_to_excel(self, side: str) -> Optional[str]:
        """
        Export semua titik ke Excel.
        Return filepath jika sukses.
        """
        st = self.state[side]
        if st.image_bgr is None:
            return None

        df = self._collect_all_points_for_excel(side)
        if df.empty:
            return None

        out_dir = self._get_output_dir()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chessboard_points_{side}_{ts}.xlsx"
        filepath = os.path.join(out_dir, filename)

        # Bisa multi-sheet jika mau. Untuk sekarang 1 sheet.
        df.to_excel(filepath, index=False, sheet_name="points")

        return filepath

    def _overlay_reprojection_opencv(
            self,
            side: str,
            cam: FisheyeCamera,
            df_tri: pd.DataFrame,  # hasil triangulasi (punya columns: x,y,z,direction,point_id)
            out_dir: str,
            ts: str,
    ) -> Tuple[str, str]:
        """
        Overlay hasil reprojection (3D -> fisheye pixel) di gambar original.
        Juga hitung error terhadap titik deteksi (fisheye uv) jika tersedia.
        Return: (overlay_image_path, error_excel_path)
        """
        st = self.state[side]
        if st.image_bgr is None:
            raise RuntimeError(f"No original image for {side}")

        # 1) Reproject semua 3D midpoint
        P_w = df_tri[["x", "y", "z"]].to_numpy(dtype=np.float64)
        uv_proj = project_world_points_to_fisheye_pixels(cam, P_w)  # (N,2)

        # 2) Kalau ingin hitung error vs detection, kita butuh uv_det untuk tiap row.
        #    Kita siapkan mapping (direction,point_id) -> uv_det dari hasil detect+remap.
        uv_det_map: Dict[Tuple[str, int], np.ndarray] = {}
        for d in DIRECTIONS:
            if d not in st.corners:
                continue
            uv_d = self._remap_rect_points_to_fisheye(side, d, st.corners[d])  # (Nd,2)
            for i in range(len(uv_d)):
                uv_det_map[(d, i)] = uv_d[i]

        # 3) Build error table + overlay image
        overlay = st.image_bgr.copy()
        rows = []
        H, W = overlay.shape[:2]

        for i in range(len(df_tri)):
            d = str(df_tri.loc[i, "direction"])
            pid = int(df_tri.loc[i, "point_id"])

            u, v = float(uv_proj[i, 0]), float(uv_proj[i, 1])

            # draw projected point
            if np.isfinite(u) and np.isfinite(v):
                x = int(round(u));
                y = int(round(v))
                if 0 <= x < W and 0 <= y < H:
                    cv2.circle(overlay, (x, y), 4, (0, 255, 255), -1)  # projected: yellow

            # compare with detection if exists
            err = None
            if (d, pid) in uv_det_map:
                ud, vd = uv_det_map[(d, pid)]
                ud = float(ud);
                vd = float(vd)
                if np.isfinite(u) and np.isfinite(v) and np.isfinite(ud) and np.isfinite(vd):
                    err = float(np.hypot(u - ud, v - vd))

                    # draw detected point (red) + line
                    xd = int(round(ud));
                    yd = int(round(vd))
                    if 0 <= xd < W and 0 <= yd < H:
                        cv2.circle(overlay, (xd, yd), 4, (0, 0, 255), -1)  # detected: red
                    if 0 <= x < W and 0 <= y < H and 0 <= xd < W and 0 <= yd < H:
                        cv2.line(overlay, (x, y), (xd, yd), (255, 0, 0), 1)  # error line: blue

            rows.append({
                "side": side,
                "direction": d,
                "point_id": pid,
                "u_proj": u,
                "v_proj": v,
                "u_det": float(uv_det_map[(d, pid)][0]) if (d, pid) in uv_det_map else np.nan,
                "v_det": float(uv_det_map[(d, pid)][1]) if (d, pid) in uv_det_map else np.nan,
                "err_px": err if err is not None else np.nan,
            })

        # 4) Save overlay image + error excel
        overlay_path = os.path.join(out_dir, f"reproj_overlay_{side}_{ts}.png")
        cv2.imwrite(overlay_path, overlay)

        err_df = pd.DataFrame(rows)
        err_xlsx = os.path.join(out_dir, f"reproj_error_{side}_{ts}.xlsx")
        err_df.to_excel(err_xlsx, index=False, sheet_name="reproj_error")

        print(f"[REPROJ] {side} overlay: {overlay_path}")
        print(f"[REPROJ] {side} error : {err_xlsx}")
        if "err_px" in err_df.columns and err_df["err_px"].notna().any():
            print(f"[REPROJ] {side} err mean(px): {float(err_df['err_px'].mean())}")

        return overlay_path, err_xlsx


class PluginOpenCv(PluginInterface):
    def __init__(self):
        super().__init__()
        self.widget = None
        self.description = "OpenCV-only fisheye rectification + chessboard detection (result remap to original)"

    def set_plugin_widget(self, model):
        self.widget = Controller(model)
        return self.widget

    def set_icon_apps(self):
        return "logo.png"

    def change_stylesheet(self):
        pass
