# contoller.py
from src.plugin_interface import PluginInterface
from PyQt6.QtWidgets import QWidget, QFileDialog, QMessageBox
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QEvent
from .ui_main import Ui_Form

import cv2
import numpy as np
import json
import os
import csv

# ------------- Utils -------------
def cv_bgr_to_qpixmap(img_bgr: np.ndarray) -> QPixmap:
    if img_bgr is None:
        return QPixmap()
    if img_bgr.ndim == 2:
        h, w = img_bgr.shape
        qimg = QImage(img_bgr.data, w, h, w, QImage.Format.Format_Grayscale8)
        return QPixmap.fromImage(qimg.copy())
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = img_rgb.shape
    qimg = QImage(img_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())

def undistort_look(img, K, D, yaw_deg=0, pitch_deg=0, roll_deg=0, zoom=1.0, keep_center=True):
    h, w = img.shape[:2]
    yaw, pitch, roll = np.deg2rad([yaw_deg, pitch_deg, roll_deg])
    Ry, _ = cv2.Rodrigues(np.array([0, yaw, 0], dtype=np.float64))
    Rx, _ = cv2.Rodrigues(np.array([pitch, 0, 0], dtype=np.float64))
    Rz, _ = cv2.Rodrigues(np.array([0, 0, 0 if roll is None else roll], dtype=np.float64))
    R = Rz @ Ry @ Rx

    Knew = K.copy().astype(np.float64)
    Knew[0, 0] *= zoom
    Knew[1, 1] *= zoom
    if keep_center:
        Knew[0, 2] = w / 2.0
        Knew[1, 2] = h / 2.0

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, R, Knew, (w, h), cv2.CV_32FC1)
    und = cv2.remap(img, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return und, map1, map2

def detect_chessboard_sb(img_bgr, pattern_size, normalize=True, exhaustive=True, accuracy=True):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    flags = 0
    if normalize:  flags |= cv2.CALIB_CB_NORMALIZE_IMAGE
    if exhaustive: flags |= cv2.CALIB_CB_EXHAUSTIVE
    if accuracy:   flags |= cv2.CALIB_CB_ACCURACY
    ok, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=flags)
    return ok, corners  # (N,1,2)

def draw_crosses(img_bgr, pts_xy, size, thickness, color=(0, 255, 0)):
    vis = img_bgr.copy()
    for (x, y) in pts_xy.astype(int):
        cv2.drawMarker(
            vis, (int(x), int(y)), color,
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=size, thickness=thickness, line_type=cv2.LINE_AA
        )
    return vis

def remap_points_und_to_fisheye(pts_und, map1, map2, w, h):
    # nearest-neighbor lookup ke peta (map1, map2 : dst->src)
    u = np.clip(np.round(pts_und[:, 0]).astype(int), 0, w - 1)
    v = np.clip(np.round(pts_und[:, 1]).astype(int), 0, h - 1)
    x_src = map1[v, u]
    y_src = map2[v, u]
    return np.stack([x_src, y_src], axis=1)

def save_csv(csv_path, headers, rows):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)

# ------------- Controller -------------
class Controller(QWidget):
    def __init__(self, model):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.model = model

        # State
        self.image_path = None
        self.param_path = None
        self.image_bgr = None
        self.K = None
        self.D = None
        self.keep_center = True
        self.roll_deg = 0.0
        self.map1 = None
        self.map2 = None

        # Cache tampilan
        self.und_bgr = None         # undistort base (tanpa marker)
        self.und_vis_bgr = None     # undistort + X
        self.overlay_vis_bgr = None # original + X (remap)
        self.detected_pts_und = None  # Nx2 (undistort coords)
        self.detected_pts_fish = None # Nx2 (fisheye coords)

        # Aspect-ratio safe labels
        for lab in (self.ui.label_image_original, self.ui.label_undistort, self.ui.label_overlay):
            lab.setScaledContents(False)
            lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lab.installEventFilter(self)

        # Cache pixmap asli (untuk rescale saat resize)
        self._pixmap_orig_last = None
        self._pixmap_und_last  = None
        self._pixmap_ovl_last  = None

        # SpinBox
        self.ui.doubleSpinBox_yaw.setDecimals(1)
        self.ui.doubleSpinBox_yaw_2.setDecimals(1)
        self.ui.doubleSpinBox_yaw_3.setDecimals(2)
        self.ui.doubleSpinBox_yaw.setSingleStep(1.0)
        self.ui.doubleSpinBox_yaw_2.setSingleStep(1.0)
        self.ui.doubleSpinBox_yaw_3.setSingleStep(0.05)
        if self.ui.doubleSpinBox_yaw_3.value() <= 0.0:
            self.ui.doubleSpinBox_yaw_3.setValue(1.0)

        # Signals
        self.ui.pushButton_opee_image.clicked.connect(self.on_open_image)
        self.ui.pushButton_opee_image_2.clicked.connect(self.on_open_param)
        self.ui.pushButton_detect.clicked.connect(self.on_detect)

        self.ui.doubleSpinBox_yaw.valueChanged.connect(self.on_params_changed)
        self.ui.doubleSpinBox_yaw_2.valueChanged.connect(self.on_params_changed)
        self.ui.doubleSpinBox_yaw_3.valueChanged.connect(self.on_params_changed)

    # -------- Aspect-ratio helpers --------
    def _set_pixmap_keep_ratio(self, label, pix: QPixmap):
        if pix is None or pix.isNull():
            label.clear()
            return
        target_size = label.contentsRect().size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            target_size = label.size()
        scaled = pix.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        label.setPixmap(scaled)

    def _show_on_label(self, label, img_bgr):
        qpix = cv_bgr_to_qpixmap(img_bgr)
        if label is self.ui.label_image_original:
            self._pixmap_orig_last = qpix
        elif label is self.ui.label_undistort:
            self._pixmap_und_last = qpix
        elif label is self.ui.label_overlay:
            self._pixmap_ovl_last = qpix
        self._set_pixmap_keep_ratio(label, qpix)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Resize:
            if obj is self.ui.label_image_original and self._pixmap_orig_last:
                self._set_pixmap_keep_ratio(obj, self._pixmap_orig_last)
            elif obj is self.ui.label_undistort and self._pixmap_und_last:
                self._set_pixmap_keep_ratio(obj, self._pixmap_und_last)
            elif obj is self.ui.label_overlay and self._pixmap_ovl_last:
                self._set_pixmap_keep_ratio(obj, self._pixmap_ovl_last)
        return super().eventFilter(obj, event)

    # -------- Handlers --------
    def on_open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not path:
            return
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            self._error(f"Gagal membaca gambar:\n{path}")
            return
        self.image_path = path
        self.image_bgr = img
        self._show_on_label(self.ui.label_image_original, img)
        self._clear_detection_overlay()
        self.update_undistort()

    def on_open_param(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Camera Param (.json)", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "r") as f:
                j = json.load(f)
            K = np.array(j["K"], dtype=np.float64)
            D = np.array(j["D"], dtype=np.float64).reshape(4, 1)
        except Exception as e:
            self._error(f"Param JSON tidak valid:\n{path}\n\n{e}")
            return
        self.param_path = path
        self.K = K
        self.D = D
        self._clear_detection_overlay()
        self.update_undistort()

    def on_params_changed(self, _=None):
        # Ubah yaw/pitch/zoom -> hapus X di undistort & overlay
        self._clear_detection_overlay()
        self.update_undistort()

    def on_detect(self):
        if self.und_bgr is None or self.map1 is None or self.map2 is None:
            self._info("Belum ada gambar undistort.\nSilakan buka image & param terlebih dahulu.")
            return

        rows, cols = self._read_pattern_size()
        pattern_size = (cols, rows)  # OpenCV expects (cols, rows)

        ok, corners = detect_chessboard_sb(self.und_bgr, pattern_size)
        if not ok or corners is None:
            print("[INFO] Chessboard tidak terdeteksi.")
            # tampilkan base undistort (tanpa X) dan kosongkan overlay
            self.detected_pts_und = None
            self.detected_pts_fish = None
            self.und_vis_bgr = self.und_bgr.copy()
            self.overlay_vis_bgr = self.image_bgr.copy() if self.image_bgr is not None else None
            self._show_on_label(self.ui.label_undistort, self.und_vis_bgr)
            if self.overlay_vis_bgr is not None:
                self._show_on_label(self.ui.label_overlay, self.overlay_vis_bgr)
            return

        # --- Simpan undistort base dulu (tanpa X) ---
        outdir = self._output_dir()
        base = self._base_name()
        und_base_path = os.path.join(outdir, f"{base}_undistort.png")
        cv2.imwrite(und_base_path, self.und_bgr)

        # --- Undistort + marker X ---
        pts_und = corners.reshape(-1, 2)
        self.detected_pts_und = pts_und.copy()
        self.und_vis_bgr = draw_crosses(self.und_bgr, pts_und, size=20, thickness=2, color=(0, 0, 255))
        self._show_on_label(self.ui.label_undistort, self.und_vis_bgr)

        und_vis_path = os.path.join(outdir, f"{base}_undistort_with_X.png")
        cv2.imwrite(und_vis_path, self.und_vis_bgr)

        # --- Remap ke fisheye & overlay di original ---
        h, w = self.image_bgr.shape[:2]
        pts_fish = remap_points_und_to_fisheye(pts_und, self.map1, self.map2, w, h)
        self.detected_pts_fish = pts_fish.copy()

        self.overlay_vis_bgr = draw_crosses(self.image_bgr, pts_fish, size=20, thickness=2, color=(0, 0, 255))
        self._show_on_label(self.ui.label_overlay, self.overlay_vis_bgr)

        ovl_path = os.path.join(outdir, f"{base}_overlay_fisheye_with_X.png")
        cv2.imwrite(ovl_path, self.overlay_vis_bgr)

        # --- Simpan CSV koordinat ---
        und_csv = os.path.join(outdir, f"{base}_und_points.csv")
        fish_csv = os.path.join(outdir, f"{base}_fish_points.csv")
        save_csv(und_csv, ["point_id", "u_und", "v_und"], [[i, float(u), float(v)] for i, (u, v) in enumerate(pts_und)])
        save_csv(fish_csv, ["point_id", "x_fish", "y_fish"], [[i, float(x), float(y)] for i, (x, y) in enumerate(pts_fish)])

        print("[OK] Chessboard terdeteksi & disimpan:")
        print(f" -> {und_base_path}")
        print(f" -> {und_vis_path}")
        print(f" -> {ovl_path}")
        print(f" -> {und_csv}")
        print(f" -> {fish_csv}")

    # -------- Core --------
    def update_undistort(self):
        if self.image_bgr is None or self.K is None or self.D is None:
            return
        yaw_deg   = float(self.ui.doubleSpinBox_yaw.value())
        pitch_deg = float(self.ui.doubleSpinBox_yaw_2.value())
        zoom      = max(float(self.ui.doubleSpinBox_yaw_3.value()), 0.01)
        try:
            und, self.map1, self.map2 = undistort_look(
                self.image_bgr, self.K, self.D,
                yaw_deg=yaw_deg, pitch_deg=pitch_deg, roll_deg=self.roll_deg,
                zoom=zoom, keep_center=self.keep_center
            )
        except Exception as e:
            self._error(f"Gagal melakukan undistort:\n{e}")
            return
        self.und_bgr = und
        self.und_vis_bgr = und.copy()
        self._show_on_label(self.ui.label_undistort, self.und_vis_bgr)

        # refresh overlay (tanpa X) agar viewer tidak kosong
        if self.image_bgr is not None:
            self.overlay_vis_bgr = self.image_bgr.copy()
            self._show_on_label(self.ui.label_overlay, self.overlay_vis_bgr)

    # -------- Helpers --------
    def _read_pattern_size(self):
        def to_int(s, default):
            try:
                return int(s)
            except Exception:
                return default
        rows_txt = self.ui.lineEdit_row.text().strip()
        cols_txt = self.ui.lineEdit_2.text().strip()
        rows = max(2, to_int(rows_txt, 15))
        cols = max(2, to_int(cols_txt, 15))
        return rows, cols

    def _clear_detection_overlay(self):
        self.detected_pts_und = None
        self.detected_pts_fish = None
        if self.und_bgr is not None:
            self.und_vis_bgr = self.und_bgr.copy()
            self._show_on_label(self.ui.label_undistort, self.und_vis_bgr)
        if self.image_bgr is not None:
            self.overlay_vis_bgr = self.image_bgr.copy()
            self._show_on_label(self.ui.label_overlay, self.overlay_vis_bgr)

    def _output_dir(self):
        # simpan ke folder "results" di lokasi plugin ini berjalan
        base_dir = os.path.dirname(os.path.abspath(__file__))
        outdir = os.path.join(base_dir, "results")
        os.makedirs(outdir, exist_ok=True)
        return outdir

    def _base_name(self):
        if self.image_path:
            return os.path.splitext(os.path.basename(self.image_path))[0]
        return "output"

    def _error(self, message: str):
        QMessageBox.critical(self, "Error", message)

    def _info(self, message: str):
        QMessageBox.information(self, "Info", message)


class OpenCV(PluginInterface):
    def __init__(self):
        super().__init__()
        self.widget = None
        self.description = "This is a plugins application"

    def set_plugin_widget(self, model):
        self.widget = Controller(model)
        return self.widget

    def set_icon_apps(self):
        return "logo.png"

    def change_stylesheet(self):
        pass
