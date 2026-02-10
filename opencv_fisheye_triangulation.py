
# opencv_fisheye_triangulation.py
# Clean OpenCV-only fisheye pipeline:
#   pixel (fisheye) -> ray -> alpha/beta (Moil-like convention optional) -> stereo triangulation (closest approach)
#
# No dependency on moil / moildev.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import math
import numpy as np
import cv2


# ------------------------- Angle / ray utilities -------------------------

def normalize_rows(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return v / n


def ray_to_alpha_beta_moil_convention(ray_cam: np.ndarray) -> Tuple[float, float]:
    """
    Moil-like convention (same as your moil_3d_rp.py):
      alpha = acos(vz) [deg]
      beta  = (90 - atan2(vy, vx)[deg]) mod 360

    ray_cam: (3,) direction in CAMERA coordinates.
    """
    v = normalize_rows(np.asarray(ray_cam, dtype=np.float64).reshape(1, 3))[0]
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])

    vz_clip = max(-1.0, min(1.0, vz))
    alpha = math.degrees(math.acos(vz_clip))

    beta_cart = math.degrees(math.atan2(vy, vx))
    beta = (90.0 - beta_cart) % 360.0
    return alpha, beta


def alpha_beta_to_ray_moil_convention(alpha_deg: float, beta_deg: float) -> np.ndarray:
    """
    Inverse of ray_to_alpha_beta_moil_convention, producing a unit ray in CAMERA coordinates.

    Derivation:
      vz = cos(alpha)
      r_xy = sin(alpha)
      beta_cart = 90 - beta
      vx = r_xy * cos(beta_cart)
      vy = r_xy * sin(beta_cart)
    """
    a = math.radians(float(alpha_deg))
    beta_cart = math.radians(90.0 - float(beta_deg))

    vz = math.cos(a)
    rxy = math.sin(a)
    vx = rxy * math.cos(beta_cart)
    vy = rxy * math.sin(beta_cart)

    v = np.array([vx, vy, vz], dtype=np.float64)
    return normalize_rows(v.reshape(1, 3))[0]


# ------------------------- Camera model -------------------------

@dataclass
class FisheyeCamera:
    """
    Minimal fisheye camera model for OpenCV fisheye.

    K, D: OpenCV fisheye intrinsics.
    R_wc: rotation from CAMERA to WORLD coordinates. (world = R_wc @ cam + C)
    C_w : camera center in WORLD coordinates.

    If you do not have extrinsics yet, you can set:
      R_wc = identity(3)
      C_w  = (0,0,0) for left, (baseline,0,0) for right (example)
    """
    K: np.ndarray              # 3x3
    D: np.ndarray              # 4,
    R_wc: np.ndarray           # 3x3
    C_w: np.ndarray            # 3,

    def __post_init__(self):
        self.K = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        self.D = np.asarray(self.D, dtype=np.float64).reshape(-1)[:4]
        if self.D.size < 4:
            self.D = np.pad(self.D, (0, 4 - self.D.size), mode="constant")
        self.R_wc = np.asarray(self.R_wc, dtype=np.float64).reshape(3, 3)
        self.C_w = np.asarray(self.C_w, dtype=np.float64).reshape(3)

    def pixel_to_ray_cam(self, uv: np.ndarray) -> np.ndarray:
        """
        uv: (N,2) fisheye pixels in ORIGINAL image.
        returns: (N,3) unit rays in CAMERA coordinates.
        """
        pts = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
        # normalized points on z=1 plane
        xy = cv2.fisheye.undistortPoints(pts, self.K, self.D, R=None, P=None).reshape(-1, 2)
        rays = np.concatenate([xy, np.ones((xy.shape[0], 1), dtype=np.float64)], axis=1)
        return normalize_rows(rays)

    def ray_cam_to_world(self, ray_cam: np.ndarray) -> np.ndarray:
        """
        ray_cam: (N,3) or (3,)
        """
        r = np.asarray(ray_cam, dtype=np.float64)
        if r.ndim == 1:
            r = r.reshape(1, 3)
        r = normalize_rows(r)
        # direction transforms with rotation only
        r_w = (self.R_wc @ r.T).T
        return normalize_rows(r_w)

    def pixel_to_ray_world(self, uv: np.ndarray) -> np.ndarray:
        """
        uv: (N,2) fisheye pixels -> unit rays in WORLD coordinates
        """
        r_cam = self.pixel_to_ray_cam(uv)
        return self.ray_cam_to_world(r_cam)

    def pixel_to_alpha_beta(self, uv: np.ndarray) -> np.ndarray:
        """
        Return (N,2) alpha,beta in degrees using Moil-like convention,
        computed from CAMERA-ray directions.
        """
        r_cam = self.pixel_to_ray_cam(uv)
        out = np.zeros((r_cam.shape[0], 2), dtype=np.float64)
        for i in range(r_cam.shape[0]):
            a, b = ray_to_alpha_beta_moil_convention(r_cam[i])
            out[i, 0] = a
            out[i, 1] = b
        return out


# ------------------------- Triangulation (closest approach) -------------------------

def closest_points_between_rays(
    C1: np.ndarray, d1: np.ndarray,
    C2: np.ndarray, d2: np.ndarray,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute closest points on two rays (infinite lines) L1(s)=C1+s d1, L2(t)=C2+t d2.
    Returns (P1, P2, distance).

    Robust for near-parallel rays via small eps.
    """
    C1 = np.asarray(C1, dtype=np.float64).reshape(3)
    C2 = np.asarray(C2, dtype=np.float64).reshape(3)
    d1 = normalize_rows(np.asarray(d1, dtype=np.float64).reshape(1, 3))[0]
    d2 = normalize_rows(np.asarray(d2, dtype=np.float64).reshape(1, 3))[0]

    w0 = C1 - C2
    a = float(np.dot(d1, d1))  # = 1
    b = float(np.dot(d1, d2))
    c = float(np.dot(d2, d2))  # = 1
    d = float(np.dot(d1, w0))
    e = float(np.dot(d2, w0))

    denom = a * c - b * b
    if abs(denom) < eps:
        # nearly parallel: pick s=0 and project w0 onto d2 for t
        s = 0.0
        t = e / c
    else:
        s = (b * e - c * d) / denom
        t = (a * e - b * d) / denom

    P1 = C1 + s * d1
    P2 = C2 + t * d2
    dist = float(np.linalg.norm(P1 - P2))
    return P1, P2, dist


def triangulate_midpoint_from_pixels(
    cam1: FisheyeCamera,
    cam2: FisheyeCamera,
    uv1: np.ndarray,   # (N,2)
    uv2: np.ndarray,   # (N,2) matched
) -> Dict[str, np.ndarray]:
    """
    Triangulate matched pixel observations uv1<->uv2 using OpenCV fisheye model.
    Returns dict with:
      - P_mid  : (N,3) midpoint
      - P1     : (N,3) closest point on ray1
      - P2     : (N,3) closest point on ray2
      - dist12 : (N,)  closest distance between rays
      - rays1_w: (N,3) unit rays world
      - rays2_w: (N,3) unit rays world
    """
    uv1 = np.asarray(uv1, dtype=np.float64).reshape(-1, 2)
    uv2 = np.asarray(uv2, dtype=np.float64).reshape(-1, 2)
    if uv1.shape[0] != uv2.shape[0]:
        raise ValueError("uv1 and uv2 must have the same number of points (already matched).")

    rays1_w = cam1.pixel_to_ray_world(uv1)
    rays2_w = cam2.pixel_to_ray_world(uv2)

    P1 = np.zeros((uv1.shape[0], 3), dtype=np.float64)
    P2 = np.zeros((uv1.shape[0], 3), dtype=np.float64)
    Pm = np.zeros((uv1.shape[0], 3), dtype=np.float64)
    dist = np.zeros((uv1.shape[0],), dtype=np.float64)

    for i in range(uv1.shape[0]):
        p1, p2, dd = closest_points_between_rays(cam1.C_w, rays1_w[i], cam2.C_w, rays2_w[i])
        P1[i] = p1
        P2[i] = p2
        Pm[i] = 0.5 * (p1 + p2)
        dist[i] = dd

    return {
        "P_mid": Pm,
        "P1": P1,
        "P2": P2,
        "dist12": dist,
        "rays1_w": rays1_w,
        "rays2_w": rays2_w,
    }

def project_world_points_to_fisheye_pixels(
    cam: FisheyeCamera,
    P_w: np.ndarray,                 # (N,3) world points
) -> np.ndarray:
    """
    Reproject world points to fisheye pixel coordinates using OpenCV fisheye model.

    We have cam.R_wc (camera->world). OpenCV projection uses:
        X_c = R * X_w + t
    where R is world->camera rotation (R_cw) and t is translation in camera frame.

    With camera center C_w and R_wc:
        R_cw = R_wc^T
        t = -R_cw * C_w

    Returns:
        uv: (N,2) fisheye pixels
    """
    P_w = np.asarray(P_w, dtype=np.float64).reshape(-1, 1, 3)

    R_cw = cam.R_wc.T
    rvec, _ = cv2.Rodrigues(R_cw)
    tvec = (-R_cw @ cam.C_w.reshape(3, 1)).reshape(3, 1)

    uv, _ = cv2.fisheye.projectPoints(P_w, rvec, tvec, cam.K, cam.D)  # (N,1,2)
    return uv.reshape(-1, 2)

