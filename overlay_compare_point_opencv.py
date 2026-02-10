from sklearn.decomposition import PCA
import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pyo
import numpy as np
import cv2
import os
import math
import importlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors


# ================== Plane summary helpers (from compute_plane_fit.py) ==================
XYZ_COLS = ["x", "y", "z"]

def fit_plane_pca(points: np.ndarray):
    P = np.asarray(points, dtype=float)

    # 1) buang row yang NaN / Inf
    finite = np.isfinite(P).all(axis=1)
    P = P[finite]

    # 2) butuh minimal 3 titik untuk plane
    if P.shape[0] < 3:
        return None, None, None, np.array([], dtype=float)

    p0 = P.mean(axis=0)
    centered = P - p0

    # 3) kalau semua titik sama (degenerate), skip
    scale = np.linalg.norm(centered)
    if (not np.isfinite(scale)) or scale == 0.0:
        return None, p0, None, np.array([], dtype=float)

    pca = PCA(n_components=3).fit(centered)
    n = pca.components_[-1]
    nrm = np.linalg.norm(n)
    if (not np.isfinite(nrm)) or nrm == 0.0:
        return None, p0, None, np.array([], dtype=float)

    n = n / nrm
    d = -float(np.dot(n, p0))

    # residual untuk titik yang valid saja
    res = centered @ n
    return n, p0, d, res


def summarize_residuals(res: np.ndarray):
    thickness = float(np.max(res) - np.min(res)) if res.size else np.nan
    rms = float(np.sqrt(np.mean(res ** 2))) if res.size else np.nan
    return thickness, rms

def slant_deg_from_normal(n: np.ndarray):
    nz = abs(float(n[2])) / (np.linalg.norm(n) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(nz, -1.0, 1.0))))

def compute_plane_tables(df: pd.DataFrame, scope_name: str):
    P = df[XYZ_COLS].to_numpy(dtype=float)

    n_pca, p0, d_pca, res_pca = fit_plane_pca(P)
    dist_origin_to_centroid = float(np.linalg.norm(p0))
    closest_point = (-d_pca) * n_pca  # titik di plane paling dekat ke origin

    thickness_pca, rms_pca = summarize_residuals(res_pca)
    slant_pca = slant_deg_from_normal(n_pca)

    plane_pca_row = {
        "scope": scope_name,
        "nx": float(n_pca[0]), "ny": float(n_pca[1]), "nz": float(n_pca[2]),
        "centroid_x": float(p0[0]), "centroid_y": float(p0[1]), "centroid_z": float(p0[2]),
        "signed_offset_d": float(d_pca),
        "abs_distance_origin_to_plane": float(abs(d_pca)),
        "dist_origin_to_centroid": float(dist_origin_to_centroid),
        "closest_point_x": float(closest_point[0]),
        "closest_point_y": float(closest_point[1]),
        "closest_point_z": float(closest_point[2]),
        "thickness": float(thickness_pca),
        "rms_dev": float(rms_pca),
        "slant_deg": float(slant_pca),
    }

    plane_origin_row = {
        "scope": scope_name,
        "nx": float(n_pca[0]), "ny": float(n_pca[1]), "nz": float(n_pca[2]),
        "centroid_x": float(p0[0]), "centroid_y": float(p0[1]), "centroid_z": float(p0[2]),
        "signed_offset_d": float(d_pca),
        "signed_distance_origin_to_plane": float(d_pca),
        "abs_distance_origin_to_plane": float(abs(d_pca)),
        "dist_origin_to_centroid": float(dist_origin_to_centroid),
        "closest_point_x": float(closest_point[0]),
        "closest_point_y": float(closest_point[1]),
        "closest_point_z": float(closest_point[2]),
        "slant_deg": float(slant_pca),
    }

    perpoint = df[XYZ_COLS].copy()
    perpoint["scope"] = scope_name
    perpoint["residual_signed_PCA"] = res_pca.astype(float)

    return plane_pca_row, plane_origin_row, perpoint

def export_plane_summary_to_excel(df_3d: pd.DataFrame, output_dir: str, filename: str = "summary_results.xlsx"):
    """Export overall + per-direction plane summary to Excel (or CSV fallback)."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)

    for c in XYZ_COLS:
        if c not in df_3d.columns:
            raise ValueError(f"[PLANE_SUM] Missing column '{c}' in df_3d")

    plane_pca_rows, plane_origin_rows, perpoint_list = [], [], []

    # overall
    if len(df_3d) >= 3:
        pca_row, origin_row, perpoint = compute_plane_tables(df_3d, "overall")
        plane_pca_rows.append(pca_row)
        plane_origin_rows.append(origin_row)
        perpoint_list.append(perpoint)

    # per-direction
    if "direction" in df_3d.columns:
        for d, g in df_3d.groupby("direction"):
            if len(g) < 3:
                continue
            pca_row, origin_row, perpoint = compute_plane_tables(g, f"direction={d}")
            plane_pca_rows.append(pca_row)
            plane_origin_rows.append(origin_row)
            perpoint_list.append(perpoint)

    plane_pca_table = pd.DataFrame(plane_pca_rows)
    plane_origin_table = pd.DataFrame(plane_origin_rows)
    perpoint_table = pd.concat(perpoint_list, ignore_index=True) if perpoint_list else pd.DataFrame()

    engine = None
    if importlib.util.find_spec("openpyxl"):
        engine = "openpyxl"
    elif importlib.util.find_spec("xlsxwriter"):
        engine = "xlsxwriter"

    if engine is None:
        plane_pca_table.to_csv(os.path.join(output_dir, "Plane_PCA.csv"), index=False)
        plane_origin_table.to_csv(os.path.join(output_dir, "Plane_Origin_Distance.csv"), index=False)
        perpoint_table.to_csv(os.path.join(output_dir, "PerPoint_Residuals.csv"), index=False)
        print("[PLANE_SUM] Excel engine not found. Saved CSV summaries instead.")
        return out_path

    with pd.ExcelWriter(out_path, engine=engine) as writer:
        plane_pca_table.to_excel(writer, sheet_name="Plane_PCA", index=False)
        plane_origin_table.to_excel(writer, sheet_name="Plane_Origin_Distance", index=False)
        perpoint_table.to_excel(writer, sheet_name="PerPoint_Residuals", index=False)

    print(f"[PLANE_SUM] Saved → {out_path}")
    return out_path
# ================================================================================



def _resolve_pattern_size_for_dir(pattern_size, direction, n_points=None):
    """
    pattern_size bisa:
      - tuple (cols, rows) untuk semua direction
      - dict {direction: (cols, rows)}
    """
    if isinstance(pattern_size, dict):
        cols, rows = pattern_size[direction]
    else:
        cols, rows = pattern_size
    return int(cols), int(rows)

def _pid_to_rc(pid, cols):
    """Konversi point_id -> (row, col) dengan asumsi row-major: pid = r*cols + c."""
    r = int(pid) // int(cols)
    c = int(pid) %  int(cols)
    return r, c

def compute_grid_edges_3d(df_3d, pattern_size, save_dir=None):
    """
    Buat DataFrame edges (garis) antar tetangga (kanan & bawah) untuk tiap direction.
    Kolom: [direction, edge_type, r, c, pid_a, pid_b, x1,y1,z1, x2,y2,z2, length]
    """
    records = []

    for direction in df_3d["direction"].unique():
        sub = df_3d[df_3d["direction"] == direction].copy()
        if sub.empty:
            continue

        cols, rows = _resolve_pattern_size_for_dir(pattern_size, direction, n_points=len(sub))

        # map pid -> row (x,y,z, dll.)
        by_pid = {int(r["point_id"]): r for _, r in sub.iterrows()}

        # set berisi (r,c) yang eksis
        rc_has = set()
        for pid in sub["point_id"]:
            rc_has.add(_pid_to_rc(pid, cols))

        def get_xyz(pid_):
            row = by_pid.get(int(pid_))
            if row is None: return None
            return float(row["x"]), float(row["y"]), float(row["z"])

        # edges horizontal (kanan) & vertikal (bawah)
        for pid in sub["point_id"]:
            r, c = _pid_to_rc(pid, cols)

            # kanan
            if (r, c+1) in rc_has:
                pid_b = r*cols + (c+1)
                p1 = get_xyz(pid)
                p2 = get_xyz(pid_b)
                if p1 and p2:
                    x1,y1,z1 = p1; x2,y2,z2 = p2
                    length = float(math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2))
                    records.append({
                        "direction": direction, "edge_type": "Horizontal",
                        "rows": r, "columns": c, "pid_cam1": int(pid), "pid_cam2": int(pid_b),
                        "x1": x1, "y1": y1, "z1": z1, "x2": x2, "y2": y2, "z2": z2,
                        "length(mm)": length
                    })

            # bawah
            if (r+1, c) in rc_has:
                pid_b = (r+1)*cols + c
                p1 = get_xyz(pid)
                p2 = get_xyz(pid_b)
                if p1 and p2:
                    x1,y1,z1 = p1; x2,y2,z2 = p2
                    length = float(math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2))
                    records.append({
                        "direction": direction, "edge_type": "Vertical",
                        "rows": r, "columns": c, "pid_cam1": int(pid), "pid_cam2": int(pid_b),
                        "x1": x1, "y1": y1, "z1": z1, "x2": x2, "y2": y2, "z2": z2,
                        "length(mm)": length
                    })

    edges_df = pd.DataFrame.from_records(records)

    # Simpan jika diminta
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        # coba Excel; kalau enggak ada enginenya → CSV
        engine = None
        if importlib.util.find_spec("openpyxl"): engine = "openpyxl"
        elif importlib.util.find_spec("xlsxwriter"): engine = "xlsxwriter"
        if engine:
            edges_xlsx = os.path.join(save_dir, "grid_edges_lengths.xlsx")
            with pd.ExcelWriter(edges_xlsx, engine=engine) as w:
                edges_df.to_excel(w, index=False, sheet_name="edges")
                summary = (edges_df.groupby(["direction","edge_type"])["length(mm)"]
                           .agg(["count","mean","std","min","max"]).reset_index())
                summary.to_excel(w, index=False, sheet_name="summary")
        else:
            edges_df.to_csv(os.path.join(save_dir, "grid_edges_lengths.csv"), index=False)

    return edges_df

def add_grid_edges_traces(fig, edges_df, plane_info=None, lift=0.4):
    """Tambahkan garis grid ke Plotly Figure (1 trace per direction), dengan lift kecil agar tidak z-fight."""
    if edges_df is None or edges_df.empty:
        print("[GRID] edges_df kosong saat menggambar.")
        return

    color_map = {'center':'#111111','north':'#e1ad01','south':'#8b4513',
                 'west':'#1e8e3e','east':'#6a0dad'}

    for direction, sub in edges_df.groupby("direction"):
        xs, ys, zs = [], [], []
        # normal bidang utk lift
        n = None
        if plane_info and direction in plane_info:
            n = np.asarray(plane_info[direction][0], float)
            n = n / (np.linalg.norm(n) + 1e-12)

        for _, r in sub.iterrows():
            p1 = np.array([r["x1"], r["y1"], r["z1"]], dtype=float)
            p2 = np.array([r["x2"], r["y2"], r["z2"]], dtype=float)
            if n is not None:
                p1 = p1 + lift * n
                p2 = p2 + lift * n
            xs += [p1[0], p2[0], None]
            ys += [p1[1], p2[1], None]
            zs += [p1[2], p2[2], None]

        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='lines',
            line=dict(width=4),   # biar tebal
            name=f"Grid {direction}",
            showlegend=True,
            marker=dict(color=color_map.get(direction, None))
        ))


def show_3d_point_2cam_ori_visualization(df_3d: pd.DataFrame, cam_L: np.ndarray, cam_R: np.ndarray, pattern_size=None, grid_res=3):
    # Ekstrak P dan Q point
    df_3d[['px', 'py', 'pz']] = pd.DataFrame(df_3d['point_p'].tolist(), index=df_3d.index)
    df_3d[['qx', 'qy', 'qz']] = pd.DataFrame(df_3d['point_q'].tolist(), index=df_3d.index)

    output_dir = "plugins/moilapp-plugin-opencv/output"
    os.makedirs(output_dir, exist_ok=True)

    # pusat baseline (untuk konsistensi arah normal)
    cam_L = np.asarray(cam_L, dtype=float)
    cam_R = np.asarray(cam_R, dtype=float)
    baseline_center = (cam_L + cam_R) / 2.0


    # Fungsi untuk gambar point + kamera
    def add_common_traces(fig):
        color_map = {
            'center': 'black', 'north': 'yellow', 'south': 'brown',
            'west': 'green', 'east': 'purple'
        }
        for direction in df_3d['direction'].unique():
            sub = df_3d[df_3d['direction'] == direction]
            fig.add_trace(go.Scatter3d(
                x=sub["x"], y=sub["y"], z=sub["z"],
                mode='markers',
                marker=dict(size=3, color=color_map.get(direction, 'gray')),
                text=[
                    f"ID: {pid}<br>Dir: {direction}<br>X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f}<br>Angle: {a:.2f}°<br>Conf: {c:.2f}<br>Ray gap: {d:.2f} mm<br>Mid→Baseline Center: {f:.2f} mm "
                    for pid, x, y, z, a, c, d, f in zip(
                        sub["point_id"], sub["x"], sub["y"], sub["z"],
                        sub["ray_angle_deg"], sub["confidence"], sub["ray_gap"],
                        sub["d_mid_to_center"]

                    )
                ],
                hoverinfo="text",
                name=f"Dir: {direction}"
            ))

        # Kamera
        fig.add_trace(go.Scatter3d(x=[cam_L[0]], y=[cam_L[1]], z=[cam_L[2]],
                                   mode='markers', marker=dict(color='red', size=6), name="Camera L"))
        fig.add_trace(go.Scatter3d(x=[cam_R[0]], y=[cam_R[1]], z=[cam_R[2]],
                                   mode='markers', marker=dict(color='blue', size=6), name="Camera R"))

    def add_plane_per_direction(fig, df_3d, grid_res=3, face_to=None):
        plane_info = {}

        for c in ("x", "y", "z", "direction"):
            if c not in df_3d.columns:
                print(f"[PLANE] Missing column: {c}")
                return plane_info

        for direction, sub in df_3d.groupby("direction"):
            P = sub[["x", "y", "z"]].to_numpy(dtype=np.float64, copy=True)

            finite = np.isfinite(P).all(axis=1)
            P = P[finite]
            n = P.shape[0]
            if n < 3:
                print(f"[PLANE] Skip {direction}: need >=3 valid points, got {n}")
                continue

            mean = P.mean(axis=0)
            Pc = P - mean
            scale = np.linalg.norm(Pc, ord=2)
            if not np.isfinite(scale) or scale == 0.0:
                print(f"[PLANE] Skip {direction}: bad scale")
                continue
            Pc /= scale

            try:
                U, S, Vh = np.linalg.svd(Pc, full_matrices=False)
                normal = Vh[-1, :]
                basis1, basis2 = Vh[0, :], Vh[1, :]
            except np.linalg.LinAlgError:
                print(f"[PLANE] SVD failed for {direction}, fallback to eig.")
                C = np.cov(Pc.T)
                w, V = np.linalg.eigh(C)
                idx = np.argsort(w)
                normal = V[:, idx[0]]
                basis1 = V[:, idx[2]]
                basis2 = V[:, idx[1]]

            nrm = np.linalg.norm(normal)
            if not np.isfinite(nrm) or nrm == 0.0:
                print(f"[PLANE] Skip {direction}: normal not finite")
                continue
            normal = normal / nrm

            # konsistenkan arah normal menghadap baseline_center
            if face_to is not None:
                to_face = np.asarray(face_to, float) - mean
                if np.dot(normal, to_face) < 0:
                    normal = -normal

            plane_info[direction] = (normal, mean)

            # ukuran bidang mengikuti sebaran
            proj1 = (P - mean) @ basis1
            proj2 = (P - mean) @ basis2
            range1 = float(proj1.max() - proj1.min())
            range2 = float(proj2.max() - proj2.min())
            margin = 5.0  # mm
            half1 = range1 / 2.0 + margin
            half2 = range2 / 2.0 + margin

            g = max(3, int(grid_res))
            grid1 = np.linspace(-half1, half1, g)
            grid2 = np.linspace(-half2, half2, g)
            xx, yy = np.meshgrid(grid1, grid2)

            plane_pts = mean + np.outer(xx.ravel(), basis1) + np.outer(yy.ravel(), basis2)
            X = plane_pts[:, 0].reshape(xx.shape)
            Y = plane_pts[:, 1].reshape(xx.shape)
            Z = plane_pts[:, 2].reshape(xx.shape)

            fig.add_trace(go.Surface(
                x=X, y=Y, z=Z, showscale=False, opacity=0.3,
                name=f"Plane {direction}",
                colorscale=[[0, 'gray'], [1, 'gray']],
                hoverinfo='skip'
            ))

        return plane_info


    # Tambahkan teks sudut antar bidang
    def add_plane_angle_labels(fig, plane_info):
        code = {'west': 'w', 'north': 'n', 'center': 'c', 'east': 'e', 'south': 's'}
        opposite_pairs = {frozenset({'north', 'south'}), frozenset({'west', 'east'})}

        angle_map = {}
        labels = []
        x_start = 500
        y_fixed = 500
        z_start = 300
        z_step = -30

        dirs = list(plane_info.keys())
        counter = 0

        for i in range(len(dirs)):
            for j in range(i + 1, len(dirs)):
                d1, d2 = dirs[i], dirs[j]
                n1, _ = plane_info[d1]
                n2, _ = plane_info[d2]

                theta = np.degrees(np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0)))
                if frozenset({d1, d2}) in opposite_pairs:
                    theta = theta

                k1 = f"{code[d1]}{code[d2]}"
                k2 = f"{code[d2]}{code[d1]}"
                angle_map[k1] = theta
                angle_map[k2] = theta

                # label = f"{d1.upper()} vs {d2.upper()} → {theta:.2f}°"
                # fig.add_trace(go.Scatter3d(
                #     x=[x_start], y=[y_fixed], z=[z_start + z_step * counter],
                #     mode="text", text=[label], showlegend=False, name="Plane Angle"
                # ))
                # counter += 1

        return angle_map

    def add_axes_traces(fig, axis_len=300):
        fig.add_trace(go.Scatter3d(
            x=[-axis_len, axis_len], y=[0, 0], z=[0, 0],
            mode='lines', line=dict(color='red', width=4), name='X Axis'
        ))
        fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[-axis_len, axis_len], z=[0, 0],
            mode='lines', line=dict(color='green', width=4), name='Y Axis'
        ))
        fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[0, 0], z=[-axis_len, axis_len],
            mode='lines', line=dict(color='blue', width=4), name='Z Axis'
        ))

    def add_pq_mid_to_figure(fig, df_3d):
        # P (kamera kiri, merah)
        fig.add_trace(go.Scatter3d(
            x=df_3d["point_p"].apply(lambda p: p[0]),
            y=df_3d["point_p"].apply(lambda p: p[1]),
            z=df_3d["point_p"].apply(lambda p: p[2]),
            mode='markers',
            marker=dict(color='red', size=3),
            name='Closest point (Cam L)'
        ))

        # Q (kamera kanan, biru)
        fig.add_trace(go.Scatter3d(
            x=df_3d["point_q"].apply(lambda p: p[0]),
            y=df_3d["point_q"].apply(lambda p: p[1]),
            z=df_3d["point_q"].apply(lambda p: p[2]),
            mode='markers',
            marker=dict(color='blue', size=3),
            name='Closest point (Cam R)'
        ))

        # Mid point (hasil triangulasi, hijau)
        fig.add_trace(go.Scatter3d(
            x=df_3d["x"],  # kolom hasil triangulasi
            y=df_3d["y"],
            z=df_3d["z"],
            mode='markers',
            marker=dict(color='green', size=3),
            name='Mid Point (3D Result)'
        ))

        # Garis koneksi antar P-Q
        for i in range(len(df_3d)):
            p = df_3d.iloc[i]["point_p"]
            q = df_3d.iloc[i]["point_q"]
            fig.add_trace(go.Scatter3d(
                x=[p[0], q[0]],
                y=[p[1], q[1]],
                z=[p[2], q[2]],
                mode='lines',
                line=dict(color='gray', width=1),
                showlegend=(i == 0),  # hanya tampilkan legend sekali
                name='Shortest inter-ray line' if i == 0 else ''
            ))

    def add_mean_distance_labels(fig, df_3d):
        mean_maps = {}
        x_start = 500
        y_fixed = 500
        z_start = 0
        z_step = -30

        grouped = df_3d.groupby("direction")
        counter = 0

        for direction, sub in grouped:
            if len(sub) < 1:
                continue

            mean_dist = sub["ray_gap"].mean()
            mean_maps[direction] = mean_dist
            label = f"{direction.upper()} Mean Dist: {mean_dist:.2f} mm"


            fig.add_trace(go.Scatter3d(
                x=[x_start],
                y=[y_fixed],
                z=[z_start + z_step * counter],
                mode="text",
                text=[label],
                textposition="middle right",
                showlegend=False,
                name="Mean ray_gap"
            ))
            counter += 1
        overall_mean = df_3d["ray_gap"].mean()
        mean_maps["all"] = overall_mean

        fig.add_trace(go.Scatter3d(
            x=[x_start],
            y=[y_fixed],
            z=[z_start + z_step * counter],
            mode="text",
            text=[f"ALL  Mean Dist: {overall_mean:.2f} mm"],
            textposition="middle right",
            showlegend=False,
            name="Mean Distance"
        ))

        return mean_maps

    # def compute_mean_distance_to_plane(df_3d):
    #
    #     """
    #         Hitung 'thickness' dan RMS dev tiap bidang per direction
    #         dengan PCA plane fitting.
    #
    #         Parameters
    #         ----------
    #         df_3d : DataFrame
    #             Harus punya kolom: x, y, z, direction
    #
    #         Returns
    #         -------
    #         dict_thickness : dict
    #             {direction: thickness_mm}
    #         dict_rms : dict
    #             {direction: rms_dev_mm}
    #         """
    #     dict_thickness = {}
    #     dict_rms = {}
    #
    #     # --- per-direction ---
    #     for direction, sub in df_3d.groupby("direction"):
    #         if len(sub) < 3:
    #             continue
    #
    #         pts = sub[["x", "y", "z"]].to_numpy(dtype=float)
    #         _, _, _, res = fit_plane_pca(pts)  # residual orthogonal ke plane
    #         thickness, rms = summarize_residuals(res)
    #
    #         dict_thickness[direction] = float(thickness)
    #         dict_rms[direction] = float(rms)
    #
    #     # --- gabungan semua direction: "all" ---
    #     if len(df_3d) >= 3:
    #         pts_all = df_3d[["x", "y", "z"]].to_numpy(dtype=float)
    #         _, _, _, res_all = fit_plane_pca(pts_all)
    #         t_all, rms_all = summarize_residuals(res_all)
    #         dict_thickness["all"] = float(t_all)
    #         dict_rms["all"] = float(rms_all)
    #
    #     return dict_thickness, dict_rms
    def compute_mean_distance_to_plane(df_3d: pd.DataFrame):
        XYZ_COLS = ["x", "y", "z"]

        thickness_map = {}
        rms_map = {}

        if df_3d is None or df_3d.empty:
            return thickness_map, rms_map

        for direction, sub in df_3d.groupby("direction"):
            P = sub[XYZ_COLS].to_numpy(dtype=float)

            # buang NaN/Inf
            finite = np.isfinite(P).all(axis=1)
            P = P[finite]

            if P.shape[0] < 3:
                # skip saja (atau set np.nan kalau Anda mau ditampilkan)
                thickness_map[direction] = np.nan
                rms_map[direction] = np.nan
                continue

            n, p0, d, res = fit_plane_pca(P)
            if n is None or res.size == 0:
                thickness_map[direction] = np.nan
                rms_map[direction] = np.nan
                continue

            thickness = float(np.max(res) - np.min(res))
            rms = float(np.sqrt(np.mean(res ** 2)))

            thickness_map[direction] = thickness
            rms_map[direction] = rms

        # opsional: agregat all (pakai semua titik valid)
        Pall = df_3d[XYZ_COLS].to_numpy(dtype=float)
        Pall = Pall[np.isfinite(Pall).all(axis=1)]
        if Pall.shape[0] >= 3:
            n, p0, d, res = fit_plane_pca(Pall)
            if n is not None and res.size:
                thickness_map["all"] = float(np.max(res) - np.min(res))
                rms_map["all"] = float(np.sqrt(np.mean(res ** 2)))
            else:
                thickness_map["all"] = np.nan
                rms_map["all"] = np.nan
        else:
            thickness_map["all"] = np.nan
            rms_map["all"] = np.nan

        return thickness_map, rms_map
 
    def add_distance_line_to_plane(fig, df_3d, plane_info):

        all_dists = []

        # Step 1: Hitung semua jarak dulu untuk normalisasi warna
        for i, row in df_3d.iterrows():
            direction = row["direction"]
            if direction not in plane_info:
                continue
            pt = np.array([row["x"], row["y"], row["z"]])
            normal, center = plane_info[direction]
            dist = np.abs((pt - center) @ normal)
            all_dists.append(dist)

        if not all_dists:
            return

        max_dist = max(all_dists)
        norm = mcolors.Normalize(vmin=0, vmax=max_dist)
        colormap = cm.get_cmap("Reds")

        # Step 2: Tambahkan garis per titik
        for i, row in df_3d.iterrows():
            direction = row["direction"]
            if direction not in plane_info:
                continue

            pt = np.array([row["x"], row["y"], row["z"]])
            normal, center = plane_info[direction]
            vec = pt - center
            dist_to_plane = (vec @ normal)
            projected = pt - dist_to_plane * normal

            # Warna berdasarkan jarak
            rgba = colormap(norm(abs(dist_to_plane)))
            rgb = tuple(int(c * 255) for c in rgba[:3])
            hex_color = '#%02x%02x%02x' % rgb

            fig.add_trace(go.Scatter3d(
                x=[pt[0], projected[0]],
                y=[pt[1], projected[1]],
                z=[pt[2], projected[2]],
                mode='lines',
                line=dict(color=hex_color, width=2),
                showlegend=False,
                hoverinfo='skip',
            ))



    # ========== Build figures ==========
    fig_basic = go.Figure()
    fig_full = go.Figure()

    add_common_traces(fig_basic)
    add_common_traces(fig_full)

    plane_info_basic = add_plane_per_direction(fig_basic, df_3d, grid_res=grid_res, face_to=baseline_center)
    plane_info_full = add_plane_per_direction(fig_full, df_3d, grid_res=grid_res, face_to=baseline_center)

    edges_df = None
    # >>> TAMBAHKAN BLOK INI <<<
    if pattern_size is not None:
        try:
            df_for_edges = df_3d.copy()
            # pastikan point_id numerik
            df_for_edges["point_id"] = df_for_edges["point_id"].astype(int)

            edges_df = compute_grid_edges_3d(df_for_edges, pattern_size, save_dir=output_dir)
            print("[GRID] edges_df shape:", edges_df.shape)


            if edges_df.empty:
                print("[GRID] Tidak ada edge yang terbentuk: cek pattern_size dan penomoran point_id")

            else:
                print("[GRID] ringkas:")
                print(edges_df.groupby(["direction", "edge_type"]).size().rename("count"))
        except Exception as e:
            print(f"[GRID] Gagal membuat/menggambar grid edges: {e}")

    if edges_df is not None and not edges_df.empty:
        add_grid_edges_traces(fig_full, edges_df, plane_info=plane_info_full, lift=0.4)
        add_grid_edges_traces(fig_basic, edges_df, plane_info=plane_info_basic, lift=0.4)

    # sudut (pakai plane_info_basic sebagai sumber peta sudut)
    angle_map_basic = add_plane_angle_labels(fig_basic, plane_info_basic)
    add_plane_angle_labels(fig_full, plane_info_full)

    thickness_map, rms_map = compute_mean_distance_to_plane(df_3d)

    add_axes_traces(fig_basic)
    add_axes_traces(fig_full)

    add_pq_mid_to_figure(fig_full, df_3d)

    mean_maps = add_mean_distance_labels(fig_full, df_3d)

    # garis jarak → bidang (sekali saja per figure)
    add_distance_line_to_plane(fig_full, df_3d, plane_info_full)
    add_distance_line_to_plane(fig_basic, df_3d, plane_info_basic)

    fig_basic.update_layout(
        scene=dict(xaxis=dict(title='X'), yaxis=dict(title='Y'), zaxis=dict(title='Z'), aspectmode='data'),
        title="3D Mid Points",
        margin=dict(l=10, r=10, t=40, b=10)
    )

    fig_full.update_layout(
        scene=dict(xaxis=dict(title='X'), yaxis=dict(title='Y'), zaxis=dict(title='Z'), aspectmode='data'),
        title="3D Points with Closest points on rays",
        margin=dict(l=10, r=10, t=40, b=10)
    )

    # Simpan HTML
    # path_basic = os.path.join(output_dir, "3d_points_only.html")
    # pyo.plot(fig_basic, filename=path_basic, auto_open=False)
    path_basic = os.path.join(output_dir, "3d_points_only.html")
    path_full = os.path.join(output_dir, "3d_points_with_pq.html")
    pyo.plot(fig_basic, filename=path_basic, auto_open=False)
    pyo.plot(fig_full, filename=path_full, auto_open=False)


    # --- Export plane summary (centroid, origin distance, thickness/RMS, residuals) ---
    try:
        export_plane_summary_to_excel(df_3d, output_dir, filename="summary_results.xlsx")
    except Exception as e:
        print(f"[PLANE_SUM] Skip export: {e}")

    return path_basic, path_full, angle_map_basic, mean_maps, thickness_map



def compare_reprojection_with_original_opencv(
    df_3d: pd.DataFrame,
    cam_L: np.ndarray,
    cam_R: np.ndarray,
    K_L: np.ndarray,
    D_L: np.ndarray,
    K_R: np.ndarray,
    D_R: np.ndarray,
    gt_csv_L: str | None = None,
    gt_csv_R: str | None = None,
    output_dir: str = "plugins/moilapp-plugin-opencv/output",
):
    '''
    OpenCV-only reprojection compare (no Moil).

    Assumption (current pipeline):
      - world axes aligned with camera axes (R_wc = I) for both cameras.
      - cam_L, cam_R are camera centers in WORLD coordinates.

    df_3d must contain: direction, point_id, x, y, z
    '''
    os.makedirs(output_dir, exist_ok=True)

    def _project(P_w: np.ndarray, C_w: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
        R_cw = np.eye(3, dtype=np.float64)
        rvec, _ = cv2.Rodrigues(R_cw)
        tvec = (-R_cw @ np.asarray(C_w, dtype=np.float64).reshape(3, 1)).reshape(3, 1)
        Pw = np.asarray(P_w, dtype=np.float64).reshape(-1, 1, 3)
        D = np.asarray(D, dtype=np.float64).reshape(-1)[:4]
        uv, _ = cv2.fisheye.projectPoints(Pw, rvec, tvec, np.asarray(K, np.float64), D)
        return uv.reshape(-1, 2)

    P = df_3d[["x", "y", "z"]].to_numpy(dtype=float)
    uv_L = _project(P, cam_L, K_L, D_L)
    uv_R = _project(P, cam_R, K_R, D_R)

    df_proj_L = pd.DataFrame(uv_L, columns=["u_L", "v_L"])
    df_proj_R = pd.DataFrame(uv_R, columns=["u_R", "v_R"])
    df_all = pd.concat([df_3d.reset_index(drop=True), df_proj_L, df_proj_R], axis=1)

    def _load_gt(path: str | None, prefix: str):
        if path is None or (not os.path.exists(path)):
            return None
        dfg = pd.read_csv(path)
        if "x_fish" in dfg.columns and "y_fish" in dfg.columns:
            dfg = dfg.rename(columns={"x_fish": f"u_{prefix}_gt", "y_fish": f"v_{prefix}_gt"})
        if "u" in dfg.columns and "v" in dfg.columns:
            dfg = dfg.rename(columns={"u": f"u_{prefix}_gt", "v": f"v_{prefix}_gt"})
        need = ["direction", "point_id", f"u_{prefix}_gt", f"v_{prefix}_gt"]
        for c in need:
            if c not in dfg.columns:
                raise ValueError(f"[GT] Missing column '{c}' in {path}")
        return dfg[need].drop_duplicates(subset=["direction", "point_id"])

    df_gt_L = _load_gt(gt_csv_L, "L")
    df_gt_R = _load_gt(gt_csv_R, "R")

    if df_gt_L is not None:
        df_all = df_all.merge(df_gt_L, on=["direction", "point_id"], how="left")
        df_all["error_L"] = ((df_all["u_L"] - df_all["u_L_gt"])**2 + (df_all["v_L"] - df_all["v_L_gt"])**2)**0.5

    if df_gt_R is not None:
        df_all = df_all.merge(df_gt_R, on=["direction", "point_id"], how="left")
        df_all["error_R"] = ((df_all["u_R"] - df_all["u_R_gt"])**2 + (df_all["v_R"] - df_all["v_R_gt"])**2)**0.5

    df_left_save = df_all[["direction", "point_id", "u_L", "v_L"] + (["u_L_gt", "v_L_gt", "error_L"] if df_gt_L is not None else [])] \
        .sort_values(by=["direction", "point_id"])
    df_right_save = df_all[["direction", "point_id", "u_R", "v_R"] + (["u_R_gt", "v_R_gt", "error_R"] if df_gt_R is not None else [])] \
        .sort_values(by=["direction", "point_id"])

    left_path = os.path.join(output_dir, "reprojection_compare_left_opencv.csv")
    right_path = os.path.join(output_dir, "reprojection_compare_right_opencv.csv")
    df_left_save.to_csv(left_path, index=False)
    df_right_save.to_csv(right_path, index=False)

    print("[SAVE] OpenCV reprojection compare saved:")
    print(" -", left_path)
    print(" -", right_path)

    return df_left_save, df_right_save


# def get_plane_basis_map(df_3d):
#     plane_map = {}
#     for direction in df_3d["direction"].unique():
#         sub = df_3d[df_3d["direction"] == direction]
#         if len(sub) < 3:
#             continue
#         pts = sub[["x", "y", "z"]].values
#         center = pts.mean(axis=0)
#
#         # Basis vector seperti di visualisasi
#         if direction == "center":
#             basis1 = np.array([1, 0, 0])
#             basis2 = np.array([0, 1, 0])
#         elif direction in ["north", "south"]:
#             basis1 = np.array([1, 0, 0])
#             basis2 = np.array([0, 0, 1])
#         elif direction in ["east", "west"]:
#             basis1 = np.array([0, 1, 0])
#             basis2 = np.array([0, 0, 1])
#         else:
#             pts_centered = pts - center
#             _, _, vh = np.linalg.svd(pts_centered)
#             basis1 = vh[0]
#             basis2 = vh[1]
#
#         normal = np.cross(basis1, basis2)
#         normal = normal / np.linalg.norm(normal)
#
#         plane_map[direction] = {
#             "point": center,
#             "normal": normal,
#             "basis1": basis1,
#             "basis2": basis2
#         }
#     return plane_map


