# import pandas as pd
# import numpy as np
#
# # ==== 1. BACA DATA & URUTKAN ====
# df = pd.read_csv("D:/moilapp_perseverance/src/plugins/moilapp-plugin-opencv/results/etaniya_vr220_home_z+70_und_points.csv")
#
#
# # urutkan berdasarkan point_id dulu
# df = df.sort_values("point_id").reset_index(drop=True)
#
# # ==== 2. SET UKURAN GRID ====
# rows = 13   # ganti sesuai jumlah baris corner
# cols = 9  # ganti sesuai jumlah kolom corner
#
# assert rows * cols == len(df), "rows*cols tidak sama dengan jumlah titik!"
#
# coords = df[["u_und", "v_und"]].values.reshape(rows, cols, 2)
#
# # ==== 3. JARAK HORIZONTAL (ke kanan) ====
# # shape (rows, cols-1)
# dh = np.linalg.norm(coords[:, 1:, :] - coords[:, :-1, :], axis=2)
#
# print("Horizontal distance (right neighbor) in pixels:")
# print("  min  =", dh.min())
# print("  max  =", dh.max())
# print("  mean =", dh.mean())
# print("  std  =", dh.std())
#
# # ==== 4. JARAK VERTIKAL (ke bawah) ====
# # shape (rows-1, cols)
# dv = np.linalg.norm(coords[1:, :, :] - coords[:-1, :, :], axis=2)
#
# print("\nVertical distance (down neighbor) in pixels:")
# print("  min  =", dv.min())
# print("  max  =", dv.max())
# print("  mean =", dv.mean())
# print("  std  =", dv.std())
#
# # ==== 5. OPSIONAL: MASUKKAN KE DATAFRAME UNTUK DILIHAT DI EXCEL ====
#
# dist_right = np.full(rows * cols, np.nan)
# dist_down  = np.full(rows * cols, np.nan)
#
# # isi dist_right (untuk semua kecuali kolom terakhir)
# for r in range(rows):
#     for c in range(cols - 1):
#         idx = r * cols + c
#         dist_right[idx] = dh[r, c]
#
# # isi dist_down (untuk semua kecuali baris terakhir)
# for r in range(rows - 1):
#     for c in range(cols):
#         idx = r * cols + c
#         dist_down[idx] = dv[r, c]
#
# df["dist_right"] = dist_right
# df["dist_down"]  = dist_down
#
# df.to_excel("points_with_grid_distances.xlsx", index=False)
# # atau:
# # df.to_csv("points_with_grid_distances.csv", index=False)
