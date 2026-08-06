"""Visualise the strawberry grasp: deformed shape, bounding extents, and gripper trace."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC = r"C:\isaac_soft\sweeps\strawberry.npz"
OUT = r"C:\isaac_soft\sweeps\strawberry_grasp.png"
MM = 1e3

d = np.load(SRC)
rest = d["rest"]
frames = d["frames"]
steps = d["frame_steps"]
gaps = d["gaps"]
efforts = d["efforts"]
ext = d["extents"]
rest_ext = d["rest_extents"]

# frame with the largest y-compression = the most deformed state
squeeze_idx = int(np.argmin([f[:, 1].max() - f[:, 1].min() for f in frames]))
squeezed = frames[squeeze_idx]

com_r = rest.mean(axis=0)
com_s = squeezed.mean(axis=0)

fig = plt.figure(figsize=(14, 9))
fig.suptitle(
    "Grasping a deformable strawberry (YCB 012_strawberry) — Newton MJWarp + VBD\n"
    f"E = {float(d['youngs']):.0f} Pa · constant grip force {float(d['force']):.0f} N · "
    f"{rest.shape[0]} tet nodes · fingers close along y",
    fontsize=13,
    fontweight="bold",
)

C_REST, C_SQ = "#94a3b8", "#dc2626"

# ---- (a) front view: y-z, the squeeze plane
ax = fig.add_subplot(2, 3, 1)
ax.scatter((rest[:, 1] - com_r[1]) * MM, (rest[:, 2] - com_r[2]) * MM, s=14, c=C_REST, label="rest", alpha=0.85)
ax.scatter(
    (squeezed[:, 1] - com_s[1]) * MM, (squeezed[:, 2] - com_s[2]) * MM, s=14, c=C_SQ, label="squeezed", alpha=0.85
)
ax.set_xlabel("y [mm]  (finger axis)")
ax.set_ylabel("z [mm]")
ax.set_title("(a) Front view — squeezed in y", fontsize=11, fontweight="bold")
ax.set_aspect("equal")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# ---- (b) top view: x-y
ax = fig.add_subplot(2, 3, 2)
ax.scatter((rest[:, 0] - com_r[0]) * MM, (rest[:, 1] - com_r[1]) * MM, s=14, c=C_REST, alpha=0.85)
ax.scatter((squeezed[:, 0] - com_s[0]) * MM, (squeezed[:, 1] - com_s[1]) * MM, s=14, c=C_SQ, alpha=0.85)
ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]  (finger axis)")
ax.set_title("(b) Top view", fontsize=11, fontweight="bold")
ax.set_aspect("equal")
ax.grid(alpha=0.3)

# ---- (c) 3D
ax = fig.add_subplot(2, 3, 3, projection="3d")
ax.scatter(
    (rest[:, 0] - com_r[0]) * MM, (rest[:, 1] - com_r[1]) * MM, (rest[:, 2] - com_r[2]) * MM, s=8, c=C_REST, alpha=0.35
)
ax.scatter(
    (squeezed[:, 0] - com_s[0]) * MM,
    (squeezed[:, 1] - com_s[1]) * MM,
    (squeezed[:, 2] - com_s[2]) * MM,
    s=8,
    c=C_SQ,
    alpha=0.7,
)
ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")
ax.set_zlabel("z [mm]")
ax.set_title("(c) Tet nodes, rest vs squeezed", fontsize=11, fontweight="bold")

# ---- (d) extents over time
ax = fig.add_subplot(2, 1, 2)
n = np.arange(len(ext))
for i, (lab, col) in enumerate((("x (free)", "#0891b2"), ("y (squeezed)", "#dc2626"), ("z (free)", "#059669"))):
    ax.plot(n, ext[:, i] * MM, color=col, lw=2, label=lab)
    ax.axhline(rest_ext[i] * MM, color=col, ls=":", lw=1.2, alpha=0.7)
ax.plot(n, gaps * MM, color="#7c3aed", lw=1.6, ls="--", label="finger gap")
ax.set_xlabel("squeeze step")
ax.set_ylabel("size [mm]")
ax.set_title(
    "(d) Bounding extents during the squeeze — dotted lines are the undeformed values", fontsize=11, fontweight="bold"
)
ax.legend(fontsize=9, ncol=4)
ax.grid(alpha=0.3)

final = ext[-1]
ax.annotate(
    f"squeeze in y: {(rest_ext[1] - final[1]) * MM:+.2f} mm    "
    f"bulge in x: {(final[0] - rest_ext[0]) * MM:+.2f} mm    "
    f"bulge in z: {(final[2] - rest_ext[2]) * MM:+.2f} mm    "
    f"applied force: {efforts[-1]:.1f} N",
    xy=(0.5, 0.03),
    xycoords="axes fraction",
    ha="center",
    fontsize=10,
    bbox=dict(boxstyle="round,pad=0.45", fc="#fef9c3", ec="#ca8a04"),
)

fig.tight_layout(rect=[0, 0.01, 1, 0.92])
fig.savefig(OUT, dpi=155)
print("wrote", OUT)
print(f"rest extents  [mm]: {np.round(rest_ext * MM, 2)}")
print(f"final extents [mm]: {np.round(final * MM, 2)}")
print(f"most-deformed frame: step {steps[squeeze_idx]}, y = {np.ptp(squeezed[:, 1]) * MM:.2f} mm")
