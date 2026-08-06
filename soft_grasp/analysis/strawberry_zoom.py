"""Zoomed before/during/after crops of the strawberry so the squeeze is visible."""

import matplotlib

matplotlib.use("Agg")
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np

SRC = r"C:\isaac_soft\videos\strawberry_v4\grasp_strawberry-step-0.mp4"
OUT = r"C:\isaac_soft\sweeps\strawberry_zoom.png"

frames = np.stack([f for f in iio.imiter(SRC)])
print("frames", frames.shape)

# locate the fruit by colour: warm tan, clearly separable from the yellow table and white robot
ref = frames[275].astype(int)
r, g, b = ref[..., 0], ref[..., 1], ref[..., 2]
mask = (r > 150) & (r < 245) & (g > 110) & (g < 200) & (b < 140) & ((r - b) > 60) & ((g - b) > 30)
ys, xs = np.nonzero(mask)
if len(xs) > 200:
    cx, cy = int(xs.mean()), int(ys.mean())
else:  # fall back to the lower-centre of frame
    cy, cx = frames.shape[1] * 7 // 10, frames.shape[2] // 2
print(f"fruit centre ~ ({cx}, {cy})")

half_w, half_h = 210, 150
x0, x1 = max(0, cx - half_w), min(frames.shape[2], cx + half_w)
y0, y1 = max(0, cy - half_h), min(frames.shape[1], cy + half_h)

# pregrasp 130 + settle 140 = squeeze starts at frame 270; peak compression was squeeze step ~80
picks = [(272, "before squeeze\n(fingers just closing)"), (350, "peak squeeze\n(max compression)"), (669, "settled hold")]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
fig.suptitle(
    "Deformable strawberry under a 10 N constant grip — Newton MJWarp + VBD (E = 200 kPa)",
    fontsize=13,
    fontweight="bold",
)
for ax, (idx, lab) in zip(axes, picks):
    ax.imshow(frames[min(idx, len(frames) - 1)][y0:y1, x0:x1])
    ax.set_title(lab, fontsize=11)
    ax.axis("off")

fig.tight_layout(rect=[0, 0.01, 1, 0.9])
fig.savefig(OUT, dpi=155)
print("wrote", OUT)
