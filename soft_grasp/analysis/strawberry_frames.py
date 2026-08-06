"""Pull frames out of the recorded grasp and lay them out as a strip."""

import matplotlib

matplotlib.use("Agg")
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np

SRC = r"C:\isaac_soft\videos\strawberry_v4\grasp_strawberry-step-0.mp4"
OUT = r"C:\isaac_soft\sweeps\strawberry_render_strip_v3.png"

frames = np.stack([f for f in iio.imiter(SRC)])
n = len(frames)
print(f"frames={n} size={frames.shape[1]}x{frames.shape[2]}")

# phases: pregrasp 130, settle 140, squeeze 400  (video starts at env step 0)
picks = [0, 130, 200, 270, 290, 340, 480, n - 1]
labels = [
    "reset",
    "above fruit",
    "descending",
    "at grasp pose",
    "contact",
    "squeezing",
    "held",
    "settled",
]
picks = [min(p, n - 1) for p in picks]

fig, axes = plt.subplots(2, 4, figsize=(16, 7.2))
fig.suptitle(
    "Newton render — Franka grasping the deformable YCB strawberry (E = 200 kPa, 10 N constant grip)",
    fontsize=13,
    fontweight="bold",
)
for ax, idx, lab in zip(axes.ravel(), picks, labels):
    ax.imshow(frames[idx])
    ax.set_title(f"{lab}  (frame {idx})", fontsize=10)
    ax.axis("off")

fig.tight_layout(rect=[0, 0.01, 1, 0.94])
fig.savefig(OUT, dpi=140)
print("wrote", OUT)

# also dump a clean close-up of the most-squeezed frame
mid = picks[5]
iio.imwrite(r"C:\isaac_soft\sweeps\strawberry_squeeze_frame.png", frames[mid])
print("wrote squeeze frame", mid)
