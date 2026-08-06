"""Before/after of the finger-tip jitter."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = r"C:\isaac_soft\sweeps\finger_jitter_fix.png"
MM = 1e3

runs = [
    ("Force-saturated drive (before)", "strawberry.npz", "#dc2626"),
    ("Position-hold grasp (after)", "strawberry_poshold.npz", "#059669"),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    "Finger-tip jitter: a saturated force drive has no stable equilibrium against a springy fruit",
    fontsize=13,
    fontweight="bold",
)

for ax, (label, f, col) in zip(axes, runs):
    d = np.load(rf"C:\isaac_soft\sweeps\{f}")
    g = d["gaps"] * MM
    tail = g[-200:]
    ax.plot(np.arange(len(g)), g, color=col, lw=1.1)
    ax.set_title(
        f"{label}\ntail p2p {tail.max() - tail.min():.2f} mm · "
        f"jerk {np.abs(np.diff(g)).mean():.3f} mm/step",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("squeeze step")
    ax.set_ylabel("finger gap [mm]")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 90)

fig.tight_layout(rect=[0, 0.01, 1, 0.91])
fig.savefig(OUT, dpi=155)
print("wrote", OUT)
