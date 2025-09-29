import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
from matplotlib.ticker import ScalarFormatter

plt.style.use(['science'])
SMALL_SIZE  = 24
MEDIUM_SIZE = 28
LEGEND_SIZE = 20
BIGGER_SIZE = 32

plt.rc('font', size=20)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=LEGEND_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc("lines", linewidth=2)

RESOLUTION_COLORS = {
    (64, 64): "#d08770",
    (120, 160): "#ebcb8b",
    (240, 320): "#a3be8c",
    (480, 640): "#b48ead"
}

ADD_MEM_TEST = False


def _format_axes(ax: plt.Axes, grid: bool = True):
    """Apply consistent, publication-ready formatting to axes."""
    if grid:
        ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.6)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.5)
    # Use scalar formatter without offset/scientific on axes unless necessary
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

# 1) Load CSV (no headers so we can custom-parse)
file_path = "/home/pascalr/Downloads/Isaac Lab Whitepaper Benchmarks - camera benchmark run3.csv"
raw = pd.read_csv(file_path)

# Compute FPS
df = raw.copy()
df["FPS"] = (1.0 / df["per_step_ms"]) * df["num_envs"] * 1000.0

# drop nan values
df = df.dropna()

# Parse resolution strings "(240, 320)" -> (240, 320)
df["resolution"] = df["resolution"].str.strip("()").str.split(",").apply(lambda x: tuple(map(int, x)))

data_types = df["data_types"].unique()
modes = df["mode"].unique()

fig, axes = plt.subplots(len(data_types), len(modes), figsize=(6 * len(modes), 12), sharey=True, sharex=False)

if len(modes) == 1:
    axes = axes[None, :]
    
for idx, dtype in enumerate(data_types):
    sub_dtype = df[df["data_types"] == dtype]
    
    for jdx, mode in enumerate(modes):
        ax = axes[idx, jdx]
        sub_mode = sub_dtype[sub_dtype["mode"] == mode]

        for res, g in sub_mode.groupby("resolution"):
            g = g.sort_values("num_envs")
            if g.empty:
                continue
            x = g["num_envs"]
            y = g["FPS"]
            mem = g["avg_memory"]

            color = RESOLUTION_COLORS.get(res, "gray")
            ax.plot(x, y, marker="o", label=f"{res}", color=color)
            if ADD_MEM_TEST:
                for xi, yi, mi in zip(x, y, mem):
                    ax.text(xi, yi * 1.05, f"{mi:.0f}", ha="center", va="bottom", fontsize=8)

        if idx == 0:
            ax.set_title(f"{mode.replace('_', ' ')}", weight="bold")
        if jdx == 0:
            ax.set_ylabel("FPS")
        if idx == len(data_types) - 1 and jdx == 1:
            ax.set_xlabel("Number of Environments")
        
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        _format_axes(ax)

axes[0, 0].legend(loc="upper right")

plt.tight_layout()
plt.savefig(f"camera_benchmarks.png", dpi=600)
plt.show()