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

for dtype in data_types:
    sub_dtype = df[df["data_types"] == dtype]
    modes = sub_dtype["mode"].unique()

    fig, axes = plt.subplots(1, len(modes), figsize=(6 * len(modes), 5), sharey=True)
    if len(modes) == 1:
        axes = [axes]

    for ax, mode in zip(axes, modes):
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

        ax.set_title(f"{mode.replace('_', ' ')}", weight="bold")
        ax.set_xlabel("Number of Environments")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        _format_axes(ax)

    axes[0].set_ylabel("FPS")
    axes[0].legend(loc="upper left")

    plt.tight_layout()
    dtype_name = dtype.replace("['", "").replace("']", "")
    plt.savefig(f"camera_benchmarks_{dtype_name}.pdf", dpi=600)
    plt.show()