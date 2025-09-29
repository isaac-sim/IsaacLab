import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
from matplotlib.ticker import ScalarFormatter

plt.style.use(['science'])
SMALL_SIZE  = 28
MEDIUM_SIZE = 32
LEGEND_SIZE = 23
BIGGER_SIZE = 36

plt.rc('font', size=25)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=LEGEND_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc("lines", linewidth=2)

MACHINE_NBR_GPUS = "RTX PRO 6000"

NBR_GPUS_COLORS = {  # RTX PRO 6000
    "Single-GPU": "#d08770",
    "2-GPU": "#ebcb8b",
    "4-GPU": "#a3be8c",
    "8-GPU": "#b48ead"
}

MACHINE_NBR_GPUS_COLORS = {  # L40 + 5090
    "L40": "#5e81ac",
    "5090": "#a3be8c"
}

LEGEND_BELOW_FIGURE = False
TASK_WITH_LEGEND = "Isaac-Velocity-Rough-Digit-v0"
TASKS = ["Isaac-Velocity-Rough-G1-v0", "Isaac-Velocity-Rough-Digit-v0"]  # Isaac-Factory-GearMesh-Direct-v0
# TASKS = ["Isaac-Dexsuite-Kuka-Allegro-Lift-v0", "Isaac-Open-Drawer-Franka-v0"]

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
file_path = "/home/pascalr/Downloads/Isaac Lab Whitepaper Benchmarks - benchmark result.csv"
raw = pd.read_csv(file_path, header=None)

# 2) Locate the "Mean" and "Std" block boundaries in row 0
row0 = raw.iloc[0].astype(str)
mean_start = row0[row0.str.contains("Mean", na=False)].index[0]
std_start = row0[row0.str.contains("Std", na=False)].index[0]
end_col = raw.shape[1]

# Helper to parse a block (Mean or Std) into tidy format
def parse_block(start_col, end_col, value_name):
    gpu_headers = raw.iloc[1]   # GPU names line (e.g., "Single-GPU", "2-GPU")
    env_headers = raw.iloc[2]   # Environment counts line (1024..16384)
    data = raw.iloc[3:].reset_index(drop=True)
    data = data.rename(columns={0: "Machine", 1: "Task"})

    current_gpu = None
    records = []
    for col in range(start_col, end_col):
        gpu = gpu_headers[col]
        env = env_headers[col]

        # Update GPU group when encountered
        if isinstance(gpu, str) and "GPU" in gpu:
            current_gpu = gpu

        if current_gpu is None:
            continue

        # Parse environment count
        try:
            env_val = int(float(env))
        except Exception:
            continue

        # Collect values per row (task/machine)
        for i in range(len(data)):
            v = data.iloc[i, col]
            if pd.isna(v):
                continue
            records.append({
                "Machine": data.loc[i, "Machine"],
                "Task": data.loc[i, "Task"],
                "GPU_Config": current_gpu,
                "Envs": env_val,
                value_name: float(v)
            })
    return pd.DataFrame(records)

# 3) Parse mean and std blocks
mean_df = parse_block(mean_start, std_start, "Mean")
std_df  = parse_block(std_start, end_col, "Std")

# 4) Merge into one tidy dataframe
tidy = pd.merge(
    mean_df,
    std_df,
    on=["Machine", "Task", "GPU_Config", "Envs"],
    how="left"
)

# 5) Ensure numeric types
tidy["Envs"] = pd.to_numeric(tidy["Envs"], errors="coerce").astype("Int64")
tidy["Mean"] = pd.to_numeric(tidy["Mean"], errors="coerce")
tidy["Std"]  = pd.to_numeric(tidy["Std"], errors="coerce")
tidy = tidy.dropna(subset=["Envs", "Mean"])

# 6) Plot settings
tasks = tidy["Task"].dropna().unique()
machines = tidy["Machine"].dropna().unique()

fig, axes = plt.subplots(1, len(TASKS), figsize=(12 * len(TASKS), 8), sharey=True)

if LEGEND_BELOW_FIGURE:
    all_handles = []
    all_labels = []

for i, task in enumerate(TASKS):
    ax = axes[i]
    sub = tidy[tidy["Task"] == task]

    x_ticks = set()

    # --- Plot L40 + L40S (Single-GPU only) ---
    for machine, color in MACHINE_NBR_GPUS_COLORS.items():
        g = sub[(sub["Machine"] == machine) & (sub["GPU_Config"] == "Single-GPU")].sort_values("Envs")
        if not g.empty:
            x = g["Envs"].astype(int)
            y = g["Mean"]
            yerr = g["Std"].fillna(0)
            line, =ax.plot(x, y, label=f"{machine} - Single-GPU", linestyle="dashdot", marker="o", color=color)
            ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.2)

            if LEGEND_BELOW_FIGURE and i == 0:
                all_handles.append(line)
                all_labels.append(f"{machine} - Single-GPU")
            x_ticks.update(x)

    # --- Plot RTX PRO 6000 (all GPU configs) ---
    for gpu, color in NBR_GPUS_COLORS.items():
        g = sub[(sub["Machine"] == "RTX PRO 6000") & (sub["GPU_Config"] == gpu)].sort_values("Envs")
        if not g.empty:
            x = g["Envs"].astype(int)
            y = g["Mean"]
            yerr = g["Std"].fillna(0)
            line, =ax.plot(x, y, label=f"RTX PRO 6000 - {gpu}", color=NBR_GPUS_COLORS[gpu], marker="o")
            ax.fill_between(x, y - yerr, y + yerr, color=NBR_GPUS_COLORS[gpu], alpha=0.2)

            if LEGEND_BELOW_FIGURE and i == 0:
                all_handles.append(line)
                all_labels.append(f"RTX PRO 6000 - {gpu}")

            x_ticks.update(x)

    # Axes styling
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(list(x_ticks), [str(x) for x in x_ticks])
    if task == TASK_WITH_LEGEND and not LEGEND_BELOW_FIGURE:
        ax.legend(
            loc="upper right",
            frameon=True,                 # draw a box around the legend
            facecolor='white',            # white background
            edgecolor='black',            # optional: black border
            framealpha=0.8,
            ncol=2               # transparency of the box (0.0 to 1.0)
        )
        

    _format_axes(ax)

axes[0].set_ylabel("FPS")
axes[0].set_xlabel("Number of Environments")
axes[1].set_xlabel("Number of Environments")

# --- Put legend below the plots ---
if LEGEND_BELOW_FIGURE:
    fig.legend(
        all_handles,
        all_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.10),   # place below figure
        ncol=3,                        # adjust number of columns
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=0.8
    )

    plt.subplots_adjust(wspace=0.01, bottom=0.2)  # leave room at bottom for legend
    plt.tight_layout(pad=2.0)
else:
    plt.subplots_adjust(wspace=0.01)
plt.tight_layout(pad=2.0)
plt.savefig(f"benchmark_gpu_model_nbr_comparison.png", dpi=600)
plt.close()
