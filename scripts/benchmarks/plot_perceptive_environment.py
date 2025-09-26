import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

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

MACHINE_NBR_GPUS = "RTX PRO 6000"

NBR_GPUS_COLORS = {
    "Single-GPU": "#d08770",
    "2-GPU": "#ebcb8b",
    "4-GPU": "#a3be8c",
    "8-GPU": "#b48ead"
}

MACHINE_NBR_GPUS_COLORS = {
    "L40": "#5e81ac",
    "5090": "#a3be8c",
    "RTX PRO 6000": "#d08770"
}

COMPARISON_ENVIRONMENTS = [
    {"tiled": "Isaac-Dexsuite-Kuka-Allegro-Lift-Depth-TiledCamera-v0", "raycaster": "Isaac-Dexsuite-Kuka-Allegro-Lift-Depth-RayCasterCamera-v0"},
    {"raycaster": "Isaac-Navigation-Flat-Anymal-C-v0"},
]
COMPARISON_ENVIRONMENTS_NAME = ["dexsuite", "navigation"]

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

for i, comparison_environment in enumerate(COMPARISON_ENVIRONMENTS):
    fig, axs = plt.subplots(1, 2, figsize=(16, 10), sharey=False)
    
    x_ticks_nbr_gpus = set()
    x_ticks_machine = set()

    for perception_mode, style in [("tiled", "dashed"), ("raycaster", "dotted")]:
        if perception_mode not in comparison_environment:
            continue
        
        sub = tidy[tidy["Task"] == comparison_environment[perception_mode]]

        for machine, color in MACHINE_NBR_GPUS_COLORS.items():
            g = sub[(sub["Machine"] == machine) & (sub["GPU_Config"] == "Single-GPU")].sort_values("Envs")
            if not g.empty:
                x = g["Envs"].astype(int)
                y = g["Mean"]
                yerr = g["Std"].fillna(0)
                axs[0].plot(x, y, label=f"{machine} - {perception_mode}", linestyle=style, marker="o", color=color)
                axs[0].fill_between(x, y - yerr, y + yerr, color=color, alpha=0.2)
                x_ticks_machine.update(x)

        for gpu, color in NBR_GPUS_COLORS.items():
            g = sub[(sub["Machine"] == "RTX PRO 6000") & (sub["GPU_Config"] == gpu)].sort_values("Envs")
            if not g.empty:
                x = g["Envs"].astype(int)
                y = g["Mean"]
                yerr = g["Std"].fillna(0)
                axs[1].plot(x, y, label=f"{gpu} - {perception_mode}", linestyle=style, color=NBR_GPUS_COLORS[gpu], marker="o")
                axs[1].fill_between(x, y - yerr, y + yerr, color=NBR_GPUS_COLORS[gpu], alpha=0.2)
                x_ticks_nbr_gpus.update(x)

    # Axes styling
    axs[0].set_xscale("log", base=2)
    axs[0].set_yscale("log")
    axs[0].set_xticks(list(x_ticks_machine), [str(x) for x in x_ticks_machine])
    axs[1].set_xticks(list(x_ticks_nbr_gpus), [str(x) for x in x_ticks_nbr_gpus])
    axs[0].set_xlabel("Number of Environments")
    axs[1].set_xlabel("Number of Environments")
    axs[0].set_ylabel("FPS")

    # ===== Separate legends for each subplot, below the figure =====
    # Collect handles/labels from each subplot
    handles0, labels0 = axs[0].get_legend_handles_labels()
    handles1, labels1 = axs[1].get_legend_handles_labels()

    # Legend for left subplot
    leg0 = axs[0].legend(
        handles0, labels0,
        loc="upper center",
        bbox_to_anchor=(-0.1, -0.25, 1.2, 0),  # full width of subplot
        mode="expand",                    # expand across width
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=0.8,
        fontsize=LEGEND_SIZE          # slightly smaller font
    )

    # Legend for right subplot
    leg1 = axs[1].legend(
        handles1, labels1,
        loc="upper center",
        bbox_to_anchor=(-0.1, -0.25, 1.2, 0),  # full width of subplot
        mode="expand",
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=0.8,
        fontsize=LEGEND_SIZE
    )

    # Extra space below for legends
    plt.tight_layout(pad=2.0, rect=[0, 0.05, 1, 1])

    plt.savefig(f"benchmark_{COMPARISON_ENVIRONMENTS_NAME[i]}_perceptive_environment.pdf", dpi=300,
                bbox_extra_artists=(leg0, leg1), bbox_inches="tight")
    plt.close()
