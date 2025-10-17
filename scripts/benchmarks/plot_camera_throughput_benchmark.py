import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import scienceplots
from matplotlib.ticker import ScalarFormatter
import argparse
import sys
from pathlib import Path

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

# update colors to black
plt.rcParams.update({
    "text.color": "black",
    "axes.labelcolor": "black",
    "axes.edgecolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
})

RESOLUTION_COLORS = {
    (64, 64): "#d08770",
    (120, 160): "#ebcb8b",
    (240, 320): "#a3be8c",
    (480, 640): "#b48ead"
}

ADD_MEM_TEST = False
USD_DIFFERENT_Y_AXIS = False


def load_and_combine_csv_files(file_paths):
    """Load multiple CSV files and combine them into a single DataFrame."""
    all_dataframes = []
    
    for i, file_path in enumerate(file_paths):
        print(f"Loading file {i+1}/{len(file_paths)}: {file_path}")
        df = pd.read_csv(file_path)
        df['run_id'] = i  # Add run identifier
        all_dataframes.append(df)
    
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"Combined {len(file_paths)} files with {len(combined_df)} total rows")
    return combined_df


def _format_axes(ax: Axes, grid: bool = True):
    """Apply consistent, publication-ready formatting to axes."""
    if grid:
        ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.6)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.5)
    # Use scalar formatter without offset/scientific on axes unless necessary
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

# Parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Plot camera throughput benchmark results from multiple CSV files')
    parser.add_argument('--files', nargs='+', type=str, default=[
        "/home/pascalr/Downloads/IsaacLab Camera benchmarks - run1.csv",
        "/home/pascalr/Downloads/IsaacLab Camera benchmarks - run2.csv",
        "/home/pascalr/Downloads/IsaacLab Camera benchmarks - run3.csv",
        "/home/pascalr/Downloads/IsaacLab Camera benchmarks - run4.csv",
        "/home/pascalr/Downloads/IsaacLab Camera benchmarks - run5.csv",
    ], help='CSV files to process')
    parser.add_argument('--output', '-o', type=str, default='camera_benchmarks_averaged.png', help='Output filename')
    parser.add_argument('--add-memory', action='store_true', help='Add memory annotations to plot')
    return parser.parse_args()

args = parse_arguments()

# Validate input files
file_paths = []
for file_path in args.files:
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File {file_path} does not exist")
        sys.exit(1)
    file_paths.append(str(path))

ADD_MEM_TEST = args.add_memory

# Load and combine CSV files
raw = load_and_combine_csv_files(file_paths)

# Compute FPS
df = raw.copy()
df["FPS"] = (1.0 / df["per_step_ms"]) * df["num_envs"] * 1000.0

# drop nan values
df = df.dropna()

# Parse resolution strings "240 x 320" -> (240, 320)
df["resolution"] = df["resolution"].str.split(" x ").apply(lambda x: tuple(map(int, x)))

# Group by configuration and compute averages across runs
print("Computing averages across runs...")
group_cols = ["data_types", "mode", "resolution", "num_envs"]
aggregated_data = df.groupby(group_cols).agg({
    "per_step_ms": ["mean", "std"],
    "avg_memory": ["mean", "std"],  
    "FPS": ["mean", "std"]
}).reset_index()

# Flatten column names
aggregated_data.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in aggregated_data.columns]

# Rename columns for consistency with original plotting code
df_avg = aggregated_data.rename(columns={
    "per_step_ms_mean": "per_step_ms",
    "avg_memory_mean": "avg_memory",
    "FPS_mean": "FPS",
    "per_step_ms_std": "per_step_ms_std",
    "avg_memory_std": "avg_memory_std", 
    "FPS_std": "FPS_std"
})

print(f"Averaged data has {len(df_avg)} unique configurations")

data_types = df_avg["data_types"].unique()
modes = df_avg["mode"].unique()[::-1]

fig, axes = plt.subplots(len(data_types), len(modes), figsize=(6 * len(modes), 12), sharey=not USD_DIFFERENT_Y_AXIS, sharex=False)

if len(modes) == 1:
    axes = axes[None, :]

# Identify the reference axes for modes that *should* share y-axis
shared_y_axes = {}

for idx, dtype in enumerate(data_types):
    sub_dtype = df_avg[df_avg["data_types"] == dtype]
    
    for jdx, mode in enumerate(modes):
        ax = axes[idx, jdx]
        sub_mode = sub_dtype[sub_dtype["mode"] == mode]

        # --- share y-axis manually if not "usd_camera" ---
        if mode != "usd_camera" and USD_DIFFERENT_Y_AXIS:
            # Use the first non-usd_camera column as reference
            if "shared_ref" not in shared_y_axes:
                shared_y_axes["shared_ref"] = ax
            else:
                ax.sharey(shared_y_axes["shared_ref"])
                if jdx != 1:
                    # remove redundant y ticks and labels for shared axes
                    ax.tick_params(labelleft=False)
        # ------------------------------------------------

        for res, g in sub_mode.groupby("resolution"):  # type: ignore
            g = g.sort_values("num_envs")  # type: ignore
            if g.empty:
                continue
            x = g["num_envs"]
            y = g["FPS"]
            y_err = g["FPS_std"]
            mem = g["avg_memory"]
            mem_err = g["avg_memory_std"]

            color = RESOLUTION_COLORS.get(tuple(res), "gray")  # type: ignore
            # Plot main line
            ax.plot(x, y, marker="o", label=f"{res}", color=color, linewidth=2)
            # Add fill_between for standard deviation
            ax.fill_between(x, y - y_err, y + y_err, color=color, alpha=0.2)
            
            if ADD_MEM_TEST:
                for xi, yi, mi, mei in zip(x, y, mem, mem_err):
                    ax.text(xi, yi * 1.05, f"{mi:.0f}±{mei:.0f}", ha="center", va="bottom", fontsize=8)

        if idx == 0:
            ax.set_title(f"{mode.replace('_', ' ')}", weight="bold")
        if jdx == 0:
            ax.set_ylabel("FPS")
        if idx == len(data_types) - 1 and jdx == 1:
            ax.set_xlabel("Number of Environments")
        
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        _format_axes(ax)

if USD_DIFFERENT_Y_AXIS:
    axes[0, 0].legend(loc="lower right")
else:
    axes[0, 0].legend(loc="upper right")

plt.tight_layout()
plt.savefig(args.output, dpi=600)
print(f"Plot saved as {args.output}")
plt.show()