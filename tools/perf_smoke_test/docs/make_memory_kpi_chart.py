# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render memory KPI charts from the regression evidence summary."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SCRIPT_DIR = Path(__file__).parent
EVIDENCE_JSON = SCRIPT_DIR / "regression_evidence_summary.json"

LABELS = {
    "isaaclab_2_3_2": "IsaacLab 2.3.2",
    "current_fork": "Current fork",
}
COLORS = {
    "isaaclab_2_3_2": "#54A24B",
    "current_fork": "#E45756",
}
TASK_ORDER = [
    "Isaac-Cartpole-Direct",
    "Isaac-Velocity-Flat-G1",
    "Isaac-Factory-GearMesh-Direct",
    "Isaac-Repose-Cube-Shadow-Vision-Benchmark-Direct-v0",
]
MISSING_STATUS = {
    ("Isaac-Repose-Cube-Shadow-Vision-Benchmark-Direct-v0", "isaaclab_2_3_2"): "Segfault during\nrendering startup",
}


def _task_label(task_id: str) -> str:
    return (
        task_id.removeprefix("Isaac-")
        .removesuffix("-v0")
        .replace("-Direct", "")
        .replace("-Benchmark", "")
        .replace("Velocity-Flat-G1", "G1")
        .replace("Factory-GearMesh", "Factory")
        .replace("Repose-Cube-Shadow-Vision", "Shadow Vision")
    )


def _load_rows() -> list[dict]:
    payload = json.loads(EVIDENCE_JSON.read_text(encoding="utf-8"))
    rows = payload.get("summary", [])
    return [row for row in rows if row.get("label") in LABELS]


def _ordered_tasks(rows: list[dict]) -> list[str]:
    tasks = {str(row.get("task_id")) for row in rows}
    ordered = [task for task in TASK_ORDER if task in tasks]
    ordered.extend(sorted(tasks - set(ordered)))
    return ordered


def _value_gb(row: dict | None, key: str) -> float | None:
    if not row:
        return None
    value = row.get(key)
    if value is None and key == "peak_gpu_mem_used_mb":
        value = row.get("mean_gpu_mem_used_mb")
    return float(value) / 1024.0 if isinstance(value, (int, float)) else None


def _write_csv(rows: list[dict], out: Path) -> None:
    out_rows = []
    for row in rows:
        out_rows.append(
            {
                "label": row.get("label"),
                "task_id": row.get("task_id"),
                "num_envs": row.get("num_envs"),
                "sample_count": row.get("sample_count"),
                "mean_fps": row.get("mean_fps"),
                "run_to_run_std_fps": row.get("run_to_run_std_fps"),
                "mean_vram_gb": _value_gb(row, "mean_gpu_mem_mean_mb"),
                "peak_vram_gb": _value_gb(row, "peak_gpu_mem_used_mb"),
                "mean_system_ram_gb": _value_gb(row, "mean_system_ram_used_mb"),
                "peak_system_ram_gb": _value_gb(row, "peak_system_ram_used_mb"),
            }
        )
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)


def _missing_status(task_id: str, label: str) -> str:
    return MISSING_STATUS.get((task_id, label), "N/A")


def _annotate_bars(ax: plt.Axes, bars, values: list[float | None], statuses: list[str]) -> None:
    ymax = max([value for value in values if value is not None] or [1.0])
    for bar, value, status in zip(bars, values, statuses):
        if value is None:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                ymax * 0.04,
                status,
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#666666",
                linespacing=1.1,
            )
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + ymax * 0.025,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )


def main() -> None:
    rows = _load_rows()
    labels = [label for label in LABELS if any(row.get("label") == label for row in rows)]
    tasks = _ordered_tasks(rows)
    by_key = {(str(row.get("task_id")), str(row.get("label"))): row for row in rows}

    fig, axes = plt.subplots(2, 1, figsize=(13.5, 8.2), sharex=True)
    fig.suptitle(
        "IsaacLab 2.3.2 vs current fork memory KPIs",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.935,
        "VRAM from nvidia-smi samples; system RAM from docker stats. Peak memory is the capacity-planning KPI. "
        "Segfault labels mean no valid steady-state benchmark sample.",
        ha="center",
        fontsize=10,
        color="#333333",
    )

    x = list(range(len(tasks)))
    width = 0.34
    offsets = {
        labels[0]: -width / 2 if len(labels) > 1 else 0.0,
        labels[-1]: width / 2 if len(labels) > 1 else 0.0,
    }
    panels = [
        (axes[0], "peak_gpu_mem_used_mb", "Peak VRAM [GB]"),
        (axes[1], "peak_system_ram_used_mb", "Peak system RAM [GB]"),
    ]
    for ax, key, ylabel in panels:
        panel_values: list[float | None] = []
        for label in labels:
            values = [_value_gb(by_key.get((task, label)), key) for task in tasks]
            statuses = [_missing_status(task, label) for task in tasks]
            panel_values.extend(values)
            bars = ax.bar(
                [pos + offsets[label] for pos in x],
                [value or 0.0 for value in values],
                width=width,
                color=COLORS[label],
                edgecolor="black",
                linewidth=0.55,
                label=LABELS[label],
            )
            _annotate_bars(ax, bars, values, statuses)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_ylim(0, max([value for value in panel_values if value is not None] or [1.0]) * 1.25)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([_task_label(task) for task in tasks], rotation=12, ha="right")

    handles = [Patch(facecolor=COLORS[label], edgecolor="black", label=LABELS[label]) for label in labels]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(top=0.88, bottom=0.16, hspace=0.35)

    out_png = SCRIPT_DIR / "memory_kpis_3_0_vs_2_3_2.png"
    out_csv = SCRIPT_DIR / "memory_kpis_3_0_vs_2_3_2.csv"
    fig.savefig(out_png, dpi=160, bbox_inches="tight", facecolor="white")
    _write_csv(rows, out_csv)
    print(f"wrote {out_png}")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
