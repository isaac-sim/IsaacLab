# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render a shareable 3.0-vs-2.3.2 FPS and memory comparison chart.

When ``regression_evidence_summary.json`` exists next to this script, the chart is
driven by the evidence-pack artifacts. Otherwise it falls back to the FPS numbers
used in the initial release-comparison graph and marks std/memory fields as
pending.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SCRIPT_DIR = Path(__file__).parent
EVIDENCE_JSON = SCRIPT_DIR / "regression_evidence_summary.json"

LABELS = {
    "isaaclab_2_2_0": "IsaacLab 2.2.0\nIsaac Sim 5.0",
    "isaaclab_2_3_2": "IsaacLab 2.3.2\nIsaac Sim 5.1",
    "current_fork": "Current fork 6.6.1\nIsaac Sim 6.0.0.x",
}
COLORS = {
    "isaaclab_2_2_0": "#4C78A8",
    "isaaclab_2_3_2": "#54A24B",
    "current_fork": "#E45756",
}
ORDER = ["isaaclab_2_2_0", "isaaclab_2_3_2", "current_fork"]
TASK_ALIASES = {
    "Isaac-Cartpole-Direct-v0": "Isaac-Cartpole-Direct",
    "Isaac-Factory-GearMesh-Direct-v0": "Isaac-Factory-GearMesh-Direct",
    "Isaac-Velocity-Flat-G1-v0": "Isaac-Velocity-Flat-G1",
}
MISSING_STATUS = {
    ("Isaac-Repose-Cube-Shadow-Vision-Benchmark-Direct-v0", "isaaclab_2_3_2"): "Segfault during\nrendering startup",
}

# Fallback median effective FPS (mean of "Environment step effective FPS", frames
# 101-299) from the initial RTX PRO 6000 Blackwell comparison.
FALLBACK_ROWS = [
    {
        "label": "isaaclab_2_2_0",
        "task_id": "Isaac-Cartpole-Direct",
        "backend": "physx",
        "num_envs": 4096,
        "sample_count": 10,
        "median_fps": 355910.7,
        "run_to_run_std_fps": None,
    },
    {
        "label": "isaaclab_2_3_2",
        "task_id": "Isaac-Cartpole-Direct",
        "backend": "physx",
        "num_envs": 4096,
        "sample_count": 3,
        "median_fps": 452161.9,
        "run_to_run_std_fps": None,
    },
    {
        "label": "current_fork",
        "task_id": "Isaac-Cartpole-Direct",
        "backend": "physx",
        "num_envs": 4096,
        "sample_count": 14,
        "median_fps": 281304.8,
        "run_to_run_std_fps": None,
    },
    {
        "label": "isaaclab_2_2_0",
        "task_id": "Isaac-Velocity-Flat-G1",
        "backend": "physx",
        "num_envs": 512,
        "sample_count": 10,
        "median_fps": 12679.6,
        "run_to_run_std_fps": None,
    },
    {
        "label": "isaaclab_2_3_2",
        "task_id": "Isaac-Velocity-Flat-G1",
        "backend": "physx",
        "num_envs": 512,
        "sample_count": 3,
        "median_fps": 12822.2,
        "run_to_run_std_fps": None,
    },
    {
        "label": "current_fork",
        "task_id": "Isaac-Velocity-Flat-G1",
        "backend": "physx",
        "num_envs": 512,
        "sample_count": 33,
        "median_fps": 10737.0,
        "run_to_run_std_fps": None,
    },
    {
        "label": "isaaclab_2_2_0",
        "task_id": "Isaac-Factory-GearMesh-Direct",
        "backend": "physx",
        "num_envs": 512,
        "sample_count": 10,
        "median_fps": 1211.7,
        "run_to_run_std_fps": None,
    },
    {
        "label": "isaaclab_2_3_2",
        "task_id": "Isaac-Factory-GearMesh-Direct",
        "backend": "physx",
        "num_envs": 512,
        "sample_count": 3,
        "median_fps": 1382.7,
        "run_to_run_std_fps": None,
    },
    {
        "label": "current_fork",
        "task_id": "Isaac-Factory-GearMesh-Direct",
        "backend": "physx",
        "num_envs": 512,
        "sample_count": 33,
        "median_fps": 1209.0,
        "run_to_run_std_fps": None,
    },
]


def _fmt(v: float) -> str:
    return f"{v / 1000:.0f}k" if v >= 10000 else f"{v:,.0f}"


def _task_label(task_id: str) -> str:
    return task_id.removeprefix("Isaac-").removesuffix("-v0").replace("-Direct", "")


def _canonical_task_id(task_id: str) -> str:
    return TASK_ALIASES.get(task_id, task_id)


def _missing_status(task_id: str, label: str) -> str | None:
    return MISSING_STATUS.get((task_id, label))


def _load_rows() -> tuple[list[dict], bool]:
    if not EVIDENCE_JSON.exists():
        return FALLBACK_ROWS, False
    try:
        payload = json.loads(EVIDENCE_JSON.read_text(encoding="utf-8"))
        rows = payload.get("summary", [])
    except Exception:
        return FALLBACK_ROWS, False
    if not rows:
        return FALLBACK_ROWS, False
    normalized = []
    for row in rows:
        if row.get("backend") not in {"physx", "default"}:
            continue
        normalized_row = dict(row)
        normalized_row["task_id"] = _canonical_task_id(str(row.get("task_id")))
        normalized.append(normalized_row)
    return normalized, True


def _ordered_labels(rows: list[dict]) -> list[str]:
    labels = {str(row.get("label")) for row in rows}
    ordered = [label for label in ORDER if label in labels]
    ordered.extend(sorted(labels - set(ordered)))
    return ordered


def _ordered_tasks(rows: list[dict]) -> list[str]:
    tasks = {str(row.get("task_id")) for row in rows}
    preferred = [
        "Isaac-Cartpole-Direct",
        "Isaac-Cartpole-Direct-v0",
        "Isaac-Velocity-Flat-G1",
        "Isaac-Velocity-Flat-G1-v0",
        "Isaac-Factory-GearMesh-Direct",
        "Isaac-Factory-GearMesh-Direct-v0",
        "Isaac-Repose-Cube-Shadow-Vision-Benchmark-Direct-v0",
    ]
    ordered = [task for task in preferred if task in tasks]
    ordered.extend(sorted(tasks - set(ordered)))
    return ordered


def _row_map(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(str(row.get("task_id")), str(row.get("label"))): row for row in rows}


def _first_present(rows: list[dict], key: str) -> str | None:
    for row in rows:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def main() -> None:
    rows, evidence_loaded = _load_rows()
    labels = _ordered_labels(rows)
    tasks = _ordered_tasks(rows)
    by_task_label = _row_map(rows)
    cols = max(1, len(tasks))

    fig = plt.figure(figsize=(max(13, 4.4 * cols), 9.2))
    gs = fig.add_gridspec(3, cols, height_ratios=[3.0, 1.85, 1.05], hspace=0.55, wspace=0.3, top=0.84, bottom=0.06)

    fig.suptitle(
        "3.0 fork vs IsaacLab 2.3.2 effective-FPS evidence (RTX PRO 6000 Blackwell)",
        fontsize=15,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.915,
        "Public 2.3.2 remains faster on overhead-bound tasks; memory/std panels populate from the evidence workflow",
        ha="center",
        fontsize=10.5,
        style="italic",
        color="#333333",
    )

    x = list(range(len(labels)))
    for col, task_id in enumerate(tasks):
        task_rows = [by_task_label.get((task_id, label), {}) for label in labels]
        fps_values = [row.get("median_fps") or row.get("mean_fps") or 0.0 for row in task_rows]
        std_values = [row.get("run_to_run_std_fps") or 0.0 for row in task_rows]
        envs = next((row.get("num_envs") for row in task_rows if row.get("num_envs")), "N/A")

        ax = fig.add_subplot(gs[0, col])
        bars = ax.bar(
            x,
            fps_values,
            yerr=std_values if any(std_values) else None,
            color=[COLORS.get(label, "#777777") for label in labels],
            width=0.68,
            edgecolor="black",
            linewidth=0.6,
            capsize=3,
        )
        ax.set_title(f"{_task_label(task_id)}\n({envs} envs)", fontsize=10.5, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [
                label.split("_")[-1] if label not in LABELS else LABELS[label].split("\n")[0].replace("IsaacLab ", "")
                for label in labels
            ],
            fontsize=8.8,
        )
        ax.set_ylim(0, max(fps_values or [1.0]) * 1.27)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        if col == 0:
            ax.set_ylabel("effective FPS", fontsize=10)

        current_row = by_task_label.get((task_id, "current_fork"), {})
        cur = current_row.get("median_fps") or current_row.get("mean_fps")
        for i, (bar, v, row) in enumerate(zip(bars, fps_values, task_rows)):
            status = _missing_status(task_id, labels[i])
            if v <= 0:
                if status:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        max(fps_values or [1.0]) * 0.08,
                        status,
                        ha="center",
                        va="bottom",
                        fontsize=7.2,
                        color="#666666",
                        linespacing=1.05,
                    )
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + max(fps_values) * 0.025,
                _fmt(v),
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )
            if cur and row.get("label") != "current_fork":
                delta = (v - cur) / cur * 100.0
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    max(fps_values) * 0.045,
                    f"{delta:+.0f}%\nvs fork",
                    ha="center",
                    va="bottom",
                    fontsize=8.2,
                    color="#1a1a1a",
                )

        ax_mem = fig.add_subplot(gs[1, col])
        gpu_mem = [
            row.get("peak_gpu_mem_used_mb") or row.get("mean_gpu_mem_used_mb") or 0.0 for row in task_rows
        ]
        sys_mem = [row.get("peak_system_ram_used_mb") or row.get("mean_system_ram_used_mb") or 0.0 for row in task_rows]
        width = 0.35
        mem_bars_vram = ax_mem.bar(
            [i - width / 2 for i in x],
            gpu_mem,
            width=width,
            color="#8CBCCB",
            edgecolor="black",
            linewidth=0.5,
            label="VRAM",
        )
        mem_bars_ram = ax_mem.bar(
            [i + width / 2 for i in x],
            sys_mem,
            width=width,
            color="#F2CF5B",
            edgecolor="black",
            linewidth=0.5,
            label="System RAM",
        )
        ax_mem.set_xticks(x)
        ax_mem.set_xticklabels([""] * len(labels))
        ax_mem.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax_mem.set_axisbelow(True)
        if max(gpu_mem + sys_mem) <= 0:
            ax_mem.text(
                0.5,
                0.5,
                "memory pending\n(evidence workflow)",
                ha="center",
                va="center",
                transform=ax_mem.transAxes,
                fontsize=8.5,
            )
            ax_mem.set_ylim(0, 1)
        else:
            ymax_mem = max(gpu_mem + sys_mem)
            for i, (vram_bar, ram_bar, row) in enumerate(zip(mem_bars_vram, mem_bars_ram, task_rows)):
                status = _missing_status(task_id, labels[i])
                if status and not row:
                    ax_mem.text(
                        (vram_bar.get_x() + ram_bar.get_x() + ram_bar.get_width()) / 2,
                        ymax_mem * 0.07,
                        status,
                        ha="center",
                        va="bottom",
                        fontsize=6.8,
                        color="#666666",
                        linespacing=1.05,
                    )
        if col == 0:
            ax_mem.set_ylabel("peak memory MB", fontsize=10)
            ax_mem.legend(fontsize=8, frameon=False, loc="upper left")

    # legend
    sample_counts = {
        label: max(
            (int(row.get("sample_count") or 0) for row in rows if row.get("label") == label),
            default=0,
        )
        for label in labels
    }
    handles = [
        Patch(
            facecolor=COLORS.get(label, "#777777"),
            edgecolor="black",
            label=f"{LABELS.get(label, label)}  (n={sample_counts[label] or 'pending'})",
        )
        for label in labels
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.205),
        ncol=min(3, len(handles)),
        fontsize=9,
        frameon=False,
    )

    # config panel
    ax_cfg = fig.add_subplot(gs[2, :])
    ax_cfg.axis("off")
    cpu = _first_present(rows, "cpu_name") or "CPU pending artifact extraction"
    cores = _first_present(rows, "cpu_physical_cores")
    cpu_line = f"CPU: {cpu}" + (f" ({cores} physical cores)" if cores else "")
    data_line = (
        "Data source: regression_evidence_summary.json generated from CI artifacts"
        if evidence_loaded
        else "Data source: fallback FPS values from initial graph; std/memory/CPU populate after evidence workflow"
    )
    config_lines = [
        "Benchmark: scripts/benchmarks/benchmark_non_rl.py | backend: PhysX | headless | seed fixed",
        "Metric: mean of Environment step effective FPS over kept frames (warm-up frames dropped)",
        "GPU: RTX PRO 6000 Blackwell (sm_120, ~96 GB) | " + cpu_line,
        "Nsight traces: uploaded by Performance Smoke - Regression Evidence Pack "
        "as CI artifacts for manual Drive upload",
        data_line,
    ]
    ax_cfg.text(
        0.5,
        1.02,
        "Measurement configuration",
        ha="center",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        transform=ax_cfg.transAxes,
    )
    ax_cfg.text(
        0.015,
        0.72,
        "\n".join(config_lines),
        ha="left",
        va="top",
        fontsize=8.6,
        family="monospace",
        transform=ax_cfg.transAxes,
        linespacing=1.5,
    )

    out = SCRIPT_DIR / "physx_fps_across_releases.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
