"""Build the summary figure from the grasp sweep CSVs.

The measured ``gap`` is the sum of the two finger joint positions, which includes a
constant pad-thickness offset (~9 mm) and so is NOT an absolute compression. Hooke's
law is therefore tested as *linearity of gap*:

    gap(F) = gap0 - (L / (E A)) * F        at fixed E   -> straight line in F
    gap(E) = gap0 - (F L / A) * (1/E)      at fixed F   -> straight line in 1/E

The constant offset falls into the intercept ``gap0``; the slope carries the physics
and is compared against the analytic uniaxial prediction.
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SWEEPS = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SWEEPS, "grasp_parameter_study.png")
MM = 1e3

AREA = 0.05 * 0.05  # cube cross-section [m^2]
LEN = 0.05  # cube length along the squeeze axis [m]


def load(name):
    path = os.path.join(SWEEPS, f"{name}.csv")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        for k, v in list(r.items()):
            if k in ("mode",):
                continue
            if k in ("grasped", "saturated"):
                r[k] = str(v).strip().lower() == "true"
                continue
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
        out.append(r)
    return out


def linfit(xs, ys):
    """Ordinary least squares -> (slope, intercept, r_squared)."""
    n = len(xs)
    if n < 2:
        return 0.0, 0.0, 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        return 0.0, my, 0.0
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return slope, intercept, r2


DRIFT_SETTLED = 5e-3  # gap still moving more than this over the measure window = not at equilibrium


def ok(rows):
    """Keep points where the gripper closed on the object AND is genuinely loading it.

    A gap at or above the open gap means the fingers never actually compressed the cube
    (at 2 N the block is merely nudged), so such points carry no stiffness information.
    """
    keep = []
    for r in rows:
        if not r.get("grasped", True):
            continue
        if r["gap"] >= r["open_gap"] - 0.002:
            continue
        keep.append(r)
    return keep


def settled(rows):
    return [r for r in rows if r["gap_drift"] <= DRIFT_SETTLED]


mesh = sorted(ok(load("mesh")), key=lambda r: r["nodes"])
stiff = sorted(ok(load("stiffness")), key=lambda r: r["youngs"])
force = sorted(ok(load("force")), key=lambda r: r["applied_force"])

fig, axes = plt.subplots(2, 2, figsize=(13.5, 10))
fig.suptitle(
    "Newton (MJWarp + VBD) soft-body grasp — parameter study\n"
    "0.05 m cube · constant-force gripper · fixed end-effector pose · tet mesh pinned by disk cache",
    fontsize=13,
    fontweight="bold",
)

C_SIM, C_REF, C_FIT = "#2563eb", "#dc2626", "#059669"

# ------------------------------------------------------------- (a) mesh density
ax = axes[0][0]
if mesh:
    x = [r["nodes"] for r in mesh]
    y = [r["gap"] * MM for r in mesh]
    ax.plot(x, y, "o-", color=C_SIM, lw=2, ms=8)
    lo, hi = min(y), max(y)
    ax.axhspan(lo, hi, color=C_SIM, alpha=0.08)
    ax.annotate(
        f"spread = {hi - lo:.2f} mm\nover {min(x):.0f}–{max(x):.0f} nodes\n(material & force fixed)",
        xy=(0.97, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=C_SIM, alpha=0.95),
    )
    ax.set_xscale("log")
ax.set_xlabel("tetrahedral mesh nodes")
ax.set_ylabel("grasped width (finger gap) [mm]")
ax.set_title("(a) Mesh resolution — object & force held fixed", fontsize=11, fontweight="bold")
ax.grid(alpha=0.3, which="both")

# ---------------------------------------------------------------- (b) stiffness
ax = axes[0][1]
if stiff:
    x = [r["youngs"] for r in stiff]
    ax.plot(x, [r["gap"] * MM for r in stiff], "o-", color=C_SIM, lw=2, ms=8)
    ax.set_xscale("log")
ax.set_xlabel("Young's modulus E [Pa]")
ax.set_ylabel("grasped width (finger gap) [mm]")
ax.set_title("(b) Object stiffness at fixed mesh & 10 N", fontsize=11, fontweight="bold")
ax.grid(alpha=0.3, which="both")

# ------------------------------------------------- (c) Hooke in 1/E at fixed F
ax = axes[1][0]
if len(stiff) >= 2:
    fit_rows = settled(stiff) or stiff
    inv = [1.0 / r["youngs"] for r in fit_rows]
    y = [r["gap"] * MM for r in fit_rows]
    s, b, r2 = linfit(inv, y)
    ax.plot(inv, y, "o", color=C_SIM, ms=9, label="simulated", zorder=3)
    drop = [r for r in stiff if r not in fit_rows]
    if drop:
        ax.plot(
            [1.0 / r["youngs"] for r in drop],
            [r["gap"] * MM for r in drop],
            "o",
            mfc="none",
            mec=C_SIM,
            ms=9,
            label="not settled (excluded)",
            zorder=3,
        )
    inv = [1.0 / r["youngs"] for r in stiff]
    xs = [0.0, max(inv) * 1.05]
    ax.plot(xs, [s * v + b for v in xs], "-", color=C_FIT, lw=1.8, label=f"linear fit  $R^2$={r2:.4f}")
    F = stiff[0]["applied_force"]
    s_an = -F * LEN / AREA * MM
    ax.plot(xs, [s_an * v + b for v in xs], "--", color=C_REF, lw=1.6, label="analytic slope")
    ax.legend(fontsize=9, loc="best")
    ax.annotate(
        f"fitted slope  {s:.3g} mm·Pa\nanalytic      {s_an:.3g} mm·Pa\nratio {s / s_an if s_an else float('nan'):.2f}×",
        xy=(0.03, 0.05),
        xycoords="axes fraction",
        ha="left",
        va="bottom",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=C_FIT, alpha=0.95),
    )
ax.set_xlabel("1 / E   [Pa$^{-1}$]")
ax.set_ylabel("grasped width (finger gap) [mm]")
ax.set_title("(c) Hooke test: gap linear in $1/E$ ?", fontsize=11, fontweight="bold")
ax.grid(alpha=0.3)

# --------------------------------------------------- (d) Hooke in F at fixed E
ax = axes[1][1]
if len(force) >= 2:
    fit_rows = settled(force) or force
    x = [r["applied_force"] for r in fit_rows]
    y = [r["gap"] * MM for r in fit_rows]
    s, b, r2 = linfit(x, y)
    ax.plot(x, y, "o", color=C_SIM, ms=9, label="simulated", zorder=3)
    drop = [r for r in force if r not in fit_rows]
    if drop:
        ax.plot(
            [r["applied_force"] for r in drop],
            [r["gap"] * MM for r in drop],
            "o",
            mfc="none",
            mec=C_SIM,
            ms=9,
            label="not settled (excluded)",
            zorder=3,
        )
    xs = [0.0, max(r["applied_force"] for r in force) * 1.05]
    ax.plot(xs, [s * v + b for v in xs], "-", color=C_FIT, lw=1.8, label=f"linear fit  $R^2$={r2:.4f}")
    E = force[0]["youngs"]
    s_an = -LEN / (E * AREA) * MM
    ax.plot(xs, [s_an * v + b for v in xs], "--", color=C_REF, lw=1.6, label="analytic slope")
    ax.legend(fontsize=9, loc="best")
    ax.annotate(
        f"fitted slope  {s:.3g} mm/N\nanalytic      {s_an:.3g} mm/N\nratio {s / s_an if s_an else float('nan'):.2f}×",
        xy=(0.03, 0.05),
        xycoords="axes fraction",
        ha="left",
        va="bottom",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=C_FIT, alpha=0.95),
    )
ax.set_xlabel("applied grip force [N]")
ax.set_ylabel("grasped width (finger gap) [mm]")
ax.set_title("(d) Hooke test: gap linear in $F$ ?", fontsize=11, fontweight="bold")
ax.grid(alpha=0.3)

fig.tight_layout(rect=[0, 0.01, 1, 0.93])
fig.savefig(OUT, dpi=160)
print("wrote", OUT)

for name, rows in (("mesh", mesh), ("stiffness", stiff), ("force", force)):
    print(f"\n--- {name} ({len(rows)} usable points) ---")
    for r in rows:
        print(
            f"  value={r.get('value'):<9g} nodes={int(r['nodes']):<5d} E={r['youngs']:<9g} "
            f"F={r['applied_force']:.2f}N gap={r['gap'] * MM:7.3f}mm drift={r['gap_drift']:.1e}"
        )
