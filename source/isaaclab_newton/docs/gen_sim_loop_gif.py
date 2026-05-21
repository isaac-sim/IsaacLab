"""
Generate unified_sim_loop.gif by rendering individual frames with matplotlib
and stitching them with Pillow — avoids FuncAnimation hang on headless servers.
"""
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

# ── palette ───────────────────────────────────────────────────────────────────
BG     = "#0d1117"
COLS   = ["#2a1f5f", "#1e3a5f", "#1e4a2a", "#4a2a10"]
GARR   = "#76b900"
CTXT   = "#e0e0e0"
CDIM   = "#555555"
CWHITE = "#ffffff"

W, H_FIG = 12, 6.5
DPI = 110

STAGES = 4
HOLD   = 8   # frames per stage
FPS    = 7

BOXES = [
    (0.03, 0.52, 0.18, 0.24),
    (0.27, 0.52, 0.18, 0.24),
    (0.51, 0.52, 0.22, 0.24),
    (0.79, 0.52, 0.18, 0.24),
]
TITLES = [
    ("Proximal Control", "GPU boundary condition"),
    ("XPBD Physics", "Cosserat rod + SDF collision"),
    ("Beer-Lambert Render", "Static mu-vol + catheter inline"),
    ("Synchronized Output", "Fluoro frame + catheter state"),
]
EQS = [
    ["$q_0^{n+1} = q_{tip}(d, \\theta)$",
     "$d$ = depth,   $\\theta$ = rotation"],

    ["Predict:   $\\tilde{x}_i = x_i + h v_i$",
     "Constraint:   $\\Delta x_i = \\sum_j \\lambda_j \\nabla C_j$",
     "Velocity:   $v_i^{n+1} = (x_i^{n+1} - x_i^n)/h$",
     "SDF wall:   $\\phi(x_i) \\leq 0 \\Rightarrow x_i \\to \\partial\\Omega$"],

    ["$I(u,v) = I_0 \\exp(-\\int_{ray} \\mu(r)\\, ds)$",
     "$\\mu = \\mu_{CT}(r) + \\mu_{cath} \\cdot 1_{seg}(r)$",
     "Catheter composited inline per ray"],

    ["Obs:   $O_t \\in R^{H \\times W}$ (fluoro)",
     "State:   $s_t = (x_{tip}, \\hat{e}, d, \\kappa)$",
     "Reward:   $r_t = r(s_t, a_t)$"],
]
EQ_X = [0.12, 0.36, 0.62, 0.88]
ARR_LABELS = ["$q_0^{n+1}$", "$\\{p_i,r_i,\\mu_i\\}$", "$I(u,v)$"]


def render_frame(stage: int) -> Image.Image:
    fig, ax = plt.subplots(figsize=(W, H_FIG), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    # ── title ─────────────────────────────────────────────────────────────
    ax.text(0.5, 0.955,
            "Unified Simulation Loop — X-Ray-Guided Catheter RL",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color=CWHITE)
    ax.text(0.5, 0.915,
            "XPBD Cosserat Physics  +  Beer-Lambert GPU Renderer  +  Domain Randomisation",
            ha="center", va="center", fontsize=8.5, color=CTXT)

    # ── loop-back arrow ───────────────────────────────────────────────────
    loop = FancyArrowPatch(
        (0.90, 0.50), (0.10, 0.50),
        connectionstyle="arc3,rad=-0.30",
        arrowstyle="-|>", color=GARR, lw=1.6,
        mutation_scale=12, linestyle="dashed", zorder=4
    )
    ax.add_patch(loop)
    ax.text(0.5, 0.432,
            "$\\leftarrow$   $t \\mapsto t+1$   $\\leftarrow$   (repeat per RL timestep)",
            ha="center", va="center", fontsize=8, color=GARR)

    # ── GPU memory + DR note ──────────────────────────────────────────────
    ax.text(0.5, 0.178,
            "GPU memory:   $\\mu_{CT}^{D \\times H \\times W}$ [static, loaded once]"
            "   +   $\\{p_i, r_i, \\mu_i\\}_{i=1}^{N}$ [dynamic, per step]",
            ha="center", va="center", fontsize=8, color=CTXT)
    ax.text(0.5, 0.125,
            "Domain randomisation:  C-arm angle,   $\\mu$-noise,   vessel geometry variants",
            ha="center", va="center", fontsize=8, color=CTXT)

    # ── timing bar ────────────────────────────────────────────────────────
    bar_specs = [("Physics  0.77ms", 0.77, COLS[1]),
                 ("Render  40ms",    40.0, COLS[2])]
    tot = sum(t for _, t, _ in bar_specs)
    bx, by, bw, bh = 0.28, 0.040, 0.44, 0.042
    cum = 0.0
    for lbl, t, col in bar_specs:
        frac = t / tot
        b = FancyBboxPatch((bx+cum*bw, by), frac*bw, bh,
                           boxstyle="square,pad=0", facecolor=col,
                           edgecolor=BG, lw=0.8)
        ax.add_patch(b)
        ax.text(bx+(cum+frac/2)*bw, by+bh/2, lbl,
                ha="center", va="center", fontsize=6.5, color=CWHITE)
        cum += frac
    ax.text(bx+bw/2, by+bh+0.010, "Per-timestep wall-clock  (512 parallel envs)",
            ha="center", va="bottom", fontsize=7, color=CDIM)

    # ── boxes + equations ─────────────────────────────────────────────────
    for i, ((bx_, by_, bw_, bh_), col) in enumerate(zip(BOXES, COLS)):
        active = (i == stage)
        done   = (i  < stage)
        alpha  = 1.0 if (active or done) else 0.22
        ew     = 2.5 if active else 1.0
        ec     = GARR if (active or done) else CDIM

        box = FancyBboxPatch((bx_, by_), bw_, bh_,
                             boxstyle="round,pad=0.012",
                             facecolor=col, edgecolor=ec,
                             linewidth=ew, alpha=alpha, zorder=3)
        ax.add_patch(box)

        ttl, sub = TITLES[i]
        ax.text(bx_+bw_/2, by_+bh_*0.67, ttl,
                ha="center", va="center", fontsize=8.5, fontweight="bold",
                color=CWHITE, alpha=alpha, zorder=4)
        ax.text(bx_+bw_/2, by_+bh_*0.30, sub,
                ha="center", va="center", fontsize=6.8, color=CTXT,
                alpha=alpha, style="italic", zorder=4)

        eq_alpha = 1.0 if active else (0.45 if done else 0.10)
        for li, line in enumerate(EQS[i]):
            ax.text(EQ_X[i], 0.47 - li*0.048, line,
                    ha="center", va="top",
                    fontsize=7.2 if li == 0 else 6.8,
                    color=GARR if (li == 0 and active) else CTXT,
                    alpha=eq_alpha, zorder=6,
                    fontweight="bold" if (li == 0 and active) else "normal")

    # ── inter-box arrows ──────────────────────────────────────────────────
    pairs = [(0,1),(1,2),(2,3)]
    for k, (src, dst) in enumerate(pairs):
        bsrc = BOXES[src]; bdst = BOXES[dst]
        x0 = bsrc[0]+bsrc[2]; y0 = bsrc[1]+bsrc[3]/2
        x1 = bdst[0];          y1 = bdst[1]+bdst[3]/2
        a_alpha = 1.0 if stage > src else (0.3 if stage == src else 0.08)
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>",
                                   color=GARR if a_alpha > 0.4 else CDIM,
                                   lw=1.8, mutation_scale=12),
                    alpha=a_alpha, zorder=5)
        ax.text((x0+x1)/2, (y0+y1)/2+0.030, ARR_LABELS[k],
                ha="center", va="bottom", fontsize=7.5,
                color=GARR, alpha=a_alpha, zorder=6)

    fig.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# ── build frames ──────────────────────────────────────────────────────────────
frames = []
for stage in range(STAGES):
    img = render_frame(stage)
    frames.extend([img] * HOLD)

# one final frame showing all active
frames.extend([render_frame(STAGES - 1)] * (HOLD * 2))

out = "/home/cdinea/directsolver/IsaacLab/source/isaaclab_newton/docs/unified_sim_loop.gif"
frames[0].save(
    out,
    save_all=True,
    append_images=frames[1:],
    loop=0,
    duration=int(1000 / FPS),
    optimize=False,
)
print(f"Saved {out}  ({len(frames)} frames @ {FPS} fps)")
