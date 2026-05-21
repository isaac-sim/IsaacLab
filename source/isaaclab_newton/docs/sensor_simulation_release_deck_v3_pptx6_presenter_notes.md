# Sensor Simulation Release Deck — Presenter Notes

**Source deck:** `sensor_simulation_release_deck (3).pptx (6).pdf` (13 slides in this export; footers reference a longer 32-slide PPTX).

**Audience:** Engineering leadership, clinical partners (XCATH / neurointerventional), I4H program stakeholders.

**Audit basis:** Code and docs in `source/isaaclab_newton/` (Isaac Lab Newton catheter stack) plus `i4h-sensor-simulation-internal/fluoro-simulator` (referenced as `fluorosim` — not vendored in this worktree; claims for `dsa.py`, `realism.py`, batched Slang kernels are cross-checked against `docs/build_release_deck.py` and `docs/gradio_ui_technical_walkthrough.md`).

**Legend for accuracy flags**

| Flag | Meaning |
|------|---------|
| ✅ | Matches current implementation |
| ⚠️ | Partially true, oversimplified, or split across packages |
| ❌ | Inaccurate or not wired in the demo path described |

---

## Slide 1 — Title

**On slide:** *X-Ray–Guided Robotic Catheter Intervention — Simulation Workflow — Release Status & Roadmap*

### Presenter notes

Open with scope: this is the **X-ray catheter intervention** simulation stack inside **Isaac for Healthcare**, not the full multi-modality sensor portfolio in one sitting.

The workflow chain is: **patient CT/CTA → digital twin (μ volume + vessel mesh + centerline) → physics + fluoroscopy render loop → training / human demo → (future) deployment on real C-arm hardware.**

**Say:** “This deck is a release audit: what ships today, what we measured, and what Sprint 2+ still owes. Numbers on later slides are from A6000 benches unless noted.”

**Accuracy:** ✅ Title and scope are correct.

---

## Slide 2 — Why This Workflow Matters (Clinical & Strategic Context)

**On slide:**

- Neurointerventional navigation is **2D fluoroscopy**; vessel lumen is invisible without contrast.
- Robotic catheter AI needs **paired** (CTA + fluoro + pose) data — scarce clinically.
- Simulator yields **synthetic paired data** with exact projection geometry.
- **512 parallel environments**, physics-accurate X-ray, Isaac scalability.

### Presenter notes

**Clinical frame (30 s):** Operators see bones and the radio-opaque wire/catheter. Vessels appear briefly after iodine injection; many cases use a **roadmap** (registered static angiogram overlaid on live fluoro). Training a policy on pixels requires millions of consistent (image, action, state) tuples — hospitals cannot supply that at scale.

**Simulation value:** We control ground truth: catheter centerline state, C-arm pose, and (when wired) contrast timing. No manual fluoro annotation.

**Compute frame:** Vectorised RL (`CatheterStateEnv`, default **512 envs**) turns data scarcity into a throughput problem. That is distinct from the **interactive demo**, which is intentionally **single-environment** for human validation.

**Implementation pointers:**

| Claim | Status | Detail |
|-------|--------|--------|
| Synthetic paired fluoro + pose | ✅ | `CatheterStateEnv` exposes tip/centerline state; rendering path can consume solver positions |
| 512 parallel envs | ✅ | `CatheterStateEnvCfg.num_envs = 512` in `isaaclab_newton/envs/catheter_state_env.py` |
| “Physics-accurate X-ray” | ⚠️ | Beer–Lambert DRR + catheter fused in Slang; full detector chain (Poisson/scatter/PSF) is **CPU post-process** on many paths, not inside the fused Slang march |
| Clinical-scale DSA cine in viewport | ⚠️ | Full `DSAPipeline` lives in **fluorosim**; viewport uses **precomputed DSA difference overlays** (cyan roadmap), not live bolus per frame |

**Accuracy flags**

- ⚠️ **512 envs:** True for **RL training**, not for the Stage 1/2 interactive apps (they use `num_envs=1` on the renderer).
- ⚠️ **“No manual annotation”:** True for **state**; pixel labels still need a defined observation pipeline (Sprint 2 for image obs).

---

## Slide 3 — Architecture Overview

**On slide:** Diagram only — “Architecture overview”.

### Presenter notes

Walk the four blocks (use the expanded diagram in `docs/sensor_simulation_release_deck.md` if the projected image is too small):

1. **CT ingestion** — DICOM/NIfTI → HU → μ volume; vessel mask → mesh + centerline graph; optional arrival map for bolus (`fluorosim` vasculature).
2. **Simulation loop** — Cosserat rod XPBD (`XPBDRodSolver` / `XCathRodSolver` / production `RodSolver`) + `SlangDiffDRRRenderer` in one timestep.
3. **Training** — demonstrations → (planned IL) → PPO (`examples/train_catheter_state.py`); **state-based today**.
4. **Deployment target** — Holoscan IGX + real C-arm (integration narrative; not demoed in viewport).

**Data contract between physics and render:** segment endpoints in **mm**, packed for Slang as `CatheterSegmentData` → fused Beer–Lambert in `render_batch_with_catheter` / `renderDRR_withCatheter_forward`.

**Say:** “Synchronization matters: the polyline rendered is the same particle state the solver held at that timestep — no async handoff.”

**Accuracy:** ✅ Architecture story is correct; ⚠️ slide is visual-only in PDF — do not invent components not on the diagram.

---

## Slide 4 — Beer–Lambert Catheter Compositing (Fused GPU Path)

**On slide:** Five-step pipeline — DRR background, catheter segment buffer, inline perpendicular test, fused integral, detector realism (Poisson, PSF, gamma, jitter).

### Presenter notes

**Core physics (accurate):** One ray march, one exponent:

\[
I = I_0 \exp\left(-\int \mu_{\mathrm{total}}(s)\,ds\right),\quad
\mu_{\mathrm{total}} = \mu_{\mathrm{CT}} + \sum_i \mu_i \sqrt{1 - d_i^2/r_i^2}
\]

The \(\sqrt{1-d^2/r^2}\) term is the **circular chord** weight at each sample — implemented in `fluorosim` Slang (`diffdrr_slang.slang`), consumed via `SlangDiffDRRRenderer.render_batch_with_catheter`.

**Step-by-step talking points:**

| Step | Say | Code / notes |
|------|-----|----------------|
| 1 DRR background | Trilinear μ volume on GPU; fixed step (e.g. 1 mm in demos) | `SlangDiffDRRConfig.step_mm` |
| 2 Segment buffer | Proximal/distal mm positions + radius + μ per segment | `CatheterSegmentData`; viewport uses **uniform** μ (~0.28–0.34 in fluoro), not full 5-zone table |
| 3 Perpendicular test | Per-ray, per-segment distance test | Fused in Slang — not a second compositing pass |
| 4 Fused integral | Multiplicative attenuation, not alpha blend | Self-crossings darken correctly |
| 5 Detector realism | Poisson, PSF, gamma, jitter | ⚠️ See flag below |

**Accuracy flags**

- ✅ Fused Beer–Lambert in a **single** GPU ray march — correct.
- ⚠️ **5-zone μ (W/NiTi/polymer/Pt)** — documented for CPU / segment-profile demos; **interactive viewport** passes a **scalar** `render_mu` per frame unless extended.
- ❌ **Detector realism on the same slide as “fused integral”** — misleading if read as “all in-shader.” In practice:
  - **Slang path:** Beer–Lambert (+ optional normalize) on GPU.
  - **`realism.apply_realism()`:** Poisson, veiling glare, PSF, gamma — **CPU** on `fluorosim` (`interactive_catheter_fluoro.py` can call this; **viewport** relies on precomputed DSA diff + style pass, not full `apply_realism` every frame).
- ⚠️ **HU → μ “piecewise linear”** — configurable in `fluorosim/config.py`; patient pack uses precomputed `mu_volume.npy`.

**Pre-empt Q&A:** “Why 25 FPS at 512²?” — Ray count × step size × catheter segment tests; 256² interactive targets ~263 FPS render-only per internal notes.

---

## Slide 5 — Part 1 Divider: Current Status

**On slide:** *PART 1 — Current Status — What is built, integrated, and measured today.*

### Presenter notes

Transition: move from physics/compositing concepts to **benchmarks and subsystems**. Emphasise **measured** vs **target** rows on slide 6.

**Accuracy:** ✅ Divider only.

---

## Slide 6 — X-Ray Performance Baseline

**On slide:** Table — physics ~1,300 Hz; Slang ~40 ms (25 FPS); CPU Beer–Lambert 200–500 ms; multi-env physics ✓; multi-env render ◐ (~25 FPS @ N≤4).

### Presenter notes

| Metric | Target | Slide “Current” | Presenter detail | Flag |
|--------|--------|-----------------|------------------|------|
| Single-env physics | >1,000 Hz | ~1,300 Hz @ 20 seg, A6000 | `XPBDRodSolver` / `XCathRodSolver` single rod; segment count affects cost | ✅ |
| Slang fused @ 512² | <5 ms | ~40 ms (~25 FPS) | Standalone bench; viewport often 320–512 det | ✅ order of magnitude |
| CPU Beer–Lambert | <2 ms | 200–500 ms | NumPy chord map — regression / offline | ✅ |
| Multi-env physics @ 512 | >60 Hz | ✓ Available | `RodSolver` + `XPBDRodSolver._BatchedWorkspace` + CUDA graphs; **RL env uses `RodSolver`**, not `XCathRodSolver` | ⚠️ |
| Multi-env Slang @ 512 | >60 Hz | ◐ ~25 FPS @ N≤4 | `renderDRR_forward_batched` exists in fluorosim; **not** 60 Hz @ 512² today | ⚠️ |

**Say:** “Physics batching is largely solved; **render batching at training resolution** is the Sprint 2 bottleneck (Texture2DArray / cache behaviour at N>8).”

**Accuracy flags**

- ⚠️ Checkmark on **multi-env physics @ 512 envs** — code paths exist; quote **~60 Hz** only if you have a recent benchmark log for your exact segment count and substep config.
- ⚠️ **“Similar at N≤4”** for render — plausible for small N; **N=512 @ 512² is not production today** despite batched kernel existing.

---

## Slide 7 — Multi-Environment Batched Physics — 512 Parallel Rods

**On slide:** `_BatchedWorkspace`, CUDA graphs, `apply_proximal_control_gpu`, batched `renderDRR_forward_batched`, L2 cache / Texture2DArray Sprint 2 fix; performance table (physics 1 env/512 env, render 256²/512², full loop ~63 FPS).

### Presenter notes

**Batching model:** Flat GPU buffers + `rod_offsets` / `edge_offsets` → one launch over all environments. CUDA graph capture on `step()` after warmup — avoids per-substep launch storms.

**GPU proximal control:** `apply_proximal_control_gpu` + `set_root_orientation` — Warp kernels, graph-safe.

**Batched render:** `dispatchThreadID.z` selects environment; `StructuredBuffer<CatheterSegment>` with per-env offsets.

**Performance table talking points:**

- **Physics 1 env @ 1,300 Hz** — aligns with slide 6.
- **Physics 512 env @ ~60 Hz** — training-scale expectation.
- **Render 256² @ ~263 FPS** — matches Gradio/interactive sizing (`DET_SIZE=256` in `interactive_catheter_fluoro.py`).
- **Render 512² @ ~25 FPS** — single-env fused path.
- **Full loop ~63 FPS @ 256²** — physics + render at demo resolution.

**Accuracy flags**

- ✅ `_BatchedWorkspace`, CUDA graphs, GPU root control — in `xpbd_rod_solver.py` (verified in repo docs/code references).
- ✅ `renderDRR_forward_batched` — claimed in `fluorosim` (see `build_release_deck.py`).
- ⚠️ **“All 3 backends”** for multi-env physics — production `RodSolver` (PyTorch batched tree) and self-contained `XPBDRodSolver` differ; **Newton wrapper** is a third bridge — clarify which backend you benchmarked.
- ⚠️ **Full loop 63 FPS** — depends on physics substeps and whether render includes CPU post-processing; cite measurement conditions.

---

## Slide 8 — RL Training Pipeline (State-Based)

**On slide:** Environment, RSL-RL wrapper, PPO config, 512 envs / 1500 iterations, smoke test; footnote: state-based only, pixels Sprint 2+.

### Presenter notes

| Component | Implementation | Notes |
|-----------|----------------|-------|
| Environment | `isaaclab_newton/envs/catheter_state_env.py` | `RodSolver`, **not** `XCathRodSolver`; **no vessel mesh collision** in default cfg — straight/target reward |
| Action | Normalised push / rotate → `apply_proximal_control` | 2-D continuous |
| Observation | Flat: positions, tip vel, target, insertion depth | **No fluoroscopy image tensor** |
| Reward | Distance + time penalty + reach bonus | **Not** the 7-term clinical decomposition on slide (that is roadmap) |
| Training | `examples/train_catheter_state.py` + RSL-RL PPO | `num_envs=512` default in cfg |
| Smoke | `examples/run_catheter_state_smoke.py` (if present) | CI validation |

**Say:** “PPO at 512 envs is **state-based** navigation in open space / target point — not yet image-based fluoro policy training. Pixel observations require Sprint 2 render throughput.”

**Accuracy flags**

- ✅ State-based PPO @ 512 — matches `CatheterStateEnvCfg`.
- ⚠️ Slide implies full vessel navigation — default env may **not** include `XCathRodSolver` mesh containment; collision realism for RL is a **integration gap** unless a separate env cfg wires mesh in.
- ✅ “Pixel/fluoro observations Sprint 2+” — accurate.

---

## Slide 9 — Part 2 Divider: This Release

**On slide:** *PART 2 — This Release — Sprint 1 — features shipping now.*

### Presenter notes

Frame as **audit of fluorosim + IsaacLab integration**, not only greenfield IsaacLab code. Many items on slide 11 live under `i4h-sensor-simulation-internal/fluoro-simulator`.

**Accuracy:** ✅ Divider.

---

## Slide 10 — This Release — Stage 1 & 2 Demo (Interactive Viewport)

**On slide:** Title only — *Stage 1 & 2 demo* (visual screenshot in PPTX; not extractable from PDF text).

### Presenter notes — **Interactive viewport (primary demo for this slide)**

Use this slide to show **human-in-the-loop** fluoro, not bulk RL. Two apps implement “Stage 1” vs “Stage 2”:

| Stage | App | UI | Purpose |
|-------|-----|-----|---------|
| **Stage 1** | `examples/interactive_catheter_fluoro.py` | **Gradio** web UI | Browser demo, 256² fast render, synthetic elliptical vessel mesh |
| **Stage 2** | `examples/interactive_catheter_slang_viewport.py` | **Native OpenCV** split viewport | Patient CT (`mu_volume.npy`), real vessel mesh collision, DSA roadmap overlay, three guidance modes |

**Recommended live demo:** Stage 2 viewport — it is what partners should see for **patient-specific** behaviour.

#### Launch (Stage 2)

```bash
cd <IsaacLab-root>
source <conda>/isaaclab
export PYTHONNOUSERSITE=1
python source/isaaclab_newton/examples/interactive_catheter_slang_viewport.py \
  --ct-dir /tmp/patient_001 \
  --det-size 512 \
  --num-segments 80 \
  --style default
```

Requires display (`DISPLAY` or Wayland). Expect ~30 s startup: vessel mesh extract, DSA overlay precompute (4 projections), mesh-nav green overlays.

#### UI layout

- **Left pane:** Slang DRR fluoro + catheter Beer–Lambert composite (`render_batch_with_catheter`).
- **Right pane:** Telemetry HUD + **top-down catheter map** (XY in CT mm).

#### Guidance modes (press **G** to cycle)

| Mode | Name | Physics / visuals | Clinical analogy |
|------|------|-------------------|------------------|
| 1 | `guided` | Full rod projected onto **centerline** each step | Roadmap / planning teleport — **not** real device physics |
| 2 | `mesh_rail` | Shaft **snapped** to centerline post-step; free tip (~8 edges); `[` `]` tip bend; W/S along rail | “Wire on centerline” visualization; bend/rest mismatch causes seam artefacts (known) |
| 3 | `fluoro` | **No centerline snap**; `XCathRodSolver` mesh collision; hub W/S push, A/D torque; fixed **18°** tip in `rest_darboux` (8 edges); optional wall friction (default 0.28); **cyan** = DSA roadmap only | Closest to **live fluoro + registered roadmap** |

**Say on fluoro mode:** “Cyan is the static vessel roadmap — like registered DSA. The white wire is physics + mesh contact only. Push and torque are applied **only at the hub** (particle 0), matching proximal insertion and rotation.”

#### Controls (Stage 2)

| Input | Action |
|-------|--------|
| **W / S** | Advance / retract (proximal velocity) |
| **A / D** | Rotate hub (torque) — rotates fixed tip plane in fluoro |
| **Mouse drag** | Horizontal = insertion, vertical = torque |
| **G** | Cycle guidance: guided → mesh_rail → fluoro |
| **[ / ]** | Tip bend rate — **mesh_rail only** |
| **1–4** | AP, LAO-45, Lateral, RAO-30 |
| **F** | Toggle vessel overlay (cyan roadmap / green mesh-nav) |
| **R** | Reset (in fluoro: re-seat straight rod + settle in mesh) |
| **Space** | Pause |
| **Q / Esc** | Quit |

#### Technical stack (accurate one-liner)

`XCathRodSolver` (Warp XPBD Cosserat rod + **mesh-edge or SDF** vessel containment) → positions in solver-local metres → mm for `CatheterSegmentData` → `SlangDiffDRRRenderer` fused Beer–Lambert → optional cyan **precomputed** DSA diff overlay (Method 3: unnormalized mask DRR − normal DRR, **not** live `DSAPipeline` per frame).

#### What to point at on the screenshot

1. **Bone + soft-tissue DRR** background from patient μ volume.
2. **Bright catheter** polyline (smooth Catmull-Rom in fluoro for clean wire).
3. **Cyan corridor** — precomputed guide-path DSA difference (toggle with F).
4. HUD: `Guidance mode`, containment %, sign_scale, tip CT mm, FPS.

#### Demo script (90 s)

1. Start in **guided** — show rod locked to centerline; cyan roadmap on.
2. **G** → **mesh_rail** — W to advance along rail; `[` `]` to show tip steering; mention shaft snap vs bend constraint trade-off.
3. **G** → **fluoro** — wait for settle; gentle W/S and A/D; emphasize wire **not** glued to centerline.
4. **1 / 2** — change projection; fluoro image rotates, physics unchanged in world frame.
5. If instability: **R** in fluoro; avoid entering fluoro from a heavily bent mesh_rail state without reset.

#### Accuracy flags for slide 10

- ⚠️ Slide title “Stage 1 & 2” — confirm audience knows **Gradio = Stage 1**, **OpenCV viewport = Stage 2**.
- ❌ Do **not** claim live **bolus DSA cine** in the viewport loop — overlays are **precomputed**; full temporal bolus is `fluorosim` `render_cine` + `volume_callback`.
- ✅ Patient mesh collision in viewport — `XCathRodSolver` + `extract_vessel_mesh` on downsampled mask.
- ⚠️ **5-zone catheter μ** — viewport uses tuned scalar μ (~0.28–0.34), not per-zone segment table.

---

## Slide 11 — This Release — Completed Deliverables

**On slide:** Long checklist (Beer–Lambert, Slang fused, DSA, vessel boost, VMTK, bolus, instrument injection, C-arm presets, differentiable render, XPBD batching, RL PPO, E2E 35 mm traversal, etc.).

### Presenter notes

Group into four buckets when presenting:

**A — Rendering / fluorosim (package: `fluorosim`)**

| Deliverable | Status | Location (claimed) |
|-------------|--------|---------------------|
| Beer–Lambert CPU + Slang | ✅ | `diffdrr_slang.slang`, CPU compositor |
| Fused DRR + catheter | ✅ | `render_batch_with_catheter` |
| Detector physics + jitter | ✅ | `realism.py` — **CPU post** |
| DSA 4-step | ✅ | `dsa.py` `DSAPipeline` |
| Vessel boost μ×8 | ✅ | `vasculature.apply_vessel_boost` |
| Bolus / arrival map | ✅ | `gamma_variate`, `build_contrast_volume`, VMTK centreline |
| C-arm presets (9 vendors) | ✅ | `CarmGeometry` classmethods |
| Differentiable backward | ✅ | `renderDRR_backward` |
| Batched render | ✅ | `renderDRR_forward_batched` |
| Instrument injection | ✅ | `instrument-injection` Warp `atomic_max` |

**B — Physics / IsaacLab Newton**

| Deliverable | Status | Location |
|-------------|--------|----------|
| Proximal push/rotate | ✅ | `apply_proximal_control` / `_gpu` |
| XPBD batched + CUDA graphs | ✅ | `xpbd_rod_solver.py` |
| Floor restitution | ✅ | XPBD solver |
| Vessel **mesh** collision (viewport) | ✅ | `xcath_rod_solver.py` + viewport |
| RL PPO 512 | ✅ | `catheter_state_env.py` — **RodSolver**, open target |

**C — Interactive / E2E**

| Deliverable | Status | Notes |
|-------------|--------|-------|
| E2E 35 mm traversal in Slang | ⚠️ | Cite your bench log; viewport demonstrates motion qualitatively |
| Gradio fluoro demo | ✅ | `interactive_catheter_fluoro.py` |
| Native viewport | ✅ | `interactive_catheter_slang_viewport.py` |

**Accuracy flags**

- ✅ Most rows are fair for **combined release** (IsaacLab + fluorosim sibling repo).
- ❌ **Do not imply every row is exercised in the Stage 2 viewport loop** (e.g. live `DSAPipeline`, 5-zone μ, `apply_realism` every frame).
- ⚠️ **“SDF / mesh collision”** in summary slide 13 — **viewport yes**; **default RL env** may not use `XCathRodSolver`.
- ⚠️ **9 vendor presets** — implemented in fluorosim; viewport uses fixed `PROJECTIONS` dict (AP/LAO/Lateral/RAO), not vendor factory selectors in UI.

---

## Slide 12 — Part 3 Divider: Roadmap

**On slide:** *PART 3 — Roadmap*

### Presenter notes

Transition to Sprint 2/3 and agentic workflow. Use slide 13 summary columns.

**Accuracy:** ✅ Divider.

---

## Slide 13 — Summary (TODAY / THIS RELEASE / NEXT / FOLLOWING)

**On slide:** Four columns — pipeline implemented; DSA/bolus/presets; Sprint 1 deliverables; Sprint 2 Texture2DArray, pixel obs, GPU detector physics, beam hardening; Sprint 3 DR, Gymnasium, agent skills, FID metrics, F/T, CBCT.

### Presenter notes

**TODAY column — say:**

- End-to-end **X-ray** path: μ volume → fused render → interactive demo.
- **~25 FPS** @ 512² render, **~1,300 Hz** single-env physics (20 segments).
- **State-based PPO @ 512** envs runs; not image-based policy training yet.

**THIS RELEASE column — say:**

- Fluorosim fidelity pack: DSA, bolus, vessel boost, presets, batched Slang dispatch, XPBD GPU control.
- IsaacLab: proximal control, RL env, **viewport** with patient mesh + three guidance modes.

**NEXT RELEASE — say:**

- **Texture2DArray** (or equivalent) for render cache at **N>8**.
- **Image observations** for RL.
- **GPU detector chain** on Slang output (remove CPU readback bottleneck).
- **Beam hardening** — still monoenergetic today.

**FOLLOWING — say:**

- Gymnasium hardening, domain randomisation, agentic skills (Phase 2/3), realism metrics (FID/SSIM), force/torque and CBCT extensions.

**Accuracy flags**

- ⚠️ **“SDF / mesh collision for vessel walls” under THIS RELEASE** — true for **XCathRodSolver / viewport**; clarify **RL training env** may still use simpler collision until integrated.
- ✅ Pixel obs / Texture2DArray / beam hardening as **next** — consistent with deck and code comments.
- ⚠️ **Ultrasound** and full multi-sensor story — **not in this 13-slide PDF**; do not claim from this export alone.

---

## Appendix A — Slide / PDF export mismatch

The PDF export contains **13 slides** while footers reference **“5 / 32”**, **“15 / 32”**, **“16 / 32”**. For a full 32-slide walkthrough, use `docs/sensor_simulation_release_deck_notes.md` (older numbering) or the source `.pptx`. This document maps **PDF slide index 1–13** only.

---

## Appendix B — Quick reference: key file paths

| Artifact | Path |
|----------|------|
| Native interactive viewport | `source/isaaclab_newton/examples/interactive_catheter_slang_viewport.py` |
| Gradio interactive fluoro | `source/isaaclab_newton/examples/interactive_catheter_fluoro.py` |
| Vessel + collision solver | `source/isaaclab_newton/isaaclab_newton/solvers/xcath_rod_solver.py` |
| RL environment | `source/isaaclab_newton/isaaclab_newton/envs/catheter_state_env.py` |
| Gradio technical walkthrough | `source/isaaclab_newton/docs/gradio_ui_technical_walkthrough.md` |
| Fluorosim (external) | `../i4h-sensor-simulation-internal/fluoro-simulator/` |

---

## Appendix C — Consolidated inaccuracy list (for slide authors)

| Slide | Issue | Recommendation |
|-------|--------|----------------|
| 2 | “512 envs” without “training vs demo” | Add footnote: interactive = 1 env |
| 4 | Detector steps implied in fused GPU march | Split box: “GPU: Beer–Lambert” vs “CPU: realism.py” |
| 6–7 | Multi-env render ✓/◐ vs 60 Hz target | Label “batched kernel live; 512²@512 env not met” |
| 8 | Implies vessel navigation in RL | State “RodSolver, target point; mesh collision in viewport” |
| 10 | “Demo” without naming viewport | Add subtitle: `interactive_catheter_slang_viewport.py` |
| 11 | All items in one demo | Tag each row: fluorosim / IsaacLab / viewport |
| 13 | Mesh collision in THIS RELEASE | Qualify: viewport + XCath; RL integration pending |

---

*Generated for presenter use. Re-verify benchmark numbers before external talks if hardware or commit changes.*
