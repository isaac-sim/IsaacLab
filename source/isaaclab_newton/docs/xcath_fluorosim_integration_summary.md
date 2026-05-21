# XCATH FluoroSim Integration — Summary

> Summary of XCATH's April 2026 presentation on integrating the FluoroSim package
> into their neurointerventional DSA simulation pipeline.
> Source: *202604_XCATH_NVIDIA_Presentation_JungeunPark*
> Author: Jung-eun Park, AI/ML Engineer, XCATH Robotics

---

## What They Built on Top of FluoroSim

XCATH has taken the FluoroSim Slang DiffDRR renderer and built a complete DSA
(Digital Subtraction Angiography) simulation pipeline around it. Over the April
W1–W3 sprint, they went from a monolithic prototype to a modular, tested system
with 17 of 19 deliverables completed, 125 tests, and a unified Docker image.

---

## Key Capabilities Implemented Using FluoroSim

### 1. Mask Subtraction DSA Workflow

Full 4-step DSA: mask DRR, contrast DRR, log subtraction, post-processing. Built
on FluoroSim's Beer-Lambert ray march.

### 2. Selective Contrast Injection (Feeding Trunk Excision, F-8)

The headline feature. Real DSA catheterizes one ICA and shows only the ipsilateral
hemisphere. Their previous sim injected all 4 roots simultaneously, which is
non-physical. They solved this by:

1. Partitioning the vessel graph via multi-source Dijkstra (Voronoi territories)
2. Identifying non-selected feeding artery trunks (ICA/VA proximal segments)
3. Excising trunks from the graph (models antegrade pressure barrier)
4. Running Dijkstra from the selected root on the excised graph

Results for Root 0 (L-ICA) selective injection:

| Metric | Baseline (4-root) | Root 0 (L-ICA) | Real DSA |
|--------|-------------------|-----------------|----------|
| Nodes reached | 5,243 (100%) | 2,259 (43.1%) | — |
| Coverage | 156k voxels | 69k voxels | — |
| Peak delay | 2.20 s | — | 2.13 s |
| Distal TTP | 3.10 s | 3.30 s | 3.47 s |

### 3. Hagen-Poiseuille Blood Flow Model

Replaced the initial allometric velocity model (power=0.5) with a Hagen-Poiseuille
model (power=2.0):

```
v(r) = v_ref × (r / r_ref)^p
v_ref = 350 mm/s,  r_ref = 2.0 mm,  p = 2.0
```

| Metric | March (power=0.5) | April W1 (power=2.0) |
|--------|-------------------|----------------------|
| Distal MCA velocity (r=1mm) | 71% of ICA (too fast) | 25% of ICA (realistic) |
| Real DSA delay match | 5.6s vs 2.13s (2.6x off) | 1.90s vs 2.13s (89% match) |

### 4. Dispersion Correction (Two-Phase)

Per-voxel TDC (time-density curve) width modulation for realistic proximal-distal
asymmetry:

**Phase 1 — Distance-based:**

```
β_eff = β₀ + k × arrival_time
k = 0.15,  β₀ = 0.4
Proximal: β = 0.4  (FWHM ~0.7s)
Distal:   β ≈ 1.0  (FWHM ~2.8s)
```

**Phase 2 — Radius-based (additive):**

```
β_eff += α_res / max(r_voxel, 0.1)
α_res = 0.1
Thin vessels → wider TDC
```

**Combined:** `β_eff = β₀ + k·T + α/r`

k Sweep Results:

| k | Proximal FWHM | Distal FWHM | Ratio |
|---|---------------|-------------|-------|
| 0.00 | 1.00s | 1.60s | 1.60x |
| 0.10 | 1.10s | 2.40s | 2.18x |
| **0.15** | **1.10s** | **2.80s** | **2.55x** (selected) |
| 0.20 | 1.10s | 3.20s | 2.91x |

Gate passed: distinguishability 0.607 (target >0.3), asymmetry 2.55x (target >=1.5).

### 5. Iodine as a 4th Material

For the DeepDRR comparison, they patched in iodine as a 4th material with 8.4x
contrast amplification, enabling quantitative CNR comparison. Sim CNR reached 26.85
(previously 0 due to a bug).

### 6. Connected Vessel Graph

Endpoint merging of the VMTK centerline from 176 disconnected segments into 1
connected component with 320 branch points.

---

## FluoroSim vs DeepDRR Validation — The Key Finding

XCATH ran a rigorous 3-phase comparison between FluoroSim (monoenergetic, 70 keV)
and DeepDRR (polyenergetic, 90 kVp spectrum):

| Phase | Test | Result |
|-------|------|--------|
| Spike | Synthetic phantom (256^3) — feasibility + timing | SSIM=0.855, rebuild=1.72s/frame |
| Phase 1 | Real CTA (HUMI029) — static DRR (aligned) | SSIM=0.964, MAE=0.093, Corr=0.981 |
| Phase 2 | Dynamic bolus DSA (100 frames, 10s) | Iodine 4th material: 8.4x contrast |

**Conclusion: Monoenergetic is practically sufficient for cerebral DSA.** The 9.3%
bone-edge difference from beam hardening washes out in mask subtraction because it
affects both mask and contrast DRRs equally. FluoroSim runs at 249 FPS vs DeepDRR's
31 minutes for 100 frames.

---

## Monoenergetic Limitations Documented

XCATH identified 6 fundamental limits of the monoenergetic approach that cannot be
fixed by parameter tuning — they require polyenergetic rendering architecture:

| # | Limitation | Why Mono Cannot Solve It | Measured Impact |
|---|-----------|--------------------------|-----------------|
| 1 | Beam hardening | μ changes along ray as low-energy photons are absorbed; mono treats μ as constant | Noise CV: sim 0.47 vs real 1.46 (3.1x gap) |
| 2 | Iodine K-edge (33.2 keV) | μ jumps 2.5x at K-edge; 70 keV misses this; spectrum integration required | DSA vessel signal weaker than real |
| 3 | Energy-dependent scatter | Compton scatter varies with energy; single-energy scatter model is wrong | NPS low-frequency mismatch |
| 4 | Bone subtraction artifact | Beam hardening differs between mask/contrast; perfect cancellation → too clean background | Sim background too clean; CNR overestimated |
| 5 | Detector energy response | Real flat-panel DQE varies with energy; all photons weighted equally | Contrast and noise characteristics differ |
| 6 | Heel effect | Anode angle → spatially varying spectrum; uniform field assumption | Intensity gradient + noise variation not modeled |

**Workaround for #4:** Inject misregistration jitter (rotation sigma=0.05 deg,
translation sigma=0.1 mm) between mask/contrast DRRs to artificially create
bone-edge artifacts mimicking real DSA.

---

## Bolus Evolution

100 frames at 10 fps, 10-second simulation window, gamma-variate alpha=1.5, beta=0.6,
v_ref=350 mm/s (post-tuning config):

| Phase | Time | Behavior |
|-------|------|----------|
| Early (<1s) | Cervical ICA fills first — proximal bolus entry. Arrival time T(voxel) smallest here; low alpha means sharp rise. |
| Peak (2–4s) | Circle of Willis and MCA branches fully enhanced. Injection pulse fully convolved → maximum vessel contrast. |
| Washout (>5s) | Proximal darkens, distal still holds contrast briefly. Single-curve model limits sequential-washout realism. |

Remaining visual gap: vessel dispersion correction (partially addressed by the
two-phase beta modulation).

---

## Real vs Sim Visual Comparison

Side-by-side observations on Patient #1, same acquisition geometry:

| Feature | Real DSA | Simulated DSA |
|---------|----------|---------------|
| Injection mode | Selective ICA injection (single hemisphere) | Previously bilateral; now mitigated via selective root point (F-8) |
| Vessel edges | Focal-spot blur softens edges naturally | Point-source renderer → crisper vessels. Mitigation: multiple X-ray sources (?) |
| Background | Bone subtraction residual visible | Bone removed via clean mask subtraction. Mitigated: misregistration jitter added |
| Distal branches | Fill finely | Rendered but less dense. Limitation: finer branch annotation needed |

---

## What Remains

Only 2 items out of 19:

| Item | Description | Estimate | Dependency |
|------|-------------|----------|------------|
| Multi-patient validation | Currently Patient #1 only; need 2–3 additional datasets | 2–3 days | Blocked on data availability |
| Cardiac pulsation | TDC double-peak from heartbeat modulation | 3 hours | None (optional) |

---

## Strategic Implications for Our Pipeline

1. **Monoenergetic sufficiency validated.** XCATH's SSIM=0.964 result is strong
   evidence that FluoroSim's monoenergetic Slang renderer is sufficient for DSA
   training data generation. We do not need polyenergetic rendering for the catheter
   navigation use case.

2. **Architectural compatibility.** XCATH's bolus model, selective injection, and
   dispersion correction are all built as layers on top of FluoroSim's core
   Beer-Lambert ray march. Our fused DRR+catheter Slang path is architecturally
   compatible with their pipeline — the catheter compositing and the bolus dynamics
   operate on the same mu-volume and can coexist.

3. **Gap convergence.** The features XCATH has closed — vessel boost via iodine 4th
   material, DSA mask subtraction, misregistration jitter, Hagen-Poiseuille blood
   flow, dispersion correction — are exactly the features listed as "MISSING" in our
   gap analysis. XCATH is building them from their side, which means integration
   rather than reimplementation.

4. **Performance advantage confirmed.** FluoroSim at 249 FPS vs DeepDRR at 31
   minutes for 100 frames. For training data generation at scale (10,000+ frames),
   the monoenergetic path is the only viable option.

5. **Known limits are acceptable.** The 6 fundamental monoenergetic limitations are
   documented and bounded. The 3.1x noise CV gap is the largest, but for RL policy
   training with domain randomization, noise statistics are randomized anyway. The
   practical sim-to-real gap for DSA is manageable.
