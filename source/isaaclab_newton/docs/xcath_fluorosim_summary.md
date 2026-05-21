# XCATH Robotics — FluoroSim Collaboration Summary

> Summary of XCATH's work with NVIDIA's FluoroSim platform for
> neurointerventional DSA simulation, based on March and April 2026
> presentations by Jung-eun Park, AI/ML Engineer.

---

## What XCATH Is Building

A **real-time monitoring system for neurointerventional surgery** that overlays
3D vessel reconstructions onto live fluoroscopy during catheter procedures. The
AI components require:

1. **Vessel segmentation** on intra-operative fluoroscopy
2. **2D/3D registration**: pre-op CTA to intra-op fluoroscopy

The bottleneck: **paired training data** (CTA + fluoroscopy + ground-truth pose)
does not exist and cannot be acquired at scale — different modalities, acquisition
times, and patient positions. NVIDIA's FluoroSim generates synthetic paired data
to solve this.

---

## What They Built on FluoroSim

### 5 Simulation Methods (All Completed)

| Method | Description | Status |
|--------|-------------|--------|
| **DRR** | Base fluoroscopy rendering from CTA (HU → mu → GPU ray march) | Done |
| **Vessel Boost** | Vessel mu ×8 amplification — makes vessels visible in standard DRR without subtraction | Done |
| **DSA** | Mask − Contrast subtraction + scatter/misregistration realism | Done |
| **Bolus Label** | Voxel-level label propagation (replaced — did not follow branch structure) | Replaced |
| **Bolus Centerline** | VMTK centerline + Dijkstra + blood flow model + DSA rendering | Done |

### DSA Mathematical Pipeline

The full DSA pipeline replicates clinical acquisition:

1. **Contrast DRR**: I_contrast = I_0 × exp(−∫[mu_tissue + mu_vessel] ds)
   — includes iodine contrast enhancement in vessel voxels.

2. **Mask DRR**: I_mask = I_0 × exp(−∫ mu_tissue ds) + jitter
   — vessels replaced with surrounding tissue mu; misregistration jitter added
   (rotation sigma = 0.05 deg, translation sigma = 0.1 mm) to simulate patient
   motion between mask and contrast acquisitions.

3. **Digital Subtraction**: DSA = I_mask − I_contrast
   — isolates vessel-only signal by canceling background anatomy.

4. **Post-processing**: DSA_final = gamma(k × DSA + scatter + noise)
   — k = 20 (contrast amplification), gamma = 0.8 (display correction),
   Gaussian scatter convolution, Poisson photon noise.

### Clinical C-arm Presets Validated

| System | SDD (mm) | SID (mm) | Resolution | Pixel Spacing |
|--------|----------|----------|------------|---------------|
| GE OEC 9900 | 1020 | 510 | 1024×1024 | vendor spec |
| GE Innova IGS 540 | 1200 | 750 | 2048×2048 | vendor spec |
| Siemens Artis zee | 1250 | 780 | 2480×1920 | vendor spec |
| Philips Azurion 7 | 1240 | 780 | 2480×1920 | vendor spec |

---

## Blood Flow / Bolus Dynamics

### Two-Stage Pipeline

**Stage 1 (VMTK conda env — CPU):**
1. VMTK IsoSurface + NetworkExtraction from vessel segmentation
2. Graph build with adjacency, radii, endpoints
3. Injection root selection (lowest-Z endpoint per branch)
4. Blood flow velocity model
5. Travel-time computation via Dijkstra + disconnected branch bridging
6. Voxel-level arrival map via KDTree (RAS→LPS)

**Stage 2 (system Python + GPU):**
1. Load arrival_map.npy
2. Gamma-variate × injection pulse convolution
3. Uniform iodine delta_mu per frame
4. Per-frame mu update → GPU ray-march → DRR → DSA

### Velocity Model Evolution

| Version | Model | Formula | Calibration |
|---------|-------|---------|-------------|
| March 2026 | Allometric (Murray's Law) | v = 150 × (r/r_ref)^0.5 | Distal MCA at 71% of ICA (too fast), bolus delay 5.6 s vs real 2.13 s (2.6× off) |
| April 2026 | Hagen-Poiseuille | v = 350 × (r/2.0)^2.0 | Distal MCA at 25% of ICA (realistic), bolus delay 1.90 s vs real 2.13 s (89% match) |

### Bolus Temporal Profile

Gamma-variate concentration curve at each voxel:

    C(t) = A × (t − T)^alpha × exp(−(t − T) / beta)

| Parameter | March | April | Meaning |
|-----------|-------|-------|---------|
| alpha | 2.0 | 1.5 | Rise steepness |
| beta | 1.4 s | 0.4 s | Washout rate |
| t_peak | 2.8 s | 0.6 s | Time to peak |
| FWHM | ~2.5 s | ~0.7 s | Bolus width |

### Dispersion Correction (April W2–W3)

Two-phase per-voxel TDC width modulation for realistic proximal-distal asymmetry:

- **Phase 1 (distance-based)**: beta_eff = beta_0 + k × arrival_time
  - k = 0.15, beta_0 = 0.4
  - Proximal: beta = 0.4 (FWHM ~0.7 s), Distal: beta ≈ 1.0 (FWHM ~2.8 s)
  - Asymmetry ratio: 2.55×

- **Phase 2 (radius-based)**: beta_eff += alpha_res / max(r_voxel, 0.1)
  - alpha_res = 0.1
  - Thin vessels → wider TDC (longer residence time)

### Selective Injection — Feeding Trunk Excision (F-8)

Solves the problem that real DSA injects at ONE vessel root (single ICA),
but the connected vessel graph allows retrograde flow via Circle of Willis:

1. Territory partition via multi-source Dijkstra (Voronoi)
2. Identify non-selected feeding artery trunks (ICA/VA proximal segments)
3. Excise trunks from graph (models antegrade pressure barrier)
4. Dijkstra from selected root on excised graph

Results: Root 1 (R-ICA) selective injection matches real DSA territory coverage
(43.1% of nodes reached, 69K voxels). Peak delay: sim 2.40 s vs real 2.13 s.

---

## FluoroSim vs DeepDRR Comparison

### Key Findings

| Metric | FluoroSim (mono 70 keV) | DeepDRR (poly 90 kVp) |
|--------|-------------------------|----------------------|
| Speed | 249 FPS | 31 min / 100 frames |
| Static DRR SSIM | — | 0.964 |
| Static DRR MAE | — | 9.3% |
| Pearson Correlation | — | 0.981 |
| Beam hardening | Not modeled | Modeled |

**Conclusion**: Monoenergetic (FluoroSim) is structurally equivalent to
polyenergetic (DeepDRR) for cerebral DSA. The 9.3% difference is concentrated
at bone edges; soft tissue is nearly identical. For DSA specifically, beam
hardening cancels in subtraction (affects mask and contrast equally).

### Monoenergetic Limitations (Cannot Fix by Tuning)

| Limitation | Impact | Evidence |
|------------|--------|----------|
| Beam hardening | mu treated as constant along ray | Noise CV: sim 0.47 vs real 1.46 (3.1×) |
| Iodine K-edge (33.2 keV) | mu jump at K-edge missed; 70 keV misses this | Bolus delta-mu weaker than real at spectrum-avg |
| Energy-dependent scatter | Single-energy scatter model is wrong | NPS low-frequency mismatch |
| Bone subtraction artifact | Perfect cancellation → too clean background | CNR overestimated |
| Detector energy response | DQE varies with energy; all photons weighted equally | Contrast/noise characteristics differ |
| Heel effect | Spatially varying spectrum not modeled | Intensity gradient + noise variation not modeled |

**Workaround for #4**: Inject misregistration jitter (rotation sigma = 0.05 deg,
translation sigma = 0.1 mm) between mask and contrast DRRs to artificially
create bone edge artifacts mimicking real DSA.

---

## XCATH's Feedback on FluoroSim

From their email:

1. **Contrast injection point selection** — they experimented with selecting
   injection points and observing distribution dynamics over time. This is the
   selective injection / F-8 feature they built.

2. **Noise and scatter validation** — compared FluoroSim with DeepDRR, found
   "high degree of similarity" (SSIM = 0.964). Confirms monoenergetic is
   sufficient for DSA.

3. **Beam hardening** — they could NOT validate beam hardening effects because
   their synthetic phantom and real CTA data are not expected to exhibit it.
   They want to discuss how we plan to handle beam hardening.

### Beam Hardening — What They're Asking

Beam hardening is the artifact caused by polychromatic X-ray spectra: as the
beam passes through dense material (bone), low-energy photons are preferentially
absorbed, shifting the remaining spectrum toward higher energies. This causes:

- **Cupping artifacts** in CT reconstruction
- **Dark streaks** between dense structures in DRR
- **Non-linear attenuation** — mu is not constant along the ray path

FluoroSim is monoenergetic (single mu per voxel at 70 keV), so beam hardening
is fundamentally not modeled. To add it, FluoroSim would need:

- **Polyenergetic rendering**: discretize the X-ray spectrum into 5–10 energy
  bins (20–120 keV), compute energy-dependent mu for each tissue type at each
  bin, ray-march once per bin, and sum the transmitted intensities weighted by
  the spectral distribution.
- **Material decomposition**: decompose each voxel into basis materials (water,
  bone, iodine) with known energy-dependent attenuation curves, rather than
  a single monoenergetic mu value.

This is a significant architectural change to the Slang shader — the current
`computePixelIntensity` would need to loop over energy bins inside the ray
march, multiplying the computation cost by the number of bins.

**However**, XCATH's own analysis shows that for DSA (their primary use case),
beam hardening cancels in subtraction. So the practical question is: do they
need polyenergetic for **non-subtracted** fluoroscopy (standard fluoro during
catheter navigation), or only for DSA?

---

## Remaining Work (from April presentation)

| Item | Status | Estimate |
|------|--------|----------|
| Multi-patient validation (2–3 additional datasets) | Blocked on data availability | 2–3 days |
| Cardiac pulsation (TDC double-peak from heartbeat) | Optional | 3 hours |

### Next Steps (from March presentation)

| Task | Description |
|------|-------------|
| Synthetic dataset generation | Large-scale paired data: CTA + synth DSA + GT pose |
| Registration algorithm | PoseNet / DeepFluoro / DiffDRR, target <100 ms/frame |
| Realism evaluation | FID / perceptual metrics, vessel visibility, registration accuracy |

XCATH has 50 internal patients with CTA + clinical DSA (cannot share externally).
The chicken-and-egg: need registration to evaluate, need evaluation to validate
registration. Strategy: synthetic data for training, real data for validation,
iterative refinement.

---

## Engineering Artifacts Delivered

| Artifact | Description |
|----------|-------------|
| `adapter.py` | DeepDRR wrapper for unified API |
| `_inject_iodine_material` | 4th material monkey-patch for DeepDRR iodine |
| Unified Docker | FluoroSim + DeepDRR in single container |
| 125 tests | Modular test suite for the simulation pipeline |
| Selective injection (F-8) | Feeding Trunk Excision algorithm |
| Dispersion correction | Per-voxel TDC width modulation (2-phase) |
| Code modularization | From monolithic scripts to modular architecture |
