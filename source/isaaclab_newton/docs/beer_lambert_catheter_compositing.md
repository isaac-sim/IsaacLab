# Beer-Lambert Catheter Compositing on Fluoroscopy DRR

> Presentation notes — technical audience.
> Source: `visualize_rod_fluoroscopy.py`, `newton_xpbd_rod_wrapper.py`, `xpbd_rod_solver.py`

---

## 1 — Problem Statement

We need to render a simulated catheter/guidewire on top of patient-specific
X-ray imagery so the composite is **radiometrically indistinguishable** from a
real fluoroscopy acquisition.  Naive opaque-line overlays fail because:

- Real radio-opaque devices *attenuate* the beam — they darken the image rather
  than occlude it.
- Background anatomy remains visible through the device.
- Attenuation is additive in log-space: self-crossings produce deeper shadows.
- Different catheter zones (marker bands, braided shaft, polymer tip) have
  distinct radio-opacity.

---

## 2 — Background Image: DRR from Patient CT

The fluoroscopy background is a **Digitally Reconstructed Radiograph (DRR)**
generated offline from volumetric CT data via DiffDRR+Slang (a differentiable
volume renderer).

Four standard C-arm angulations are pre-rendered:

| View     | Filename                            | C-arm angle |
|----------|-------------------------------------|-------------|
| AP       | `xray_diffdrr_slang_AP.png`         | 0°          |
| LAO 30°  | `xray_diffdrr_slang_LAO_30.png`     | +30°        |
| Lateral  | `xray_diffdrr_slang_Lateral.png`    | +90°        |
| RAO 30°  | `xray_diffdrr_slang_RAO_30.png`     | −30°        |

At runtime the closest DRR is selected by angular distance to the current
virtual C-arm angle.  The extrinsic matrix is recomputed to match, so the
catheter's 3D→2D projection is geometrically consistent with the view.

---

## 3 — Catheter State Extraction from the Rod Solver

### 3.1  Newton XPBD path (`NewtonXPBDRodSolver`)

The catheter is modelled as a Cosserat rod discretised into `N` particles,
solved by Newton's `SolverXPBDRod` using a block-Thomas direct solve on the
6×6 block-tridiagonal JMJT system.

```
state.particle_q   →   (N, 3) float64 positions  [metres]
         │
         ▼
_sync_positions_from_state()
         │
         ▼
torch.Tensor  (1, N, 3)  float32   [metres]
```

Newton stores only positions; orientations are reconstructed from polyline
tangents via central differences (`orientations_xyzw_along_polyline()`).

### 3.2  Self-contained XPBD path (`XPBDRodSolver`)

All Warp kernels are embedded — no external Newton dependency.  The solver
natively maintains both positions and quaternion orientations in Warp arrays,
exposed to PyTorch via zero-copy `wp.to_torch()`.

```
_Workspace.positions      →   (N+1, 3) wp.vec3   [metres]
_Workspace.orientations   →   (N+1, 4) wp.quat   [xyzw]
```

### 3.3  Unit conversion

Solver output is in **SI metres**.  The compositing pipeline scales positions
by ×1000 to millimetres, matching clinical C-arm geometry conventions (SID, SOD,
pixel spacing all specified in mm).

---

## 4 — C-arm Projection Model

A standard pinhole camera model parameterised by interventional C-arm geometry.

### 4.1  Intrinsic matrix

```
        ┌ f   0   cx ┐        f  = SID / pixel_spacing
  K  =  │ 0   f   cy │        cx = image_width  / 2
        └ 0   0   1  ┘        cy = image_height / 2
```

Typical values: SID = 1000 mm, pixel spacing = 0.81 mm/px → f ≈ 1235 px.

### 4.2  Extrinsic matrix

```
  R  = Rx(cran/caud) · Ry(LAO/RAO)
  t  = −R · R^T · [0, 0, −SOD]^T
  [R | t]  : 3×4
```

The iso-centre is at the world origin; the X-ray source sits at distance SOD
(600 mm) along the rotated optical axis.

### 4.3  3D → 2D projection

```
  P = K · [R | t]
  uv = (P · [X; 1]) / z        homogeneous divide
```

### 4.4  Cone-beam magnification of catheter radius

Each segment's projected pixel radius accounts for depth-dependent
magnification:

```
  r_px = r_physical · f_px / z_cam
```

where `z_cam` is the segment midpoint's depth in camera coordinates.  Proximal
segments (closer to source) appear larger — exactly matching real cone-beam
geometry.

---

## 5 — Per-Segment Attenuation Profile

The catheter is not uniformly radio-opaque.  A piecewise linear attenuation
profile models real device construction:

| Zone                   | Segments    | μ value | Physical material      |
|------------------------|-------------|---------|------------------------|
| Proximal marker band   | 0–1         | 3.0     | Tungsten               |
| Braided shaft          | 2 → 60%    | 0.8     | Nitinol braid          |
| Transition zone        | 60% → 85%  | 0.8→0.2 | Sparse braid + polymer |
| Soft polymer tip       | 85% → 95%  | 0.15    | Pure polymer           |
| Distal tip marker      | last 3      | 5.0     | Platinum coil          |

The μ values are dimensionless effective attenuation coefficients (μ × nominal
diameter).  Higher values produce darker shadows on the fluoroscopy image.

---

## 6 — Beer-Lambert Compositing

This is the core rendering step.  For each of the `N−1` segments:

### 6.1  Cylinder chord thickness

The projected cross-section of a cylinder at perpendicular pixel distance `d`
from the segment centreline has chord length:

```
  t(d) = 2 · √(r² − d²)       for d < r
  t(d) = 0                     for d ≥ r
```

This is the exact ray–cylinder intersection length — yielding physically correct
smooth edge falloff rather than hard-edged aliased lines.

### 6.2  Attenuation map accumulation

Chord thickness is normalised by diameter and weighted by the segment's μ:

```
  atten_map(u,v)  +=  μ_i · t_i(u,v) / (2·r_i)
```

Accumulation is **additive in the exponent** — self-crossings where the catheter
overlaps itself produce deeper shadows, correctly modelling the physics of
multiple attenuating layers.

### 6.3  Transmission and compositing

The DRR background is multiplicatively darkened via Beer-Lambert:

```
  T(u,v) = exp(−atten_map(u,v))
  I_final(u,v) = I_DRR(u,v) · T(u,v)
```

This produces:

- Physically correct darkening proportional to path length through the device
- Smooth sub-pixel edge profiles from the chord geometry
- Visible background anatomy through the catheter body
- Deeper shadows at crossings and marker bands
- Near-zero attenuation through the polymer tip

---

## 7 — Detector Physics Simulation

Three post-processing stages model real flat-panel detector characteristics on
top of the Beer-Lambert composite.

### 7.1  Veiling glare / X-ray scatter

```
  I_blocked = I_DRR · (1 − T)
  I_scatter = GaussianBlur(I_blocked, σ=18 px)
  I_final  += 0.03 · I_scatter
```

Models photon scatter in patient tissue and detector housing.  The large kernel
(σ=18 px) simulates the long-range scatter halo that softens shadows in real
fluoroscopy.

### 7.2  Detector point-spread function

```
  I_final = GaussianBlur(I_final, σ=0.7 px)
```

Models the finite spatial resolution of the CsI scintillator / flat-panel
detector.

### 7.3  Poisson quantum noise

```
  photons(u,v) = I_final(u,v) · N_0           N_0 = 2000 photons/px
  noisy(u,v)   ~ Poisson(photons(u,v))
  I_final(u,v) = noisy(u,v) / N_0
```

Reproduces shot noise characteristic of low-dose fluoroscopy.  Catheter shadow
regions receive fewer photons → proportionally higher relative noise, matching
real image statistics.

---

## 8 — End-to-End Pipeline

```
┌───────────────────────────────────────────────────────┐
│  XPBD Cosserat Rod Solver  (Newton or self-contained) │
│  step() → state.particle_q → (N, 3) positions [m]    │
└─────────────────────┬─────────────────────────────────┘
                      │  ×1000  →  [mm]
                      ▼
        ┌─────────────────────────────┐
        │  C-arm Projection (K, [R|t])│
        │  project_points_3d_to_2d()  │──→ (N, 2)  pixel coords
        │  compute_projected_radii()  │──→ (N−1,)  px radii
        └─────────────┬───────────────┘
                      │
                      ▼
        ┌─────────────────────────────────────────────┐
        │  Beer-Lambert Compositing                    │
        │                                              │
        │  for each segment i:                         │
        │    chord(d) = 2√(r² − d²)                   │
        │    atten_map += μ_i · chord / 2r             │
        │                                              │
        │  I = I_DRR · exp(−atten_map)                 │
        └─────────────┬───────────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────────────────────┐
        │  Detector Physics                            │
        │  ① Scatter:  +3% GaussBlur(blocked, σ=18)   │
        │  ② PSF:      GaussBlur(σ=0.7)               │
        │  ③ Noise:    Poisson(2000 photons/px)        │
        └─────────────┬───────────────────────────────┘
                      │
                      ▼
              BGR uint8 fluoroscopy frame
              (saved as fluoro_frame_XXXX.png)
```

---

## 9 — Output Frames

The `fluoro_frames/` directory contains snapshots at key simulation timesteps
(frame 0, 22, 37, 89, 150, 450, 899, 1799, …).  Each frame shows:

- **HUD overlay** — simulation time, FPS, catheter tip position (mm), C-arm
  angle
- **View label** — AP / LAO 30 / Lateral / RAO 30
- **Catheter shadow** — Beer-Lambert attenuation visible as a dark silhouette
  with smooth edges, overlaid on the DRR skull anatomy

The catheter is visible drooping under gravity from its fixed proximal end,
settling into a steady-state configuration by ~15 s of simulation time.

---

## 10 — Key Design Decisions

1. **Multiplicative (not additive) compositing** — the catheter darkens the
   background rather than painting over it.  This is physically correct and
   preserves anatomical context through the device.

2. **Chord-thickness model instead of line rendering** — using the exact
   ray-cylinder intersection produces smooth sub-pixel edges and correct
   intensity profiles across the catheter diameter.

3. **Additive attenuation in log-space** — multiple overlapping segments
   accumulate in the exponent before exponentiation, correctly modelling
   stacked attenuators without double-counting.

4. **Zone-specific μ profile** — marker bands, braided shaft, and polymer tip
   have distinct attenuation values matching real device construction, producing
   the characteristic bright-dark-bright pattern visible on clinical
   fluoroscopy.

5. **Full detector physics chain** — scatter, PSF, and Poisson noise transform
   a "perfect" composite into a realistic noisy fluoroscopy image, critical for
   training perception models or validating image-guided navigation algorithms.
