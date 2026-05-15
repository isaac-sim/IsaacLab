# Unified Sim Loop — Beer-Lambert Catheter Compositing Frames

> Technical presenter notes for the "Unified sim loop" slide showing three
> generated fluoroscopy frames at different simulation timesteps.
> Reference implementation: `visualize_rod_fluoroscopy.py`

---

## Slide Overview

"This slide shows three frames from the unified simulation loop at different
timesteps — t = 0.03 s, t = 0.77 s, and t = 7.53 s. Each frame is a
Beer-Lambert composite: the dark catheter shadow is physically derived from
X-ray absorption through a cylindrical cross-section, overlaid on a DRR of a
real patient's head CT. The green HUD reports simulation time, tip position in
mm, and the C-arm angle. These are not artist renderings — every pixel is
computed from the physics."

---

## What Each Frame Represents

### Left Frame — t = 0.03 s (Initial State)

"This is essentially frame zero. The HUD reads SIM 0.03s, TIP (118.0, −6.5, −0.0) mm,
C-ARM 0 deg. The catheter has just been initialized and the XPBD solver has run
only a few substeps. The rod starts in a nearly straight configuration positioned
near the skull base, with the proximal root fixed (inv_mass = 0) and the distal
tip free. The catheter shadow appears as a short dark line within the cranial
anatomy — notice that the underlying bone structures (skull, petrous ridges,
orbital walls) remain fully visible through the catheter. This is the Beer-Lambert
transmission property: the catheter attenuates the background rather than
occluding it."

### Center Frame — t = 0.77 s (Dynamic Deformation)

"At t = 0.77 s (frame 22), the XPBD solver has completed approximately 46 substeps
(2 substeps/frame × 23 frames). The HUD reports TIP (117.0, −63.3, −0.0) mm —
the tip has dropped ~57 mm inferiorly under gravity. The catheter is now visibly
deflected into a curved shape as the Cosserat rod bends under gravitational load
balanced by the bend stiffness constraints (E = 10^8 Pa). The dark shadow tracks
the rod's curvature — each segment's projected cylinder generates its own
attenuation footprint, and the per-segment mu-profile is visible: the distal
marker band (last 2–3 segments, platinum, mu = 5.0) appears as the darkest
portion near the tip, the braided shaft (mu = 0.8) is moderately opaque, and the
transition zone fades to near-transparency."

### Right Frame — t = 7.53 s (Steady State)

"By t = 7.53 s (frame ~226), the rod has reached a damped equilibrium. The HUD
shows TIP (118.0, −142.9, −0.0) mm — the tip has settled approximately 137 mm
below its initial position. The catheter hangs in a gravitational catenary
constrained by its internal bend/twist stiffness. The curvature is smooth because
the XPBD solver enforces Darboux vector constraints that couple adjacent segment
orientations. The Beer-Lambert attenuation profile is clearly visible: high-opacity
marker bands at both ends (proximal tungsten at mu = 3.0, distal platinum at
mu = 5.0) frame the more transparent polymer/nitinol shaft. The Poisson quantum
noise is visible as grain across the entire image — this is generated at 2000
photons/pixel, matching a typical low-dose fluoroscopy acquisition."

---

## Beer-Lambert Compositing — How It Works

### Step 1: DRR Background Generation

"The background X-ray image is a Digitally Reconstructed Radiograph (DRR) computed
from the patient's CT volume using the Slang GPU renderer (DiffDRR). The CT is
converted from Hounsfield Units to linear attenuation coefficients mu via
piecewise-linear mapping: HU [−1000, 3000] → mu [0, 0.02] mm^−1. A fixed-step
ray march (0.5 mm step, up to 2048 steps) through the 3D mu-volume computes the
transmitted intensity via Beer-Lambert: I = I_0 × exp(−∫mu ds). The DRR is
pre-rendered for four C-arm views — AP (0°), LAO 30° (+30°), Lateral (+90°),
RAO 30° (−30°) — and loaded as PNG backgrounds. The closest DRR is selected
based on the current C-arm angle."

### Step 2: Catheter State Extraction

"The XPBD solver exposes particle positions as a (N, 3) tensor in world
coordinates (metres). For compositing, these are converted to mm (×1000). The
solver runs via `RodSolver.step()` or `XPBDRodSolver.step(dt)`, executing
`num_substeps` (default 2) block-Thomas direct solves per call. Each substep
advances the rod state through predict → assemble JMJT → direct solve → apply
corrections → integrate. The output is the updated `positions[0, :, :]` tensor
(env 0, all N particles, xyz)."

### Step 3: C-arm Projection (3D → 2D)

"World-frame 3D positions are projected to 2D detector pixel coordinates using
a pinhole camera model:

**Intrinsic matrix K**: Focal length f_px = SID / pixel_spacing = 1000 / 0.81 ≈ 1235
pixels. Principal point at detector center (256, 256) for a 512×512 image.

**Extrinsic matrix [R|t]**: Rotation R = R_x(cranio-caudal) × R_y(LAO/RAO), with
the X-ray source at (0, 0, −SOD) in the camera frame. SOD = 600 mm.

**Projection**: P = K × [R|t] is the 3×4 projection matrix. Homogeneous world
points (x, y, z, 1) are projected to (u, v) = (P × X)[:2] / (P × X)[2].

**Cone-beam magnification**: Each segment's projected pixel radius accounts for
depth-dependent magnification: r_px = r_physical × f_px / z_cam, where z_cam is
the segment midpoint's depth in camera coordinates. Segments closer to the X-ray
source appear larger. The radius is clamped to [1, 50] pixels."

### Step 4: Per-Segment Attenuation Profile

"The `_segment_attenuation_profile()` function assigns a mu × 2r value (effective
attenuation × diameter, dimensionless) to each segment, modeling the physical
construction of a real catheter:

| Region               | Segments           | mu Value | Material          |
|----------------------|--------------------|----------|-------------------|
| Proximal marker band | 0–1                | 3.0      | Tungsten          |
| Braided shaft        | 2 – 60% of N      | 0.8      | Nitinol braid     |
| Transition zone      | 60% – 85% of N    | 0.8→0.2  | Sparse braid + polymer |
| Soft tip             | 85% – 95% of N    | 0.15     | Polymer (PEBAX)   |
| Distal marker        | Last 3 segments    | 5.0      | Platinum coil     |

This profile controls the visual appearance: high-mu segments (markers) produce
dense, clearly visible shadows; low-mu segments (soft tip) are nearly
transparent — matching real fluoroscopy where only the radio-opaque markers are
clearly visible."

### Step 5: Beer-Lambert Attenuation Map Construction

"For each segment i, the compositing function computes the per-pixel attenuation
contribution within a bounding box around the projected segment line:

1. **Parameterise the segment** as P(t) = p0 + t × (p1 − p0), t ∈ [0, 1],
   where p0 and p1 are the projected 2D endpoints.

2. **Compute perpendicular distance** d(u,v) from each pixel to the nearest
   point on the segment line.

3. **Compute cylinder chord thickness**: For a circular cross-section of
   projected radius r_px, the X-ray path length through the cylinder at
   perpendicular distance d is:

   t(d) = 2 × sqrt(r_px^2 − d^2)   for d < r_px
   t(d) = 0                          for d >= r_px

   This is the chord of a circle — the exact analytical solution for a
   cylindrical cross-section viewed in projection.

4. **Normalize** by diameter: chord_norm = t(d) / (2 × r_px), giving values
   in [0, 1] with 1.0 at the centreline and smooth falloff to 0 at the edges.

5. **Accumulate** into the attenuation map:
   atten_map[v, u] += mu_i × chord_norm(u, v)

   The accumulation is **additive in the exponent** — this is critical. Where
   the catheter crosses itself (e.g., a loop), the attenuation values from
   overlapping segments add, producing physically correct increased opacity
   at self-crossings. This cannot be achieved with alpha blending."

### Step 6: Beer-Lambert Transmission

"The final composited image applies the exponential attenuation:

   I_final(u,v) = I_DRR(u,v) × exp(−atten_map(u,v))

This is the Beer-Lambert law: the transmitted intensity decreases exponentially
with the total attenuation along the X-ray path. Key properties:

- **Multiplicative darkening**: The catheter darkens the background, never
  brightens it — just like a real radiopaque object in an X-ray.
- **Background visibility**: Anatomical structures behind the catheter remain
  visible because exp(−mu) < 1 but > 0. At mu = 0.8, transmission is ~45%.
  At mu = 5.0 (platinum marker), transmission is ~0.7% — nearly opaque.
- **Smooth edges**: The circular chord function provides sub-pixel-quality
  smooth edge falloff — no aliasing or hard boundaries.
- **Additive at crossings**: Self-intersecting catheter paths produce
  physically correct increased attenuation where segments overlap."

### Step 7: Detector Physics Simulation

"Three post-processing effects simulate real detector physics:

1. **Veiling glare / scatter** (scatter_sigma = 18 px, scatter_fraction = 0.03):
   The intensity blocked by the catheter (I_DRR × (1 − transmission)) is blurred
   with a large Gaussian kernel (sigma = 18 px, ksize = 109 px) and 3% of the
   result is added back to the image. This models X-ray scatter: photons that
   interact with the catheter material are re-radiated in all directions,
   producing a faint haze around the catheter shadow. The 3% fraction and 18 px
   sigma approximate the scatter-to-primary ratio and scatter kernel width of a
   typical flat-panel detector with an anti-scatter grid.

2. **Detector PSF** (sigma = 0.7 px): A small Gaussian blur (sigma = 0.7 px,
   ksize = 5 px) models the detector's point-spread function — the slight
   blurring from phosphor light spread in the scintillator layer of a flat-panel
   detector.

3. **Poisson quantum noise** (2000 photons/pixel): The composited intensity is
   scaled to photon counts (I × 2000), Poisson-sampled, and scaled back. This
   models the fundamental quantum noise in X-ray imaging: each pixel's photon
   count follows a Poisson distribution whose mean is proportional to the
   transmitted intensity. The 2000 photons/pixel count corresponds to a typical
   low-dose pulsed fluoroscopy mode (~1–3 mR/frame). Higher counts (5000+)
   would simulate cine acquisition; lower counts (500) simulate ultra-low-dose
   navigation mode."

---

## HUD Telemetry Explained

"Each frame displays a green heads-up display in the upper-left corner:

- **SIM**: Simulation wall-clock time in seconds. Advances at dt = 1/60 s per
  physics step, 2 steps per rendered frame.
- **FPS**: Rendering throughput. On the headless GPU machine, ~2 FPS because the
  compositing runs on CPU (NumPy). A Warp GPU kernel version would achieve
  30+ FPS.
- **TIP**: Distal tip position in world coordinates (mm). The coordinate frame
  is X = patient-left, Y = superior (cranial), Z = anterior.
- **C-ARM**: Current C-arm LAO/RAO angulation in degrees. All three frames show
  0 deg = AP (antero-posterior) view.

The **AP** label in the upper-right confirms the DRR view selection — the closest
pre-rendered DRR to the current C-arm angle."

---

## Physical Parameters for These Frames

| Parameter | Value | Notes |
|-----------|-------|-------|
| Rod segments | 30 | N = 30 particles, 29 edges |
| Rod length | 120 mm (0.12 m) | Visible catheter segment in FOV |
| Catheter radius | 1.0 mm | ~6-French (2 mm OD) |
| Young's modulus | 10^8 Pa | Typical nitinol microcatheter |
| Density | 7800 kg/m^3 | Nitinol / stainless steel |
| Damping | 0.05 | Velocity-proportional damping coefficient |
| Substeps | 2 per frame | Block-Thomas direct solve each substep |
| Gravity | (0, −9.81, 0) m/s^2 | Y-down in production solver |
| SID | 1000 mm | Source-to-image distance |
| SOD | 600 mm | Source-to-object distance |
| Pixel spacing | 0.81 mm/px | Yields ~250 mm FOV at object plane |
| Image size | 512 × 512 px | Matches DRR resolution |
| Poisson noise | 2000 photons/px | Low-dose pulsed fluoroscopy |
| Scatter sigma | 18 px | Veiling glare kernel width |
| Scatter fraction | 0.03 | 3% scatter-to-primary ratio |
| Detector PSF | 0.7 px sigma | Scintillator light spread |

---

## Key Talking Points for This Slide

- "These are not rendered 3D graphics — they are simulated fluoroscopy images
  generated from the same physics that drives a real C-arm. The background is a
  DRR from an actual patient CT (DiffDRR Slang renderer), and the catheter
  shadow is computed via Beer-Lambert X-ray absorption through a cylindrical
  cross-section model."

- "The temporal progression shows the XPBD solver in action: at t = 0.03 s the
  rod is nearly straight (initial condition); by t = 0.77 s gravitational
  deflection has begun; by t = 7.53 s the rod has reached damped equilibrium
  in a gravitational catenary — the steady-state shape balances gravity against
  the bend stiffness (Darboux vector constraints) of the Cosserat rod model."

- "The per-segment attenuation profile is visible in the images: look at the
  darkest portions near the distal tip — those are the platinum marker bands
  (mu = 5.0). The shaft is more translucent (mu = 0.8), and the soft polymer
  tip is nearly invisible (mu = 0.15). This matches real clinical fluoroscopy
  where only the radio-opaque markers are clearly distinguishable."

- "The Poisson noise visible as grain throughout the images is physically
  modeled at 2000 photons/pixel — the same order of magnitude as real pulsed
  fluoroscopy. This noise is critical for domain randomization: a policy
  trained on clean DRRs will fail on real noisy fluoroscopy. By matching the
  noise statistics, we close the sim-to-real gap."

- "The compositing is additive in the exponent — if the catheter loops and
  crosses itself, the overlapping segments produce physically correct increased
  attenuation. This is not achievable with simple alpha-blending overlays.
  The Beer-Lambert formulation ensures photometric consistency with the DRR
  background."
