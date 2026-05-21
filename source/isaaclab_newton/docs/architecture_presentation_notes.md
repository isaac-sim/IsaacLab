# X-Ray-Guided Robotic Catheter Interventional System — Architecture Presentation Notes

> Technical presenter notes for the system architecture overview slide.
> References: `fluoro-simulator` (Slang DiffDRR renderer), `xpbd_rod_solver.py` (Warp XPBD Cosserat rod solver).

---

## Slide Title

**X-Ray-Guided Robotic Catheter Interventional System — Architecture Overview**

---

## Opening

"This diagram shows the end-to-end architecture for sim-to-real autonomous catheter
navigation. The system takes a patient CT scan and produces a fully trained neural
policy deployed on edge hardware for closed-loop robotic catheter control under
live fluoroscopy. Four stages form the pipeline: CT ingestion, physics-based
simulation, policy training, and runtime deployment."

---

## Section 1: CT Ingestion Pipeline

### HU-to-mu Conversion

"The first stage ingests the patient's CT or CTA volume in DICOM or NIfTI format.
We convert from Hounsfield Units to linear attenuation coefficients mu — this is
the physical quantity that governs X-ray absorption.

The mapping is piecewise-linear: we clip HU to [−1000, 3000] and map linearly to
mu in [0, 0.02] mm^−1. This is implemented in the `HuToMuMapping` configuration
class — HU −1000 corresponds to air (mu ≈ 0), HU 0 corresponds to water
(mu ≈ 0.02 cm^−1 at ~70 keV), and HU 3000 covers dense bone and metal.

The output is a 3D float32 volume of mu values with associated voxel spacing
(typically 0.5–1.0 mm isotropic) and spatial origin from the DICOM header. This
mu-volume is the data structure the GPU renderer ray-marches through."

### Vessel Segmentation

"Vessel segmentation uses TotalSegmentator or a nnU-Net variant to produce a
binary mask of the vascular tree. This segmentation defines the navigable lumen —
the physical free-space envelope the catheter tip must remain within during
autonomous navigation."

### Mesh Generation (Marching Cubes)

"The binary segmentation is iso-surfaced via Marching Cubes at the 0.5 level set,
producing a triangle mesh of the vessel wall. This mesh is decimated and smoothed
(Laplacian) to 50K–200K triangles for real-time collision queries. It serves
double duty: visual rendering in the Omniverse viewport and collision geometry for
the physics solver."

### Centerline Extraction

"Topological thinning or Voronoi-based skeletonization of the vessel segmentation
produces a 1D graph of centerlines with branch points. Each edge carries a local
inscribed-sphere radius estimate. These centerlines serve as navigation waypoints
for target sampling during RL training and define the 'reachable workspace' for
the catheter tip."

### SDF Generation (Collision Field)

"From the triangle mesh we compute a signed distance field on a regular 3D grid
(256^3 or 512^3). The SDF stores the signed Euclidean distance to the nearest
vessel wall: negative inside the lumen, positive outside. At runtime, the XPBD
solver evaluates collision via trilinear interpolation of the SDF — O(1) per
particle, with the SDF gradient providing the surface normal for contact response.
This replaces BVH-based mesh queries and is significantly faster for the dense
particle-to-surface interactions needed by the rod solver."

### Output

"The complete asset bundle — mu-volume, triangle mesh, SDF volume, centerline
graph — is exported as USD with custom schema extensions, forming the
patient-specific digital anatomy for that procedure."

---

## Section 2: Simulation Environment

### Fluoroscopy Renderer (GPU Ray-Casting)

"The fluoroscopy renderer produces synthetic DRRs (Digitally Reconstructed
Radiographs) using a cone-beam projection model implemented as a Slang GPU shader
with CUDA kernel dispatch.

**C-arm Geometry**: The virtual C-arm is parameterized by source-to-detector
distance (SDD, default 1020 mm), source-to-isocenter distance (SID, 510 mm),
detector dimensions (512×512 pixels at 0.5 mm/pixel spacing), and C-arm pose as
Euler angles (ZXY convention) plus a translation vector. There is no explicit
pinhole intrinsic matrix K — the projection is built procedurally: the X-ray source
is placed at local (0, 0, −SID) and detector pixels are mapped to physical mm
offsets from the detector center at local z = SDD − SID.

**Ray-Marching Algorithm**: For each detector pixel the shader computes a ray from
source to detector point, performs a slab AABB intersection against the mu-volume
bounding box, then executes a fixed-step ray march with step size 0.5 mm (up to
MAX_STEPS = 2048 iterations). At each step the mu value is sampled via custom
trilinear interpolation from the 3D r32_float texture (sampling is wrapped in
`no_diff()` so autodiff gradients flow only through pose, not volume data). The
attenuation integral is accumulated as sum(mu_i × step_mm).

**Beer-Lambert Output**: The transmitted intensity is I(u,v) = I_0 × exp(−integral),
where I_0 is the unattenuated beam intensity (configurable, default 1.0). The
resulting image is min-max normalized to [0, 1] and optionally inverted (1 − I)
for the bone-bright convention used clinically.

**Autodiff Support**: The Slang shader provides forward (`renderDRR_forward`) and
backward (`renderDRR_backward`) CUDA kernels. The backward pass uses
`bwd_diff(computePixelIntensity)` to compute gradients of pixel intensity with
respect to C-arm pose parameters (rotation and translation), enabling
differentiable rendering for pose optimization. Per-pixel float4 buffers
accumulate the rotation and translation gradients.

**PyTorch Integration**: An optional `SlangDiffDRRFunction` / `TorchSlangDiffDRR`
wrapper exposes the differentiable renderer as a custom `torch.autograd.Function`,
enabling end-to-end gradient flow from image-space losses through to C-arm pose
parameters."

### Newton Physics — Catheter (Cosserat Rod XPBD Solver)

"The catheter is modeled as a Cosserat elastic rod — a chain of N rigid segments
connected by stretch, bend, and twist constraints. The solver uses the XPBD
(eXtended Position-Based Dynamics) framework with a **block-Thomas direct solve**
on the constraint system, achieving O(N) complexity per substep.

**State Representation**: Each particle (segment endpoint) carries a position
(vec3) and orientation (quaternion, xyzw convention). Per-particle arrays include
inverse mass, inverse inertia (3×3 stored as flat float arrays), and linear/angular
velocities. Per-edge arrays include rest lengths, rest Darboux vectors (encoding
rest curvature and twist), and bend/twist stiffness coefficients.

**Substep Algorithm** (per substep, implemented as Warp GPU kernels):

1. **Predict** (`_xr_predict_pos`, `_xr_predict_rot`): Explicit Euler integration
   of positions and orientations with gravity, external forces/torques, and
   linear/angular damping. Root particle has inv_mass = 0 (kinematically
   controlled by the robotic actuator).

2. **Prepare constraints** (`_xr_zero_f`, `_xr_prepare_compliance`): Zero the
   Lagrange multiplier accumulators. Compute per-constraint compliance
   alpha = 1 / (stiffness × dt²) from the material's Young's modulus and torsion
   modulus.

3. **Evaluate constraints** (`_xr_update_constraints`): Compute the 6D constraint
   value C_i for each edge: stretch error (||x_{i+1} − x_i|| − L_rest) and
   Darboux vector error (current curvature/twist minus rest Darboux). The Darboux
   vector is extracted from the relative quaternion between adjacent segments:
   omega = 2 × im(q_i^* ⊗ q_{i+1}) / L_rest.

4. **Compute Jacobians** (`_xr_compute_jacobians`): Analytical Jacobians
   J_pos (3×6 per edge) and J_rot (3×6 per edge) mapping constraint-space
   corrections to position and rotation corrections. The rotational Jacobian
   accounts for the quaternion-to-axis-angle relationship.

5. **Compute inverse inertia** (`_xr_compute_inv_inertia`): Transform the
   body-frame inverse inertia tensor to world frame: I_world^−1 = R × I_local^−1 × R^T.

6. **Assemble JMJT system** (`_xr_assemble_jmjt`): Build the 6×6
   block-tridiagonal system JMJT + alpha × I. Each diagonal block D_i is 6×6
   (stored as 36 floats), each off-diagonal block E_i is 6×6. The block structure
   arises from the chain topology: constraint i couples particles i and i+1 only.

7. **Build RHS** (`_xr_build_rhs`): RHS = −C − alpha × lambda_accumulated (the
   XPBD Lagrange multiplier warm-starting).

8. **Block-Thomas direct solve** (`_xr_block_thomas`): A single-thread serial
   kernel (dim=1) performs the forward elimination and backward substitution on the
   6×6 block-tridiagonal system. Each step involves 3×3 Cholesky factorization
   (`_xr_chol`), triangular solve (`_xr_solvL`, `_xr_solvU`), and 3×3 matrix
   multiplication. This is the key algorithmic advantage — iterative Gauss-Seidel
   XPBD needs hundreds of iterations for stiff rods; the direct solve converges in
   a single pass, yielding two orders of magnitude speedup.

9. **Apply corrections** (`_xr_compute_corrections`, `_xr_apply_corrections`):
   Scatter the solved delta-lambda back to position corrections (M^−1 × J^T × dlambda)
   and rotation corrections (I^−1 × J_rot^T × dlambda). Corrections are accumulated
   per-particle from adjacent edges, then applied: x_pred += dx,
   q_pred = q_pred ⊕ dtheta (quaternion correction via the _xr_qcorr helper).

10. **Floor collision** (`_xr_floor_collision`): Optional ground-plane projection.

11. **Integrate** (`_xr_integrate_pos`, `_xr_integrate_rot`): Derive new velocities
    from position changes (v = (x_new − x_old) / dt) and angular velocities from
    orientation changes (omega = 2 × im(q_new ⊗ q_old^*) / dt).

**Proximal Control Interface**: The `apply_proximal_control(push_vel, rotate_vel, dt)`
method kinematically updates the root particle's position along the local tangent
direction and applies an axial rotation via quaternion composition. Since the root
has inv_mass = 0, the XPBD solver preserves this kinematic update. This is the
exact interface the robotic catheter actuator uses — push/rotate at the proximal end.

**Material Parameterization**: The `RodConfig` dataclass encodes Young's modulus
(Pa), shear modulus (Pa), density (kg/m³), Poisson ratio, and normalized stiffness
multipliers for stretch, bend, and twist (0–1 scale). Geometry is defined by
segment count, segment length, and per-segment radius. A `RodTipConfig` allows
specifying rest curvature on the distal tip segments, enabling J-tip or angled
catheter shapes."

### Unified Sim Loop

"The three subsystems execute in lock-step each frame:

1. **Policy outputs control action** — push velocity, rotate velocity, and
   optionally tendon tensions (for bi-directional catheters).
2. **Newton steps catheter physics** — the XPBD solver advances the rod state by
   dt (typically 1/60 s with 2 substeps, giving a physics dt of 1/120 s).
3. **Renderer produces fluoroscopy frame** — the updated particle positions are
   projected through the cone-beam model and composited onto the DRR of the
   patient anatomy via Beer-Lambert attenuation.

The synchronized output is a (fluoroscopy_frame, catheter_state) pair — the
observation tensor that feeds back to the policy network."

### Domain Randomization

"Per-episode randomization is applied to close the sim-to-real gap:

- **C-arm angles**: LAO/RAO and cranial/caudal rotation sampled from clinically
  relevant ranges (±30°).
- **X-ray dose/noise**: The post-processing pipeline applies gain/bias scaling,
  Poisson quantum noise (photon counts parameterized by `poisson_photons`,
  modeling dose variation), additive Gaussian noise, and Gaussian blur
  (sigma in pixels, approximating X-ray scatter and detector PSF). Seeds are
  deterministic per episode for reproducibility.
- **Vessel geometries**: Different patient CT volumes produce different anatomical
  environments, providing geometric diversity. Within a single anatomy,
  target locations are randomized within the reachable workspace."

---

## Section 3: Training Pipeline

### Demonstration Collection

"Expert electrophysiologists perform catheter navigation procedures in the
simulation environment, generating trajectory demonstrations. Each demonstration
records the full (observation, action) sequence: fluoroscopy frames, catheter
state vectors, and the expert's push/rotate control commands at each timestep.
These trajectories capture the implicit knowledge of skilled human operators —
which anatomical landmarks to track, how to handle vessel branches, and when to
advance versus retract."

### GR00T-H Training (IL → RL)

"Training follows a two-stage curriculum using the GR00T-H framework:

**Stage 1 — Imitation Learning (IL)**: Behavioral cloning from the expert
demonstrations. The policy network (CNN encoder for fluoroscopy images + MLP head
for continuous control) is trained via supervised learning to predict the expert's
action given the expert's observation. This initializes the policy to human-level
competence and provides a stable starting point for RL fine-tuning. Without IL
initialization, RL from scratch on this task would face severe exploration
challenges — the catheter must navigate through narrow, branching vessels where
random actions are almost always non-productive.

**Stage 2 — Reinforcement Learning (RL)**: PPO fine-tuning of the IL-initialized
policy. The reward function combines:

- Distance-to-target: −d(tip, waypoint) for the current navigation waypoint along
  the centerline.
- Contact penalty: −lambda × F_contact, penalizing excessive vessel wall forces
  (from SDF collision queries).
- Time penalty: −tau per step, incentivizing efficiency.
- Reached bonus: large sparse reward for reaching the target location within
  threshold distance.

RL enables the policy to discover strategies the expert demonstrations did not
cover — alternative navigation paths, more efficient insertion sequences, and
recovery from perturbations."

### Curriculum Learning

"Training progresses from simple to complex:

- **Phase 1**: Straight vessel segments (no branches) — the policy learns basic
  insertion control.
- **Phase 2**: Single-branch anatomies — the policy learns to select and navigate
  branch points.
- **Phase 3**: Full patient anatomies with multiple branches, tortuous vessels,
  and clinically realistic C-arm views.

Each phase increases the difficulty while preserving the learned behaviors from
previous phases. This stabilizes training convergence and reduces the total
wall-clock time to a deployable policy."

### Output

"The trained policy (GR00T-H) is a neural network that maps (fluoroscopy_frame,
catheter_state) → (push_vel, rotate_vel). It runs on standard GPU hardware and
is exported for edge deployment."

---

## Section 4: Runtime Deployment

### Holoscan IGX

"NVIDIA's medical-grade edge computing platform runs the trained policy with
low-latency inference. The Holoscan pipeline ingests the real-time fluoroscopy
video stream (typically 15–30 fps), runs the CNN encoder + MLP policy forward
pass, and outputs control commands within the frame budget (~30 ms end-to-end).
The IGX platform provides medical-grade isolation, deterministic scheduling, and
audit logging required for clinical deployment."

### Real C-arm Fluoroscopy

"The live X-ray stream from the procedure room C-arm replaces the synthetic
fluoroscopy renderer. Because the policy was trained on physically-based DRRs with
domain randomization (varying dose, noise, scatter, C-arm angles), it generalizes
to real fluoroscopy images without fine-tuning. The C-arm's SID/SDD and pixel
spacing are calibrated to match the simulation's projection model."

### Robotic Catheter (XCath)

"The physical catheter actuator receives the push/rotate commands from the Holoscan
inference pipeline and executes them on the real catheter. The control interface
is identical to the simulation's `apply_proximal_control(push_vel, rotate_vel, dt)`
API — this is by design. The robotic actuator sits at the proximal end of the
catheter (outside the patient), mimicking the physician's hand movements."

### Safety Layer

"A non-learned safety envelope wraps the entire deployment:

- **Motion limits**: Maximum push/rotate velocities and accelerations are clamped
  to clinically validated bounds, regardless of the policy's output.
- **Collision guards**: If the inferred catheter tip position approaches known
  danger zones (e.g., aortic wall, valve structures), the safety layer overrides
  the policy with a retract/stop command.
- **Watchdog timeout**: If the Holoscan inference pipeline misses a frame deadline,
  the safety layer holds the last safe state (zero velocity) until inference
  recovers.

The safety layer is a hard constraint envelope — it is never learned or tuned.
It is a regulatory requirement for autonomous medical device deployment."

---

## System Flow Summary (Right Panel)

"The right panel condenses the four-phase flow:

1. **CT Ingestion**: DICOM → HU-to-mu → vessel segmentation → mesh + SDF +
   centerline → USD physics assets.
2. **Simulation Loop**: Each frame — policy outputs control action → Newton
   steps catheter physics → renderer produces fluoroscopy observation.
3. **Training (Closed Loop)**: Fluoroscopy frames + catheter state feed GR00T-H
   for IL then RL; policy control actions flow back into the simulation.
   Output: trained policy optimized for navigation.
4. **Deployment**: Trained policy on Holoscan IGX, real C-arm fluoroscopy feeds
   inference, robot actuates catheter, safety layer active — closed-loop
   autonomous catheter navigation."

---

## Key Talking Points

- The entire pipeline is **patient-specific** — each new CT produces a unique
  digital twin with that patient's vascular anatomy, attenuation properties, and
  navigation targets.

- The **sim-to-real bridge** is the fluoroscopy imaging modality. We train on
  physically-based synthetic DRRs (Beer-Lambert cone-beam projection through the
  patient's mu-volume), and deploy on real C-arm X-ray. Domain randomization
  (dose, noise, scatter, C-arm angle) closes the appearance gap.

- The **proximal control interface** — push velocity and rotate velocity at the
  catheter root — is identical between simulation and the robotic actuator.
  This is the physical interface; the XPBD solver's root particle with
  inv_mass = 0 enforces the kinematic boundary condition.

- The **block-Thomas direct XPBD solver** is critical for training throughput.
  It converges in a single pass where iterative Gauss-Seidel XPBD needs hundreds
  of iterations for stiff rods. At 512 parallel environments with 20-segment
  catheters, this enables >1000 environment-steps/second on a single A6000.

- **GR00T-H with IL+RL** is the training recipe: imitation learning initializes
  the policy to human-level navigation competence; reinforcement learning
  fine-tunes it to exceed human performance by exploring strategies not present
  in the demonstration dataset.

- The **safety layer is non-negotiable** for clinical deployment — it is a hard
  constraint envelope, not a learned behavior. Motion limits, collision guards,
  and watchdog timeouts provide defense-in-depth against policy failures.

---

## Technical References

| Component | Implementation | Key File |
|-----------|----------------|----------|
| HU-to-mu conversion | Piecewise-linear mapping, HU [−1000, 3000] → mu [0, 0.02] mm^−1 | `fluorosim/preprocessor.py` |
| DRR ray-marching | Slang GPU shader, fixed-step march, MAX_STEPS=2048, trilinear mu sampling | `fluorosim/rendering/diffdrr_slang.slang` |
| C-arm model | SDD=1020 mm, SID=510 mm, 512×512 @ 0.5 mm/px, Euler ZXY pose | `fluorosim/config.py` |
| Autodiff DRR | Forward/backward CUDA kernels, gradients w.r.t. C-arm pose | `fluorosim/rendering/diffdrr_slang_renderer.py` |
| Detector noise | Poisson quantum + Gaussian + Gaussian blur (scatter proxy) | `fluorosim/rendering/realism.py` |
| XPBD Cosserat rod solver | Block-Thomas direct solve, 15 Warp GPU kernels | `isaaclab_newton/solvers/xpbd_rod_solver.py` |
| Proximal control API | Kinematic root update: push along tangent + quaternion axial rotation | `XPBDRodSolver.apply_proximal_control()` |
| RL environment | gymnasium.Env, RodSolver with 512 parallel envs, PPO via rsl_rl | `isaaclab_newton/envs/catheter_state_env.py` |
