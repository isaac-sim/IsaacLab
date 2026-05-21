# J&J MedTech — Technical Discovery Questions

> Questions to scope feature requirements for the catheter simulation workflow.
> Organized by domain. Each question includes the rationale (why we need to
> know this) tied to a specific engineering decision in our stack.

---

## 1. Catheter Mechanics & Actuation

### 1.1 Tendon Architecture

- How many pull-wires / tendons does the catheter have, and what is the routing
  topology? (Single tendon pair for uni-directional deflection, two orthogonal
  pairs for bi-directional, or asymmetric routing?)

  *Rationale: Determines whether we model tendons as cable constraints within
  the XPBD solver or approximate with external forces at discrete attachment
  points. A single pair may be approximatable; four tendons with complex
  routing requires a full cable constraint type.*

- Are the tendons position-controlled (displacement at the proximal spool) or
  force-controlled (tension commanded directly)? What is the actuation
  bandwidth — what update rate do tendon commands arrive at?

  *Rationale: Position-controlled tendons can be modeled as kinematic
  constraints; force-controlled tendons require tension-to-deflection dynamics.
  This determines the action space representation for RL.*

- Where are the tendon attachment and routing guide points along the catheter
  shaft? Are there intermediate guide rings, or do tendons run freely inside
  the lumen?

  *Rationale: Guide points create piecewise-linear tendon paths that
  concentrate bending at specific segments. Free-running tendons produce
  distributed curvature. The XPBD constraint formulation differs for each.*

- Does the catheter have a steerable sheath (inner catheter steered inside an
  outer deflectable sheath), or is it a single-body device?

  *Rationale: Coaxial multi-instrument simulation (guidewire inside
  microcatheter inside guide catheter) requires sliding joint constraints
  between concentric rods — a feature we have identified as a gap.*

### 1.2 Material & Geometry

- What are the stiffness variations along the catheter length? Can you provide
  approximate Young's modulus values (or relative stiffness ratios) for the
  proximal shaft, transition zone, and distal tip?

  *Rationale: We currently model uniform material along the rod. Per-segment
  Young's modulus requires extending the `_xr_prepare_compliance` kernel with
  per-edge stiffness arrays — a 1-week change, but we need the target values.*

- What is the catheter outer diameter (French size), and does it vary along the
  length (tapered tip)?

  *Rationale: Our solver supports per-segment radius via `RodGeometryConfig`.
  We need the actual profile to match their device.*

- Is anisotropic bending stiffness important? (e.g., braided shaft with
  preferential bend direction vs. isotropic tube)

  *Rationale: Our solver assumes isotropic bending. Anisotropic stiffness
  would require separate bend compliance per axis in the Darboux constraint —
  a non-trivial solver modification.*

- What tip shapes do you need? (Fixed-curve J-tip, Simmons, or dynamically
  variable via tendons?) What are the target deflection angles?

  *Rationale: `RodSolver` supports static tip shaping via rest Darboux vectors
  (`RodTipConfig`). `XPBDRodSolver` does not yet. Dynamic shaping via tendons
  is a separate feature.*

---

## 2. Robotic Manipulator

### 2.1 Actuator Architecture

- How many degrees of freedom does the robotic manipulator have? Specifically:
  which DOFs control catheter insertion (linear push/pull), axial rotation, and
  tendon tensions?

  *Rationale: This determines the action space dimensionality for RL and the
  coupling constraint between the articulation solver and the rod solver.*

- Is the manipulator a cassette-based system (catheter loaded into a motorized
  cartridge) or a direct-drive gripper mechanism?

  *Rationale: Cassette systems have well-defined kinematic models; direct-drive
  grippers introduce contact-dependent slip dynamics that are harder to model.*

- What is the control interface — what commands does the robot accept? (Joint
  positions, joint velocities, joint torques, or task-space Cartesian commands?)

  *Rationale: This maps directly to how we couple the articulation output to
  the rod solver's `apply_proximal_control`. Velocity commands map cleanly;
  torque commands require force feedback from the rod.*

### 2.2 Force Feedback

- Does the manipulator measure insertion resistance (axial force)? Is this
  signal used in the AI control loop?

  *Rationale: If yes, we need to compute and expose the axial reaction force
  at the rod root as an observation. This is the force the vessel anatomy
  exerts on the catheter tip, propagated back through the rod to the root —
  available from our solver's constraint forces but not currently exposed.*

- Is torque feedback (rotational resistance) measured and used?

  *Rationale: Same as above — rotational reaction torque at the rod root from
  torsional constraint forces.*

---

## 3. Sensor Simulation

### 3.1 Sensor Priority

- Which sensor modalities are primary inputs to the AI control loop? Can you
  rank these by importance for the initial training pipeline:
  - [ ] Catheter tip position/orientation (EM tracking / CARTO)
  - [ ] Intracardiac electrograms (unipolar / bipolar ECG)
  - [ ] Tissue impedance (contact quality)
  - [ ] Contact force (TPI or force-sensing catheter)
  - [ ] Fluoroscopy images (X-ray)
  - [ ] Other (specify)

  *Rationale: This directly prioritizes which sensor simulation modules we
  build first. Contact force is partially available from our collision solver;
  ECG and impedance are not implemented and require significant development.*

### 3.2 ECG / Electrograms

- Is the AI loop consuming raw intracardiac electrograms, or derived features
  (e.g., local activation time, voltage amplitude, fractionation index)?

  *Rationale: Raw electrograms require a cardiac EP model running on the
  atrial surface mesh; derived features could be approximated with
  lookup tables based on electrode position relative to known scar/fiber maps.*

- How many electrodes does the catheter carry, and what is the electrode
  spacing? Are they ring electrodes (circumferential) or point electrodes?

  *Rationale: Electrode count and geometry determine the dimensionality of the
  ECG observation vector and the spatial resolution required from the EP model.*

- Do you need the full CARTO-style 3D electroanatomical map as a training
  observation, or just the local signals at the catheter's current position?

  *Rationale: A full 3D map observation is expensive (volumetric data); local
  signals are a fixed-size vector proportional to electrode count.*

### 3.3 Impedance

- Is impedance used as a real-time control signal (continuous observation), or
  as a post-hoc lesion quality metric?

  *Rationale: Real-time use means it must be computed every simulation step
  (~16 ms budget). Post-hoc analysis can be batched offline.*

- What impedance model do you use clinically? (Baseline impedance, impedance
  drop during ablation, or complex impedance spectrum?)

  *Rationale: Determines the complexity of the tissue impedance model we need
  to implement — a simple lookup vs. a full electrical model.*

### 3.4 Contact Force

- Is contact force sensing from the catheter tip a primary observation for the
  AI loop? What are the relevant force magnitudes (typical target contact
  force in grams)?

  *Rationale: Our `RodSolver` computes contact forces from the BVH mesh
  collision kernel. We need to know the target force range (typically 5–40 g
  for ablation) to calibrate the collision stiffness/damping parameters.*

### 3.5 Location Tracking (EM / CARTO)

- Is the AI policy expected to consume CARTO-derived 3D positions directly, or
  does it learn from raw magnetic field sensor signals?

  *Rationale: If CARTO positions, we can expose solver state with added
  Gaussian noise to simulate tracking error. If raw sensor signals, we need a
  magnetic field forward model — significantly more complex.*

- What is the typical CARTO tracking accuracy and update rate? (Needed to
  parameterize our noise model.)

  *Rationale: CARTO 3 typically reports ±1 mm accuracy at 60 Hz. We need to
  confirm these are the correct specs to model.*

---

## 4. Anatomy & Tissue Interaction

### 4.1 Target Anatomy

- Which anatomical chambers and vessels does the catheter navigate through for
  the target procedure? (e.g., femoral vein → IVC → right atrium → trans-septal
  → left atrium → pulmonary vein ostia?)

  *Rationale: This defines the collision geometry we need to generate —
  vessel meshes and chamber meshes from CT segmentation. Different anatomies
  have very different contact dynamics (tubular vessels vs. open chambers).*

- Do you have patient-specific CT/MRI datasets you intend to use for the
  digital twin, or are you using atlas/template anatomies?

  *Rationale: Patient-specific requires our CT ingestion pipeline (HU → mu,
  segmentation, mesh generation, SDF). Atlas-based is simpler but less
  clinically representative.*

### 4.2 Tissue Contact Model

- How important is tissue deformation (compliant vessel walls) vs. rigid
  collision? Do you need hydroelastic or soft-body contact?

  *Rationale: Our current collision is rigid (position-level projection to the
  mesh surface). Compliant walls require either a deformable mesh solver
  coupled to the rod, or a compliant contact model with configurable
  stiffness/damping — the latter is simpler and likely sufficient.*

- Is tissue perforation / penetration detection required (e.g., trans-septal
  puncture)?

  *Rationale: Puncture mechanics require modeling tissue failure — a
  threshold-based constraint removal or mesh modification that we do not
  currently support.*

- Do you model blood flow drag on the catheter? At what fidelity?

  *Rationale: Our POR identifies blood drag as a lower-priority gap. If J&J
  considers it important for their maneuvers (e.g., catheter whip in the
  atrium), we need to reprioritize.*

---

## 5. Clinical Data & Validation

### 5.1 Recorded Procedure Dataset

- What is the recording format for clinical procedure data? (CARTO export
  format, proprietary binary, DICOM SR, ROS bags, CSV?)

  *Rationale: We need to build a data loader. Format determines scope — CARTO
  exports have well-documented schemas; proprietary formats require
  documentation from J&J.*

- What signals are recorded per timestep? (Catheter position/orientation,
  tendon tensions, ECG channels, impedance, fluoroscopy frames, contact force?)

  *Rationale: Each signal type requires a corresponding simulation output
  for comparison metrics. This defines the observation dict for validation.*

- What is the recording sample rate? (Different sensors may be sampled at
  different rates — CARTO at 60 Hz, ECG at 1–2 kHz, fluoroscopy at 15 fps.)

  *Rationale: Our solver runs at 60 Hz (physics dt = 1/60 s). Signals recorded
  at higher rates require interpolation or solver upsampling.*

- How many recorded procedures are available, and how are they annotated?
  (Success/failure labels, expert quality scores, anatomical landmarks?)

  *Rationale: The number and annotation quality determine whether these are
  useful for imitation learning (IL) training or only for validation.*

### 5.2 Sim-to-Real Validation Metrics

- What quantitative metrics do you use to compare simulated vs. real catheter
  behavior? (Tip position error, shape similarity, insertion force profile,
  contact force correlation?)

  *Rationale: Defines the acceptance criteria for our simulation fidelity and
  the automated regression tests we need to build.*

- Do you have bench-top phantom experiments with ground-truth measurements
  that we can use as initial validation targets (before moving to clinical
  data)?

  *Rationale: Phantom data is cleaner (no patient variability) and easier to
  reproduce in simulation. It's the natural first validation step.*

---

## 6. Training & Deployment

### 6.1 Policy Architecture

- What observations does the AI policy consume? Is it a state-based policy
  (joint angles, catheter state, sensor values) or an image-based policy
  (fluoroscopy frames)?

  *Rationale: State-based policies use our existing `CatheterState-v0` env.
  Image-based policies require the fluoroscopy compositing in the training
  loop — a significant performance consideration at 512+ envs.*

- What actions does the policy output? (Push/rotate only, or push/rotate +
  tendon tensions? Continuous or discrete action space?)

  *Rationale: Determines the action space dimensionality and the mapping to
  our solver's control API. Tendon tensions require the tendon actuation
  feature.*

- What is the target treatment routine? (Point-to-point navigation, sequential
  waypoint following, ablation lesion coverage, or full PVI procedure?)

  *Rationale: Single-point navigation is our current env. Sequential
  waypoints require episode structure changes. Ablation coverage requires
  a tissue state model (lesion formation) that we do not have.*

### 6.2 Training Infrastructure

- How many parallel environments do you need for training throughput? What GPU
  hardware is available?

  *Rationale: Our `RodSolver` supports multi-env on a single GPU. At 4096
  envs we estimate ~200 MB physics memory on A6000. If they need multi-GPU
  or multi-node, that's additional infrastructure.*

- What RL/IL framework are you using or planning to use? (rsl_rl, Stable
  Baselines 3, rl_games, CleanRL, custom?)

  *Rationale: Our env is integrated with rsl_rl. Other frameworks require
  different wrappers.*

### 6.3 Hardware-in-the-Loop

- What is the HIL communication protocol between the inference engine and the
  robotic controller? (ROS 2, gRPC, custom UDP, shared memory?)

  *Rationale: Determines the real-time bridge implementation. ROS 2 is
  straightforward; custom protocols require documentation.*

- What is the control loop latency budget? (End-to-end from sensor input to
  motor command — typical surgical robotics targets 1–10 ms.)

  *Rationale: Determines whether we can use standard GPU inference or need
  TensorRT optimization / CUDA graph capture for the policy forward pass.*

- What is the phantom setup? (Silicone vascular model, 3D-printed anatomy,
  ex-vivo tissue? Embedded sensors for ground truth?)

  *Rationale: Phantom geometry needs to match the simulation mesh. If they
  have CAD files for the phantom, we can use those directly as collision
  meshes.*

---

## 7. Solver Architecture

### 7.1 Current Implementation

- In your from-scratch Cosserat rod XPBD in Warp — is the constraint solver
  iterative (Gauss-Seidel / Jacobi) or do you use a direct solve?

  *Rationale: If iterative, our block-Thomas direct solver is a significant
  upgrade for stiff rods — single-pass convergence vs. hundreds of iterations.
  This could be our highest-value technical contribution.*

- How many segments do you use per catheter? What physics timestep and substep
  count?

  *Rationale: Segment count and substep count are the primary performance
  knobs. We need to benchmark our solver against their configuration to
  demonstrate throughput improvements.*

- What collision model do you use currently? (Mesh BVH, SDF, PhysX
  scene query, or none?)

  *Rationale: Determines whether they can use our BVH mesh collision path
  directly or need an SDF-based alternative.*

### 7.2 Multi-Solver Strategy

- How are you coupling PhysX articulations with the Warp XPBD catheter? Is it
  one-directional (articulation drives rod root) or bidirectional (rod forces
  feed back to articulation)?

  *Rationale: We have the same coupling gap. Understanding their approach
  (or lack thereof) determines whether we co-develop the coupling layer or
  provide an existing solution.*

- Are you using PhysX for anything beyond the robotic arm? (e.g., rigid body
  anatomy, soft tissue deformation?)

  *Rationale: Determines whether the simulation needs a PhysX scene running
  alongside the Warp XPBD solver in the same frame, requiring careful
  synchronization.*

---

## 8. Omniverse & Application Architecture

- Is the target deployment an Omniverse Kit Extension (standalone app with
  custom UI panels), or an Isaac Lab environment (headless training +
  optional viewport)?

  *Rationale: Our current integration is Isaac Lab (headless-first). An
  Omniverse Kit Extension requires a different packaging and UI approach.*

- Do you need real-time visualization during training, or is headless training
  with post-hoc replay sufficient?

  *Rationale: Real-time visualization at 512+ envs requires a viewport
  rendering budget that competes with training compute. Headless training with
  checkpoint replay is more practical.*

- What USD schema do you use for the catheter and anatomy? Do you have
  existing USD assets for the robotic arm?

  *Rationale: Interoperability requires matching USD schema conventions. Our
  catheter is rendered as `VisualizationMarkers` (capsule prims), not a
  dedicated catheter USD schema.*
