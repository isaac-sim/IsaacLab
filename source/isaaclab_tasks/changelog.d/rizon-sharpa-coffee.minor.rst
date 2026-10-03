Added
^^^^^

* Added a manager-based Rizon–Sharpa coffee-scooping task with hollow procedural
  tableware, coupled Newton MPM/MJWarp physics, scripted IK and Apple Vision Pro
  hand retargeting. Retained the original shallow 50 by 32 by 10 mm spoon,
  1.8 mm shell, 10 mm handle height and original mount frame. Kept the larger
  centered bowl and 456,836-particle coffee seed, totaling 205.5762 g at the
  unchanged 450 kg/m³ density. Added two successive localized scoops with
  60-degree entries, approximately 15 mm forward draws, zero roll, rotation to
  level within the pile and level lift over each 13.2-second profile. Preserved
  the 29-dimensional action interface and measured arrival gates, with the glass
  at (0.60, 0.04) m.
  Configured native ``P1d`` strain, particle-sampled ``pic27`` contacts and
  ``backward`` collider velocities with 100 iterations, replacing routine hard
  projection after numerical floor clustering in projected trials.
  Defaulted dry mounted 2 mm demos to 400 Hz with a 98,304-cell sparse-grid
  budget; retained 800 Hz for other configurations and exposed timestep,
  material-substep, grid-budget and contact-proxy inertia overrides.
  Added GL previews, RTX recordings, passive grain rendering, a following
  camera and MP4 capture at simulation speed. Reused constant dry-grain
  instance attributes and prepared malformed robot visual normals through a
  cached USD override. Added optional visual coffee tint transfer into water.
  Reported carried and delivered grams, while qualifying the carried mask's
  bounded heap coverage and measuring capture doses by glass-mass increments
  after emptying. Added an experimental free-spoon contact mode with authored
  reset candidates, shared finger-contact settings and two-way MPM proxy
  coupling, plus loaded-hold, release and partial-reset regression coverage.
  Checked the mounted localized motion in a contact-disabled CPU rollout at
  400 Hz lasting 60.24 simulated seconds, with sampled tool/bowl and hand/bowl
  gaps of 3.956 and 6.667 mm. Kept the intruding contact-hand alternative and
  moving contact grasp uncertified. Documented the rejected enlarged-spoon
  captures and calibration trials as historical evidence in the task README.
  Completed the restored spoon's mounted 400 Hz RTX recording in 60.25 simulated
  seconds, measuring 2.78730 g and 2.74275 g glass-mass increments after emptying,
  about 60% more delivery than the preceding shallow-entry motion,
  with zero watched native/pose-rebase floor crossings and continuous floor flags.
  Qualified 21 radial watch exits per cohort, 0.17505 g classified spill and
  0.01080 g unclassified remainder. Confirmed complete decoding of the 1,507-frame, 25 fps
  MP4 and stated that the measured teaspoon doses did not meet the earlier
  8–10 g target, retaining the original spoon shape and native particle masses.
  Added an optional initial water fill and an eight-second, three-turn scripted
  stir after the two pours. Replaced uniform water tint with local passive
  concentration advection and bounded, conservative neighbour diffusion,
  displayed through GL face textures and RTX colour primvars. Kept physical
  coffee grains visible and preserved their mass and granular material.
  Preserved tiny late tracer releases with a double-precision counter and
  reported the effect separately as ``visual_tracer_g``. Exposed an optional
  MPM iteration cap and validated a complete 78.37-second coarse wet GL preview,
  qualifying its 0.63% water containment error and 0.154 combined-capture RTF.
  Added Kit viewport recordings with native grain displays, reconstructed local
  water colours, shared studio lighting and render-only camera warmup.
  Raised the wet demo's following camera after the first pour and kept the
  cup interior visible through the second scoop and a closer stirring view.
  Replaced the reflective glass with an opaque sage ceramic cup, softened
  the studio key and fill, and added closer elevated pouring and stirring views.
  Smoothed the native liquid reconstruction without changing particle motion,
  and used local tracer colours for refracting water absorption in Kit.
  Sized the render surface buffer for the fine grid and late stirring splashes,
  and validated the complete two-pour, three-turn Kit close-up recording.
  Converted coffee after accumulated contact with water into the contacted
  liquid's MPM material while preserving mass, position, velocity and particle
  identity. Updated reference volume and granular stress inside captured
  physics, independently of rendering, and restored dry material on partial
  resets. Included converted coffee in the liquid surface, removed its grain
  display, and conserved its colour tracer over the shared liquid carriers.
  Isolated captured wetting queries from rendering-grid descriptor updates in
  the installed Warp build and covered that interference during graph replay.
  Reconstructed only liquid carriers with automatic render-buffer sizing and
  the coupled MPM entry's current particle radii, handling dispersed droplets
  without changing the captured physics grid.

Fixed
^^^^^

* Reduced the spoon's MPM collision margin near its rim to one-ninth of the
  broad-face margin (0.4 mm on the 3 mm grid) with native per-face
  material bands, preserving the broad-face and handle margins and the original
  closed spoon mesh. Configured the taper before capture and exposed its scale
  in the demo, with contact-clearance and loaded-spoon regression coverage.
