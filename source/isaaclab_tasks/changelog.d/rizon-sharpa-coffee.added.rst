* Added the manager-based ``IsaacContrib-Rizon-Sharpa-Coffee`` environment with
  procedural tableware, coupled Newton MPM/MJWarp physics, a single-scoop IK demo,
  optional stirring, Apple Vision Pro retargeting and named grasp-reset authoring.
  The default uses a measured-palm tool attachment; free-spoon contact grasping
  remains experimental. The licensed robot bundle must be provided separately.
* Registered a state-based RSL-RL PPO baseline with normalized palm and finger
  actions. Scripted IK and teleoperation preserve their absolute trajectories
  through the task's target encoder, sharing the same manager action interface.
  Source/cup target observations and a measured-load approach reward support
  learning without a scripted phase oracle.
* Kept the Sharpa retargeting URDF external with pinned checksum validation and
  included the license for the packaged IsaacTeleop retargeting configuration.
* Added mass-preserving soluble-coffee conversion into liquid during captured
  physics, world-local material resets and conservative colour transport.
  Optional GL, RTX and Kit presentation includes native rendering grains,
  reconstructed liquid, a following camera and simulation-time MP4 capture.
* Consolidated joint calibration and tracking validation, reused the task's
  measured spoon pose, and gave streamed recording one resource owner.
  Recording metadata now includes resolved simulation settings and CLI options.
* Added explicit coarse-grid experiments up to 10 mm with 100 Hz material
  stepping, independently selectable mounted-robot integration, physical spoon
  margin overrides and up to 128 native visual grains per physical particle.
  Fine defaults were retained; the experimental wet profile trades contact
  resolution and some liquid-level accuracy for higher throughput.
* Selected a 4 mm demo grid for finer scoop motion and reconstructed the liquid
  only at recorded frames, preserving simulation-time surface filtering.
* Added a 120 Hz mounted-demo profile with 60 Hz control, 840 Hz rigid integration
  and 30 fps recording, keeping the spoon collider aligned with material steps.
* Added an optional public NVIDIA kitchen backdrop for Kit with an enclosed
  room, SimReady appliances, warm lights and matte cabinetry. The backdrop is
  composed after physics construction and adds no solver bodies or collisions.
* Smoothed the demo camera with continuous acceleration and gentler close-up
  framing, added a front/right stirring view, and removed distracting kitchen countertop props.
  Kit presentation omits dissolved grains instead of submitting zero-width points.
* Shortened the scripted demo to one scoop followed directly by stirring, with
  a slightly shifted entry, a lower pour and a 20 ms demo wetting delay.
