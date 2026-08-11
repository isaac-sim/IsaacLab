Added
^^^^^

* Added a goal-conditioned bimanual YAM cable-routing environment with relative joint-position
  actions, smooth one-meter self-avoiding cable resets, single-span wrap validation, heterogeneous
  fixture resets, and Newton MJWarp/VBD proxy coupling.
* Added a pinned Robot Menagerie YAM USD package with native Newton collision and mimic schemas,
  plus contact-capacity stress validation for dense cable interactions.
* Added pinned ManipulationNet task-board and F1 USD visuals with primitive-only Newton collision,
  reproducible STL conversion metadata, and front-edge dual-YAM placement.
* Added three staged round-peg routing goals plus an explicit seven-goal task spanning both
  fixtures and winding directions, with an RSL-RL PPO configuration and shared route encoder for
  the actor and critic.
* Added an Apple Vision Pro XR teleoperation variant with engagement-calibrated right-hand
  pinch-point retargeting, native Newton IK, tracking-loss recovery, responsive right-arm drives,
  and an inactive left-YAM hold. The ``teleop`` installation extra provides IsaacTeleop,
  CloudXR, and Kit/OpenXR together without requiring the optional IsaacTeleop tuning UI.

Fixed
^^^^^

* Fixed Apple Vision Pro startup diagnostics and added an explicit host-side auto-start option for
  clients that cannot send commands on the named teleoperation message channel.
* Fixed reversed YAM open/close targets and applied current Newton gravity-compensation schemas so
  static relative-joint commands hold the robot without accumulating gravity sag.
* Fixed Apple Vision Pro pose jumps, semantic wrist-to-gripper alignment, and control latency by
  synchronizing right-hand retargeting, deriving the YAM grasp frame from anatomical hand
  landmarks, clutching only translation, and prewarming Newton IK before live input.
* Fixed policy geometry and shaping to use the physical YAM pad midpoint, cable strain to use
  connected capsule endpoints, and route completion to reject cable loops outside the peg height.
* Fixed non-finite terminal cable states so they are reset with finite route metrics and rewards
  instead of aborting synchronized multi-GPU training.
* Avoided redundant per-step route evaluation and expensive ordinary cable generation when a
  full-scene replay reset will replace it; AVP teleoperation now explicitly bypasses the training
  replay-bank build.
* Replaced legacy dense tangent-point optimization with a deterministic, fixed-sweep XPBD-style
  Warp projector using Chebyshev acceleration, a Taubin smoothing tail, and current Torch CUDA
  streams. Added a 12 mm bend gate, bend-limited exact-length reconstruction, and replay-forward
  revalidation; cable material frames are now constructed in board-local coordinates before world
  translation so distant environment clones cannot lose bend precision. Replay entries now survive
  a complete Newton reset/relative-restore round trip and are accepted from live route semantics,
  while stored progress remains diagnostic. YAM reset contacts also select reachable downstream
  cable workspace instead of a fixed material offset.
