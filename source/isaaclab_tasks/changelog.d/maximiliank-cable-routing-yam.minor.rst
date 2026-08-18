Added
^^^^^

* Added a goal-conditioned bimanual YAM cable-routing environment with relative joint-position
  actions, smooth one-meter self-avoiding cable resets, single-span wrap validation, heterogeneous
  fixture resets, and Newton MJWarp/VBD proxy coupling.
* Added a pinned Robot Menagerie YAM USD package with native Newton collision and mimic schemas,
  plus explicit contact-buffer headroom for dense cable interactions.
* Added pinned ManipulationNet task-board and F1 USD visuals with primitive-only Newton collision,
  reproducible STL conversion metadata, and front-edge dual-YAM placement.
* Added three staged round-peg routing goals plus an explicit seven-goal task spanning both
  fixtures and winding directions, with an RSL-RL PPO configuration and shared route encoder for
  the actor and critic.
* Split cable routing into a robot-neutral board scene and task configuration plus a bimanual YAM
  embodiment, with a reusable two-manipulator contract for actions, contact frames, and reset IK.
Fixed
^^^^^

* Fixed reversed YAM open/close targets and applied current Newton gravity-compensation schemas so
  static relative-joint commands hold the robot without accumulating gravity sag.
* Fixed policy geometry and shaping to use the physical YAM pad midpoint, cable strain to use
  connected capsule endpoints, and route completion to reject cable loops outside the peg height.
* Fixed non-finite terminal cable states so they are reset with finite route metrics and rewards
  instead of aborting synchronized multi-GPU training; robot/action failures are now sanitized at
  the physics boundary and terminated with finite policy observations and rewards.
* Held each relative joint target across control decimation, clamped it to the authored limits,
  represented gripper actions by their binary command state, and made terminal success and failure
  rewards independent of control frequency. Newton actuator graph capture and complete-episode PPO
  startup are now enabled for training.
* Applied explicit Newton contact materials to the fixtures and a task calibration layer that keeps
  the Menagerie actuator dynamics while targeting high friction to YAM fingers. Limited the cable
  solver proxy to each wrist/gripper subtree, and declared the task's Newton and contrib runtime
  dependencies.
* Prevented simultaneous success and invalid-state terminations from receiving successful reset
  replay credit.
* Avoided redundant or manager-order-dependent route evaluation and expensive ordinary cable
  generation when a full-scene replay reset will replace it.
* Softened cable bending and twist for rope-like motion, and required replay states to settle for
  one simulated second under tighter residual-speed limits before acceptance.
* Replaced legacy dense tangent-point optimization with a deterministic, fixed-sweep XPBD-style
  Warp projector using Chebyshev acceleration, a Taubin smoothing tail, and current Torch CUDA
  streams. Added a 12 mm bend gate, bend-limited exact-length reconstruction, and replay-forward
  revalidation; cable material frames are now constructed in board-local coordinates before world
  translation so distant environment clones cannot lose bend precision. Replay entries now survive
  a complete Newton reset/relative-restore round trip and are accepted from live route semantics,
  while stored progress remains diagnostic. YAM reset contacts also select reachable downstream
  cable workspace instead of a fixed material offset. Replay construction now bounds and fills
  rare exhausted-row tails from already validated snapshots for the same route.
