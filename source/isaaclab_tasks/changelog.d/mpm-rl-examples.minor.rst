Added
^^^^^

* Added Newton MPM Franka pouring and UR10 particle-pushing reinforcement
  learning tasks with reset-safe particle randomization and rigid-particle
  coupling. The pouring task uses compact current-state observations and an
  outcome-aware reset curriculum.
* Added bounded sparse MPM configurations with CUDA graph capture and
  fixed-payload resets for both tasks.
* Added a compact one-file generator for reproducible Franka pouring reset
  artifacts.
* Added randomized pile shape and table placement, paired collision-screened
  robot starts, and a weighted reset mixture to the UR10
  particle-pushing task. Every level keeps the same single-pile, single-sweep
  objective.
* Added an opt-in Newton GL visualization of the particle-pushing policy
  heightmap.

Changed
^^^^^^^

* Simplified Franka Pour artifacts and runtime validation around the one-file
  generator. Regenerate legacy calibrated artifacts with
  ``scripts/tools/generate_franka_pour_reset_dataset.py`` and pin the printed
  digest when an exact artifact is required.
* Loaded the Franka Pour robot and cups from the standard Isaac Lab Nucleus asset root,
  while retaining their environment variables for local overrides.
* Resolved the default Franka Pour reset artifact relative to the repository
  root and included the generator command in missing-artifact errors.

Fixed
^^^^^

* Checked every UR10 reset candidate for contact-buffer overflow and copied
  mutable Newton prototype geometry before offline IK and collision screening.
