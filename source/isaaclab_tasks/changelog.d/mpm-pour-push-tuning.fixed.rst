* Matched Franka Pour's invisible particle-only spill floor to the ground plane and extended
  the particle workspace to include it, so particles falling off the table remained counted as spills.
* Kept the UR10 Particle Push sparse-grid capacity hierarchy valid for small environment
  counts, including single-environment play and evaluation.
* Fixed the UR10 particle-push paddle and its visual becoming world-fixed during per-asset
  Newton cloning by authoring both inside the robot prototype. Paddle geometry overrides
  now live under ``scene.robot.spawn.paddle`` and ``scene.robot.spawn.paddle_visual``.
