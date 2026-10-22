* Placed generic reward, clearance termination and step-scheduled curriculum terms in the core MDP,
  keeping foot-support geometry in the velocity MDP modules.
* Declared the G1 teacher's static reward settings in ``G1Rewards`` and preserved the flat task's
  reward settings through explicit configuration overrides.
* Composed the G1 environments from scene, event, curriculum, termination and observation configurations,
  nesting the depth-camera configuration in its student scene and leaving environment initialization to supply
  asset-derived action scales and sole geometry.
* Moved G1 symmetry helpers alongside the other velocity-task robot symmetry modules.
* Reused the common depth-image observation with a G1 image-processing modifier, keeping observation
  group definitions inline in the observations configurations. Named groups by their contents: current
  proprioception, proprioception history, ray-cast height scans and depth images. Apply student supervision
  noise and reflected-camera overrides in the observations configuration's post-initialization hook.
* Declared the teacher's symmetry settings directly in its PPO algorithm configuration.
* **Breaking:** Renamed the flat G1 observation group from ``policy`` to ``proprio``. Update observation-group
  references and configuration overrides to use ``proprio``.
* Moved the G1 mesh-terrain preset into the G1 terrain configuration module.
* Removed the rough G1 actuator-path override, using the current default shared across physics backends.
  The configured implicit actuators retain their solver-owned drives.
* Read G1 sole geometry and reference torques from the asset, reusing configured actuator stiffnesses.
