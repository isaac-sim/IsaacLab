* Placed generic reward, clearance termination and step-scheduled curriculum terms in the core MDP,
  keeping foot-support geometry in the velocity MDP modules.
* Declared the G1 teacher's static reward settings in ``G1Rewards`` and preserved the flat task's
  reward settings through explicit configuration overrides.
* Composed the G1 environments from scene, event, curriculum, termination and observation configurations,
  leaving initialization to supply asset-derived action scales and sole geometry.
* Moved G1 symmetry helpers alongside the other velocity-task robot symmetry modules.
* Reused the common depth-image observation with a G1 image-processing modifier, assembling observation
  groups directly in the student environment configuration.
* Read G1 sole geometry and reference torques from the asset, reusing configured actuator stiffnesses.
