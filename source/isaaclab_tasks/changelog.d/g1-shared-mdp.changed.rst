* Moved G1 rewards and the clearance termination into the shared velocity MDP modules,
  scheduling the termination warmup through the curriculum manager.
* Moved G1 symmetry helpers alongside the other velocity-task robot symmetry modules.
* Reused the common depth-image observation with a G1 image-processing modifier, assembling observation
  groups directly in the student environment configuration.
* Read G1 sole geometry and reference torques from the asset, reusing configured actuator stiffnesses.
