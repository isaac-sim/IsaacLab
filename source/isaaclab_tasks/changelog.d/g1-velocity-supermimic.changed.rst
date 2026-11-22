* Changed the motion tracking ``ArticulationKinematics`` to compute an optional selection of bodies, so only the joints
  between them and the floating base must be revolute.
* Changed the flat G1 SuperMimic agent to the settings of the official SuperMimic code: no world-model-only warm-up, no
  collection noise and BF16 matrix operations.
