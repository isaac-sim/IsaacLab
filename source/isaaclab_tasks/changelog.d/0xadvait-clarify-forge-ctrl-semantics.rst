Changed
^^^^^^^

* Clarified the documentation of the action-space semantics in the Factory and FORGE control
  configurations. In the Factory environments, the action thresholds scale per-step end-effector
  displacements and the action bounds clip the target relative to the fixed asset, while in the
  FORGE environments the action bounds map actions onto the operational volume around the fixed
  asset and the randomized action thresholds clip the per-step motion, following the FORGE paper.
