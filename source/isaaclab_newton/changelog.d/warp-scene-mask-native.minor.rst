Added
^^^^^

* Added masked body-write kernels so the rigid-object-collection
  ``write_*_to_sim_mask`` methods execute mask-natively and become eligible for
  CUDA-graph capture.
* Added ``reset_capture_safe`` declarations to Newton assets for the scene-level
  capture-eligibility query.

Changed
^^^^^^^

* Changed masked articulation resets to forward the boolean mask to Lab
  actuators through :meth:`~isaaclab.actuators.ActuatorBase.reset_mask` instead
  of materializing compact environment IDs in the asset.
