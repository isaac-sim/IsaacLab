Added
^^^^^

* Added an RSL-RL training configuration and behavioral-success metrics to the
  Shadow handover Direct task.
* Added renderer presets and configuration validation to the Shadow camera
  Direct task.

Changed
^^^^^^^

* Changed the camera Direct feature-extractor keypoints helper to delegate to
  the shared :func:`~isaaclab_tasks.core.reorient.mdp.observations.compute_cube_keypoints`;
  the old ``compute_keypoints`` name is deprecated and warns.

Fixed
^^^^^

* Fixed handover construction on Newton, broken by renamed distal joints in
  the current Shadow Newton asset.
