Added
^^^^^

* Added an RSL-RL training configuration and behavioral-success metrics to the
  Shadow handover Direct task.
* Added renderer presets and configuration validation to the Shadow camera
  Direct task, including an RGB-depth preset for training with the Newton Warp
  renderer.
* Added OVPhysX physics presets to the handover and camera Direct
  environments.

Deprecated
^^^^^^^^^^

* Deprecated ``shadow_hand_camera_env.compute_keypoints`` in favor of
  :func:`~isaaclab_tasks.core.reorient.mdp.observations.compute_cube_keypoints`.
* Deprecated the ``Isaac-Reorient-Cube-Shadow-Camera-Benchmark-Direct``
  registration in favor of the regular camera task with the
  ``env.feature_extractor.enabled=False`` override.

Fixed
^^^^^

* Fixed handover construction on Newton, broken by renamed distal joints in
  the current Shadow Newton asset.
