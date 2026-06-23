Fixed
^^^^^

* Fixed the camera-based Cartpole task failing to converge under Newton physics with the RTX
  ``depth``, ``albedo``, and ``simple_shading`` AOV observations. These AOVs bypass DLSS temporal
  accumulation, so the observation carried no temporal cue for the policy to infer velocity from
  (Newton's symplectic integrator has no implicit damping). The ``frame_stack`` default resolver
  now enables 2-frame stacking for these Newton + RTX AOVs, matching the existing Newton + Warp
  behavior; Newton + RTX ``rgb`` keeps single-frame observations as DLSS already supplies the cue.
  The resolver reads backend capability classmethods
  (:meth:`~isaaclab.physics.physics_manager.PhysicsManager.provides_implicit_damping`,
  :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.provides_temporal_camera_data`) resolved
  from the configs, instead of hard-coding backend types in the task.
