* Routed frame-transformer, PVA, and frame-view state reads through the existing lazy
  scene-data state accessor so the Newton manager resolves pending kinematics.
* Initialized PVA pose and quaternion buffers to their identity reset defaults.
