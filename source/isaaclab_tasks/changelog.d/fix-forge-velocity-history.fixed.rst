* Fixed FORGE policy linear and angular velocity observations to use independently maintained
  noisy-pose history, synchronized with the final reset observations, and corrected the XYZW
  quaternion projection to retain yaw motion. Existing policy checkpoints may need retraining
  because the velocity inputs changed.
