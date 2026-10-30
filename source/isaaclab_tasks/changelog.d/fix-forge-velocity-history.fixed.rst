* Fixed FORGE policy linear and angular velocity observations to use consecutive noisy poses
  instead of history overwritten by the parent update, and corrected the XYZW quaternion projection
  to retain yaw motion. Existing policy checkpoints may need retraining because the velocity inputs changed.
