* Fixed NaN gradients in ``axis_angle_from_quat`` at identity rotations by keeping the
  unselected exact-form division finite during the Taylor approximation.
