* Fixed ``OperationalSpaceController.set_command`` raising for the ``variable`` and ``variable_kp`` impedance
  modes when the command is broadcast across environments or carries a gradient. The impedance parameters are
  now clamped out of place, so the caller's command tensor is neither written to nor treated as writable.
