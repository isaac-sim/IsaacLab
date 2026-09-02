Changed
^^^^^^^

* Changed :class:`~isaaclab.controllers.OperationalSpaceController` to evaluate its task-space
  impedance, contact-wrench and null-space laws through Newton's model-free operational-space
  controller (:class:`newton.controllers.ControllerOperationalSpaceModelFree`), preserving its
  public configuration, command, and output contracts. The task frame is now handed to Newton as
  the operational frame, so gains, selection axes and targets stay expressed in it instead of being
  rotated into the root frame first. Solves now use float32 internal buffers.

Fixed
^^^^^

* Fixed the ``variable_kp`` impedance mode rebinding the motion damping-gain buffer instead of
  writing it in place, which left previously captured references, such as the LeApp export
  annotator's gain tensors, reading a stale buffer.
