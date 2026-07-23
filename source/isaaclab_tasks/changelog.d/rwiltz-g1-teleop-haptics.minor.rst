Added
^^^^^

* Added controller haptic feedback to the two G1 locomanipulation teleop example
  environments (``IsaacContrib-PickPlace-Locomanipulation-G1-Abs`` and
  ``IsaacContrib-PickPlace-FixedBaseUpperBodyIK-G1-Abs``). Each env now has per-hand
  finger ``ContactSensor`` s and a ``haptic_feedback`` config so the XR controller
  vibrates when the corresponding G1 hand applies force to an object.
