Fixed
^^^^^

* Fixed the Unitree Go1 and Go2 leg actuator limits ignoring the knee reduction. Both robots applied
  the hip and thigh limits to the calf joints, which capped calf torque well below its rated value and
  let the torque-speed curve keep motoring past its rated speed. The calf joints now use the limits
  authored in ``go1.usd`` and ``go2.usd`` (Go1: 35.55 N·m, 20.06 rad/s; Go2: 45.43 N·m, 15.70 rad/s),
  and the hip and thigh limits were aligned with the same assets (23.7 N·m, 30.1 rad/s).

  These robots now produce more calf torque at lower calf speeds, so policies trained on the previous
  configuration should be retrained rather than reused directly.
