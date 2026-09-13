Added
^^^^^

* Added ``Isaac-Velocity-Rough-G1-29Dof-AirTime100-Nf-DepthDistill``, which distils the
  fingerless G1 velocity teacher into a student reading a chest depth camera. Its action space is
  the 29 joints the hardware drives rather than the asset's 43, so the teacher observation group
  is scoped to the same joints; existing distillation tasks are unchanged.
