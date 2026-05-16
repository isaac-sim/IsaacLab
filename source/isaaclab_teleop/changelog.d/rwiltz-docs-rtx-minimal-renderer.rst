Added
^^^^^

* Expanded the **Optimize XR Performance** documentation with guidance for
  lower-spec GPUs and complex scenes: a walkthrough for switching the
  Isaac Lab viewport to the RTX - Minimal renderer (including the
  ``DistantLight``-only lighting limitation), notes on the
  ``sim.dt`` / ``sim.render_interval`` trade-off, a description of the
  XR **Resolution Multiplier** slider for trading image sharpness for GPU
  headroom, guidance on ``RetargetingExecutionConfig`` (sync vs pipelined
  modes and ``DeadlinePacingConfig.safety_margin_s``), and a CloudXR
  frame-pacing diagnostic note. See :ref:`isaac-teleop-performance`.
