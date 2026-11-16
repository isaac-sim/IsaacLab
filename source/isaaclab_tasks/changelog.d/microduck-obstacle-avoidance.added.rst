* Added ToF-only MicroDuck assisted navigation with native PPO, a tanh actor, full-support
  Beta residuals, privileged critic and frozen TorchScript walker. The task defaults support
  one uninterrupted training command with gamma 0.998 and mirror loss, without batch augmentation.
* Added weighted obstacle layouts with seven training families and scattered front
  courses comprising 65 percent of resets. Joystick-like requests steer
  toward fixed forward goals while the actor handles avoidance. Wide blocked approaches receive 2 percent
  of resets; all six prop roles shuffle, including cones in either encounter position.
  Legacy ring/slalom layouts are excluded until goal placement is corrected. Two-obstacle
  courses leave the straight-ahead goal free, and blocked approaches retain lateral exits.
  Half of scattered courses align two obstacles along the goal corridor. Half of passage
  resets contain one shuffled prop between wider walls with clearance on both sides.
  Episodes last 20 seconds.
* Added task-local optical-axis ToF sensing with cached range/validity/age history, delayed
  packets, floor-return calibration and balanced left/right floor-edge randomization.
* Added approach-risk-gated progress, movement-gated tracking, sparse safe-arrival credit
  with revocation after failure, geometry-based bypass rewards and native contact terminations.
* Added stale-sensor stopping and CPU coverage for rewards, shuffled goal anchors, partial
  resets, action limits, distribution behavior and native mirror loss.
* Added one mixed SimReady/primitive navigation scene with public box, bin, cabinet and
  cone assets alongside a low block and walls, with separate layout sampling and simulator
  writes, shared passage geometry, and explicit bypass/goal stages.
* Referenced the upstream MicroDuck and scaled Isaac Sim cone, with revision-pinned Hugging Face
  SimReady packages downloaded to ``assets/simready`` before training. Scene-layer spawning
  consolidates and centers fixed props without modified source files, a preparation tool
  or a local asset manifest.
* Added laptop-based ONNX gamepad navigation tools in ``scripts/deploy/microduck_tof_nav``,
  with dry-run logging, command limits, stale-input stopping and offline regression tests.
