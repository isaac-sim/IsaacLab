Fixed
^^^^^

* Fixed excessive per-step overhead in the direct Cartpole task by batching actuator
  writes and fusing termination, reward, and observation calculations.
