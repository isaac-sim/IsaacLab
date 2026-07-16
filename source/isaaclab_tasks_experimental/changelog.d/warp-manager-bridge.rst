Fixed
^^^^^

* Fixed the direct Warp Cartpole task to match the stable task's observations,
  reset ranges, termination condition, reward scaling, and scene configuration.
* Fixed the Warp Cartpole ``survival_success_rate`` twin to report the
  ``Metrics/success_rate`` value on-device instead of silently dropping the
  metric.
