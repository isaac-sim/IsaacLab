* Fixed :class:`~isaaclab.sensors.SensorBase` marking environments as updated after a read that captured
  nothing. Sensors that skip environments not stepped since their last reset returned the reset values for one
  more update period after a read right after the reset. These environments now stay outdated and get one extra
  capture after the next physics step. Their update period still counts from the reset, so their later updates
  keep their steps.
