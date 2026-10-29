* Changed Newton-executed BAM servos to run on Newton's :class:`~newton.actuators.DriveBAM`,
  :class:`~newton.actuators.Battery`, :class:`~newton.actuators.InputProcessorDelay` and effort clamp.
* Changed the BAM supply to one battery per actuator group and environment. Separate BAM groups of one
  robot now sag independently, also when Newton merges their actuators.
