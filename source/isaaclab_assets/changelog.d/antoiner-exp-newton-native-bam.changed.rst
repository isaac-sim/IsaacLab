* Changed the MicroDuck servos' supply sag to a battery resistance range of ``(0, 0.2 * kt)`` [Ohm], the former
  sag gain range of ``(0, 0.2)`` [V/(N.m)] times the XL330 torque constant, since BAM now runs on Newton's
  battery, whose voltage drops with the supply current. The duty cycle weights that current, so the supply sags
  about 9 times less than with the sag gain (0.009 V instead of 0.082 V on average over a 32-environment
  rollout). A range wide enough to match would exceed the 0.2 Ohm under which the 14 servos' supply settles.
