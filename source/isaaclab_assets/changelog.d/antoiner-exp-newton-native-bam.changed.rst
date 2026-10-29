* Changed the MicroDuck servos' supply sag to a battery resistance range of ``(0, 0.2 * kt)`` [Ohm], the former
  sag gain range of ``(0, 0.2)`` [V/(N.m)] times the XL330 torque constant, since BAM now runs on Newton's
  battery, whose voltage drops with the supply current.
