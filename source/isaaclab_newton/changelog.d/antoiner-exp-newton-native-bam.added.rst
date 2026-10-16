* Added :class:`~isaaclab_newton.actuators.BamCommandLag` and :class:`~isaaclab_newton.actuators.BamFriction`,
  the BAM servo parts Newton's native BAM path lacks: the per-environment random command lag and the gearbox
  friction budget.
* Added :func:`~isaaclab_newton.actuators.get_bam_group` and :class:`~isaaclab_newton.actuators.BamGroup`
  to read and write a Newton-executed BAM group's friction scale and supply.
