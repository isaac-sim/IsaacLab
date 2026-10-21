* Added :class:`~isaaclab_newton.actuators.BamCommandLag`, :class:`~isaaclab_newton.actuators.BamFriction`
  and :class:`~isaaclab_newton.actuators.InputProcessorPositionBacklash`, the BAM servo parts Newton's native
  BAM path lacks: the per-environment random command lag, the gearbox friction budget, and an encoder that
  reads the play angle but not its velocity.
* Added :func:`~isaaclab_newton.actuators.get_bam_group` and :class:`~isaaclab_newton.actuators.BamGroup`
  to read and write a Newton-executed BAM group's friction scale and supply.
