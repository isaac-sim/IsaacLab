* **Breaking:** Changed ``Isaac-Shadow-Handover-Direct`` and ``Isaac-Shadow-Handover`` to alternate
  their goal between the two hands after each completed transfer, continuing until
  timeout or an object drop.
* Added ``success_dwell_steps`` (20 cumulative steps inside the success distance) to
  both workflows and ``goal_position_offset`` to the Direct configuration. The manager
  command's ``position_offset`` now uses each hand's local root frame instead of the
  object's initial position; update custom offsets for this frame change.
* Changed ``Metrics/success_rate`` to report ``goals / (goals + 1)`` and added
  ``Metrics/consecutive_success`` for the number of transfers per episode. Success rates
  from the previous single-goal task are not directly comparable; retrain policies for
  the alternating task objective.
