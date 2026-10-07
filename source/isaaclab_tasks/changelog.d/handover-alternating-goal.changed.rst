* Changed ``Isaac-Shadow-Handover-Direct`` to alternate its goal between the two hands
  after each completed transfer, continuing until timeout or an object drop.
* Added ``success_dwell_steps`` (20 cumulative steps inside the success distance) and
  ``goal_position_offset`` (a shared offset in each hand's local frame).
* Changed ``Metrics/success_rate`` to report ``goals / (goals + 1)`` and added
  ``Metrics/consecutive_success`` for the number of transfers per episode. Success rates
  from the previous single-goal task are not directly comparable; retrain policies for
  the alternating task objective.
