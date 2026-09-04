Changed
^^^^^^^

* Removed per-step host-device synchronizations from the environment hot path: scalar index
  assignments in reset paths (:class:`~isaaclab.utils.buffers.CircularBuffer`,
  :class:`~isaaclab.managers.RewardManager`, :class:`~isaaclab.managers.EventManager`, joint action
  terms, :class:`~isaaclab.sensors.SensorBase`, :class:`~isaaclab.envs.ManagerBasedRLEnv`) now use
  device-side fills, :class:`~isaaclab.managers.TerminationManager` updates its last-episode bookkeeping
  with a masked write instead of ``nonzero``, :class:`~isaaclab.markers.VisualizationMarkers` skips the
  synchronizing environment-index validation when no visualizer backend consumes marker state, :class:`~isaaclab.sensors.Camera` skips the
  device-to-host outdated-mask readback for sensors with ``update_period == 0``, and zero-range uniform
  noise is treated as the identity.
