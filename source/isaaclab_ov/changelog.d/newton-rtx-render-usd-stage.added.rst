* Added :class:`~isaaclab_ov.stage.OvstageBackend`, a simulation-owned rendering stage populated by
  :class:`~isaaclab_ov.cloner.OvrtxReplicateContext` using the same native clone operations as OVRTX cameras.
  Consumers such as Newton's ``ViewerRTX`` can borrow the stage with its authored materials. It requires OVStage 0.2 or newer.
