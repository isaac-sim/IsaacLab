* Added :attr:`~isaaclab.managers.ObservationTermCfg.history_newest_first` and
  :attr:`~isaaclab.managers.ObservationGroupCfg.history_newest_first`, backed by the ``history_newest_first``
  option of :class:`~isaaclab.utils.buffers.CircularBuffer`, to place the current step first along the history
  dimension for policies whose deployment runtime builds observation history in that order.
