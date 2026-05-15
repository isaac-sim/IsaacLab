Fixed
^^^^^

* Fixed ``AttributeError: 'NoneType' object has no attribute 'shape'`` raised
  when instantiating skrl PPO models for the ``Isaac-TrackPositionNoObstacles-ARL-Robot-1-*``
  and ``Isaac-Navigation-3DObstacles-ARL-Robot-1-*`` tasks. The drone-ARL skrl
  configs used ``input: STATES`` for both policy and value networks, which
  skrl 2.0 resolves against ``state_space`` (``None`` for single-agent
  environments). Updated the configs to use ``input: OBSERVATIONS`` to match
  the rest of the single-agent skrl configs in IsaacLab.
