* Added the ``isaaclab_model_bridge`` package, which connects Isaac Lab to a model served by its own process and
  Python environment. :mod:`isaaclab_model_bridge.protocol` carries bounded messages over a Unix socket or TCP,
  :mod:`isaaclab_model_bridge.ipc` keeps images on the GPU through Warp's CUDA IPC between processes on one Linux
  machine, :class:`isaaclab_model_bridge.BridgeSession` runs one generation session from the Isaac Lab side, and
  :func:`isaaclab_model_bridge.serve` keeps a model resident and passes each session's settings to it.
