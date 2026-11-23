* Added the ``isaaclab_neural_bridge`` package, which connects Isaac Lab to a model served by its own process and
  Python environment. :mod:`isaaclab_neural_bridge.protocol` carries bounded messages over a Unix socket or TCP,
  :mod:`isaaclab_neural_bridge.ipc` keeps images on the GPU through Warp's CUDA IPC between processes on one Linux
  machine, :class:`isaaclab_neural_bridge.BridgeSession` runs one generation session from the Isaac Lab side, and
  :func:`isaaclab_neural_bridge.serve` keeps a model resident and passes each session's settings to it.
