* Added a CUDA IPC transport to the Cosmos service: on the same Linux machine and GPU, controls and generated images
  move between the camera and service through shared GPU memory, ordered by interprocess CUDA events, without
  host copies or host synchronization for that transfer. Control preprocessing can still use the CPU: edge
  extraction runs in the camera process, and blur filtering copies RGB to the CPU and back in the service.
  :attr:`~isaaclab_experimental.cosmos.CosmosModelCfg.transport` selects ``auto`` (default), ``cuda_ipc`` or
  ``socket``.
* Added Unix socket endpoints (``unix:///path``) to the Cosmos service. On Linux the service and cameras default
  to ``/tmp/isaaclab-cosmos-<uid>.sock``, which only the user can open; ``tcp://host:port`` remains available for
  Isaac Lab on another machine. ``isaaclab-cosmos-server --endpoint`` replaces ``--host`` and ``--port``.
