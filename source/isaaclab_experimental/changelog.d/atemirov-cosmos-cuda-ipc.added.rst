* Added a CUDA IPC transport to the Cosmos service: on the same Linux machine and GPU, controls and generated images
  stay in shared GPU memory, ordered by interprocess CUDA events, with no host copies or synchronization.
  :attr:`~isaaclab_experimental.cosmos.CosmosModelCfg.transport` selects ``auto`` (default), ``cuda_ipc`` or
  ``socket``.
* Added Unix socket endpoints (``unix:///path``) to the Cosmos service. On Linux the service and cameras default
  to ``/tmp/isaaclab-cosmos-<uid>.sock``, which only the user can open; ``tcp://host:port`` remains available for
  Isaac Lab on another machine.
