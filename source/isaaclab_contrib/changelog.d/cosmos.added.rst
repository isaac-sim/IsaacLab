* Added :mod:`isaaclab_contrib.sensors.cosmos`, which puts Cosmos3-Nano-Sim-Transfer on a camera: the camera renders
  depth, and Cosmos generates the ``rgb`` the task reads. :func:`~isaaclab_contrib.sensors.cosmos.cosmos_camera`
  converts a pinhole RGB camera. The ``control`` argument of ``cosmos_camera`` takes other processors that write the
  control image, for the segmentation, edge and blur controls the checkpoint also accepts.
  :func:`~isaaclab_contrib.sensors.cosmos.episode_frame_budget` sizes a session to the episode.
* Added :mod:`isaaclab_contrib.sensors.image_transfer`, camera tensor processors for any image-generation model,
  which Cosmos builds on. :class:`~isaaclab_contrib.sensors.image_transfer.ImageTransferProcessorCfg` queues
  captures per view, following the camera's ``env_mask``, resets each view with its own deterministic seed (from a
  base seed or a list with one seed per view), and shares one model per configuration through the simulation
  context. It requires an explicit capture ``env_mask`` and rejects asynchronous camera calls before queuing
  controls. :class:`~isaaclab_contrib.sensors.image_transfer.DepthControlProcessorCfg` prepares depth controls.
* Added :class:`~isaaclab_contrib.sensors.cosmos.CosmosClientCfg` settings: one prompt per environment, or prompt files
  read with :func:`~isaaclab_contrib.sensors.cosmos.read_prompt_files`; several cameras sharing one session; several
  services, one per GPU, each generating a share of the environments; the transport, CUDA IPC through Warp or the
  socket; and ``fps``, the frame rate the model is told.
* Added single-frame Cosmos generation, ``CosmosClientCfg.generation_mode="single_frame"``, which generates each image
  from the current capture alone, with no history. ``decision_interval`` and ``augmentation_probability`` generate
  once per action chunk or for a share of the captures and publish the rendered ``rgb`` otherwise, and each
  environment's episode moves to its next prompt. ``CosmosClient.offload`` and ``activate`` move a service's weights
  to host memory and back between rollouts.
* Added ``python -m isaaclab_contrib.sensors.cosmos.cli status``, which reports whether a running Cosmos service is
  ready and has an active camera session, and its ``offload`` and ``activate`` commands.
* Added the Cosmos Transfer camera integration guide, which covers serving Cosmos, and the Shadow Hand demo.
