* Added :mod:`isaaclab_experimental.image_transfer`, which generates camera images with an application-owned model
  as camera tensor processors: :class:`~isaaclab_experimental.image_transfer.ImageTransferProcessorCfg` queues
  captures per view, following the camera's ``env_mask``, resets each view with its own deterministic seed, and
  shares one model per configuration through the simulation context.
  :class:`~isaaclab_experimental.image_transfer.FunctionProcessorCfg` and ``depth_to_control`` prepare controls.
* Added :mod:`isaaclab_experimental.cosmos`, which puts Cosmos3-Nano-Sim-Transfer on a camera: the camera renders
  depth, and Cosmos generates the ``rgb`` the task reads. :func:`~isaaclab_experimental.cosmos.cosmos_camera`
  converts a pinhole RGB camera and :func:`~isaaclab_experimental.cosmos.apply_cosmos` converts a task's camera in
  place, checking first that the service is running and idle. Their ``control`` argument takes other processors
  that write the control image, for the segmentation, edge and blur controls the checkpoint also accepts.
  :func:`~isaaclab_experimental.cosmos.episode_frame_budget` sizes a session to the episode.
* Added :class:`~isaaclab_experimental.cosmos.CosmosModelCfg` settings: one prompt per environment, or prompt
  files read with :func:`~isaaclab_experimental.cosmos.read_prompt_files`; several cameras sharing one session;
  several services, one per GPU, each generating a share of the environments; and the transport.
* Added the standalone ``isaaclab-cosmos`` package with the ``isaaclab-cosmos-server`` command. It serves the
  public Cosmos Framework from its own environment, keeps the model loaded between sessions, and generates all of a
  session's views as one batch. Images move over a Unix socket or TCP, or, on one Linux machine, over CUDA IPC
  between GPUs that can access each other.
* Added the ``Isaac-Reorient-Cube-Shadow-Camera-Cosmos-Direct`` and ``Isaac-Reorient-Cube-Shadow-Camera-Cosmos``
  tasks: the Shadow Hand camera tasks with a camera image that Cosmos generates from depth, one prompt per
  environment. ``env.cosmos`` holds the camera's Cosmos settings, so overrides such as ``env.cosmos.prompt=...``
  set them. The package registers them through the ``isaaclab.tasks`` entry point.
* Added ``python -m isaaclab_experimental.cosmos.client.cli status``, which reports whether a running Cosmos
  service is ready and has an active camera session.
* Added the Cosmos service setup guide, the camera integration guide, and a demo walkthrough.
