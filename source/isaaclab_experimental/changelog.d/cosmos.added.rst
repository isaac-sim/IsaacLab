* Added :mod:`isaaclab_experimental.image_transfer`, which generates camera images with an application-owned model
  as camera tensor processors: :class:`~isaaclab_experimental.image_transfer.ImageTransferProcessorCfg` queues
  captures per view, following the camera's ``env_mask``, resets each view with its own deterministic seed
  (from a base seed or a list with one seed per view), and
  shares one model per configuration through the simulation context.
  :class:`~isaaclab_experimental.image_transfer.DepthControlProcessorCfg` prepares depth controls.
* Added :mod:`isaaclab_experimental.cosmos`, which puts Cosmos3-Nano-Sim-Transfer on a camera: the camera renders
  depth, and Cosmos generates the ``rgb`` the task reads. :func:`~isaaclab_experimental.cosmos.cosmos_camera`
  converts a pinhole RGB camera. Image transfer processors require an explicit capture ``env_mask`` and reject
  asynchronous camera calls before queuing controls. The ``control`` argument of ``cosmos_camera`` takes other processors that
  write the control image, for the segmentation, edge and blur controls the checkpoint also accepts.
  :func:`~isaaclab_experimental.cosmos.episode_frame_budget` sizes a session to the episode.
* Added :class:`~isaaclab_experimental.cosmos.CosmosModelCfg` settings: one prompt per environment, or prompt
  files read with :func:`~isaaclab_experimental.cosmos.read_prompt_files`; several cameras sharing one session;
  several services, one per GPU, each generating a share of the environments; the transport; and ``fps``, the
  frame rate the model is told.
* Added the standalone ``isaaclab-cosmos`` package with the ``isaaclab-cosmos-server`` command. It serves the
  public Cosmos Framework from its own environment, keeps the model loaded between sessions, and generates all of a
  session's views as one batch. Images move over a Unix socket or TCP, or, on one Linux machine, over CUDA IPC
  between GPUs that can access each other.
* Added the ``Isaac-Reorient-Cube-Shadow-Camera-Cosmos-Direct`` and ``Isaac-Reorient-Cube-Shadow-Camera-Cosmos``
  tasks: the Shadow Hand camera tasks with a camera image that Cosmos generates from depth, one prompt per
  environment. ``env.cosmos`` holds the camera's Cosmos settings, so overrides such as ``env.cosmos.prompt=...``
  set them. The package registers them through the ``isaaclab.tasks`` entry point.
* Added single-frame Cosmos generation, ``CosmosModelCfg.generation_mode="single_frame"``, which generates each
  image from the current capture alone, with no history. ``decision_interval`` and ``augmentation_probability``
  generate once per action chunk or for a share of the captures and publish the rendered ``rgb`` otherwise, and
  each environment's episode moves to its next prompt. ``CosmosModel.offload`` and ``activate`` move a service's
  weights to host memory and back between rollouts.
* Added ``python -m isaaclab_experimental.cosmos.client.cli status``, which reports whether a running Cosmos
  service is ready and has an active camera session, and its ``offload`` and ``activate`` commands.
* Added the Cosmos Transfer camera integration guide, which covers serving Cosmos, and the Shadow Hand demo.
