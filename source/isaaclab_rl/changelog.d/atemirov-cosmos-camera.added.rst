* Added ``--cosmos`` and ``--cosmos_config FILE`` to ``isaaclab train`` and ``isaaclab play`` to generate a camera
  task's policy images with a running Cosmos service, without changes to the task. The YAML file sets the prompt,
  camera, control, depth range, endpoint, and transport. ``--num_envs`` environments are requested from the service
  and generated as one batch.
