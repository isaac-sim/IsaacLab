* Added ``--cosmos`` and related options to ``isaaclab train`` and ``isaaclab play`` to generate a camera task's
  policy images with a running Cosmos service, without changes to the task. ``--num_envs`` environments are
  requested from the service and generated as one batch; ``--cosmos_control`` takes ``depth``,
  ``edge``, or ``blur``, and ``--cosmos_transport`` ``auto``, ``cuda_ipc``, or ``socket``.
