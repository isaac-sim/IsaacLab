* Added :func:`~isaaclab.app.validate_launch` to check whether a configuration's physics, renderer and
  visualizer combination can launch, without starting a runtime. :func:`~isaaclab.app.launch_simulation`
  runs it first.
* Added ``--resolve`` to ``isaaclab list_envs`` to list the preset combinations each task can launch headless.
