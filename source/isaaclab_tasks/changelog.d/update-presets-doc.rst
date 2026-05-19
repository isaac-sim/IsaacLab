Added
^^^^^

* Added :func:`~isaaclab_tasks.utils.preset_cli.enumerate_task_presets` public helper that
  returns the available preset names for a registered task, bucketed by selector type
  (``physics=``, ``renderer=``, ``presets=``). Used by tooling such as ``list_envs.py``.
* Added ``--show_presets`` flag to ``scripts/environments/list_envs.py``. When set, a
  **Presets** column is added to the environment table showing physics, renderer, and domain
  preset names available for each environment.
