* **Breaking:** The UR10 particle-push task no longer opens a Newton GL visualizer in play mode. Pass
  ``--visualizer newton_gl`` to open one.
* :func:`~isaaclab_tasks.utils.setup_preset_cli` never takes a Hydra override as the value of an option whose
  value is optional: ``--video presets=newton_mjwarp`` uses the ``--video`` default and passes the override on.
