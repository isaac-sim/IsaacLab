Fixed
^^^^^

* Fixed nested :class:`~isaaclab_tasks.utils.hydra.PresetCfg` resolution so
  child preset choices are scoped to the selected parent branch.
* Improved task config resolution time by bypassing Hydra composition when only
  preset selections or plain scalar overrides are used.
