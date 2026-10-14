* Added :meth:`~isaaclab.scene.InteractiveScene.export_to_usd` and
  :attr:`~isaaclab.scene.InteractiveSceneCfg.export_usd_path` to export one initialized environment as USD.
  The file keeps assets as references to their source files and writes every physics value of the
  active backend that the USD does not already state, such as actuator gains and armature. Files the run
  wrote to the temporary directory, such as re-saved actuator networks, are copied next to the export.
