Added
^^^^^

* Added ``PpispProcessorCfg`` for ordered visual observation processing,
  including USD configuration discovery and controller buffer initialization.

Changed
^^^^^^^

* Moved PPISP execution and controller state from renderer instances to processing
  chains owned by ``mdp.processed_image`` observation terms. Existing
  ``CameraCfg.isp_cfg`` configurations remained supported through a compatibility
  adapter. Managed environments can migrate by setting ``CameraCfg.isp_cfg=None``
  and passing ``processors=[PpispProcessorCfg(isp_cfg=existing_cfg)]`` in an
  ``ObservationTermCfg(func=mdp.processed_image, ...)``.
