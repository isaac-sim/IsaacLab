Added
^^^^^

* Added ``RendererCfg.supported_output_types()`` and recursive configclass validation so every ``CameraCfg`` detects
  renderer/data-type incompatibilities without task-specific guards, starting the simulator, or importing renderer
  implementations.
