* Deprecated the modifiers API in favor of ``tensor_processors``, ``TensorProcessor``, and
  ``TensorProcessorCfg``. Use ``ObservationTermCfg.tensor_processors`` for observation processing;
  legacy imports and configuration remain supported with the shared core migration warning on first use.
