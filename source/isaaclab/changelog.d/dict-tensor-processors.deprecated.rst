* Deprecated ``isaaclab.utils.modifiers``, ``ModifierBase``, and ``ModifierCfg`` in favor of
  ``isaaclab.utils.tensor_processors``, ``TensorProcessor``, and ``TensorProcessorCfg``. Replace
  ``ObservationTermCfg.modifiers`` with ``ObservationTermCfg.tensor_processors``; legacy imports
  and configuration remain supported with one shared warning on first use.
