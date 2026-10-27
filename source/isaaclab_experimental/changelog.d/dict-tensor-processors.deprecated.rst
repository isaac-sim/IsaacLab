* Deprecated ``ObservationTermCfg.modifiers`` in favor of ``ObservationTermCfg.tensor_processors``.
  Retained the legacy field with a targeted warning and forwarding during manager initialization.
  Setting both fields now raises ``ValueError`` to prevent silently overriding either pipeline.
