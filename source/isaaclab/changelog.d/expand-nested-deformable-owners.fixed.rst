* Expanded deformable geometry through every imported ancestor source when world compositions selected different nested assets.
* Preserved the deprecated sensor-frame spawning APIs for one compatibility cycle; new ray casters should set
  ``RayCasterCfg.spawn`` to ``None`` and track an existing frame.
