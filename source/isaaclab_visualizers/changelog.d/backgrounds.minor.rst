Changed
^^^^^^^

* Unified Kit, Newton GL, and Newton RTX scene backgrounds behind ``VisualizerCfg.background_mode``. They now use a solid sky-blue background by default; set ``background_mode="sky"`` to show the native dome or procedural sky. Newton RTX uses the hosted ``blue_sky`` HDR by default.
* Deprecated ``NewtonGLVisualizerCfg.enable_sky`` in favor of ``background_mode``.
* Matched Kit and Newton RTX physical camera exposure defaults and brightened both by one EV.

Fixed
^^^^^

* Applied ``VisualizerCfg.focal_length`` to the Kit viewport camera as well as the Newton visualizers.
