* Added ``VideoRecorderCfg(view=...)`` to share an ``ImageViewCfg`` with visualizer windows.
  Recording reused the composed device frame and host readback, replacing duplicate CPU colorization.
  Existing ``sensor:<name>`` and ``viz:<type>[:streaming_view]`` source strings remained supported.
* Selected display and recording producers together during launch. Headless recording retained the
  configured perspective producer, including when multiple viewers used the same backend.
* **Breaking:** Invalid recording sources and channel selections raised errors when bound or first
  captured instead of silently disabling recording. Configure a declared sensor or a capture-capable
  visualizer with the requested outputs. Empty warmup frames continued to be skipped.
* Detected physics configuration subclasses by type during launch, preserving runtime compatibility
  checks independently of the configuration class name.
