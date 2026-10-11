* Added ``VideoRecorderCfg(view=...)`` to share an ``ImageViewCfg`` with visualizer windows.
  Recording reused the composed device frame and host readback, replacing duplicate CPU colorization.
  Existing ``sensor:<name>`` and ``viz:<type>[:streaming_view]`` source strings remained supported.
* Selected display and recording producers together during launch. Headless recording retained the
  configured perspective producer, including when multiple viewers used the same backend.
* **Breaking:** Invalid recording sources and channel selections raised errors during recorder
  construction instead of silently disabling recording. Configure a declared sensor or a capture-capable
  visualizer with the requested outputs. Empty warmup frames continued to be skipped.
* Detected physics configuration subclasses by type during launch, preserving runtime compatibility
  checks independently of the configuration class name.
* Preserved automatic recording frame rates for non-Gym environments and used shared image composition
  for renderer gallery captures after removing the CPU colorizer.
* Bound recorder capture functions after the initial simulation reset, removing deferred source discovery
  from frame capture while preserving recordings that followed the window's camera selection.
* **Breaking:** Propagated original import errors from ``string_to_callable`` and visualizer loading. Code handling
  missing modules should catch ``ModuleNotFoundError`` instead of the former ``ValueError`` wrapper.
