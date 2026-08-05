Added
^^^^^

* Added ``OVRTX_SHADER_CACHE_PATH`` environment variable support to
  :class:`~isaaclab_ov.renderers.OVRTXRenderer`. When set, the NVIDIA Vulkan
  driver PSO cache (``nv_shadercache``) is redirected to the specified path via
  the ``/rtx/shaderDb/driverShaderCachePath`` carb setting before
  :class:`~ovrtx.Renderer` construction. This allows CI to persist the compiled
  shader cache across runs and eliminates the ~600 s cold-compile cost on
  repeated test executions.
