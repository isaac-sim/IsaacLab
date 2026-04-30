Changelog
---------

0.1.3 (2026-04-30)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Simple-shading outputs, with RTX Minimal mode resolved from the requested camera data types and written on
  the injected render product in USD.
* Expanded unit tests for OVRTX Warp kernels in ``test_ovrtx_renderer_kernels.py``.

Changed
^^^^^^^

* OVRTX integration now branches ``read_gpu_transforms``, depth tile extraction, and semantic ID coloring kernels on
  ovrtx **0.3.0** vs older versions so tiled buffers and transforms stay correct across ovrtx versions.
* RGB tiling reads ``LdrColor`` and supports both 3- and 4-channel buffers.

Removed
^^^^^^^

* Removed ``OVRTXRendererCfg.simple_shading_mode``. Request simple shading via the simple-shading data types on the
  camera instead; the renderer derives RTX minimal mode from the data types.

0.1.2 (2026-03-23)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Semantic segmentation in :class:`~isaaclab_ov.renderers.OVRTXRenderer` maps
  semantic instance IDs to RGBA using the same pseudo-random per-ID HSV scheme as the
  Isaac Sim RTX render backend, so OVRTX and Isaac RTX produce matching colors for the
  same IDs. Numeric IDs ``0`` (BACKGROUND) and ``1`` (UNLABELLED) use fixed RGBA.

0.1.1 (2026-03-07)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ``ovrtx>=0.2.0,<0.3.0`` as a declared dependency, installable from the
  public NVIDIA package index (``pypi.nvidia.com``).
* Added ``ov`` to the list of valid sub-packages for selective installation via
  ``./isaaclab.sh -i ov``.

0.1.0 (2026-03-04)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :mod:`isaaclab_ov.renderers` module with OVRTX renderer for tiled camera
  rendering:

  * :class:`~isaaclab_ov.renderers.OVRTXRenderer` and
    :class:`~isaaclab_ov.renderers.OVRTXRendererCfg`: RTX-based rendering via the
    ovrtx library, with stage export, USD cloning, and camera/object bindings.

  * :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.prepare_stage`: Base
    interface hook for stage preprocessing before create_render_data (OVRTX
    exports USD stage; Isaac RTX and Newton Warp use no-op implementations).
