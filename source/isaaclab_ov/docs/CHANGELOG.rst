Changelog
---------

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
