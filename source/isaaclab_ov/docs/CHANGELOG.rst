Changelog
---------

0.1.1 (2026-03-05)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Semantic segmentation output: semantic IDs are now hashed (CRC32) before
  conversion to RGBA so that consecutive IDs (e.g. 1, 2, 3) map to visually
  distinct colors, improving distinguishability in the rendered segmentation.

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
