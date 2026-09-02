Fixed
^^^^^

* Fixed ``IsaacRtxRenderer.render()`` crashing with ``RuntimeError: Invalid indexing in slice``
  when an annotator's channel buffer has not warmed up yet (e.g. right after attach, at env
  creation) for the ``motion_vectors``, ``normals``, simple-shading, and RGB HDR data types.
* Added a pre-flight check in ``IsaacRtxRenderer.create_render_data()`` that raises a clear,
  actionable error when the requested ``num_envs``/camera resolution would overflow Warp's
  signed-32-bit-representable array shape limit, instead of an opaque ``ValueError`` raised deep
  inside ``render()``.
