Fixed
^^^^^

* Fixed :class:`~isaaclab_physx.renderers.IsaacRtxRenderer` failing with
  ``Annotator SimpleShadingSD is not attached to any render products`` when
  multiple environments sequentially request ``simple_shading_*`` camera
  outputs in the same Kit process. Each tiled render product is given a
  unique UUID name, and the owned HydraTexture is destroyed on cleanup to
  avoid leaking render products across env create/destroy cycles.
