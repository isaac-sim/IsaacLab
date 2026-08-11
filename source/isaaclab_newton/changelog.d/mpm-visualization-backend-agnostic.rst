Fixed
^^^^^

* Fixed :class:`~isaaclab_newton.assets.MPMObject` not creating particle visualization
  prims outside the Kit viewport visualizer, which left MPM particles invisible to Kit RTX
  cameras. The spawner now creates ``UsdGeom.Points`` prims whenever the object is configured
  visible, without inspecting the active render backend.
* Fixed MPM particle visualization prims being authored outside the environment hierarchy,
  which left them without an ``omni:scenePartition`` and therefore invisible to tiled
  renderers (Kit RTX cameras, OVRTX). The ``UsdGeom.Points`` prims are now authored as a
  ``Particles`` child of the asset prim (``/World/envs/env_{idx}/<Asset>/Particles``) with the
  reset-xform-stack flag set, so they inherit the environment's scene partition while keeping
  their world-frame positions.
