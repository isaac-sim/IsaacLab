* Deprecated the ``isaaclab_physx`` package name. Import from :mod:`isaaclab_isaacsim` instead. The old name still
  imports, emits a :class:`DeprecationWarning`, and is removed in 3.2.
* Deprecated the whole package, which is the Isaac Sim (Kit) path, in 3.1 for removal in 3.2 together with the Kit
  visualizer and the ``isaacsim_physx`` and ``isaacsim_rtx`` selectors. Use :mod:`isaaclab_newton` or
  :mod:`isaaclab_ov` for Kit-free runs.
