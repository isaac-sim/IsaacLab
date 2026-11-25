* **Breaking:** Removed the deprecated OVRTX-owned scene API and ``ovrtx_replicate``. All OVRTX cameras,
  lidars, materials, meshes, and markers now use OvStage. Remove ``ISAAC_LAB_OVRTX_USE_OVSTAGE`` from launch
  scripts and use ``ovstage_replicate`` for native rendering copies. Sharing with OVPhysX remains opt-in
  through ``ISAAC_LAB_SHARE_OVSTAGE=1``. Rendering is synchronous; ``async_rendering=True`` warns and no
  longer pipelines captures.
