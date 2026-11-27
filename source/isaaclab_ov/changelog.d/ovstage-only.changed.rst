* **Breaking:** Removed the OVRTX-owned scene path, whose native scene API OVRTX deprecated in 0.4. All OVRTX
  cameras, lidars, materials, meshes, and markers now use OvStage. Remove ``ISAAC_LAB_OVRTX_USE_OVSTAGE`` from
  launch scripts. Sharing with OVPhysX remains opt-in through ``ISAAC_LAB_SHARE_OVSTAGE=1``. Rendering is
  synchronous and no longer pipelines captures.
