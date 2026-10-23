* Bound teleop target frames and XR anchor references once during setup instead of rediscovering the
  Fabric stage and prim every update. Live pose updates and explicit target-matrix overrides were preserved.
* Rejected nonexistent teleop target-frame paths before starting the session instead of silently emitting
  poses in the world frame. Ensure ``target_frame_prim_path`` exists before entering the device context.
