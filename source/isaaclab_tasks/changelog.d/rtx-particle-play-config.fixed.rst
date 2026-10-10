* Removed GL-only particle options from the Franka pour and UR10 particle push RTX playback configs.
  RTX used the particle appearance authored in the shared scene.
* Kept playback camera defaults on their existing configuration instead of rebuilding an RTX configuration
  from the same values. Removed the soft-body lift task's redundant window configuration subclass.
