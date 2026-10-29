* Added a Franka suction parcel-sorting task with twelve cuboid shapes, five physical
  collection lanes, and success-guided sampling through the shared reset-state API.
  This experimental task uses the upstream Newton surface-gripper prototype and trains
  a new policy; finger-gripper conveyor checkpoints are not compatible.
  Invalid policy actions and non-finite physical states cannot count as successful episodes.
  Twelve-parcel training also samples partially sorted physical batches without rewarding
  the pre-sorted inventory or including those resets in full-start success metrics.
