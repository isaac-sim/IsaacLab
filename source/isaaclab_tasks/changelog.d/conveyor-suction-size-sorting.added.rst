* Added a Franka suction parcel-sorting task with twelve cuboid shapes, five physical
  collection lanes, and success-guided sampling through the shared reset-state API.
  Parcels measured 8.5–19 cm long, with wider receiving lanes and a visible four-cup suction tool
  whose authored contact surfaces matched the Newton seal geometry.
  This experimental task uses the upstream Newton surface-gripper prototype and trains
  a new policy; finger-gripper conveyor checkpoints are not compatible.
  Invalid policy actions and non-finite physical states cannot count as successful episodes.
  Policy observations include pickup qualification so carry/release resets do not hide
  the lift history required for a physical receipt.
  Training defaulted to twelve-parcel, 240-second episodes and sampled partially sorted
  physical batches without rewarding pre-sorted inventory or including those resets in
  full-start success metrics.
