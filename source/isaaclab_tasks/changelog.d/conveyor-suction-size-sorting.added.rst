* Added a Franka suction parcel-sorting task with twelve cuboid shapes, five physical
  collection lanes, and success-guided sampling through the shared reset-state API.
  Parcels span 8.5–19 cm, with wider receiving lanes and a visible four-cup suction tool
  whose authored contact surfaces match the Newton seal geometry.
  This experimental task uses the upstream Newton surface-gripper prototype and trains
  a new policy; finger-gripper conveyor checkpoints are not compatible.
  Invalid policy actions and non-finite physical states cannot earn a receipt.
  Policy observations include pickup qualification so carry/release resets do not hide
  the lift history required for a physical receipt.
  Each verified delivery earns a reward and success event before the next target is selected
  in the same physical scene. The first receipt from a sampled reset immediately updates
  its success history; subsequent receipts cannot duplicate that outcome or be erased by a
  later failure. All twelve parcels start unsorted, and complete-batch metrics remain separate
  from per-delivery success within 240-second episodes.
