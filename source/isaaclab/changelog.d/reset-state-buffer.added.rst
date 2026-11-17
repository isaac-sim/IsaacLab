* Added Warp-native ``ResetStateBuffer`` and ``ResetStateSampler`` in ``isaaclab.envs.reset``
  with preallocated masked replay, explicit catalog IDs, deterministic recent-outcome tracking,
  and CUDA graph capture. Buffers supported empty state trees and broadcasting one default state.
* Added metric-based sampling scores with an analytic unit-peak target-success kernel and
  uniform coverage. Torch inputs and environment IDs were boundary shims into native arrays and masks.
* Added an explicit event-term reseeding lifecycle before seeded environment resets.
  Native reset streams were reseeded in place, preserving graph bindings and outcome history.
