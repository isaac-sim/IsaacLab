Fixed
^^^^^

* Reduced PhysX ray-caster update overhead by caching the PhysX transform view and replaying its Warp kernels through
  a CUDA graph. Updating the sensor while an outer CUDA graph capture is active now raises an error, since replays of
  such a graph would consume stale transforms.
