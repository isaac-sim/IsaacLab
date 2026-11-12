* Kept OVStage selection local to the borrowed renderer when ordinary and leased
  renderers coexist, retaining both required clone routes.
* Validated auxiliary camera samples before starting scene writes and matched cameras
  under the actual cloned environment namespace.
* Reused camera readback staging buffers while preserving stream ordering and retaining
  storage when its consumers cannot be drained.
