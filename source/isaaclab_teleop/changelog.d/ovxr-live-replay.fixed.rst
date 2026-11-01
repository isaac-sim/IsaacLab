* Applied the same filtered tracking pose to separate render anchors in Kit and OVXR.
* Deferred anchor USD imports until use so Kit can initialize its own USD provider.
* Fixed teleoperation cleanup stopping at the first failure: a retargeting UI or XR anchor
  cleanup failure no longer skips the Isaac Capture session and CloudXR shutdown, and a failed
  device entry stops a runtime it already launched. The first failure is still reported.
