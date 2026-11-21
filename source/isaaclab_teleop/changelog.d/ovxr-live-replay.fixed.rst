* Fixed replay ending at a recorded RESET; it now resets and waits for the next recorded START.
* Fixed teleoperation cleanup stopping at the first failure: a retargeting UI or XR anchor
  cleanup failure no longer skips the Isaac Capture session and CloudXR shutdown, and a failed
  device entry stops a runtime it already launched. The first failure is still reported.
* Fixed demonstration recording without a Kit window dropping its instructions; they are now logged.
