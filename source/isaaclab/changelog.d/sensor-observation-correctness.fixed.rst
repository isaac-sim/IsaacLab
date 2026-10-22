* Kept physics-backed sensors due after a reset read skipped capture, so their first
  post-step measurement was not delayed by the sensor update period.
* Kept sensor environment rows owned by the shared scheduler and rejected incomplete
  native bindings before indexed updates or resets.
