* Kept cleared sensor outputs valid after reset and refreshed them on the first update,
  without delaying the first measurement by the sensor update period. Derived-query
  sensors explicitly invalidate their cached poses, ray hits, or images.
* Kept sensor environment rows owned by the shared scheduler and rejected incomplete
  native bindings before indexed updates or resets.
