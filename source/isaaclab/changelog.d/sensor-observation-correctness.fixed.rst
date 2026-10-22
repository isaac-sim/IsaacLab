* Centralized capture readiness in the sensor scheduler, retaining reset values until
  the first physics step without delaying capture by the sensor update period.
* Kept sensor environment rows owned by the shared scheduler and rejected incomplete
  native bindings before indexed updates or resets.
