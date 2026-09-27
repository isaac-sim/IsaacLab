Fixed
^^^^^

* Fixed model finalization, solver initialization, and contact-sensor construction timings to synchronize
  only the physics device at both measurement boundaries, excluding previously queued work from each phase.
