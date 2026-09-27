Fixed
^^^^^

* Fixed PVA and contact-sensor diagnostic timings to measure device-synchronized update durations.
  Contact-sensor samples were recorded after synchronization rather than while GPU work could still be pending.
