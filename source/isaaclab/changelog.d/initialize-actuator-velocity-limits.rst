Fixed
^^^^^

* Initialized actuator-resolved soft joint velocity limits during articulation construction, including unbounded
  ``RemotizedPDActuator`` semantics, so reset terms see final values before the first actuator update.
