Fixed
^^^^^

* Removed a redundant ``joint_limits`` read from every reorientation and handover reset. The limits are a
  static model property already available from initialization; re-reading them per reset cost a
  pinned-host staging read and a host-to-device copy on backends that keep the property on the CPU.
