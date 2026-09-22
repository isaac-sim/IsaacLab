Changed
^^^^^^^

* Published PhysX rigid transforms and their dirty state through SDP, and routed Isaac RTX
  transform updates through its shared Fabric transport while preserving native PhysX Fabric updates.
  Kit app updates requested current SDP transforms without an additional physics ``forward()`` call.
