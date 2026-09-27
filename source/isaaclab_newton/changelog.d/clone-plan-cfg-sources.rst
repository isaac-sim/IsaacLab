Changed
^^^^^^^

* Composed reusable asset builders into world prototypes before native replication,
  preserving repeated instances without widening the declared USD import scope.
  Removed the redundant post-clone label pass. Direct ``newton_physics_replicate`` calls
  retained their path-based inputs.
