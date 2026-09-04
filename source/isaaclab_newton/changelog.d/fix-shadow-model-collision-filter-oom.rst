Fixed
^^^^^

* Fixed PhysX-backend Newton visualization models replicating unused collision
  filters and contact pairs, which could exhaust memory during ``ModelBuilder.finalize()``.
