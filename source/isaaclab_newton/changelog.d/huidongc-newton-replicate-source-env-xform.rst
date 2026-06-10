Fixed
^^^^^

* Fixed Newton replication to apply per-source world transforms when adding
  prototypes to cloned worlds, instead of always offsetting from ``env_0``.
* Fixed Newton replication to resolve each source env from its destination
  template slot instead of assuming ``/World/envs/env_<id>`` paths.
