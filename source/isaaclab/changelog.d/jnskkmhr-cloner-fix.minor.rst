Fixed
^^^^^

* Fixed :func:`~isaaclab.cloner.make_clone_plan` failing to clone prims whose
  ``prim_path`` contains more than one ``.*`` wildcard (for example
  ``/World/envs/env_.*/Robot/.*/link``). Only the leading wildcard is now
  replaced with the environment index, leaving inner wildcards intact so
  tree-structured USD prims can be cloned.
