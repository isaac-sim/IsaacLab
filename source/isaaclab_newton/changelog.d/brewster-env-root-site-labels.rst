Fixed
^^^^^

* Fixed a bodyless per-environment site carrying the label ``ft_0`` in every environment. Such a
  site is now registered with the destination template of the clone-plan row that requested it and
  labelled from the environment it lands in, so it reads e.g. ``/World/envs/env_3/ft_0``. Sites are
  still resolved by index, so consumers that look them up by label and index are unaffected.
