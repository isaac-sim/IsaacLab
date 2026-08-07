Fixed
^^^^^

* Fixed OmniHub failing to launch in the Isaac Sim based containers, which stalled Kit startup by
  roughly ten seconds with repeated ``Hub failed to launch`` warnings. Its cache directory
  ``/var/cache/hub`` was owned by the image's ``isaac-sim`` user and therefore not writable by the
  container's runtime user.
