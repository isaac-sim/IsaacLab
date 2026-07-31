Fixed
^^^^^

* Deleted stale Newton-tiled golden images (``*-newton-tiled.png``) for all scenes. The golden
  images were invalidated by a Newton version bump in ``43187ba712`` which changed the Warp
  renderer's visual output. The test framework will regenerate the golden images on the next CI
  run, after which the new images should be reviewed and committed.
