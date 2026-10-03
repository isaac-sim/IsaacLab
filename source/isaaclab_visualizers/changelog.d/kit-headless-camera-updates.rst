Fixed
^^^^^

* Kept headless Kit recording cameras under the direct USD pose publisher.
  Avoided viewport initialization creating stronger session-layer transforms
  that froze subsequent camera movement in captured frames.
  Published direct camera poses in Kit's session layer so they override the
  renderer's default camera transforms.
