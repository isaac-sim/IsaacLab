Fixed
^^^^^

* Fixed LEAPP deployment environments failing when resetting persistent inference state after a policy step.
* Fixed LEAPP deployment to use play-mode task configuration, deterministic seeding, the standard
  event lifecycle, and backend-managed physics decimation.
* Added fail-closed controller-owned articulation-write handling, including simulated gravity
  compensation and overlap validation.
