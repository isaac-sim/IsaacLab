Added
^^^^^

* Exposed the RSL-RL Gaussian distribution's standard-deviation range in the
  model configuration.
* Added configurable relative and absolute tolerances for LEAPP export parity
  validation.

Fixed
^^^^^

* Fixed RSL-RL distillation startup to require a concrete teacher checkpoint
  for a new run while allowing configured resume requests and selectors to load
  existing distillation runs.
* Fixed RSL-RL export to resolve task play-mode settings so training-only
  curricula and observation corruption are disabled while tracing.
* Fixed RSL-RL distillation export to trace the student observation groups,
  including camera images, instead of looking for an actor group.
