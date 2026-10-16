Fixed
^^^^^

* Fixed the runtime benchmark label lookup so that the FeatherPGS physics presets run through
  :mod:`scripts.benchmarks.runtime` and are labelled ``newton_feather_pgs`` in the output bundle. The lookup
  matched none of its markers for the FeatherPGS manager path and raised ``Unsupported concrete physics config``
  before any physics ran; the marker is ``feather_pgs`` so that the Featherstone manager is not mislabelled, and
  the label was added to :data:`~isaaclab.benchmark.schema.PhysicsBackend`.
