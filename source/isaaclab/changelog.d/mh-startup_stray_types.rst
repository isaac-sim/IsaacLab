Added
^^^^^

* Added :meth:`~isaaclab.sim.SimulationContext.require_visual_shapes` and
  :attr:`~isaaclab.sim.SimulationContext.visual_shapes_required`, which let a sensor declare
  before cloning that it draws visual-only geometry. :class:`~isaaclab.sensors.Camera` calls it
  for every renderer backend, so a headless run with cameras still imports the geometry those
  cameras render.

Changed
^^^^^^^

* Changed :func:`~isaaclab.utils.assets.read_file` to read remote assets through the local
  download cache instead of re-reading them from the server on every call. This cuts repeated
  downloads of payloads such as actuator networks at startup. Cached copies are validated
  against the server before use: the revision the copy came from (content hash and version when
  the provider reports them, otherwise size and modification time) is recorded alongside it and
  compared on every run, so an asset that changed on the server is downloaded again. Copies
  cached by earlier versions carry no recorded revision and are re-fetched once. When the server
  cannot be reached, or reports no revision metadata at all, the local copy is still used and a
  warning says so. Using a local copy is logged, with one warning per cache directory naming the
  directory and one info message per asset.
* Changed :func:`~isaaclab.cloner.replicate` to drop :class:`~isaaclab.cloner.UsdReplicateContext`
  when Kit is unavailable, since nothing composes or renders the replicated prims in that case.
