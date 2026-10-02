* **Breaking:** Required clone contexts to derive from :class:`~isaaclab.cloner.ReplicateContext`.
  Replaced optional method discovery with explicit preparation before construction and dispatch.
  Custom contexts must inherit the base class and implement ``replicate``; contexts changing routes
  may override the static ``prepare`` method. Context implementations access their simulation through
  ``self._sim``, including ``self._sim.stage`` for the USD stage.
