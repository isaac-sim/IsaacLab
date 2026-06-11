Changed
^^^^^^^

* Changed :class:`~isaaclab.controllers.rmp_flow.RmpFlowController` to load RMPFlow directly from the
  ``lula`` library on every backend, instead of going through the Isaac Sim Kit motion-generation
  extension. ``lula`` is importable both under Kit and kitless (e.g. the Newton visualizer), giving a
  single code path. The ``rmp_flow_smoothed`` variant is now available in both modes as well.
