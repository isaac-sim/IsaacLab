* Deprecated USDA text inputs to ``populate_stage``, ``OvstageBackend.populate``, and
  ``OvPhysxBackend.load`` while keeping existing positional inputs and ``usda`` or ``stage_usda``
  keywords functional until Isaac Lab 4.0. Call ``populate_stage`` with ``usd_path`` pointing to
  a binary USD file. Pass an ``Sdf.Layer``, such as the result of ``export_stage_to_layer``, to
  ``OvstageBackend.populate`` and ``OvPhysxBackend.load`` instead of serializing the layer to text.
