Changed
^^^^^^^

* Changed Newton RTX's default lighting environment to consume scene lights imported by the cloner,
  including their HDR texture, light properties, and initial transform. Set ``rtx_environment="studio"`` or ``"none"``
  to select an explicit renderer-owned alternative.
