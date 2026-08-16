Changed
^^^^^^^

* **Breaking:** Changed ``load_torchscript_model`` to raise ``RuntimeError`` when TorchScript loading fails instead
  of returning ``None``. Callers that previously checked for a ``None`` return should catch ``RuntimeError`` at
  the load call instead.
