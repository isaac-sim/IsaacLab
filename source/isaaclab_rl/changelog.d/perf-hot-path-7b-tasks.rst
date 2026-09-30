Changed
^^^^^^^

* Computed the Stable-Baselines3 wrapper's done flags from the host copies of the termination and
  truncation flags, saving one device-to-host transfer per step.
