Fixed
^^^^^

* Fixed the Mimic generation loop never bounding a run by failure count. ``env_loop`` now stops when
  ``datagen_config.max_num_failures`` failed attempts have accumulated, alongside the existing stop
  on enough successes or attempts.

Removed
^^^^^^^

* Removed the ``datagen_config.max_num_failures = 25`` assignment from the shipped Mimic environment
  configs. The field was never read when those lines were written, so honouring it now would newly
  cap every shipped task at 25 failed attempts and cut short any run asking for a large number of
  demos. Set the field explicitly to opt into a cap.
