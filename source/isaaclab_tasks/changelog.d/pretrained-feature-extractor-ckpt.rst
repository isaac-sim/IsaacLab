Fixed
^^^^^

* Fixed ``play --checkpoint pretrained`` on the Shadow Hand camera tasks, which raised
  ``ValueError: max() iterable argument is empty`` because the vision CNN the policy was trained with
  was never published. Both tasks now declare it through ``companion_checkpoints``, so it is published
  beside the policy and fetched with it, and a missing checkpoint reports the directory that was searched.
