Fixed
^^^^^

* Fixed ``ValueError: max() iterable argument is empty`` when playing the Shadow Hand camera tasks with no
  feature-extractor checkpoint in the log directory. The missing checkpoint is now reported together with the
  directory that was searched.

* Fixed the feature-extractor checkpoint path being resolved against the log directory twice, which raised
  ``FileNotFoundError`` whenever the log directory was relative.
