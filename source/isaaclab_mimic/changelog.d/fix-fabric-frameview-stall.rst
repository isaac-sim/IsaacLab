Fixed
^^^^^

* Fixed :class:`SceneAsset` leaking its cached frame view when the view is rebuilt,
  which left the view's backend state to be released on garbage collection.
