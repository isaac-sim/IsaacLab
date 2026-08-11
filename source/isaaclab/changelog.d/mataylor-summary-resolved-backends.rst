Changed
^^^^^^^

* Changed :func:`~isaaclab.app.scan` to request the Kit visualizer for livestreamed runs itself,
  instead of relying on :func:`~isaaclab.app.launch_simulation` to do so beforehand. Callers that
  scan a config directly now resolve the ``rtx`` renderer preset the same way the launcher does.
