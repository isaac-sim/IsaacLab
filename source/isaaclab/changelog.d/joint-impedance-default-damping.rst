Fixed
^^^^^

* Fixed :class:`~isaaclab.controllers.JointImpedanceController` construction failing when
  ``damping_ratio`` was omitted by defaulting it to 1.0 in the configuration. Explicit ``None``
  remains unsupported; omit the field or set a numeric damping ratio instead.
