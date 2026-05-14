Fixed
^^^^^

* Fixed :class:`~envs.LeappDeploymentEnv` crashing on ``reset()`` with
  ``AttributeError: 'LeappDeploymentEnv' object has no attribute 'extras'``
  by initializing ``self.extras`` in ``__init__``.
