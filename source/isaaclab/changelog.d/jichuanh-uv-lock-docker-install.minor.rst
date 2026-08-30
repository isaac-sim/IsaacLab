Changed
^^^^^^^

* Changed the Docker images to install from ``uv.lock`` into a virtual environment at
  ``/opt/isaaclab-venv`` instead of into Isaac Sim's own site-packages, leaving the shipped
  Isaac Sim environment untouched and applying ``[tool.uv] override-dependencies``.

Added
^^^^^

* Added the ``teleop-no-isaacsim`` extra, carrying the teleop stack without the Isaac Sim
  wheel for environments that already provide Kit. ``teleop`` now composes from it and is
  unchanged for existing users.
* Added the ``kit-image`` extra, naming the package set the Isaac Sim container installs.
* Added the ``test-runtime`` extra, carrying the test runners (``pytest``, ``pytest-mock``,
  ``junitparser``, ``flaky``, ``coverage``) without the Sphinx documentation stack. ``test``
  now composes from it and is unchanged for existing users. Both Docker images install it, so
  in-container ``pytest`` works without a separate install step.
