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
