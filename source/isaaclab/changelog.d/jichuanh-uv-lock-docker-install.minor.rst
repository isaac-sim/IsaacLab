Changed
^^^^^^^

* Changed the Docker images to install from ``uv.lock`` into a virtual environment at
  ``/opt/isaaclab-venv`` instead of into Isaac Sim's own site-packages, leaving the shipped
  Isaac Sim environment untouched and applying ``[tool.uv] override-dependencies``.
* Changed the ``teleop`` extra to carry only the teleop stack; Isaac Sim is no longer bundled
  with it. Environments that already provide Kit, such as the container images, no longer
  install a second copy of it. Install ``--extra teleop,isaacsim`` to get the Isaac Sim wheel
  as before.
* Renamed the ``test`` extra to ``dev``, which still carries the test suite plus the
  documentation toolchain. Use ``--extra dev`` where ``--extra test`` was used to build docs.
* Changed the ``test`` extra to carry the test suite alone, without the documentation
  toolchain, so the container images can install it without shipping Sphinx and its
  GPL-3.0-or-later ``docutils`` dependency.
