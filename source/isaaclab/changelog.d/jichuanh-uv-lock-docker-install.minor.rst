Changed
^^^^^^^

* Changed the Docker images to install from ``uv.lock`` into a virtual environment at
  ``/opt/isaaclab-venv`` instead of into Isaac Sim's own site-packages, leaving the shipped
  Isaac Sim environment untouched and applying ``[tool.uv] override-dependencies``.
