Changed
^^^^^^^

* Changed ``isaaclab.sh --install`` to install from ``uv.lock`` when the target is a
  ``uv``-managed virtual environment, making the install reproducible and applying
  ``[tool.uv].override-dependencies``. Other environments keep the existing pip path.
* Changed the Docker images to install into a virtual environment at ``/opt/isaaclab-venv``
  instead of into Isaac Sim's own site-packages, leaving the shipped Isaac Sim environment
  untouched.
