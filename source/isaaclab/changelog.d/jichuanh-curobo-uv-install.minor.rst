Changed
^^^^^^^

* Changed the cuRobo Docker image to install from ``uv.lock`` into the virtual environment at
  ``/opt/isaaclab-venv``, matching the base image. Isaac Sim's site-packages is left untouched,
  so the image no longer deletes the prebundled torch, re-bootstraps ``pip`` afterwards, or
  uninstalls ``quadprog``. cuRobo itself is still built from its pinned commit against the
  environment's torch.
