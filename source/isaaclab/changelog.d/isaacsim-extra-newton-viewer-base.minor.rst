Added
^^^^^

* Added an ``isaacsim`` extra to the root ``pyproject.toml`` so the PhysX
  backend can be pulled in directly under uv, e.g.
  ``uv run --extra isaacsim isaaclab train --task Isaac-Cartpole-Direct presets=physx``.
  Isaac Sim narrows ``newton`` to its own pinned version, while the base install
  otherwise tracks the latest ``newton[sim]>=1.2.0`` from the package index.

Changed
^^^^^^^

* The Newton interactive viewer GUI (``imgui-bundle``, ``typing-extensions``) is
  now part of the base install, so the viewer's HUD controls work without
  ``--extra newton``. The ``newton`` extra / ``-i newton`` token is retained as a
  backwards-compatible alias.
* The Isaac Sim version is now declared once in the root ``pyproject.toml``
  ``isaacsim`` extra and read from there by the documentation build and the
  license-check CI workflow, instead of being hard-coded in each location.
