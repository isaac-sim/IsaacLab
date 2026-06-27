Changed
^^^^^^^

* Centralized all Isaac Lab third-party dependencies (required and optional)
  into the root ``pyproject.toml`` as the single source of truth. The wheel
  builder (``tools/wheel_builder/gen_pyproject.py``) and the ``./isaaclab.sh -i``
  install CLI now read the root project's ``dependencies`` and
  ``optional-dependencies`` instead of ``tools/wheel_builder/res/python_packages.toml``
  (removed). The ``./isaaclab.sh -i`` token syntax is unchanged.
* Each sub-package's ``pyproject.toml`` keeps complete, self-contained
  (relaxed, ``>=``) dependency metadata, but the declarations are generated from
  the root ``[tool.isaaclab.packages]`` table by ``tools/gen_package_pyproject.py``
  so they are still edited in one place. Exact, known-good versions live in the
  root ``[tool.uv]`` configuration and ``uv.lock`` (not in the published package
  metadata). A pre-commit hook / unit test fails if a sub-package drifts out of
  sync with the root table.
* **Changed:** Newton (``newton[sim]``) is now a core dependency installed in
  every environment as the default physics engine, rather than an opt-in extra.
  The Newton interactive viewer GUI is also part of the base install, so the
  ``newton`` optional extra has been removed; the ``newton`` install token /
  ``--extra newton`` is now a no-op kept for backward compatibility.
* Added the ``[tool.isaaclab.versions]`` table to the root ``pyproject.toml`` as
  the single source of truth for externally-pinned versions (Isaac Sim, the
  torch stack, and the OV renderer/physics wheels). The install CLI, docs, and
  CI read these values; a unit test enforces that the literal pins in the extras
  and ``[tool.uv].constraint-dependencies`` stay in sync with the table.
* **Changed:** The aggregate ``all`` extra now contains only packages that can
  co-resolve with ``isaacsim`` (the documented ``[all,isaacsim]`` install).
  ``ov`` (OVRTX / OvPhysX), ``viser``, and the mimic USD-to-URDF converter
  (``nvidia-srl-usd-to-urdf``) are no longer pulled in by ``all`` because their
  pins conflict with isaacsim's; install them explicitly with ``--extra ov`` /
  ``--extra viser`` / ``--extra mimic`` when not using isaacsim.
