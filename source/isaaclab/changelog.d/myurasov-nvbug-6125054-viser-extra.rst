Fixed
^^^^^

* Fixed ``pip install isaaclab[isaacsim,all]==3.0.0`` failing with
  ``No solution found`` (UV) or ``error: resolution-too-deep`` (pip) when
  resolving against ``isaacsim==6.0.0.0``. ``viser>=1.0.16`` was a base
  dependency of the built ``isaaclab`` wheel and transitively requires
  ``websockets>=13.1``, but ``isaacsim-kernel==6.0.0.0`` pins
  ``websockets==12.0``. Moved ``viser`` to an opt-in ``viser`` extra in
  ``tools/wheel_builder/res/python_packages.toml`` so the base wheel is
  installable alongside ``isaacsim==6.0.0.0``. Users who want the Viser
  visualizer can request it explicitly with ``isaaclab[viser]``.
