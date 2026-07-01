Fixed
^^^^^

* Fixed ``./isaaclab.sh -i`` (PhysX path) to install the Isaac Sim requirement
  pinned in the root ``pyproject.toml`` ``isaacsim`` extra, instead of a
  hard-coded ``isaacsim[all]>=6.0.0``. The install CLI now shares the single
  source of truth used by the docs, CI, and ``uv run --extra isaacsim``.
