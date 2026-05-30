Added
^^^^^

* Added ``kitless`` pip extra for kit-less installs (git Newton and RL frameworks).

Changed
^^^^^^^

* **Breaking:** The ``all`` pip extra now installs RL frameworks only. Use
  ``isaaclab[kitless]`` for kit-less pip installs or ``isaaclab[isaacsim,all]`` for
  the full Isaac Sim workflow. Do not combine ``kitless`` with ``isaacsim``.
* Updated installation, quickstart, and pip documentation to match ``./isaaclab.sh -i``
  install tokens, pip wheel extras, and kit-less setup paths.
* Regenerated the environments overview table from task configuration metadata.
