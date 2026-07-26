Fixed
^^^^^

* Bounded the ``psutil`` dependency to ``>=5.9.8,<6`` and capped ``click`` at ``8.1.7`` in the
  ``rl-games`` extra. Both packages are pinned exactly by ``isaacsim-kernel`` (``psutil==5.9.8``,
  ``click==8.1.7``), but the installer resolves Isaac Sim and the Isaac Lab dependencies in
  separate pip invocations, so the unbounded requirements let the later resolve install
  ``psutil`` 7.x and ``click`` 8.4.x over the pinned copies.
