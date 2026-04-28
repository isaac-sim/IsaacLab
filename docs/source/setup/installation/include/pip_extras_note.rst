.. note::

   The bare ``isaaclab`` install ships only the core extension. To run
   the bundled training scripts under ``scripts/reinforcement_learning/``
   you must install with the ``[all]`` extras (or the per-framework
   extras ``[skrl]`` / ``[sb3]`` / ``[rsl-rl]``); otherwise commands such
   as ``python scripts/reinforcement_learning/skrl/train.py ...`` fail
   at import time with ``ModuleNotFoundError: No module named 'skrl'``.
