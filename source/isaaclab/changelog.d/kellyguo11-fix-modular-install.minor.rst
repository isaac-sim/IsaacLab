Changed
^^^^^^^

* Changed the installation model of :meth:`~isaaclab.cli.commands.install.command_install`
  from per-submodule selection to a three-tier system. All core submodules
  (``isaaclab``, ``isaaclab_assets``, ``isaaclab_contrib``, ``isaaclab_experimental``,
  ``isaaclab_newton``, ``isaaclab_ov``, ``isaaclab_ovphysx``, ``isaaclab_physx``,
  ``isaaclab_rl``, ``isaaclab_tasks``, ``isaaclab_tasks_experimental``,
  ``isaaclab_visualizers``)
  are now always installed by ``./isaaclab.sh -i``. Optional submodules
  (``mimic``, ``teleop``) and automatic extra feature sets
  (``newton``, ``rl[...]``, ``visualizer[...]``) are installed by ``./isaaclab.sh -i``
  / ``./isaaclab.sh -i all``.
  Optional dependency extras require selectors, so rlinf dependencies are
  installed with ``contrib[rlinf]`` and the ``ovrtx`` / ``ovphysx`` wheels are installed
  with ``ov[ovrtx]``, ``ov[ovphysx]``, or ``ov[all]``. Old per-submodule tokens (e.g.
  ``assets``, ``tasks``, ``physx``) now emit a warning and are skipped gracefully.
  Migrate using the table below:

  +----------------------------------------------+-------------------------------------------+
  | Old command                                  | New command                               |
  +==============================================+===========================================+
  | ``./isaaclab.sh -i assets,tasks,physx``      | ``./isaaclab.sh -i none``                 |
  +----------------------------------------------+-------------------------------------------+
  | ``./isaaclab.sh -i assets,tasks,ov,rl[...]`` | ``./isaaclab.sh -i ov[all],rl[...]``      |
  +----------------------------------------------+-------------------------------------------+
  | ``./isaaclab.sh -i newton,rl[all]``          | unchanged                                 |
  +----------------------------------------------+-------------------------------------------+
  | ``./isaaclab.sh -i mimic,teleop``            | unchanged                                 |
  +----------------------------------------------+-------------------------------------------+
  | ``uv pip install isaaclab[tasks,rl,assets]`` | ``uv pip install isaaclab[all]``          |
  +----------------------------------------------+-------------------------------------------+

* Simplified :mod:`isaaclab` package extras to ``isaacsim`` and ``all``; removed the old
  per-submodule extras (``tasks``, ``rl``, ``assets``, etc.) from ``pip install isaaclab[...]``.
