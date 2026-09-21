Added
^^^^^

* Added ``isaaclab list_envs`` to list registered environments and optional presets. From a downstream project,
  the command automatically limits its output to tasks declared by the nearest ``pyproject.toml``; pass ``--all``
  to list every installed task or ``--keyword`` to select task ids explicitly.
* Added a ``Blank`` initial-content option to the external project generator. Blank projects contain packaging,
  task discovery, tests, and development tooling without the cart-pole example files.
* Added a package-relative asset directory and path constant to external projects so project-owned USD files can be
  referenced consistently from editable checkouts and installed wheels.
* Added an opt-in ``--non-interactive`` template-generator mode with validated arguments for project metadata, initial
  content, workflows, RL libraries, algorithms, and the optional Isaac Sim UI extension.

Fixed
^^^^^

* Fixed external projects to forward the complete set of optional extras from the active Isaac Lab package, including
  visualizer extras such as ``rerun`` and ``viser`` and the aggregate ``all`` extra.
