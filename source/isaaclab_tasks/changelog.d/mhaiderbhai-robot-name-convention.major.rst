Changed
^^^^^^^

* **Breaking:** Multi-word robot names in Gym environment IDs are now written in CamelCase, since
  ``-`` is reserved for separating task-name aspects (task / object / robot / workflow / variant).
  This applies across the core, contributed, and experimental (Warp) tasks. Update ``gym.make`` /
  ``--task`` calls; the surrounding ID structure and any ``-v0`` / ``-Warp-v0`` suffix are otherwise
  unchanged:

  * ``Anymal-B`` → ``AnymalB``, ``Anymal-C`` → ``AnymalC``, ``Anymal-D`` → ``AnymalD``.
  * ``Unitree-A1`` → ``UnitreeA1``, ``Unitree-Go1`` → ``UnitreeGo1``, ``Unitree-Go2`` → ``UnitreeGo2``.
  * ``Kuka-Allegro`` → ``KukaAllegro``.
  * ``OpenArm-Bi`` → ``OpenArmBi``.

  For example ``Isaac-Velocity-Rough-Anymal-C-v0`` → ``Isaac-Velocity-Rough-AnymalC-v0`` and
  ``Isaac-Reach-OpenArm-Bi-v0`` → ``Isaac-Reach-OpenArmBi-v0``.
