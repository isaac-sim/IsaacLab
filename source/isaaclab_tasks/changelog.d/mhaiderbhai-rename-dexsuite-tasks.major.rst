Changed
^^^^^^^

* **Breaking:** Renamed the Dexsuite Kuka-Allegro environment IDs to drop the ``Dexsuite`` prefix and
  the ``-v0`` version suffix. Update ``gym.make`` / ``--task`` calls:

  * ``Isaac-Dexsuite-Kuka-Allegro-Reorient-v0`` → ``Isaac-Reorient-KukaAllegro``.
  * ``Isaac-Dexsuite-Kuka-Allegro-Reorient-Play-v0`` → ``Isaac-Reorient-KukaAllegro-Play``.
  * ``Isaac-Dexsuite-Kuka-Allegro-Lift-v0`` → ``Isaac-Lift-KukaAllegro``.
  * ``Isaac-Dexsuite-Kuka-Allegro-Lift-Play-v0`` → ``Isaac-Lift-KukaAllegro-Play``.
