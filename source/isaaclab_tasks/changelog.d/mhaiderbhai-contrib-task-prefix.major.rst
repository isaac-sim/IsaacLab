Changed
^^^^^^^

* **Breaking:** Renamed all contributed environment IDs (under ``isaaclab_tasks.contrib``) to use the
  ``IsaacContrib-`` prefix instead of ``Isaac-`` and dropped the trailing ``-v0`` version suffix, so
  they follow the same naming convention as the refactored core tasks. The Python API
  (config classes, ``mdp`` modules, agent configs) is unchanged. Update ``gym.make`` / ``--task``
  calls accordingly, for example ``Isaac-Velocity-Rough-AnymalC-v0`` →
  ``IsaacContrib-Velocity-Rough-AnymalC`` and ``Isaac-Cartpole-Showcase-Direct`` →
  ``IsaacContrib-Cartpole-Showcase-Direct``.
