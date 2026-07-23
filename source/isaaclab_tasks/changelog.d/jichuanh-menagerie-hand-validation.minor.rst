Changed
^^^^^^^

* **Breaking:** Changed the ``Isaac-Reorient-Cube-Shadow-Direct`` and
  ``Isaac-Reorient-Cube-Allegro-Direct`` tasks to use the Mujoco Menagerie hand assets.
  Joint and body names now follow the official robot naming: use ``rh_FFJ3`` instead of
  ``robot0_FFJ3`` for the Shadow Hand, and ``ffj0``/``ff_distal`` instead of
  ``index_joint_0``/``index_link_3`` for the Allegro Hand.
