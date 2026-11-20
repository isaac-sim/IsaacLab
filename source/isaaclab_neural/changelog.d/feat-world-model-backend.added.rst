* Added the ``isaaclab_neural`` package. Its ``WorldModelManager`` physics backend, ``WorldModelRenderer`` and virtual
  ``Articulation`` let a learned world model supply the joint state and camera frames of a task instead of a physics
  engine, through the task's ``physics=`` and ``renderer=`` presets.
* Added the ``WorldModel`` interface, the ``reset_from_records`` reset event and ``at_chunk_boundary`` for terminations
  that fire where environments can reset.
