* Added the ``isaaclab_neural`` package. Its ``NeuralManager`` physics backend, ``NeuralRenderer`` and virtual
  ``Articulation`` let a learned world model supply the joint state and camera frames of a task instead of a physics
  engine, through the task's ``physics=`` and ``renderer=`` presets.
* Added the ``WorldModel`` interface for models that predict one or more camera outputs, such as RGB or depth, of
  one or more named camera views, each shown by its own camera, reset records that the virtual ``Articulation`` starts episodes from, and ``at_chunk_boundary`` for
  terminations that fire where environments can reset.
