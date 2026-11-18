* Added the ``isaaclab_neural`` package. Its ``WorldModelManager`` physics backend, ``WorldModelRenderer`` and virtual
  ``Articulation`` let a learned world model supply the joint state and camera frames of a task instead of a physics
  engine, through the task's ``physics=`` and ``renderer=`` presets.
* Added the ``WorldModel`` interface, the ``reset_from_records`` reset event and ``at_chunk_boundary`` for terminations
  that fire where environments can reset.
* Added ``Cosmos3Model`` for ``unified_v1`` Cosmos3 checkpoints, the ``cosmos3`` extra and a replay script. The extra
  overrides the Cosmos Framework's ``transformers<5`` pin, and loading lifts the read-only flag that OmegaConf 2.4
  puts on the saved config while the Framework rewrites it.
