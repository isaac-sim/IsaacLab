Changed
^^^^^^^

* Changed the default ``model_path`` in ``isaaclab_ppo_gr00t_assemble_trocar.yaml`` from an absolute
  container-specific path to ``.pretrained_checkpoints/rlinf/Assemble_Trocar``, the location that
  ``scripts/reinforcement_learning/rlinf/setup_rlinf.py`` downloads the checkpoint to. Pass
  ``--model_path`` to use a checkpoint stored elsewhere.
