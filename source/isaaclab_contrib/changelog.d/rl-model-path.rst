Added
^^^^^

* Added ``--rl_model_path`` CLI flag to ``play.py`` for evaluating RL-finetuned checkpoints.
  The base model architecture is loaded via ``--model_path`` and the RL-trained weights
  (``full_weights.pt``) are overlaid from the checkpoint directory.
