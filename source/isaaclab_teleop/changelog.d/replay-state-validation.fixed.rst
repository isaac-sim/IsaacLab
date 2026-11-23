* Fixed ``replay_demos.py --validate_states``, which raised a state shape mismatch on its first comparison
  because recorded episode states keep their batch dimension.
