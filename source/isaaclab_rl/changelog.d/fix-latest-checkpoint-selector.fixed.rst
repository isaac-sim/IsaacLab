* Fixed ``--checkpoint latest`` loading the best-reward checkpoint instead of the newest one for skrl runs and for
  rl_games runs whose config name sorts after ``last_``, such as ``shadow_hand`` or ``reach_franka``. The selector
  now prefers the libraries' periodic checkpoints and falls back to the previous behavior when there are none.
