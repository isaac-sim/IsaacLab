Fixed
^^^^^

* Fixed checkpoint resolution so omitting a checkpoint selects the latest available checkpoint.
  ``rl_games`` and ``sb3`` play now load available short-run checkpoints by default, while
  explicit checkpoint requests remain strict. Numbered checkpoint filenames are now sorted
  naturally so epoch 10 is selected after epoch 9.
