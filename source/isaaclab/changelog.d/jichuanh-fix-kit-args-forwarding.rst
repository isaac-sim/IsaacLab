Fixed
^^^^^

* Fixed the multi-GPU training launcher (``train_multigpu``) failing with argparse error
  "argument --kit_args: expected one argument" on every rank when a single Kit argument was
  passed in the space-separated form (e.g. ``--kit_args "--ext-folder=/path/to/ext"``). The
  launcher now forwards it to the training script as a single ``--kit_args=<value>`` token.
