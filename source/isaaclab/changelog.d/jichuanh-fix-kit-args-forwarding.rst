Fixed
^^^^^

* Fixed passing a single Kit argument in the space-separated form (e.g.
  ``--kit_args "--ext-folder=/path/to/ext"``) failing with the argparse error
  "argument --kit_args: expected one argument" on every entry point, including all ranks of
  the multi-GPU training launcher. :meth:`~isaaclab.app.AppLauncher.add_app_launcher_args`
  now normalizes such pairs into single ``--kit_args=<value>`` tokens before parsing.
