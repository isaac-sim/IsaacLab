Fixed
^^^^^

* Fixed :func:`~isaaclab.cli.commands.install.command_install` leaving extras-qualified
  prebundle mirrors (``<pkg>[extras]/<pkg>-<version>-*/<pkg>``) pointing at the previously
  shipped file list, which broke Isaac Sim extensions whenever a package was repointed to a
  version that renamed or dropped files.
