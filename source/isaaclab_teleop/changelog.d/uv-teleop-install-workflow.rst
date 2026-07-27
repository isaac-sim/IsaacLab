Fixed
^^^^^

* Fixed a spurious warning during ``./isaaclab.sh -i teleop`` by removing the
  ``[tool.isaaclab] pip_upgrade_dependencies`` entry for ``isaacteleop``. The dependency is
  declared by the root ``teleop`` extra rather than by this package, so the targeted upgrade
  could never resolve it from the installed package metadata.
