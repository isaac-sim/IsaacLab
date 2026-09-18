Fixed
^^^^^

* Registered the installed Newton USD schema alongside the PhysX schemas before
  OVStage population when ovphysx exposes its discovery helper, preserving
  authored ``newton:*`` attributes when that schema is installed. Scenes without
  those attributes do not require the optional package.
