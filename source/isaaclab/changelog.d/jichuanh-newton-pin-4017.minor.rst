Changed
^^^^^^^

* Changed the pinned Newton build to the commit merging
  `newton#4017 <https://github.com/newton-physics/newton/pull/4017>`_, which makes
  ``ArticulationView`` generic over custom frequencies. Reading a MuJoCo actuator attribute such as
  ``mujoco.actuator_trntype`` through an articulation view previously raised *"has custom frequency
  'mujoco:actuator' which is not supported by ArticulationView"*.

Fixed
^^^^^

* Fixed ``isaaclab.sh --install`` aborting whenever the pinned Newton commit changes. The installer
  uninstalled Newton before reinstalling it, but Isaac Sim's ``isaacsim.pip.newton`` prebundle
  symlinks into the installed tree, so removing the distribution left every link dangling and the
  install failed its own prebundle check (nvbugs 6343978). The pinned build is now force-reinstalled
  over the existing tree instead.
