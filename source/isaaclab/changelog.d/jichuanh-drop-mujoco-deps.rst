Removed
^^^^^^^

* Removed explicit ``mujoco`` and ``mujoco-warp`` dependencies from
  :mod:`isaaclab`. These packages are not used by ``isaaclab`` core and are
  now resolved transitively through Newton's ``[sim]`` extra in
  :mod:`isaaclab_newton`. Users installing only the PhysX or Kit backends no
  longer pull in MuJoCo.
