Deprecated
^^^^^^^^^^

* Deprecated implicit material, collider-offset, and gravity event selection in
  ``isaaclab.envs.mdp``. Existing terms retained their signatures and behavior;
  new configurations should select backend terms through ``EventTermCfg.func``.
  Shared asset-property randomization remained in core.
