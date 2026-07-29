Added
^^^^^

* Added ``joint_pos``, ``diffik``, and ``newton_ik`` action presets to ``Isaac-Reach-Franka``, with ``joint_pos`` as
  the default. Select the action independently from the ``isaacsim_physx``, ``ovphysx``, or ``newton_mjwarp``
  physics preset; ``diffik`` uses the same action configuration across backends, while ``newton_ik`` requires
  Newton.

Changed
^^^^^^^

* **Breaking:** Renamed the canonical Franka Reach configuration module from ``joint_pos_env_cfg`` to
  ``franka_reach_env_cfg`` and the OSC module from ``osc_env_cfg`` to ``franka_reach_osc_env_cfg``.
* **Breaking:** Removed ``Isaac-Reach-Franka-Newton-IK-Rel`` and
  ``Isaac-Reach-Franka-Newton-IK-Rel-v0``. Use
  ``Isaac-Reach-Franka presets=newton_mjwarp,newton_ik`` instead.
* **Breaking:** Removed ``IsaacContrib-Reach-Franka-IK-Rel``. Use
  ``Isaac-Reach-Franka physics=isaacsim_physx presets=diffik`` instead.
* Changed Reach to use one linear position term, one orientation term, a combined pose-success termination,
  and shared action-rate, action-magnitude, and arm-velocity penalties. To retain the previous MDP, restore
  ``end_effector_position_tracking_fine_grained`` and remove the success reward, success termination, and
  action-magnitude term.
* Changed Reach to use a common static table across physics backends and the MuJoCo Menagerie Franka, with
  angular-drive gains and solver velocity limits normalized across backends. To retain the previous Franka, set
  ``scene.robot`` to ``FRANKA_PANDA_CFG``.
* Changed Reach to use a 120 Hz simulation rate with four control-decimation steps. The Newton MJWarp preset
  uses two solver substeps and a data-update interval of two. To retain the previous timing, use a 60 Hz
  simulation rate, two control-decimation steps, and one Newton solver substep.
* Changed Franka Reach IK controllers to scale translation by ``0.05`` m and rotation by ``0.5`` rad, while
  retaining the joint-position controller's ``0.5`` rad scale. To retain the previous IK scale, set all six
  task-space coordinates to ``0.5``.
