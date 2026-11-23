* Added the H2 + Sharpa tasks ``IsaacContrib-Pick-And-Place-Apple-H2-Sharpa`` and
  ``IsaacContrib-Pack-AGX-Orin-H2-Sharpa``, their ``-Eval`` registrations, and task-local GR00T N1.7 PPO agent configurations
  ``rlinf_gr00t_pick_and_place_apple_ppo_cfg`` and ``rlinf_gr00t_pack_agx_ppo_cfg``.
* Added shared H2 joint ordering and calibrated wrist cameras in the shared ``isaaclab_tasks.contrib.h2_sharpa.tabletop_env_cfg`` module.
  Newton's joint impedance controller tracks the 14 arm position targets with gravity compensation,
  and the joint-position action controls the 44 hand joints.
  Camera observations use the core RGB term with a fitted task-specific color response.
* Added task-owned phase tracking through stateful termination terms, with sparse rewards reading
  phase transitions independently of reward weights.
* Added attribution and license documentation for the Lightwheel CC BY-NC 4.0 scene assets.
