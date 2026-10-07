* Added the H2 + Sharpa tasks ``IsaacContrib-Pick-And-Place-Apple-H2-Sharpa`` and
  ``IsaacContrib-Pack-AGX-Orin-H2-Sharpa``, their ``-Eval`` registrations, and GR00T N1.7 PPO configurations.
* Added shared H2 joint ordering and calibrated wrist cameras in ``isaaclab_tasks.contrib.h2_sharpa``.
  Actions controlled the 58 policy joints directly; reset targets held the remaining body joints.
* Added task-owned phase tracking through stateful termination terms, with sparse rewards reading
  phase transitions independently of reward weights.
* Added attribution and license documentation for the Lightwheel CC BY-NC 4.0 scene assets.
