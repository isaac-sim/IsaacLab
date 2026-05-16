Fixed
^^^^^

* Fixed ``calculate_rotation_error`` in
  ``source/isaaclab/test/controllers/test_pink_ik.py`` composing rotation matrices
  with element-wise ``*`` instead of matrix multiplication ``@`` — a latent bug
  from `isaac-sim/IsaacLab#3149
  <https://github.com/isaac-sim/IsaacLab/pull/3149>`_ that surfaced as NaN after
  `isaac-sim/IsaacLab#5609
  <https://github.com/isaac-sim/IsaacLab/pull/5609>`_ added the unit-norm guard to
  ``quat_from_matrix``.
* Made ``test_pink_ik`` deterministic by seeding the env (``env_cfg.seed = 42``)
  in ``create_test_env``.
* Loosened the G1 Pink IK rotation tolerance from ``0.030`` rad to ``0.100`` rad
  in ``pink_ik_g1_test_configs.json`` to accommodate G1's intentionally smooth IK
  tuning (slower-converging than GR1T2). GR1T2 tolerance unchanged at ``0.020`` rad.
