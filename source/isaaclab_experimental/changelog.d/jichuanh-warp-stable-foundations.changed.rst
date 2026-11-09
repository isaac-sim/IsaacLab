* Changed the Warp joint penalties ``joint_torques_l2``, ``joint_vel_l1``, ``joint_vel_l2``, ``joint_acc_l2``,
  ``joint_deviation_l1`` and ``joint_pos_limits``, and the termination ``joint_pos_out_of_manual_limit``, to loop over
  the selected joints in the order of ``joint_ids``. A joint selected twice now counts twice, as in the torch terms.
