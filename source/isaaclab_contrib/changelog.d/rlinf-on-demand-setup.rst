Added
^^^^^

* Added GR00T N1.7 support to the RLinf extension. When ``actor.model.model_type`` is
  ``gr00t_n1d7`` the extension defers model construction to RLinf's native loader, imports the task's
  ``env.train.isaaclab.modality_config_module`` to register the embodiment's modalities, and uses
  that module's ``convert_gr00t_to_isaaclab_action`` when it defines one.

Fixed
^^^^^

* Fixed the RLinf extension applying stale actions to a freshly reset sub-environment. ``ManagerBasedRLEnv.step`` resets a terminated sub-environment before returning, so the rest of the action chunk — predicted from the episode that just ended — was driving the new one. Tasks that set ``env.train.isaaclab.hold_pose_on_midchunk_reset`` now hold the post-reset pose until the next chunk boundary. The stale first camera frame after a reset is handled by the task configurations through ``num_rerenders_on_reset`` rather than by the extension.
* Fixed the RLinf extension rejecting a direct ``full_weights.pt`` path in ``rl_model_path``. It now
  accepts either the file or the ``global_step_<N>`` directory holding it, as its error message
  already stated.
* Fixed the RLinf extension against RLinf 0.3, which moved the N1.5 model under
  ``gr00t_n1d5`` and split the action-converter registry per GR00T generation.
